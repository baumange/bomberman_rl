import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import events as e

from .callbacks import ACTIONS, get_features, MODEL_FILE, valid_actions_mask
from tensordict import TensorDict
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import (
    ActorValueOperator,
    MaskedCategorical,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

lr = 1e-5
max_grad_norm = 1.0

frames_per_batch = 400
# For a complete training, bring the number of frames up to 1M
# total_frames = 40_000
rounds = 10000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.98
lmbda = 0.94
entropy_eps = 6e-2


def setup_training(self):
    # ---------------------------------------------------------
    # Shared CNN
    # state -> features
    # ---------------------------------------------------------
    self.common_module = TensorDictModule(
        self.model.common_module,
        in_keys=["state"],
        out_keys=["features"],
    )

    # ---------------------------------------------------------
    # Policy head
    # features -> logits
    # ---------------------------------------------------------
    self.actor_module = TensorDictModule(
        self.model.policy_head,
        in_keys=["features"],
        out_keys=["logits"],
    )

    # ---------------------------------------------------------
    # Value head
    # features -> state_value
    # ---------------------------------------------------------
    self.value_module = ValueOperator(
        self.model.value_head,
        in_keys=["features"],
    )

    # ---------------------------------------------------------
    # Shared actor/value structure
    # ---------------------------------------------------------
    self.model_operator = ActorValueOperator(
        common_operator=self.common_module,
        policy_operator=self.actor_module,
        value_operator=self.value_module,
    )

    # ---------------------------------------------------------
    # PPO actor
    # ---------------------------------------------------------
    self.actor = ProbabilisticActor(
        module=ProbabilisticTensorDictSequential(
            self.model_operator,
            self.actor_module,
        ),
        in_keys={"logits": "logits", "mask": "action_mask"},
        # spec=self.action_spec,
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )

    # ---------------------------------------------------------
    # GAE
    # ---------------------------------------------------------
    self.advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        # value_network=self.model_operator.get_value_operator(),
        value_network=None,
        average_gae=True,
    )

    # ---------------------------------------------------------
    # PPO loss
    #
    # IMPORTANT:
    # Pass value_module directly rather than
    # model_operator.get_value_operator()
    # ---------------------------------------------------------
    self.loss_module = ClipPPOLoss(
        actor_network=self.actor,
        critic_network=self.value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coeff=entropy_eps,
        critic_coeff=1.0,
        loss_critic_type="smooth_l1",
    )

    # ---------------------------------------------------------
    # Replay buffer
    # ---------------------------------------------------------
    self.replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=frames_per_batch,
        ),
        sampler=SamplerWithoutReplacement(),
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    self.optim = torch.optim.Adam(
        self.loss_module.parameters(),
        lr=lr,
    )
    if hasattr(self, "optimizer_state_dict"):
        self.optim.load_state_dict(self.optimizer_state_dict)

    # One scheduler step per collected batch/update
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optim,
        T_max=rounds,
        eta_min=0.0,
    )
    if hasattr(self, "scheduler_state_dict"):
        self.scheduler.load_state_dict(self.scheduler_state_dict)
    if hasattr(self, "step"):
        self.scheduler.last_epoch = self.step

    self.trajectory = []

    self.writer = SummaryWriter(log_dir="runs/ppo_training")

    # Additional training bookkeeping
    # map action strings to indices
    self.action_map = {v: k for k, v in ACTIONS.items()}
    self.gamma = gamma
    self.entropy_coeff = entropy_eps
    self.max_grad_norm = max_grad_norm
    self.model.train()


def reward_from_events(self, events):
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.CRATE_DESTROYED: 5,
        e.KILLED_OPPONENT: 100,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -100,
        e.INVALID_ACTION: -5,
        e.WAITED: -1,
        e.BOMB_DROPPED: 0,
        e.SURVIVED_ROUND: 0,
    }
    reward = sum(game_rewards.get(event, 0) for event in events)
    self.logger.debug(
        f'Awarded {reward} for events {events}, action {ACTIONS[self.last_action] if hasattr(self, "last_action") else "N/A"}'
    )
    return reward


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Collect transitions during the episode."""
    # ignore initial call without state
    if old_game_state is None:
        return

    reward = reward_from_events(self, events)

    # map action string to index
    if self_action not in self.action_map:
        return

    # store features (without moving to device yet)
    state_feat = get_features(old_game_state).to(self.device)
    next_feat = None if new_game_state is None else get_features(new_game_state).to(self.device)

    td = TensorDict(
        {
            "state": state_feat,
            "action": torch.tensor(self.last_action, device=self.device),
            "action_mask": self.last_action_mask,
            "action_log_prob": torch.tensor([self.last_log_prob], device=self.device),
            "action_entropy": torch.tensor([self.last_entropy], device=self.device),
            "state_value": torch.tensor([self.last_value], device=self.device),
            "next": {
                "state": next_feat,
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([new_game_state is None], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([new_game_state is None], dtype=torch.bool, device=self.device),
            },
        }
    )
    if len(self.trajectory) > 0:
        self.trajectory[-1]["next", "state_value"] = td["state_value"]

    self.trajectory.append(td)


def end_of_round(self, last_game_state, last_action, events):
    """At the end of an episode, compute returns and perform a PPO-style update (simple actor-critic)."""
    # add final transition
    final_reward = reward_from_events(self, events)

    td = TensorDict(
        {
            "state": self.trajectory[-1]["next"]["state"],
            "action": torch.tensor(self.last_action, device=self.device),
            "action_mask": self.last_action_mask,
            "action_log_prob": torch.tensor([self.last_log_prob], device=self.device),
            "action_entropy": torch.tensor([self.last_entropy], device=self.device),
            "state_value": torch.tensor([self.last_value], device=self.device),
            "next": {
                "state": get_features(last_game_state).to(self.device),
                "state_value": self.model.get_action(
                    get_features(last_game_state).to(self.device),
                    valid_mask=valid_actions_mask(last_game_state).to(self.device),
                )[3],
                "reward": torch.tensor([final_reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([True], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([True], dtype=torch.bool, device=self.device),
            },
        }
    )

    if len(self.trajectory) > 0:
        self.trajectory[-1]["next", "state_value"] = td["state_value"]

    self.trajectory.append(td)

    batch = torch.stack(self.trajectory).to(self.device)

    self.logger.info(
        f"End of round: collected {len(batch)} transitions, final reward: {final_reward}"
    )

    loss_logs = []

    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        self.advantage_module(batch)
        data_view = batch.reshape(-1)
        self.replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = self.replay_buffer.sample(sub_batch_size)
            loss_vals = self.loss_module(subdata.to(self.device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_logs.append(loss_value.item())

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_grad_norm)
            self.optim.step()
            self.optim.zero_grad()

    self.scheduler.step()
    self.trajectory.clear()

    self.writer.add_scalar(
        "Training/Reward",
        batch["next", "reward"].mean().item(),
        self.scheduler.last_epoch,
    )
    self.writer.add_scalar(
        "Training/Loss", sum(loss_logs) / len(loss_logs), self.scheduler.last_epoch
    )
    self.writer.add_scalar(
        "Training/Learning_Rate",
        self.optim.param_groups[0]["lr"],
        self.scheduler.last_epoch,
    )
    self.writer.add_scalar(
        "Training/Entropy",
        batch["action_entropy"].mean().item(),
        self.scheduler.last_epoch,
    )
    self.writer.add_graph(self.model, batch["state"])

    self.logger.info(
        f"Training update: reward: {batch['next', 'reward'].mean().item():.4f}, lr: {self.optim.param_groups[0]['lr']:.6f} loss: {sum(loss_logs) / len(loss_logs):.4f}"
    )

    loss_logs.clear()

    torch.save(
        {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.scheduler.last_epoch,
        },
        MODEL_FILE,
    )


#    if last_game_state is not None and last_action in self.action_map:
#        self.episode_buffer.append((get_features(last_game_state), self.action_map[last_action], final_reward, None))
#
#    if len(self.episode_buffer) == 0:
#        return
#
#    # Build tensors
#    states = torch.cat([t[0] for t in self.episode_buffer], dim=0).to(self.optim.param_groups[0]['params'][0].device)
#    actions = torch.tensor([t[1] for t in self.episode_buffer], dtype=torch.long, device=states.device)
#    rewards = [t[2] for t in self.episode_buffer]
#
#    # compute discounted returns
#    returns = []
#    R = 0.0
#    for r in reversed(rewards):
#        R = r + self.gamma * R
#        returns.insert(0, R)
#    returns = torch.tensor(returns, dtype=torch.float32, device=states.device)
#
#    # forward pass
#    logits, values = self.model(states)
#    values = values.squeeze(-1)
#
#    dist = Categorical(logits=logits)
#    log_probs = dist.log_prob(actions)
#    entropy = dist.entropy().mean()
#
#    advantages = returns - values.detach()
#
#    actor_loss = -(log_probs * advantages).mean()
#    critic_loss = F.mse_loss(values, returns)
#
#    loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy
#
#    # optimize
#    self.optim.zero_grad()
#    loss.backward()
#    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
#    self.optim.step()
#    try:
#        self.scheduler.step()
#    except Exception:
#        pass
#
#    # save model
#    torch.save({
#        'model_state_dict': self.model.state_dict(),
#        'optimizer_state_dict': self.optim.state_dict(),
#    }, MODEL_FILE)
#
#    self.logger.info(f'Training update finished: loss={loss.item():.4f}, actor={actor_loss.item():.4f}, critic={critic_loss.item():.4f}')
#
#    # clear buffer
#    self.episode_buffer.clear()
