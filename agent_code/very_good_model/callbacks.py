from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from torch import multiprocessing

is_fork = multiprocessing.get_start_method() == "fork"


class VeryGoodModelCommon(nn.Module):
    def __init__(
        self,
        in_channels: int,
        board_height: int,
        board_width: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.board_height = board_height
        self.board_width = board_width

        # ---------------------------------------------------------
        # Spatial feature extractor
        # ---------------------------------------------------------

        self.cnn = nn.Sequential(
            nn.Conv2d( # 17x17 -> 15x15
                in_channels,
                64,
                kernel_size=3,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d( # 15x15 -> 13x13
                64,
                64,
                kernel_size=3,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d( # 13x13 -> 11x11
                64,
                64,
                kernel_size=3,
                padding=0,
            ),
            nn.ReLU(),
        )

        # Calculate the size after the CNN automatically.
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                in_channels,
                board_height,
                board_width,
            )

            cnn_output = self.cnn(dummy)

            self.cnn_output_size = cnn_output.flatten(1).shape[1]

        # ---------------------------------------------------------
        # Shared fully-connected representation
        # ---------------------------------------------------------

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

    def forward(self, obs):

        x = self.cnn(obs)

        x = x.flatten(start_dim=1)

        x = self.fc(x)

        return x


class VeryGoodModel(nn.Module):

    def __init__(
        self,
        in_channels: int,
        board_height: int,
        board_width: int,
        num_actions: int,
    ):
        super().__init__()

        self.common_module = VeryGoodModelCommon(
            in_channels=in_channels,
            board_height=board_height,
            board_width=board_width,
        )

        self.num_actions = num_actions

        # ---------------------------------------------------------
        # Actor
        # ---------------------------------------------------------

        self.policy_head = nn.Linear(
            256,
            num_actions,
        )

        # ---------------------------------------------------------
        # Critic
        # ---------------------------------------------------------

        self.value_head = nn.Linear(
            256,
            1,
        )

    def forward(self, obs):
        # Use the shared common module to produce features, then
        # apply the actor and critic heads.
        features = self.common_module(obs)

        logits = self.policy_head(features)

        value = self.value_head(features)

        return logits, value

    @torch.no_grad()
    def get_action(self, obs, valid_mask=None, deterministic=False, logger=None):
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        logits, value = self.forward(obs)

        if valid_mask is not None:
            # Mask out invalid actions by setting their logits to a very low value.
            # This ensures that the softmax will assign them near-zero probability.
            invalid_mask = ~valid_mask
            if logits.dim() == 2:
                invalid_mask = invalid_mask.unsqueeze(0)
            logits[invalid_mask] = -1e9

        distribution = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy().mean()

        if logger:
            logger.debug(
                f"Logits: {logits} probs: {torch.softmax(logits, dim=-1)} Action: {action} Log Prob: {log_prob} Value: {value} Entropy: {entropy}"
            )

        return action, log_prob, entropy, value.squeeze(-1)


CHANNELS = 11  # walls, boxes, self, enemies, bombs, bomb_timer, explosion, explosion_timer, coins, danger, walkable

HEIGHT = 17
WIDTH = 17

NUM_ACTIONS = 6

ACTIONS = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
    4: "BOMB",
    5: "WAIT",
}

ACTIONS_INV = {v: k for k, v in ACTIONS.items()}

BOMB_TIMER = 4
BOMB_DURATION = 2
BOMB_RANGE = 3

MODEL_FILE = Path(__file__).with_name("very_good_model.pt")


def get_features(game_state):
    if game_state is None:
        return torch.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)

    field = torch.tensor(game_state["field"], dtype=torch.float32)
    explosion_map = torch.tensor(game_state["explosion_map"], dtype=torch.float32)
    self_info = game_state["self"]
    bombs = game_state.get("bombs", [])
    others = game_state.get("others", [])
    coins = game_state.get("coins", [])

    # Create a tensor for the channels
    channels = torch.zeros((CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)

    # Channel 0: walls
    channels[0] = (field == -1).float()

    # Channel 1: boxes
    channels[1] = (field == 1).float()

    # Channel 2: self
    x, y = self_info[3]
    channels[2, x, y] = 1.0

    # Channel 3: enemies
    for other in others:
        ox, oy = other[3]
        channels[3, ox, oy] = 1.0

    # Channel 4, 5: bomb timer (normalized)
    for bomb_pos, timer in bombs:
        bx, by = bomb_pos
        channels[4, bx, by] = 1.0
        channels[5, bx, by] = timer / BOMB_TIMER

    # Channel 6: explosion map
    channels[6] = (explosion_map > 0).float()

    # Channel 7: explosion timer (normalized)
    channels[7] = (explosion_map > 0).float() * (explosion_map / BOMB_DURATION)

    # Channel 8: coins
    for coin_x, coin_y in coins:
        channels[8, coin_x, coin_y] = 1.0

    # Channel 9: danger (if in explosion range, stops at walls)
    danger_map = torch.zeros((HEIGHT, WIDTH), dtype=torch.float32)
    for bomb_pos, timer in bombs:
        bx, by = bomb_pos
        danger = (BOMB_TIMER - timer) / BOMB_TIMER

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            for step in range(1, BOMB_RANGE + 1):
                nx = bx + dx * step
                ny = by + dy * step

                if not (0 <= nx < HEIGHT and 0 <= ny < WIDTH):
                    break

                tile = field[nx, ny].item()
                if tile == -1:
                    break

                danger_map[nx, ny] = max(float(danger_map[nx, ny]), float(danger))

        danger_map[bx, by] = max(float(danger_map[bx, by]), float(danger))

    channels[9] = danger_map

    # Channel 10: walkable (empty space)
    channels[10] = (field == 0).float()

    # Return a batch of size 1 to be consistent with training/evaluation code.
    return channels


def valid_actions(game_state):
    if game_state is None:
        return []

    if "field" not in game_state or "self" not in game_state:
        return []

    # `self` position is stored as (x, y) -> (col, row)
    x, y = game_state["self"][3]

    actions = []

    # LEFT: (x-1, y)
    if (
        x - 1 >= 0
        and game_state["field"][x - 1, y] == 0
        and (x - 1, y) not in [bomb[0] for bomb in game_state.get("bombs", [])]
    ):
        actions.append("LEFT")

    # RIGHT: (x+1, y)
    if (
        x + 1 < game_state["field"].shape[1]
        and game_state["field"][x + 1, y] == 0
        and (x + 1, y) not in [bomb[0] for bomb in game_state.get("bombs", [])]
    ):
        actions.append("RIGHT")

    # UP: (x, y-1)
    if (
        y - 1 >= 0
        and game_state["field"][x, y - 1] == 0
        and (x, y - 1) not in [bomb[0] for bomb in game_state.get("bombs", [])]
    ):
        actions.append("UP")

    # DOWN: (x, y+1)
    if (
        y + 1 < game_state["field"].shape[0]
        and game_state["field"][x, y + 1] == 0
        and (x, y + 1) not in [bomb[0] for bomb in game_state.get("bombs", [])]
    ):
        actions.append("DOWN")

    # Can place bomb
    if game_state["self"][2]:
        actions.append("BOMB")

    # Always allow WAIT
    actions.append("WAIT")

    return actions


def valid_actions_mask(game_state):
    mask = torch.zeros(len(ACTIONS), dtype=torch.bool)

    valid = valid_actions(game_state)
    for action in valid:
        mask[ACTIONS_INV[action]] = True

    return mask


def setup(self):
    if self.train:
        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )
    else:
        self.device = torch.device("cpu")
    self.model = VeryGoodModel(
        in_channels=CHANNELS,
        board_height=HEIGHT,
        board_width=WIDTH,
        num_actions=len(ACTIONS),
    ).to(self.device)

    if MODEL_FILE.exists():
        try:
            self.logger.info("Loading model from disk.")
            checkpoint = torch.load(MODEL_FILE, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if self.train and "optimizer_state_dict" in checkpoint:
                    self.optimizer_state_dict = checkpoint["optimizer_state_dict"]
                if self.train and "scheduler_state_dict" in checkpoint:
                    self.scheduler_state_dict = checkpoint["scheduler_state_dict"]
                if self.train and "step" in checkpoint:
                    self.step = checkpoint["step"]
            else:
                self.model.load_state_dict(checkpoint)
        except Exception:
            self.logger.exception(
                "Failed to load model checkpoint, starting from scratch."
            )
    else:
        self.logger.info("Initializing model from scratch.")

    self.model.eval()


def act(self, game_state):
    features = get_features(game_state).to(self.device)
    action_mask = valid_actions_mask(game_state).to(self.device)
    action, log_prob, entropy, value = self.model.get_action(
        features, valid_mask=action_mask, logger=self.logger, deterministic=not self.train
    )
    if self.train:
        self.last_action = action.item()
        self.last_log_prob = log_prob
        self.last_entropy = entropy
        self.last_value = value
        self.logger.debug(f"mask: {action_mask}")
        self.last_action_mask = action_mask
    return ACTIONS[action.item()]
