# -*- coding: utf-8 -*-

# Standard Library Imports
import random
import math
import time
from enum import Enum, auto
import io  # Keep for potential future web streaming integration
import traceback  # For detailed error reporting

# Third-Party Imports
import pygame
import numpy as np


# --- Enums ---
class AntState(Enum):
    """Defines the possible states of an ant."""

    SEARCHING = auto()
    RETURNING_TO_NEST = auto()
    ESCAPING = auto()
    PATROLLING = auto()
    DEFENDING = auto()
    HUNTING = auto()
    TENDING_BROOD = auto()  # Placeholder


class BroodStage(Enum):
    """Defines the developmental stages of ant brood."""

    EGG = auto()
    LARVA = auto()
    PUPA = auto()


class AntCaste(Enum):
    """Defines the castes within the ant colony."""

    WORKER = auto()
    SOLDIER = auto()


class FoodType(Enum):
    """Defines the types of food available."""

    SUGAR = 0
    PROTEIN = 1


# --- Configuration Constants ---
# World & Grid
GRID_WIDTH = 150
GRID_HEIGHT = 80
CELL_SIZE = 8
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE
MAP_BG_COLOR = (20, 20, 10)

# Nest
NEST_POS = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
NEST_RADIUS = 6

# Food
NUM_FOOD_TYPES = len(FoodType)
INITIAL_FOOD_CLUSTERS = 6
FOOD_PER_CLUSTER = 250
FOOD_CLUSTER_RADIUS = 5
MIN_FOOD_DIST_FROM_NEST = 30
MAX_FOOD_PER_CELL = 100.0
INITIAL_COLONY_FOOD_SUGAR = 80.0
INITIAL_COLONY_FOOD_PROTEIN = 80.0
RICH_FOOD_THRESHOLD = 50.0
CRITICAL_FOOD_THRESHOLD = 25.0

# Obstacles
NUM_OBSTACLES = 10
MIN_OBSTACLE_SIZE = 3
MAX_OBSTACLE_SIZE = 10
OBSTACLE_COLOR = (100, 100, 100)

# Pheromones
PHEROMONE_MAX = 1000.0
PHEROMONE_DECAY = 0.9985
PHEROMONE_DIFFUSION_RATE = 0.04
NEGATIVE_PHEROMONE_DECAY = 0.992
NEGATIVE_PHEROMONE_DIFFUSION_RATE = 0.06
RECRUITMENT_PHEROMONE_DECAY = 0.96
RECRUITMENT_PHEROMONE_DIFFUSION_RATE = 0.12
RECRUITMENT_PHEROMONE_MAX = 500.0

# Weights (Influence on ant decision-making)
W_HOME_PHEROMONE_RETURN = 45.0
W_FOOD_PHEROMONE_SEARCH_BASE = 40.0
W_FOOD_PHEROMONE_SEARCH_LOW_NEED = 5.0
W_FOOD_PHEROMONE_SEARCH_AVOID = -10.0
W_HOME_PHEROMONE_SEARCH = 0.0
W_ALARM_PHEROMONE = -35.0
W_NEST_DIRECTION_RETURN = 85.0
W_NEST_DIRECTION_PATROL = -10.0
W_ALARM_SOURCE_DEFEND = 500.0
W_PERSISTENCE = 1.5
W_RANDOM_NOISE = 0.2
W_NEGATIVE_PHEROMONE = -50.0
W_RECRUITMENT_PHEROMONE = 200.0
W_AVOID_NEST_SEARCHING = -80.0
W_HUNTING_TARGET = 300.0

# Probabilistic Choice Parameters
PROBABILISTIC_CHOICE_TEMP = 1.0
MIN_SCORE_FOR_PROB_CHOICE = 0.01

# Pheromone Drop Amounts
P_HOME_RETURNING = 100.0
P_FOOD_RETURNING_TRAIL = 60.0
P_FOOD_AT_SOURCE = 500.0
P_ALARM_FIGHT = 100.0
P_NEGATIVE_SEARCH = 10.0
P_RECRUIT_FOOD = 400.0
P_RECRUIT_DAMAGE = 250.0
P_RECRUIT_DAMAGE_SOLDIER = 400.0
P_RECRUIT_PREY = 300.0

# --- Other Fixed Constants ---
P_FOOD_SEARCHING = 0.0
P_FOOD_AT_NEST = 0.0
W_AVOID_HISTORY = -1000.0

# Ant Parameters
INITIAL_ANTS = 10
QUEEN_HP = 1000
WORKER_MAX_AGE_MEAN = 12000
WORKER_MAX_AGE_STDDEV = 2000
WORKER_PATH_HISTORY_LENGTH = 8
WORKER_STUCK_THRESHOLD = 60
WORKER_ESCAPE_DURATION = 30
WORKER_FOOD_CONSUMPTION_INTERVAL = 100
SOLDIER_PATROL_RADIUS_SQ = (NEST_RADIUS * 2) ** 2
SOLDIER_DEFEND_ALARM_THRESHOLD = 300.0

# Ant Caste Attributes
ANT_ATTRIBUTES = {
    AntCaste.WORKER: {
        "hp": 50,
        "attack": 3,
        "capacity": 1.5,
        "speed_delay": 0,
        "color": (0, 150, 255),
        "return_color": (0, 255, 100),
        "food_consumption_sugar": 0.02,
        "food_consumption_protein": 0.005,
        "description": "Worker",
        "size_factor": 2.5,
    },
    AntCaste.SOLDIER: {
        "hp": 90,
        "attack": 10,
        "capacity": 0.2,
        "speed_delay": 1,
        "color": (0, 50, 255),
        "return_color": (255, 150, 50),
        "food_consumption_sugar": 0.025,
        "food_consumption_protein": 0.01,
        "description": "Soldier",
        "size_factor": 2.0,
    },
}

# Brood Cycle Parameters
QUEEN_EGG_LAY_RATE = 60
QUEEN_FOOD_PER_EGG_SUGAR = 1.0
QUEEN_FOOD_PER_EGG_PROTEIN = 1.5
QUEEN_SOLDIER_RATIO_TARGET = 0.15
EGG_DURATION = 500
LARVA_DURATION = 800
PUPA_DURATION = 600
LARVA_FOOD_CONSUMPTION_PROTEIN = 0.06
LARVA_FOOD_CONSUMPTION_SUGAR = 0.01
LARVA_FEED_INTERVAL = 50

# Enemy Parameters
INITIAL_ENEMIES = 1
ENEMY_HP = 60
ENEMY_ATTACK = 10
ENEMY_MOVE_DELAY = 4
ENEMY_SPAWN_RATE = 1000
ENEMY_TO_FOOD_ON_DEATH_SUGAR = 10.0
ENEMY_TO_FOOD_ON_DEATH_PROTEIN = 50.0
ENEMY_NEST_ATTRACTION = 0.05

# Prey Parameters
INITIAL_PREY = 5
PREY_HP = 25
PREY_MOVE_DELAY = 2
PREY_SPAWN_RATE = 600
PROTEIN_ON_DEATH = 30.0
PREY_COLOR = (0, 200, 0)
PREY_FLEE_RADIUS_SQ = 5 * 5

# Simulation Speed Control
BASE_FPS = 40
SPEED_MULTIPLIERS = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0]
TARGET_FPS_LIST = [10] + [
    max(1, int(m * BASE_FPS)) for m in SPEED_MULTIPLIERS[1:]
]
DEFAULT_SPEED_INDEX = SPEED_MULTIPLIERS.index(1.0)

# Colors
QUEEN_COLOR = (255, 0, 255)
WORKER_ESCAPE_COLOR = (255, 165, 0)
ENEMY_COLOR = (200, 0, 0)
FOOD_COLORS = {
    FoodType.SUGAR: (200, 200, 255),
    FoodType.PROTEIN: (255, 180, 180),
}
FOOD_COLOR_MIX = (230, 200, 230)
PHEROMONE_HOME_COLOR = (0, 0, 255, 150)
PHEROMONE_FOOD_SUGAR_COLOR = (150, 150, 255, 150)
PHEROMONE_FOOD_PROTEIN_COLOR = (255, 150, 150, 150)
PHEROMONE_ALARM_COLOR = (255, 0, 0, 180)
PHEROMONE_NEGATIVE_COLOR = (150, 150, 150, 100)
PHEROMONE_RECRUITMENT_COLOR = (255, 0, 255, 180)
EGG_COLOR = (255, 255, 255, 200)
LARVA_COLOR = (255, 255, 200, 220)
PUPA_COLOR = (200, 180, 150, 220)
# UI Colors
BUTTON_COLOR = (80, 80, 150)
BUTTON_HOVER_COLOR = (100, 100, 180)
BUTTON_TEXT_COLOR = (240, 240, 240)
END_DIALOG_BG_COLOR = (0, 0, 0, 180)

# Define aliases for random functions
rnd = random.randint
rnd_gauss = random.gauss
rnd_uniform = random.uniform


# --- Helper Functions ---
def is_valid(pos):
    """Check if a position (x, y) is within the grid boundaries."""
    if not isinstance(pos, (tuple, list)) or len(pos) != 2:
        return False
    x, y = pos
    if not all(
        isinstance(coord, (int, float)) and math.isfinite(coord)
        for coord in [x, y]
    ):
        return False
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def get_neighbors(pos, include_center=False):
    """Get valid integer neighbor coordinates for a given position."""
    if not is_valid(pos):
        return []
    x_int, y_int = int(pos[0]), int(pos[1])
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0 and not include_center:
                continue
            n_pos = (x_int + dx, y_int + dy)
            if 0 <= n_pos[0] < GRID_WIDTH and 0 <= n_pos[1] < GRID_HEIGHT:
                neighbors.append(n_pos)
    return neighbors


def distance_sq(pos1, pos2):
    """Calculate squared Euclidean distance between two points (integer)."""
    if not pos1 or not pos2:
        return float("inf")
    try:
        x1, y1 = int(pos1[0]), int(pos1[1])
        x2, y2 = int(pos2[0]), int(pos2[1])
        if not (
            0 <= x1 < GRID_WIDTH
            and 0 <= y1 < GRID_HEIGHT
            and 0 <= x2 < GRID_WIDTH
            and 0 <= y2 < GRID_HEIGHT
        ):
            return float("inf")
        return (x1 - x2) ** 2 + (y1 - y2) ** 2
    except (TypeError, IndexError, ValueError):
        return float("inf")


def normalize(value, max_val):
    """Normalize a value to the range [0, 1], clamped."""
    if max_val <= 0:
        return 0.0
    norm_val = float(value) / float(max_val)
    return min(1.0, max(0.0, norm_val))


# --- Brood Class ---
class BroodItem:
    """Represents an item of brood (egg, larva, pupa) in the nest."""

    def __init__(
        self, stage: BroodStage, caste: AntCaste, position: tuple, current_tick: int
    ):
        self.stage = stage
        self.caste = caste
        self.pos = tuple(map(int, position))
        self.creation_tick = current_tick
        self.progress_timer = 0.0
        self.last_feed_check = current_tick

        if self.stage == BroodStage.EGG:
            self.duration = EGG_DURATION
            self.color = EGG_COLOR
            self.radius = CELL_SIZE // 5
        elif self.stage == BroodStage.LARVA:
            self.duration = LARVA_DURATION
            self.color = LARVA_COLOR
            self.radius = CELL_SIZE // 4
        elif self.stage == BroodStage.PUPA:
            self.duration = PUPA_DURATION
            self.color = PUPA_COLOR
            self.radius = int(CELL_SIZE / 3.5)
        else:
            self.duration = 0
            self.color = (0, 0, 0, 0)
            self.radius = 0

    def update(self, current_tick, simulation):
        """Update progress, handle feeding, and check for stage transition."""
        current_multiplier = SPEED_MULTIPLIERS[simulation.simulation_speed_index]
        update_factor = current_multiplier if current_multiplier > 0 else 0
        if update_factor == 0.0:
            return None

        self.progress_timer += update_factor

        if self.stage == BroodStage.LARVA:
            if current_tick - self.last_feed_check >= LARVA_FEED_INTERVAL:
                self.last_feed_check = current_tick
                needed_p = LARVA_FOOD_CONSUMPTION_PROTEIN
                needed_s = LARVA_FOOD_CONSUMPTION_SUGAR
                has_p = simulation.colony_food_storage_protein >= needed_p
                has_s = simulation.colony_food_storage_sugar >= needed_s

                if has_p and has_s:
                    simulation.colony_food_storage_protein -= needed_p
                    simulation.colony_food_storage_sugar -= needed_s
                else:
                    self.progress_timer = max(
                        0.0, self.progress_timer - update_factor
                    )

        if self.progress_timer >= self.duration:
            if self.stage == BroodStage.EGG:
                self.stage = BroodStage.LARVA
                self.progress_timer = 0.0
                self.duration = LARVA_DURATION
                self.color = LARVA_COLOR
                self.radius = CELL_SIZE // 4
                self.last_feed_check = current_tick
                return None
            elif self.stage == BroodStage.LARVA:
                self.stage = BroodStage.PUPA
                self.progress_timer = 0.0
                self.duration = PUPA_DURATION
                self.color = PUPA_COLOR
                self.radius = int(CELL_SIZE / 3.5)
                return None
            elif self.stage == BroodStage.PUPA:
                return self
        return None

    def draw(self, surface):
        """Draw the brood item statically centered in its cell."""
        if not is_valid(self.pos) or self.radius <= 0:
            return
        center_x = int(self.pos[0]) * CELL_SIZE + CELL_SIZE // 2
        center_y = int(self.pos[1]) * CELL_SIZE + CELL_SIZE // 2
        draw_pos = (center_x, center_y)
        pygame.draw.circle(surface, self.color, draw_pos, self.radius)
        if self.stage == BroodStage.PUPA:
            o_col = (
                (50, 50, 50)
                if self.caste == AntCaste.WORKER
                else (100, 0, 0)
            )
            pygame.draw.circle(surface, o_col, draw_pos, self.radius, 1)


# --- Grid Class ---
class WorldGrid:
    """Manages the simulation grid (food, obstacles, pheromones)."""

    def __init__(self):
        self.food = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT, NUM_FOOD_TYPES), dtype=np.float32
        )
        self.obstacles = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=bool)
        self.pheromones_home = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=np.float32
        )
        self.pheromones_alarm = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=np.float32
        )
        self.pheromones_negative = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=np.float32
        )
        self.pheromones_recruitment = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=np.float32
        )
        self.pheromones_food_sugar = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=np.float32
        )
        self.pheromones_food_protein = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=np.float32
        )

    def reset(self):
        """Resets the grid state for a new simulation."""
        self.food.fill(0)
        self.obstacles.fill(0)
        self.pheromones_home.fill(0)
        self.pheromones_alarm.fill(0)
        self.pheromones_negative.fill(0)
        self.pheromones_recruitment.fill(0)
        self.pheromones_food_sugar.fill(0)
        self.pheromones_food_protein.fill(0)
        self.place_obstacles()
        self.place_food_clusters()

    def place_food_clusters(self):
        """Place initial food clusters of alternating types."""
        nest_pos_int = tuple(map(int, NEST_POS))
        min_dist_sq = MIN_FOOD_DIST_FROM_NEST**2
        for i in range(INITIAL_FOOD_CLUSTERS):
            food_type_index = i % NUM_FOOD_TYPES
            attempts = 0
            cx = cy = 0
            found_spot = False
            while attempts < 150 and not found_spot:
                cx = rnd(0, GRID_WIDTH - 1)
                cy = rnd(0, GRID_HEIGHT - 1)
                if (
                    distance_sq((cx, cy), nest_pos_int) > min_dist_sq
                    and not self.obstacles[cx, cy]
                ):
                    found_spot = True
                attempts += 1
            if not found_spot:
                attempts = 0
                while attempts < 200:
                    cx = rnd(0, GRID_WIDTH - 1)
                    cy = rnd(0, GRID_HEIGHT - 1)
                    if not self.obstacles[cx, cy]:
                        found_spot = True
                        break
                    attempts += 1
            if not found_spot:
                cx = rnd(0, GRID_WIDTH - 1)
                cy = rnd(0, GRID_HEIGHT - 1)

            added_amount = 0.0
            for _ in range(int(FOOD_PER_CLUSTER * 1.8)):
                if added_amount >= FOOD_PER_CLUSTER:
                    break
                fx = cx + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                fy = cy + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                fx_int, fy_int = int(fx), int(fy)
                if (
                    0 <= fx_int < GRID_WIDTH
                    and 0 <= fy_int < GRID_HEIGHT
                    and not self.obstacles[fx_int, fy_int]
                ):
                    amount_to_add = rnd_uniform(0.5, 1.0) * (
                        MAX_FOOD_PER_CELL / 8
                    )
                    current_amount = self.food[fx_int, fy_int, food_type_index]
                    new_amount = min(
                        MAX_FOOD_PER_CELL, current_amount + amount_to_add
                    )
                    self.food[fx_int, fy_int, food_type_index] = new_amount
                    added_amount += new_amount - current_amount

    def place_obstacles(self):
        """Place rectangular obstacles, avoiding the immediate nest area."""
        nest_area = set()
        nest_radius_buffer = NEST_RADIUS + 3
        nest_center_int = tuple(map(int, NEST_POS))
        min_x = max(0, nest_center_int[0] - nest_radius_buffer)
        max_x = min(GRID_WIDTH - 1, nest_center_int[0] + nest_radius_buffer)
        min_y = max(0, nest_center_int[1] - nest_radius_buffer)
        max_y = min(GRID_HEIGHT - 1, nest_center_int[1] + nest_radius_buffer)
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if distance_sq((x, y), nest_center_int) <= nest_radius_buffer**2:
                    nest_area.add((x, y))

        placed_count = 0
        for _ in range(NUM_OBSTACLES * 5):
            if placed_count >= NUM_OBSTACLES:
                break
            attempts = 0
            placed_this_obstacle = False
            while attempts < 25 and not placed_this_obstacle:
                w = rnd(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                h = rnd(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                x = rnd(0, GRID_WIDTH - w - 1)
                y = rnd(0, GRID_HEIGHT - h - 1)
                overlaps_nest = False
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        if (i, j) in nest_area:
                            overlaps_nest = True
                            break
                    if overlaps_nest:
                        break
                if not overlaps_nest:
                    if x + w < GRID_WIDTH and y + h < GRID_HEIGHT:
                        self.obstacles[x : x + w, y : y + h] = True
                        placed_this_obstacle = True
                        placed_count += 1
                attempts += 1

    def is_obstacle(self, pos):
        """Check if a given position corresponds to an obstacle cell."""
        if not is_valid(pos):
            return True
        try:
            x = int(pos[0])
            y = int(pos[1])
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                return self.obstacles[x, y]
            else:
                return True
        except (IndexError, TypeError, ValueError):
            return True

    def get_pheromone(self, pos, ph_type="home", food_type: FoodType = None):
        """Get the pheromone value at a specific integer position."""
        try:
            x, y = int(pos[0]), int(pos[1])
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                return 0.0
        except (ValueError, TypeError, IndexError):
            return 0.0
        try:
            if ph_type == "home":
                return self.pheromones_home[x, y]
            elif ph_type == "alarm":
                return self.pheromones_alarm[x, y]
            elif ph_type == "negative":
                return self.pheromones_negative[x, y]
            elif ph_type == "recruitment":
                return self.pheromones_recruitment[x, y]
            elif ph_type == "food":
                if food_type == FoodType.SUGAR:
                    return self.pheromones_food_sugar[x, y]
                elif food_type == FoodType.PROTEIN:
                    return self.pheromones_food_protein[x, y]
                else:
                    return 0.0
            else:
                return 0.0
        except IndexError:
            return 0.0

    def add_pheromone(
        self, pos, amount, ph_type="home", food_type: FoodType = None
    ):
        """Add pheromone to a specific integer position."""
        if amount <= 0 or self.is_obstacle(pos):
            return
        try:
            x, y = int(pos[0]), int(pos[1])
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                return
        except (ValueError, TypeError, IndexError):
            return

        target_array = None
        max_value = PHEROMONE_MAX
        if ph_type == "home":
            target_array = self.pheromones_home
        elif ph_type == "alarm":
            target_array = self.pheromones_alarm
        elif ph_type == "negative":
            target_array = self.pheromones_negative
        elif ph_type == "recruitment":
            target_array = self.pheromones_recruitment
            max_value = RECRUITMENT_PHEROMONE_MAX
        elif ph_type == "food":
            if food_type == FoodType.SUGAR:
                target_array = self.pheromones_food_sugar
            elif food_type == FoodType.PROTEIN:
                target_array = self.pheromones_food_protein
            else:
                return

        if target_array is not None:
            try:
                target_array[x, y] = min(target_array[x, y] + amount, max_value)
            except IndexError:
                pass

    def update_pheromones(self, speed_multiplier):
        """Update pheromones: apply decay and diffusion."""
        effective_multiplier = max(0.0, speed_multiplier)
        if effective_multiplier == 0.0:
            return

        min_decay_factor = 0.1
        decay_factor_common = max(
            min_decay_factor, PHEROMONE_DECAY**effective_multiplier
        )
        decay_factor_neg = max(
            min_decay_factor, NEGATIVE_PHEROMONE_DECAY**effective_multiplier
        )
        decay_factor_rec = max(
            min_decay_factor, RECRUITMENT_PHEROMONE_DECAY**effective_multiplier
        )

        self.pheromones_home *= decay_factor_common
        self.pheromones_alarm *= decay_factor_common
        self.pheromones_negative *= decay_factor_neg
        self.pheromones_recruitment *= decay_factor_rec
        self.pheromones_food_sugar *= decay_factor_common
        self.pheromones_food_protein *= decay_factor_common

        diffusion_rate_common = PHEROMONE_DIFFUSION_RATE * effective_multiplier
        diffusion_rate_neg = (
            NEGATIVE_PHEROMONE_DIFFUSION_RATE * effective_multiplier
        )
        diffusion_rate_rec = (
            RECRUITMENT_PHEROMONE_DIFFUSION_RATE * effective_multiplier
        )

        max_diffusion = 0.124
        diffusion_rate_common = min(max_diffusion, max(0.0, diffusion_rate_common))
        diffusion_rate_neg = min(max_diffusion, max(0.0, diffusion_rate_neg))
        diffusion_rate_rec = min(max_diffusion, max(0.0, diffusion_rate_rec))

        mask = ~self.obstacles
        arrays_rates = [
            (self.pheromones_home, diffusion_rate_common),
            (self.pheromones_food_sugar, diffusion_rate_common),
            (self.pheromones_food_protein, diffusion_rate_common),
            (self.pheromones_alarm, diffusion_rate_common),
            (self.pheromones_negative, diffusion_rate_neg),
            (self.pheromones_recruitment, diffusion_rate_rec),
        ]

        for arr, rate in arrays_rates:
            if rate > 0:
                masked = arr * mask
                pad = np.pad(masked, 1, mode="constant")
                neighbors_sum = (
                    pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                    pad[1:-1, :-2] + pad[1:-1, 2:] +
                    pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
                )
                diffused = masked * (1.0 - rate) + (neighbors_sum / 8.0) * rate
                arr[:] = np.where(mask, diffused, 0)

        min_ph = 0.01
        all_arrays = [
            self.pheromones_home,
            self.pheromones_food_sugar,
            self.pheromones_food_protein,
            self.pheromones_alarm,
            self.pheromones_negative,
            self.pheromones_recruitment,
        ]
        for arr in all_arrays:
            max_val = (
                RECRUITMENT_PHEROMONE_MAX
                if arr is self.pheromones_recruitment
                else PHEROMONE_MAX
            )
            np.clip(arr, 0, max_val, out=arr)
            arr[arr < min_ph] = 0


# --- Prey Class ---
class Prey:
    """Represents a small creature that ants can hunt for protein."""

    def __init__(self, pos, sim):
        self.pos = tuple(map(int, pos))
        self.simulation = sim
        self.hp = float(PREY_HP)
        self.max_hp = float(PREY_HP)
        self.move_delay_base = PREY_MOVE_DELAY
        self.move_delay_timer = rnd_uniform(0, self.move_delay_base)
        self.color = PREY_COLOR
        self.fleeing_ant = None

    def update(self, speed_multiplier):
        """Update prey state: flee or wander."""
        if speed_multiplier == 0.0:
            return

        grid = self.simulation.grid
        pos_int = self.pos

        # --- Fleeing Behavior ---
        nearest_ant = None
        min_dist_sq = PREY_FLEE_RADIUS_SQ

        check_radius = int(PREY_FLEE_RADIUS_SQ**0.5) + 1
        min_x = max(0, pos_int[0] - check_radius)
        max_x = min(GRID_WIDTH - 1, pos_int[0] + check_radius)
        min_y = max(0, pos_int[1] - check_radius)
        max_y = min(GRID_HEIGHT - 1, pos_int[1] + check_radius)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) == pos_int:
                    continue
                ant = self.simulation.get_ant_at((x, y))
                if ant:
                    d_sq = distance_sq(pos_int, ant.pos)
                    if d_sq < min_dist_sq:
                        min_dist_sq = d_sq
                        nearest_ant = ant

        # --- Movement Delay ---
        self.move_delay_timer -= speed_multiplier
        if self.move_delay_timer > 0:
            return
        self.move_delay_timer += self.move_delay_base

        # --- Movement Logic ---
        possible_moves = get_neighbors(pos_int)
        valid_moves = [
            m
            for m in possible_moves
            if not grid.is_obstacle(m)
            and not self.simulation.is_enemy_at(m)
            and not self.simulation.is_prey_at(m, self)
            and not self.simulation.is_ant_at(m)
        ]

        if not valid_moves:
            return

        chosen_move = None
        if nearest_ant:  # Fleeing
            flee_dx = pos_int[0] - nearest_ant.pos[0]
            flee_dy = pos_int[1] - nearest_ant.pos[1]
            dist = max(1, abs(flee_dx) + abs(flee_dy))
            target_dx = round(flee_dx / dist)
            target_dy = round(flee_dy / dist)

            best_flee_move = None
            max_flee_score = -float("inf")

            for move in valid_moves:
                move_dx = move[0] - pos_int[0]
                move_dy = move[1] - pos_int[1]
                score = move_dx * target_dx + move_dy * target_dy
                score += distance_sq(move, nearest_ant.pos) * 0.1
                if score > max_flee_score:
                    max_flee_score = score
                    best_flee_move = move
            chosen_move = (
                best_flee_move if best_flee_move else random.choice(valid_moves)
            )
        else:  # Wander randomly
            chosen_move = random.choice(valid_moves)

        if chosen_move:
            self.pos = chosen_move

    def take_damage(self, amount, attacker):
        """Process damage taken by the prey."""
        if self.hp <= 0:
            return
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0

    def draw(self, surface):
        """Draw the prey."""
        if not is_valid(self.pos):
            return
        pos_px = (
            int(self.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos[1] * CELL_SIZE + CELL_SIZE / 2),
        )
        radius = int(CELL_SIZE / 2.8)
        pygame.draw.circle(surface, self.color, pos_px, radius)


# --- Ant Class ---
class Ant:
    """Represents a worker or soldier ant."""

    def __init__(self, pos, simulation, caste: AntCaste):
        self.pos = tuple(map(int, pos))
        self.simulation = simulation
        self.caste = caste
        attrs = ANT_ATTRIBUTES[caste]
        self.hp = float(attrs["hp"])
        self.max_hp = float(attrs["hp"])
        self.attack_power = attrs["attack"]
        self.max_capacity = attrs["capacity"]
        self.move_delay_base = attrs["speed_delay"]
        self.search_color = attrs["color"]
        self.return_color = attrs["return_color"]
        self.food_consumption_sugar = attrs["food_consumption_sugar"]
        self.food_consumption_protein = attrs["food_consumption_protein"]
        self.size_factor = attrs["size_factor"]
        self.state = AntState.SEARCHING
        self.carry_amount = 0.0
        self.carry_type = None
        self.age = 0.0
        self.max_age_ticks = int(
            rnd_gauss(WORKER_MAX_AGE_MEAN, WORKER_MAX_AGE_STDDEV)
        )
        self.path_history = []
        self.history_timestamps = []
        self.move_delay_timer = 0
        self.last_move_direction = (0, 0)
        self.stuck_timer = 0
        self.escape_timer = 0.0
        self.last_move_info = "Born"
        self.just_picked_food = False
        self.food_consumption_timer = rnd_uniform(
            0, WORKER_FOOD_CONSUMPTION_INTERVAL
        )
        self.last_known_alarm_pos = None
        self.target_prey = None

    def _update_path_history(self, new_pos_int):
        """Adds integer position to history if different, trims old entries."""
        t = self.simulation.ticks
        if not self.path_history or self.path_history[-1] != new_pos_int:
            self.path_history.append(new_pos_int)
            self.history_timestamps.append(t)
            cutoff_time = t - WORKER_PATH_HISTORY_LENGTH
            cutoff_index = 0
            while (
                cutoff_index < len(self.history_timestamps)
                and self.history_timestamps[cutoff_index] < cutoff_time
            ):
                cutoff_index += 1
            self.path_history = self.path_history[cutoff_index:]
            self.history_timestamps = self.history_timestamps[cutoff_index:]

    def _is_in_history(self, pos_int):
        """Check if an integer position is in the recent path history."""
        return pos_int in self.path_history

    def _clear_path_history(self):
        """Clears the path history."""
        self.path_history = []
        self.history_timestamps = []

    def _filter_valid_moves(
        self, potential_neighbors_int, ignore_history_near_nest=False
    ):
        """Filter potential integer moves for obstacles, history, queen, ants."""
        valid_moves_int = []
        q_pos_int = self.simulation.queen.pos if self.simulation.queen else None
        pos_int = self.pos
        nest_pos_int = tuple(map(int, NEST_POS))
        is_near_nest_now = distance_sq(pos_int, nest_pos_int) <= (
            NEST_RADIUS + 2
        ) ** 2

        for n_pos_int in potential_neighbors_int:
            history_block = False
            check_hist = not ignore_history_near_nest or (
                ignore_history_near_nest and not is_near_nest_now
            )
            if check_hist and self._is_in_history(n_pos_int):
                history_block = True

            is_queen_pos = n_pos_int == q_pos_int
            is_obstacle_pos = self.simulation.grid.is_obstacle(n_pos_int)
            is_ant_pos = self.simulation.is_ant_at(n_pos_int, exclude_self=self)

            if (
                not history_block
                and not is_queen_pos
                and not is_obstacle_pos
                and not is_ant_pos
            ):
                valid_moves_int.append(n_pos_int)

        return valid_moves_int

    def _choose_move(self):
        """Determine the next integer move based on state, goals, environment."""
        potential_neighbors_int = get_neighbors(self.pos)
        if not potential_neighbors_int:
            self.last_move_info = "No neighbors"
            return None

        ignore_hist_near_nest = self.state == AntState.RETURNING_TO_NEST
        valid_neighbors_int = self._filter_valid_moves(
            potential_neighbors_int, ignore_hist_near_nest
        )

        if not valid_neighbors_int:
            self.last_move_info = "Blocked"
            fallback_neighbors_int = []
            q_pos_int = self.simulation.queen.pos if self.simulation.queen else None
            for n_pos_int in potential_neighbors_int:
                if (
                    n_pos_int != q_pos_int
                    and not self.simulation.grid.is_obstacle(n_pos_int)
                    and not self.simulation.is_ant_at(n_pos_int, exclude_self=self)
                ):
                    fallback_neighbors_int.append(n_pos_int)

            if fallback_neighbors_int:
                fallback_neighbors_int.sort(
                    key=lambda p: self.path_history.index(p)
                    if p in self.path_history
                    else -1
                )
                return fallback_neighbors_int[0]
            return None

        if self.state == AntState.ESCAPING:
            escape_moves_int = [
                p for p in valid_neighbors_int if not self._is_in_history(p)
            ]
            return (
                random.choice(escape_moves_int)
                if escape_moves_int
                else random.choice(valid_neighbors_int)
            )

        move_scores = {}
        if self.state == AntState.RETURNING_TO_NEST:
            move_scores = self._score_moves_returning(
                valid_neighbors_int, self.just_picked_food
            )
        elif self.state == AntState.SEARCHING:
            move_scores = self._score_moves_searching(valid_neighbors_int)
        elif self.state == AntState.PATROLLING:
            move_scores = self._score_moves_patrolling(valid_neighbors_int)
        elif self.state == AntState.DEFENDING:
            move_scores = self._score_moves_defending(valid_neighbors_int)
        elif self.state == AntState.HUNTING:
            move_scores = self._score_moves_hunting(valid_neighbors_int)
        else:
            move_scores = self._score_moves_searching(valid_neighbors_int)

        if not move_scores:
            self.last_move_info = f"No scores({self.state})"
            return random.choice(valid_neighbors_int)

        selected_move_int = None
        if self.state == AntState.RETURNING_TO_NEST:
            selected_move_int = self._select_best_move_returning(
                move_scores, valid_neighbors_int, self.just_picked_food
            )
        elif self.state == AntState.DEFENDING or self.state == AntState.HUNTING:
            selected_move_int = self._select_best_move(
                move_scores, valid_neighbors_int
            )
        else:
            selected_move_int = self._select_probabilistic_move(
                move_scores, valid_neighbors_int
            )

        return selected_move_int if selected_move_int else random.choice(valid_neighbors_int)


    def _score_moves_base(self, neighbor_pos_int):
        """Calculates a base score for moving to a neighbor."""
        score = 0.0
        move_dir = (
            neighbor_pos_int[0] - self.pos[0],
            neighbor_pos_int[1] - self.pos[1],
        )
        if move_dir == self.last_move_direction and move_dir != (0, 0):
            score += W_PERSISTENCE
        score += rnd_uniform(-W_RANDOM_NOISE, W_RANDOM_NOISE)
        return score

    def _score_moves_returning(self, valid_neighbors_int, just_picked):
        """Scores potential integer moves for returning to the nest."""
        scores = {}
        pos_int = self.pos
        nest_pos_int = tuple(map(int, NEST_POS))
        dist_sq_now = distance_sq(pos_int, nest_pos_int)
        grid = self.simulation.grid

        for n_pos_int in valid_neighbors_int:
            score = 0.0
            home_ph = grid.get_pheromone(n_pos_int, "home")
            sugar_ph = grid.get_pheromone(n_pos_int, "food", FoodType.SUGAR)
            protein_ph = grid.get_pheromone(
                n_pos_int, "food", FoodType.PROTEIN
            )
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            neg_ph = grid.get_pheromone(n_pos_int, "negative")

            score += home_ph * W_HOME_PHEROMONE_RETURN
            if distance_sq(n_pos_int, nest_pos_int) < dist_sq_now:
                score += W_NEST_DIRECTION_RETURN

            score += alarm_ph * W_ALARM_PHEROMONE * 0.3
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.4

            if just_picked:
                if self.carry_type == FoodType.SUGAR:
                    score -= sugar_ph * W_FOOD_PHEROMONE_SEARCH_BASE * 0.2
                elif self.carry_type == FoodType.PROTEIN:
                    score -= protein_ph * W_FOOD_PHEROMONE_SEARCH_BASE * 0.2

            score += self._score_moves_base(n_pos_int)
            scores[n_pos_int] = score
        return scores

    def _score_moves_searching(self, valid_neighbors_int):
        """Scores potential integer moves for searching based on food needs."""
        scores = {}
        grid = self.simulation.grid
        nest_pos_int = tuple(map(int, NEST_POS))
        sim = self.simulation

        sugar_critical = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD
        protein_critical = (
            sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD
        )

        w_sugar = W_FOOD_PHEROMONE_SEARCH_LOW_NEED
        w_protein = W_FOOD_PHEROMONE_SEARCH_LOW_NEED

        if sugar_critical and not protein_critical:
            w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE
            w_protein = W_FOOD_PHEROMONE_SEARCH_AVOID
        elif protein_critical and not sugar_critical:
            w_protein = W_FOOD_PHEROMONE_SEARCH_BASE
            w_sugar = W_FOOD_PHEROMONE_SEARCH_AVOID
        elif sugar_critical and protein_critical:
            if (
                sim.colony_food_storage_sugar
                <= sim.colony_food_storage_protein
            ):
                w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 1.1
                w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 0.9
            else:
                w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 1.1
                w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 0.9
        else:
            if (
                sim.colony_food_storage_sugar
                <= sim.colony_food_storage_protein * 1.5
            ):
                w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 0.6
                w_protein = W_FOOD_PHEROMONE_SEARCH_LOW_NEED
            else:
                w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 0.6
                w_sugar = W_FOOD_PHEROMONE_SEARCH_LOW_NEED

        if self.caste == AntCaste.SOLDIER:
            w_sugar *= 0.1
            w_protein *= 0.1

        for n_pos_int in valid_neighbors_int:
            score = 0.0
            home_ph = grid.get_pheromone(n_pos_int, "home")
            sugar_ph = grid.get_pheromone(n_pos_int, "food", FoodType.SUGAR)
            protein_ph = grid.get_pheromone(
                n_pos_int, "food", FoodType.PROTEIN
            )
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            score += sugar_ph * w_sugar
            score += protein_ph * w_protein

            recruit_w = W_RECRUITMENT_PHEROMONE
            if self.caste == AntCaste.SOLDIER:
                recruit_w *= 1.2
            score += recr_ph * recruit_w

            score += neg_ph * W_NEGATIVE_PHEROMONE
            score += alarm_ph * W_ALARM_PHEROMONE
            score += home_ph * W_HOME_PHEROMONE_SEARCH
            if distance_sq(n_pos_int, nest_pos_int) <= (NEST_RADIUS * 1.8) ** 2:
                score += W_AVOID_NEST_SEARCHING

            score += self._score_moves_base(n_pos_int)
            scores[n_pos_int] = score
        return scores

    def _score_moves_patrolling(self, valid_neighbors_int):
        """Scores potential integer moves for patrolling (soldiers only)."""
        scores = {}
        grid = self.simulation.grid
        pos_int = self.pos
        nest_pos_int = tuple(map(int, NEST_POS))
        dist_sq_current = distance_sq(pos_int, nest_pos_int)

        for n_pos_int in valid_neighbors_int:
            score = 0.0
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            score += recr_ph * W_RECRUITMENT_PHEROMONE * 0.7
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.5
            score += alarm_ph * W_ALARM_PHEROMONE * 0.5

            dist_sq_next = distance_sq(n_pos_int, nest_pos_int)

            if dist_sq_current <= SOLDIER_PATROL_RADIUS_SQ:
                if dist_sq_next > dist_sq_current:
                    score -= W_NEST_DIRECTION_PATROL

            if dist_sq_next > (SOLDIER_PATROL_RADIUS_SQ * 1.4):
                score -= 8000

            score += self._score_moves_base(n_pos_int)
            scores[n_pos_int] = score
        return scores

    def _score_moves_defending(self, valid_neighbors_int):
        """Scores potential integer moves for defending."""
        scores = {}
        grid = self.simulation.grid
        pos_int = self.pos

        if self.last_known_alarm_pos is None or random.random() < 0.2:
            best_signal_pos = None
            max_signal_strength = -1
            search_radius_sq = 6 * 6
            x0, y0 = pos_int
            min_scan_x = max(0, x0 - int(search_radius_sq**0.5))
            max_scan_x = min(GRID_WIDTH - 1, x0 + int(search_radius_sq**0.5))
            min_scan_y = max(0, y0 - int(search_radius_sq**0.5))
            max_scan_y = min(GRID_HEIGHT - 1, y0 + int(search_radius_sq**0.5))

            for i in range(min_scan_x, max_scan_x + 1):
                for j in range(min_scan_y, max_scan_y + 1):
                    p_int = (i, j)
                    if distance_sq(pos_int, p_int) <= search_radius_sq:
                        signal = (
                            grid.get_pheromone(p_int, "alarm") * 1.2
                            + grid.get_pheromone(p_int, "recruitment") * 0.8
                        )
                        if self.simulation.get_enemy_at(p_int):
                            signal += 600
                        if signal > max_signal_strength:
                            max_signal_strength = signal
                            best_signal_pos = p_int
            if max_signal_strength > 80.0:
                self.last_known_alarm_pos = best_signal_pos
            else:
                self.last_known_alarm_pos = None

        for n_pos_int in valid_neighbors_int:
            score = 0.0
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")
            enemy_at_n_pos = self.simulation.get_enemy_at(n_pos_int)

            if enemy_at_n_pos:
                score += 15000

            if self.last_known_alarm_pos:
                dist_now_sq = distance_sq(pos_int, self.last_known_alarm_pos)
                dist_next_sq = distance_sq(n_pos_int, self.last_known_alarm_pos)
                if dist_next_sq < dist_now_sq:
                    score += W_ALARM_SOURCE_DEFEND

            score += alarm_ph * W_ALARM_PHEROMONE * -1.0
            score += recr_ph * W_RECRUITMENT_PHEROMONE * 1.1
            score += self._score_moves_base(n_pos_int)
            scores[n_pos_int] = score
        return scores

    def _score_moves_hunting(self, valid_neighbors_int):
        """Scores potential integer moves for hunting prey."""
        scores = {}
        pos_int = self.pos
        target_pos = self.target_prey.pos if self.target_prey else None

        if not target_pos:
            return {
                n_pos_int: self._score_moves_base(n_pos_int)
                for n_pos_int in valid_neighbors_int
            }

        dist_sq_now = distance_sq(pos_int, target_pos)

        for n_pos_int in valid_neighbors_int:
            score = 0.0
            dist_sq_next = distance_sq(n_pos_int, target_pos)

            if dist_sq_next < dist_sq_now:
                score += W_HUNTING_TARGET

            score += (
                self.simulation.grid.get_pheromone(
                    n_pos_int, "alarm"
                )
                * W_ALARM_PHEROMONE
                * 0.1
            )
            score += (
                self.simulation.grid.get_pheromone(
                    n_pos_int, "negative"
                )
                * W_NEGATIVE_PHEROMONE
                * 0.2
            )

            score += self._score_moves_base(n_pos_int)
            scores[n_pos_int] = score
        return scores

    def _select_best_move(self, move_scores, valid_neighbors_int):
        """Selects the integer move with the highest score (deterministic)."""
        best_score = -float("inf")
        best_moves_int = []
        for pos_int, score in move_scores.items():
            if score > best_score:
                best_score = score
                best_moves_int = [pos_int]
            elif score == best_score:
                best_moves_int.append(pos_int)
        if not best_moves_int:
            self.last_move_info += "(Best:No best?)"
            return random.choice(valid_neighbors_int)
        chosen_int = random.choice(best_moves_int)
        score = move_scores.get(chosen_int, -999)
        state_prefix = self.state.name[:4] if self.state != AntState.HUNTING else "Hunt"
        self.last_move_info = f"{state_prefix} Best->{chosen_int} (S:{score:.1f})"
        return chosen_int

    def _select_best_move_returning(
        self, move_scores, valid_neighbors_int, just_picked
    ):
        """Selects the best integer move for returning, prioritizing nest."""
        best_score = -float("inf")
        best_moves_int = []
        pos_int = self.pos
        nest_pos_int = tuple(map(int, NEST_POS))
        dist_sq_now = distance_sq(pos_int, nest_pos_int)
        closer_moves = {}
        other_moves = {}
        for pos_int_cand, score in move_scores.items():
            if distance_sq(pos_int_cand, nest_pos_int) < dist_sq_now:
                closer_moves[pos_int_cand] = score
            else:
                other_moves[pos_int_cand] = score

        target_pool = {}
        selection_type = ""
        if closer_moves:
            target_pool = closer_moves
            selection_type = "Closer"
        elif other_moves:
            target_pool = other_moves
            selection_type = "Other"
        else:
            self.last_move_info += "(R: No moves?)"
            return (
                random.choice(valid_neighbors_int)
                if valid_neighbors_int
                else None
            )

        for pos_int_cand, score in target_pool.items():
            if score > best_score:
                best_score = score
                best_moves_int = [pos_int_cand]
            elif score == best_score:
                best_moves_int.append(pos_int_cand)

        if not best_moves_int:
            self.last_move_info += f"(R: No best in {selection_type})"
            target_pool = move_scores
            best_score = -float("inf")
            best_moves_int = []
            for pos_int_cand, score in target_pool.items():
                if score > best_score:
                    best_score = score
                    best_moves_int = [pos_int_cand]
                elif score == best_score:
                    best_moves_int.append(pos_int_cand)
            if not best_moves_int:
                return (
                    random.choice(valid_neighbors_int)
                    if valid_neighbors_int
                    else None
                )

        if len(best_moves_int) == 1:
            chosen_int = best_moves_int[0]
            self.last_move_info = f"R({selection_type})Best->{chosen_int} (S:{best_score:.1f})"
        else:
            grid = self.simulation.grid
            best_moves_int.sort(
                key=lambda p: grid.get_pheromone(p, "home"), reverse=True
            )
            max_ph = grid.get_pheromone(best_moves_int[0], "home")
            top_ph_moves = [
                p for p in best_moves_int if grid.get_pheromone(p, "home") == max_ph
            ]
            chosen_int = random.choice(top_ph_moves)
            self.last_move_info = f"R({selection_type})TieBrk->{chosen_int} (S:{best_score:.1f})"
        return chosen_int

    def _select_probabilistic_move(self, move_scores, valid_neighbors_int):
        """Selects an integer move probabilistically based on scores."""
        if not move_scores or not valid_neighbors_int:
            return (
                random.choice(valid_neighbors_int)
                if valid_neighbors_int
                else None
            )
        pop_int = list(move_scores.keys())
        scores = np.array(list(move_scores.values()), dtype=np.float64)
        if len(pop_int) == 0:
            return None

        min_score = np.min(scores) if scores.size > 0 else 0
        shifted_scores = scores - min_score + 0.01
        temp = min(max(PROBABILISTIC_CHOICE_TEMP, 0.1), 5.0)
        weights = np.power(shifted_scores, temp)
        weights = np.maximum(MIN_SCORE_FOR_PROB_CHOICE, weights)
        total_weight = np.sum(weights)

        # Fallback logic for invalid weights
        if total_weight <= 1e-9 or not np.isfinite(total_weight):
            self.last_move_info += f"({self.state.name[:3]}:InvW)"
            best_s = -float("inf")
            best_p = None
            for p_int, s in move_scores.items(): # Corrected fallback loop
                if s > best_s:
                    best_s = s
                    best_p = p_int
            return best_p if best_p else random.choice(valid_neighbors_int)

        probabilities = weights / total_weight
        sum_probs = np.sum(probabilities)

        # Fallback logic for normalization issues
        if not np.isclose(sum_probs, 1.0):
            if sum_probs > 1e-9 and np.all(np.isfinite(probabilities)):
                probabilities /= sum_probs
                if not np.isclose(np.sum(probabilities), 1.0):
                    self.last_move_info += "(ProbNormErr)"
                    best_s = -float("inf")
                    best_p = None
                    for p_int, s in move_scores.items(): # Corrected fallback loop
                        if s > best_s:
                            best_s = s
                            best_p = p_int
                    return best_p if best_p else random.choice(valid_neighbors_int)
            else:
                self.last_move_info += "(ProbBadSum)"
                best_s = -float("inf")
                best_p = None
                for p_int, s in move_scores.items(): # Corrected fallback loop
                    if s > best_s:
                        best_s = s
                        best_p = p_int
                return best_p if best_p else random.choice(valid_neighbors_int)

        # Fallback for final validation
        if not np.all(np.isfinite(probabilities)) or not np.isclose(
            np.sum(probabilities), 1.0
        ):
            self.last_move_info += "(ProbErrFinal)"
            best_s = -float("inf")
            best_p = None
            for p_int, s in move_scores.items(): # Corrected fallback loop
                if s > best_s:
                    best_s = s
                    best_p = p_int
            return best_p if best_p else random.choice(valid_neighbors_int)

        try:
            chosen_index = np.random.choice(len(pop_int), p=probabilities)
            chosen_int = pop_int[chosen_index]
            score = move_scores.get(chosen_int, -999)
            self.last_move_info = f"{self.state.name[:3]} Prob->{chosen_int} (S:{score:.1f})"
            return chosen_int
        except ValueError as e:
            print(f"WARN choices ({self.state}): {e}. Sum={np.sum(probabilities)}")
            self.last_move_info += "(ProbValErr)"
            best_s = -float("inf")
            best_p = None
            for p_int, s in move_scores.items(): # Corrected fallback loop
                if s > best_s:
                    best_s = s
                    best_p = p_int
            return best_p if best_p else random.choice(valid_neighbors_int)

    def _switch_state(self, new_state: AntState, reason: str):
        """Helper to switch state and clear history/targets if needed."""
        if self.state != new_state:
            # print(f"Ant {id(self)}: {self.state.name} -> {new_state.name} ({reason})") # Debug
            self.state = new_state
            if reason:
                self.last_move_info = reason
            self._clear_path_history()
            if new_state != AntState.HUNTING:
                self.target_prey = None
            if new_state != AntState.DEFENDING:
                self.last_known_alarm_pos = None

    def _update_state(self):
        """Handle automatic state transitions for Soldiers and Workers."""
        sim = self.simulation

        # Worker Hunting Logic
        protein_needed = (
            sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * 1.5
        )
        if (
            self.caste == AntCaste.WORKER
            and protein_needed
            and self.state == AntState.SEARCHING
            and self.carry_amount == 0
            and not self.target_prey
        ):
            nearby_prey = sim.find_nearby_prey(
                self.pos, PREY_FLEE_RADIUS_SQ * 2.5
            )
            if nearby_prey:
                nearby_prey.sort(key=lambda p: distance_sq(self.pos, p.pos))
                self.target_prey = nearby_prey[0]
                self._switch_state(
                    AntState.HUNTING, f"HuntPrey@{self.target_prey.pos}"
                )
                return

        # Soldier State Logic
        if (
            self.caste == AntCaste.SOLDIER
            and self.state
            not in [AntState.ESCAPING, AntState.RETURNING_TO_NEST, AntState.HUNTING]
        ):
            pos_int = self.pos
            nest_pos_int = tuple(map(int, NEST_POS))
            max_alarm = 0
            max_recruit = 0
            search_radius_sq = 5 * 5
            grid = sim.grid
            x0, y0 = pos_int
            min_scan_x = max(0, x0 - int(search_radius_sq**0.5))
            max_scan_x = min(GRID_WIDTH - 1, x0 + int(search_radius_sq**0.5))
            min_scan_y = max(0, y0 - int(search_radius_sq**0.5))
            max_scan_y = min(GRID_HEIGHT - 1, y0 + int(search_radius_sq**0.5))

            for i in range(min_scan_x, max_scan_x + 1):
                for j in range(min_scan_y, max_scan_y + 1):
                    p_int = (i, j)
                    if distance_sq(pos_int, p_int) <= search_radius_sq:
                        max_alarm = max(max_alarm, grid.get_pheromone(p_int, "alarm"))
                        max_recruit = max(
                            max_recruit, grid.get_pheromone(p_int, "recruitment")
                        )

            threat_signal = max_alarm + max_recruit * 0.6
            is_near_nest = (
                distance_sq(pos_int, nest_pos_int) <= SOLDIER_PATROL_RADIUS_SQ
            )

            if threat_signal > SOLDIER_DEFEND_ALARM_THRESHOLD:
                if self.state != AntState.DEFENDING:
                    self._switch_state(AntState.DEFENDING, "ThreatHigh!")
                return

            if self.state != AntState.DEFENDING and not self.target_prey:
                nearby_prey = sim.find_nearby_prey(
                    self.pos, PREY_FLEE_RADIUS_SQ * 2.0
                )
                if nearby_prey:
                    nearby_prey.sort(key=lambda p: distance_sq(self.pos, p.pos))
                    self.target_prey = nearby_prey[0]
                    self._switch_state(
                        AntState.HUNTING, f"HuntPrey@{self.target_prey.pos}"
                    )
                    return

            if self.state == AntState.DEFENDING:
                self._switch_state(AntState.PATROLLING, "ThreatLow")
            elif is_near_nest and self.state != AntState.PATROLLING:
                self._switch_state(AntState.PATROLLING, "NearNest")
            elif not is_near_nest and self.state == AntState.PATROLLING:
                self._switch_state(AntState.SEARCHING, "PatrolFar")
            elif is_near_nest and self.state == AntState.SEARCHING:
                self._switch_state(AntState.PATROLLING, "SearchNear")

        # Check if HUNTING target is lost (for both castes)
        if self.state == AntState.HUNTING:
            if (
                not self.target_prey
                or self.target_prey not in sim.prey
                or distance_sq(self.pos, self.target_prey.pos)
                > PREY_FLEE_RADIUS_SQ * 4
            ):
                self.target_prey = None
                default_state = (
                    AntState.PATROLLING
                    if self.caste == AntCaste.SOLDIER
                    else AntState.SEARCHING
                )
                self._switch_state(default_state, "LostPreyTarget")

    def update(self, speed_multiplier):
        """Update ant's state, position, age, food, and interactions."""
        # Aging
        self.age += speed_multiplier
        if self.age >= self.max_age_ticks:
            self.hp = 0
            self.last_move_info = "Died of old age"
            return

        # Food Consumption
        self.food_consumption_timer += speed_multiplier
        if self.food_consumption_timer >= WORKER_FOOD_CONSUMPTION_INTERVAL:
            self.food_consumption_timer %= WORKER_FOOD_CONSUMPTION_INTERVAL
            needed_s = self.food_consumption_sugar
            needed_p = self.food_consumption_protein
            sim = self.simulation
            if not (
                sim.colony_food_storage_sugar >= needed_s
                and sim.colony_food_storage_protein >= needed_p
            ):
                self.hp = 0
                self.last_move_info = "Starved"
                return
            sim.colony_food_storage_sugar -= needed_s
            sim.colony_food_storage_protein -= needed_p

        # State Updates
        if self.state == AntState.ESCAPING:
            self.escape_timer -= speed_multiplier
            if self.escape_timer <= 0:
                next_state = (
                    AntState.PATROLLING
                    if self.caste == AntCaste.SOLDIER
                    else AntState.SEARCHING
                )
                self._switch_state(next_state, "EscapeEnd")

        self._update_state()

        # Combat/Hunting Check
        pos_int = self.pos
        neighbors_int = get_neighbors(pos_int, True)
        enemies_in_range = [
            e
            for p_int in neighbors_int
            if (e := self.simulation.get_enemy_at(p_int)) and e.hp > 0
        ]
        if enemies_in_range:
            target_enemy = random.choice(enemies_in_range)
            self.attack(target_enemy)
            self.simulation.grid.add_pheromone(
                pos_int, P_ALARM_FIGHT, "alarm"
            )
            self.stuck_timer = 0
            self.target_prey = None
            self.last_move_info = f"Fighting Enemy {target_enemy.pos}"
            if self.state != AntState.DEFENDING:
                self._switch_state(AntState.DEFENDING, "EnemyContact!")
            return

        prey_in_range = [
            p
            for p_int in neighbors_int
            if (p := self.simulation.get_prey_at(p_int)) and p.hp > 0
        ]
        should_attack_prey = False
        target_prey_to_attack = None

        if self.state == AntState.HUNTING and self.target_prey and self.target_prey in prey_in_range:
             if self.target_prey.pos in neighbors_int: # Check adjacency
                 should_attack_prey = True
                 target_prey_to_attack = self.target_prey
        elif prey_in_range and self.state != AntState.RETURNING_TO_NEST:
             can_hunt_opportunistically = (self.caste == AntCaste.WORKER and self.simulation.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * 2) or \
                                          (self.caste == AntCaste.SOLDIER and self.state != AntState.DEFENDING)
             if can_hunt_opportunistically:
                  adjacent_prey = [p for p in prey_in_range if p.pos in neighbors_int and p.pos != pos_int]
                  prey_on_cell = [p for p in prey_in_range if p.pos == pos_int]
                  if adjacent_prey:
                       should_attack_prey = True
                       target_prey_to_attack = random.choice(adjacent_prey)
                  elif prey_on_cell:
                       should_attack_prey = True
                       target_prey_to_attack = random.choice(prey_on_cell)


        if should_attack_prey and target_prey_to_attack:
            self.attack(target_prey_to_attack)
            self.stuck_timer = 0
            self.last_move_info = f"Attacking Prey {target_prey_to_attack.pos}"
            if target_prey_to_attack.hp <= 0:
                killed_prey_pos = target_prey_to_attack.pos
                self.simulation.kill_prey(target_prey_to_attack)
                self.simulation.grid.add_pheromone(
                    killed_prey_pos, P_FOOD_AT_SOURCE, "food", FoodType.PROTEIN
                )
                self.simulation.grid.add_pheromone(
                    killed_prey_pos, P_RECRUIT_PREY, "recruitment"
                )
                if self.target_prey == target_prey_to_attack:
                    self.target_prey = None
                next_s = AntState.SEARCHING if self.caste == AntCaste.WORKER else AntState.PATROLLING
                self._switch_state(next_s, "PreyKilled")
            return

        # Movement Delay
        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return
        effective_delay_updates = 0
        if self.move_delay_base > 0 and speed_multiplier > 0:
            effective_delay_updates = max(
                0, int(round(self.move_delay_base / speed_multiplier)) - 1
            )
        elif self.move_delay_base > 0 and speed_multiplier == 0:
            effective_delay_updates = float("inf")
        self.move_delay_timer = effective_delay_updates

        # Choose and Execute Move
        old_pos_int = self.pos
        local_just_picked = self.just_picked_food
        self.just_picked_food = False
        new_pos_int = self._choose_move()
        moved = False
        found_food_type = None
        food_amount = 0.0

        if new_pos_int:
            if new_pos_int != old_pos_int:
                self.pos = new_pos_int
                self.last_move_direction = (
                    new_pos_int[0] - old_pos_int[0],
                    new_pos_int[1] - old_pos_int[1],
                )
                self._update_path_history(new_pos_int)
                self.stuck_timer = 0
                moved = True
                try:
                    x_int, y_int = new_pos_int
                    if 0 <= x_int < GRID_WIDTH and 0 <= y_int < GRID_HEIGHT:
                        foods = self.simulation.grid.food[x_int, y_int]
                        if foods[FoodType.SUGAR.value] > 0.1:
                            found_food_type = FoodType.SUGAR
                            food_amount = foods[FoodType.SUGAR.value]
                        elif foods[FoodType.PROTEIN.value] > 0.1:
                            found_food_type = FoodType.PROTEIN
                            food_amount = foods[FoodType.PROTEIN.value]
                except IndexError:
                    found_food_type = None
            else:
                self.stuck_timer += 1
                self.last_move_info += "(Move->Same)"
                self.last_move_direction = (0, 0)
        else:
            self.stuck_timer += 1
            self.last_move_info += "(NoChoice)"
            self.last_move_direction = (0, 0)

        # Post-Movement Actions
        pos_int = self.pos
        nest_pos_int = tuple(map(int, NEST_POS))
        is_near_nest = distance_sq(pos_int, nest_pos_int) <= NEST_RADIUS**2
        grid = self.simulation.grid
        sim = self.simulation

        if self.state in [AntState.SEARCHING, AntState.HUNTING]:
            if (
                self.caste == AntCaste.WORKER
                and found_food_type
                and self.carry_amount == 0
            ):
                pickup_amount = min(self.max_capacity, food_amount)
                if pickup_amount > 0.01:
                    self.carry_amount = pickup_amount
                    self.carry_type = found_food_type
                    food_idx = found_food_type.value
                    try:
                        x, y = pos_int
                        grid.food[x, y, food_idx] = max(
                            0, grid.food[x, y, food_idx] - pickup_amount
                        )
                        grid.add_pheromone(
                            pos_int,
                            P_FOOD_AT_SOURCE,
                            "food",
                            food_type=found_food_type,
                        )
                        if food_amount >= RICH_FOOD_THRESHOLD:
                            grid.add_pheromone(
                                pos_int, P_RECRUIT_FOOD, "recruitment"
                            )
                        self._switch_state(
                            AntState.RETURNING_TO_NEST,
                            f"Picked {found_food_type.name}",
                        )
                        self.just_picked_food = True
                        self.target_prey = None
                    except IndexError:
                        self.carry_amount = 0
                        self.carry_type = None
                else:
                    self.last_move_info += "(FoodTrace)"
            elif (
                moved
                and not found_food_type
                and self.state != AntState.HUNTING
                and distance_sq(pos_int, nest_pos_int) > (NEST_RADIUS + 3) ** 2
            ):
                if is_valid(old_pos_int):
                    grid.add_pheromone(
                        old_pos_int, P_NEGATIVE_SEARCH, "negative"
                    )

        elif self.state == AntState.RETURNING_TO_NEST:
            if is_near_nest:
                dropped_amount = self.carry_amount
                type_dropped = self.carry_type
                if dropped_amount > 0 and type_dropped:
                    if type_dropped == FoodType.SUGAR:
                        sim.colony_food_storage_sugar += dropped_amount
                    elif type_dropped == FoodType.PROTEIN:
                        sim.colony_food_storage_protein += dropped_amount
                    self.carry_amount = 0
                    self.carry_type = None

                next_state = AntState.SEARCHING
                clear_history = True # Usually clear history after returning
                state_reason = "" # Reason is built into the switch call

                if self.caste == AntCaste.WORKER:
                    sugar_crit = (
                        sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD
                    )
                    protein_crit = (
                        sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD
                    )
                    next_state = AntState.SEARCHING
                    reason_detail = f"Dropped{type_dropped.name if type_dropped else ''}({dropped_amount:.1f})"
                    if sugar_crit or protein_crit:
                         state_reason = "SEARCH(Need!)"
                    else:
                         state_reason = "SEARCH"
                    self._switch_state(next_state, f"{reason_detail}->{state_reason}")

                elif self.caste == AntCaste.SOLDIER:
                    next_state = AntState.PATROLLING
                    self._switch_state(next_state, "Returned->PATROL")

                # History clearing is handled by _switch_state

            elif moved and not local_just_picked:
                if (
                    is_valid(old_pos_int)
                    and distance_sq(old_pos_int, nest_pos_int) > NEST_RADIUS**2
                ):
                    grid.add_pheromone(old_pos_int, P_HOME_RETURNING, "home")
                    if self.carry_amount > 0 and self.carry_type:
                        grid.add_pheromone(
                            old_pos_int,
                            P_FOOD_RETURNING_TRAIL,
                            "food",
                            food_type=self.carry_type,
                        )

        # Stuck Check
        if (
            self.stuck_timer >= WORKER_STUCK_THRESHOLD
            and self.state != AntState.ESCAPING
        ):
            neighbors_int = get_neighbors(pos_int, True)
            is_fighting = any(
                sim.get_enemy_at(p_int) for p_int in neighbors_int
            )
            is_hunting_adjacent = False
            if self.state == AntState.HUNTING and self.target_prey:
                is_hunting_adjacent = self.target_prey in sim.prey and self.target_prey.pos in neighbors_int

            if not is_fighting and not is_hunting_adjacent:
                self._switch_state(AntState.ESCAPING, "Stuck")
                self.escape_timer = WORKER_ESCAPE_DURATION
                self.stuck_timer = 0


    def attack(self, target):
        """Attack either an Enemy or Prey."""
        if isinstance(target, (Enemy, Prey)):
            target.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        """Process damage taken by the ant."""
        if self.hp <= 0:
            return
        self.hp -= amount
        if self.hp > 0:
            grid = self.simulation.grid
            pos_int = self.pos
            grid.add_pheromone(pos_int, P_ALARM_FIGHT / 2, "alarm")
            recruit_param = (
                "P_RECRUIT_DAMAGE_SOLDIER"
                if self.caste == AntCaste.SOLDIER
                else "P_RECRUIT_DAMAGE"
            )
            # Use globals() as P_... are global constants
            grid.add_pheromone(pos_int, globals()[recruit_param], "recruitment")
        else:
            self.hp = 0


# --- Queen Class ---
class Queen:
    """Manages queen state and egg laying."""
    def __init__(self, pos, sim):
        self.pos = tuple(map(int, pos))
        self.simulation = sim
        self.hp = float(QUEEN_HP)
        self.max_hp = float(QUEEN_HP)
        self.age = 0.0
        self.egg_lay_timer_progress = 0.0
        self.egg_lay_interval_ticks = QUEEN_EGG_LAY_RATE
        self.color = QUEEN_COLOR
        self.attack_power = 0
        self.carry_amount = 0

    def update(self, speed_multiplier):
        """Update Queen's age and handle egg laying based on speed."""
        if speed_multiplier == 0.0:
            return
        self.age += speed_multiplier
        self.egg_lay_timer_progress += speed_multiplier
        if self.egg_lay_timer_progress >= self.egg_lay_interval_ticks:
            self.egg_lay_timer_progress %= self.egg_lay_interval_ticks
            needed_s = QUEEN_FOOD_PER_EGG_SUGAR
            needed_p = QUEEN_FOOD_PER_EGG_PROTEIN
            sim = self.simulation
            if (
                sim.colony_food_storage_sugar >= needed_s
                and sim.colony_food_storage_protein >= needed_p
            ):
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p
                caste = self._decide_caste()
                egg_pos = self._find_egg_position()
                if egg_pos:
                    sim.brood.append(
                        BroodItem(BroodStage.EGG, caste, egg_pos, int(sim.ticks))
                    )

    def _decide_caste(self):
        """Decide the caste of the next egg based on colony needs."""
        ratio = 0.0
        ants = self.simulation.ants
        brood = self.simulation.brood
        developing_brood = [
            b for b in brood if b.stage in [BroodStage.LARVA, BroodStage.PUPA]
        ]
        total_population = len(ants) + len(developing_brood)
        if total_population > 0:
            soldier_count = sum(
                1 for a in ants if a.caste == AntCaste.SOLDIER
            ) + sum(
                1 for b in developing_brood if b.caste == AntCaste.SOLDIER
            )
            ratio = soldier_count / total_population
        if ratio < QUEEN_SOLDIER_RATIO_TARGET:
            return (
                AntCaste.SOLDIER if random.random() < 0.65 else AntCaste.WORKER
            )
        elif random.random() < 0.04:
            return AntCaste.SOLDIER
        return AntCaste.WORKER

    def _find_egg_position(self):
        """Find a valid integer position near the queen for a new egg."""
        possible_spots = get_neighbors(self.pos)
        valid_spots = [
            p for p in possible_spots if not self.simulation.grid.is_obstacle(p)
        ]
        if not valid_spots:
            return None
        brood_positions = {b.pos for b in self.simulation.brood}
        free_valid_spots = [p for p in valid_spots if p not in brood_positions]
        return (
            random.choice(free_valid_spots)
            if free_valid_spots
            else random.choice(valid_spots)
        )

    def take_damage(self, amount, attacker):
        """Process damage taken by the queen."""
        if self.hp <= 0:
            return
        self.hp -= amount
        if self.hp > 0:
            grid = self.simulation.grid
            grid.add_pheromone(self.pos, P_ALARM_FIGHT * 4, "alarm")
            grid.add_pheromone(self.pos, P_RECRUIT_DAMAGE * 4, "recruitment")
        else:
            self.hp = 0


# --- Enemy Class ---
class Enemy:
    """Represents an enemy entity."""
    def __init__(self, pos, sim):
        self.pos = tuple(map(int, pos))
        self.simulation = sim
        self.hp = float(ENEMY_HP)
        self.max_hp = float(ENEMY_HP)
        self.attack_power = ENEMY_ATTACK
        self.move_delay_base = ENEMY_MOVE_DELAY
        self.move_delay_timer = rnd_uniform(0, self.move_delay_base)
        self.color = ENEMY_COLOR

    def update(self, speed_multiplier):
        """Update enemy state: attack or move."""
        if speed_multiplier == 0.0:
            return
        pos_int = self.pos
        neighbors_int = get_neighbors(pos_int, True)
        target_ants = [
            a
            for p_int in neighbors_int
            if (a := self.simulation.get_ant_at(p_int)) and a.hp > 0
        ]
        if target_ants:
            queen_target = next(
                (a for a in target_ants if isinstance(a, Queen)), None
            )
            target = queen_target if queen_target else random.choice(target_ants)
            self.attack(target)
            return

        self.move_delay_timer -= speed_multiplier
        if self.move_delay_timer > 0:
            return
        self.move_delay_timer += self.move_delay_base

        possible_moves_int = get_neighbors(pos_int)
        valid_moves_int = []
        for m_int in possible_moves_int:
            if (
                not self.simulation.grid.is_obstacle(m_int)
                and not self.simulation.is_enemy_at(m_int, self)
                and not self.simulation.is_ant_at(m_int)
                and not self.simulation.is_prey_at(m_int)
            ):
                valid_moves_int.append(m_int)

        if valid_moves_int:
            chosen_move_int = None
            nest_pos_int = tuple(map(int, NEST_POS))
            if random.random() < ENEMY_NEST_ATTRACTION:
                best_nest_move = None
                min_dist_sq = distance_sq(pos_int, nest_pos_int)
                for move in valid_moves_int:
                    d_sq = distance_sq(move, nest_pos_int)
                    if d_sq < min_dist_sq:
                        min_dist_sq = d_sq
                        best_nest_move = move
                chosen_move_int = (
                    best_nest_move
                    if best_nest_move
                    else random.choice(valid_moves_int)
                )
            else:
                chosen_move_int = random.choice(valid_moves_int)
            if chosen_move_int:
                self.pos = chosen_move_int

    def attack(self, target_ant):
        """Attack a target ant."""
        target_ant.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        """Process damage taken by the enemy."""
        if self.hp <= 0:
            return
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0


# --- Main Simulation Class ---
class AntSimulation:
    """Manages the overall simulation state, entities, drawing, and UI."""

    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ant Simulation - Prey Update")
        self.clock = pygame.time.Clock()
        self.font = None
        self.debug_font = None
        self._init_fonts()
        self.grid = WorldGrid()
        self.simulation_running = False
        self.app_running = True
        self.end_game_reason = ""
        self.colony_generation = 0
        self.ticks = 0.0
        self.ants = []
        self.enemies = []
        self.brood = []
        self.prey = []
        self.queen = None
        self.colony_food_storage_sugar = 0.0
        self.colony_food_storage_protein = 0.0
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_interval_ticks = ENEMY_SPAWN_RATE
        self.prey_spawn_timer = 0.0
        self.prey_spawn_interval_ticks = PREY_SPAWN_RATE
        self.show_debug_info = True
        self.simulation_speed_index = DEFAULT_SPEED_INDEX
        self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
        self.buttons = self._create_buttons()
        if self.app_running:
            self._reset_simulation()

    def _init_fonts(self):
        """Initialize fonts, handling potential errors."""
        try:
            pygame.font.init()
            self.font = pygame.font.SysFont("sans", 16)
            self.debug_font = pygame.font.SysFont("monospace", 14)
            print("Using system 'sans' and 'monospace' fonts.")
        except Exception as e1:
            print(f"System font error: {e1}. Trying default font.")
            try:
                self.font = pygame.font.Font(None, 20)
                self.debug_font = pygame.font.Font(None, 16)
                print("Using Pygame default font.")
            except Exception as e2:
                print(f"FATAL: Default font error: {e2}. Cannot render text.")
                self.font = None
                self.debug_font = None
                self.app_running = False

    def _reset_simulation(self):
        """Resets the simulation state for a new game generation."""
        print(f"Resetting simulation for Kolonie {self.colony_generation + 1}...")
        self.ticks = 0.0
        self.ants.clear()
        self.enemies.clear()
        self.brood.clear()
        self.prey.clear()
        self.queen = None
        self.colony_food_storage_sugar = INITIAL_COLONY_FOOD_SUGAR
        self.colony_food_storage_protein = INITIAL_COLONY_FOOD_PROTEIN
        self.enemy_spawn_timer = 0.0
        self.prey_spawn_timer = 0.0
        self.end_game_reason = ""
        self.colony_generation += 1
        self.grid.reset()
        if not self._spawn_initial_entities():
            print("CRITICAL ERROR during simulation reset. Cannot continue.")
            self.simulation_running = False
            self.app_running = False
            self.end_game_reason = "Initialisierungsfehler"
            return
        self.simulation_speed_index = DEFAULT_SPEED_INDEX
        self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
        self.simulation_running = True
        print(
            f"Kolonie {self.colony_generation} gestartet at "
            f"{SPEED_MULTIPLIERS[self.simulation_speed_index]:.1f}x speed."
        )

    def _create_buttons(self):
        """Creates data structures for UI speed control buttons (+/-)."""
        buttons = []
        button_h = 20
        button_w = 30
        margin = 5
        btn_plus_x = WIDTH - button_w - margin
        btn_minus_x = btn_plus_x - button_w - margin
        rect_minus = pygame.Rect(btn_minus_x, margin, button_w, button_h)
        buttons.append({"rect": rect_minus, "text": "-", "action": "speed_down"})
        rect_plus = pygame.Rect(btn_plus_x, margin, button_w, button_h)
        buttons.append({"rect": rect_plus, "text": "+", "action": "speed_up"})
        return buttons

    def _spawn_initial_entities(self):
        """Spawns the queen, initial ants, enemies, and prey."""
        queen_pos = self._find_valid_queen_pos()
        if queen_pos:
            self.queen = Queen(queen_pos, self)
            print(f"Queen placed at {queen_pos}")
        else:
            print("CRITICAL: Cannot place Queen.")
            return False

        spawned_ants = 0
        attempts = 0
        max_att = INITIAL_ANTS * 20
        queen_pos_int = self.queen.pos
        while spawned_ants < INITIAL_ANTS and attempts < max_att:
            radius = NEST_RADIUS + 2
            ox = rnd(-radius, radius)
            oy = rnd(-radius, radius)
            pos = (queen_pos_int[0] + ox, queen_pos_int[1] + oy)
            caste = (
                AntCaste.SOLDIER if random.random() < 0.2 else AntCaste.WORKER
            )
            if self.add_ant(pos, caste):
                spawned_ants += 1
            attempts += 1
        if spawned_ants < INITIAL_ANTS:
            print(f"Warn: Spawned only {spawned_ants}/{INITIAL_ANTS} ants.")

        enemies_spawned = 0
        for _ in range(INITIAL_ENEMIES):
            if self.spawn_enemy():
                enemies_spawned += 1
        print(f"Spawned {enemies_spawned}/{INITIAL_ENEMIES} enemies.")

        prey_spawned = 0
        for _ in range(INITIAL_PREY):
            if self.spawn_prey():
                prey_spawned += 1
        print(f"Spawned {prey_spawned}/{INITIAL_PREY} prey.")
        return True

    def _find_valid_queen_pos(self):
        """Finds a valid, non-obstacle integer position near nest center."""
        base_int = tuple(map(int, NEST_POS))
        if is_valid(base_int) and not self.grid.is_obstacle(base_int):
            return base_int
        neighbors = get_neighbors(base_int)
        random.shuffle(neighbors)
        for p_int in neighbors:
            if not self.grid.is_obstacle(p_int):
                return p_int
        for r in range(2, 5):
            perimeter = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        p_int = (base_int[0] + dx, base_int[1] + dy)
                        if is_valid(p_int) and not self.grid.is_obstacle(p_int):
                            perimeter.append(p_int)
            if perimeter:
                return random.choice(perimeter)
        print("CRITICAL: Could not find valid spot for Queen.")
        return None

    def add_ant(self, pos_int, caste: AntCaste):
        """Creates and adds an ant if the integer position is valid."""
        if (
            is_valid(pos_int)
            and not self.grid.is_obstacle(pos_int)
            and not self.is_ant_at(pos_int)
            and not self.is_enemy_at(pos_int)
            and not self.is_prey_at(pos_int)
            and (not self.queen or pos_int != self.queen.pos)
        ):
            self.ants.append(Ant(pos_int, self, caste))
            return True
        return False

    def spawn_enemy(self):
        """Spawns a new enemy at a valid random integer location."""
        tries = 0
        nest_pos_int = self.queen.pos if self.queen else tuple(map(int, NEST_POS))
        min_dist_sq = (MIN_FOOD_DIST_FROM_NEST + 5) ** 2
        while tries < 60:
            pos_i = (rnd(0, GRID_WIDTH - 1), rnd(0, GRID_HEIGHT - 1))
            if (
                distance_sq(pos_i, nest_pos_int) > min_dist_sq
                and not self.grid.is_obstacle(pos_i)
                and not self.is_enemy_at(pos_i)
                and not self.is_ant_at(pos_i)
                and not self.is_prey_at(pos_i)
            ):
                self.enemies.append(Enemy(pos_i, self))
                return True
            tries += 1
        return False

    def spawn_prey(self):
        """Spawns a new prey item at a valid random location."""
        tries = 0
        nest_pos_int = self.queen.pos if self.queen else tuple(map(int, NEST_POS))
        min_dist_sq = (MIN_FOOD_DIST_FROM_NEST - 5) ** 2
        while tries < 50:
            pos_i = (rnd(0, GRID_WIDTH - 1), rnd(0, GRID_HEIGHT - 1))
            if (
                distance_sq(pos_i, nest_pos_int) > min_dist_sq
                and not self.grid.is_obstacle(pos_i)
                and not self.is_enemy_at(pos_i)
                and not self.is_ant_at(pos_i)
                and not self.is_prey_at(pos_i)
            ):
                self.prey.append(Prey(pos_i, self))
                return True
            tries += 1
        return False

    def kill_ant(self, ant_to_remove, reason="unknown"):
        """Removes an ant from the simulation."""
        try:
            self.ants.remove(ant_to_remove)
        except ValueError:
            pass

    def kill_enemy(self, enemy_to_remove):
        """Removes an enemy and drops food resources."""
        try:
            pos_int = enemy_to_remove.pos
            if is_valid(pos_int) and not self.grid.is_obstacle(pos_int):
                fx, fy = pos_int
                grid = self.grid
                s_idx = FoodType.SUGAR.value
                p_idx = FoodType.PROTEIN.value
                try:
                    grid.food[fx, fy, s_idx] = min(
                        MAX_FOOD_PER_CELL,
                        grid.food[fx, fy, s_idx] + ENEMY_TO_FOOD_ON_DEATH_SUGAR,
                    )
                    grid.food[fx, fy, p_idx] = min(
                        MAX_FOOD_PER_CELL,
                        grid.food[fx, fy, p_idx]
                        + ENEMY_TO_FOOD_ON_DEATH_PROTEIN,
                    )
                except IndexError:
                    pass
            self.enemies.remove(enemy_to_remove)
        except ValueError:
            pass

    def kill_prey(self, prey_to_remove):
        """Removes prey and adds protein food to the grid."""
        try:
            pos_int = prey_to_remove.pos
            if is_valid(pos_int) and not self.grid.is_obstacle(pos_int):
                fx, fy = pos_int
                grid = self.grid
                p_idx = FoodType.PROTEIN.value
                try:
                    grid.food[fx, fy, p_idx] = min(
                        MAX_FOOD_PER_CELL,
                        grid.food[fx, fy, p_idx] + PROTEIN_ON_DEATH,
                    )
                except IndexError:
                    pass
            self.prey.remove(prey_to_remove)
        except ValueError:
            pass

    def kill_queen(self, queen_to_remove):
        """Handles the death of the queen."""
        if self.queen == queen_to_remove:
            print(
                f"\n--- QUEEN DIED (Tick {int(self.ticks)}, Kolonie {self.colony_generation}) ---"
            )
            print(
                f"    Food S:{self.colony_food_storage_sugar:.1f} P:{self.colony_food_storage_protein:.1f}"
            )
            print(f"    Ants:{len(self.ants)}, Brood:{len(self.brood)}")
            self.queen = None
            self.simulation_running = False
            self.end_game_reason = "Königin gestorben"

    def is_ant_at(self, pos_int, exclude_self=None):
        """Checks if an ant is at an integer position."""
        q = self.queen
        if q and q.pos == pos_int and exclude_self != q:
            return True
        return any(a.pos == pos_int for a in self.ants if a is not exclude_self)

    def get_ant_at(self, pos_int):
        """Returns the ant object at an integer position."""
        q = self.queen
        if q and q.pos == pos_int:
            return q
        return next((a for a in self.ants if a.pos == pos_int), None)

    def is_enemy_at(self, pos_int, exclude_self=None):
        """Checks if an enemy is at an integer position."""
        return any(e.pos == pos_int for e in self.enemies if e is not exclude_self)

    def get_enemy_at(self, pos_int):
        """Returns the enemy object at an integer position."""
        return next((e for e in self.enemies if e.pos == pos_int), None)

    def is_prey_at(self, pos_int, exclude_self=None):
        """Checks if a prey item is at an integer position."""
        return any(p.pos == pos_int for p in self.prey if p is not exclude_self)

    def get_prey_at(self, pos_int):
        """Returns the prey object at an integer position."""
        return next((p for p in self.prey if p.pos == pos_int), None)

    def find_nearby_prey(self, pos_int, radius_sq):
        """Finds prey within a certain squared radius of a position."""
        return [
            p
            for p in self.prey
            if distance_sq(pos_int, p.pos) <= radius_sq and p.hp > 0
        ]

    def update(self):
        """Runs one simulation tick."""
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]
        if current_multiplier == 0.0:
            self.ticks += 0.001
            return

        self.ticks += current_multiplier

        # Pre-Update Removal
        ants_to_remove = [
            a
            for a in self.ants
            if a.hp <= 0
            or a.age >= a.max_age_ticks
            or self.grid.is_obstacle(a.pos)
        ]
        for a in ants_to_remove:
            self.kill_ant(a, "pre-update")
        enemies_to_remove = [
            e
            for e in self.enemies
            if e.hp <= 0 or self.grid.is_obstacle(e.pos)
        ]
        for e in enemies_to_remove:
            self.kill_enemy(e)
        prey_to_remove = [
            p
            for p in self.prey
            if p.hp <= 0 or self.grid.is_obstacle(p.pos)
        ]
        for p in prey_to_remove:
            self.kill_prey(p)
        if self.queen and (
            self.queen.hp <= 0 or self.grid.is_obstacle(self.queen.pos)
        ):
            self.kill_queen(self.queen)

        if not self.simulation_running:
            return

        # Update Entities
        if self.queen:
            self.queen.update(current_multiplier)
        if not self.simulation_running:
            return

        hatched = []
        brood_copy = list(self.brood)
        for item in brood_copy:
            if item in self.brood:
                hatch_signal = item.update(self.ticks, self)
                if hatch_signal:
                    hatched.append(hatch_signal)
        for pupa in hatched:
            if pupa in self.brood:
                self.brood.remove(pupa)
                self._spawn_hatched_ant(pupa.caste, pupa.pos)

        ants_copy = list(self.ants)
        random.shuffle(ants_copy)
        enemies_copy = list(self.enemies)
        random.shuffle(enemies_copy)
        prey_copy = list(self.prey)
        random.shuffle(prey_copy)

        for a in ants_copy:
            if a in self.ants:
                a.update(current_multiplier)
        for e in enemies_copy:
            if e in self.enemies:
                e.update(current_multiplier)
        for p in prey_copy:
            if p in self.prey:
                p.update(current_multiplier)

        # Post-Update Removal
        final_ants_invalid = [a for a in self.ants if a.hp <= 0]
        for a in final_ants_invalid:
            self.kill_ant(a, "post-update")
        final_enemies_invalid = [e for e in self.enemies if e.hp <= 0]
        for e in final_enemies_invalid:
            self.kill_enemy(e)
        final_prey_invalid = [p for p in self.prey if p.hp <= 0]
        for p in final_prey_invalid:
            self.kill_prey(p)
        if self.queen and self.queen.hp <= 0:
            self.kill_queen(self.queen)

        if not self.simulation_running:
            return

        # Update Environment
        self.grid.update_pheromones(current_multiplier)
        self.enemy_spawn_timer += current_multiplier
        if self.enemy_spawn_timer >= self.enemy_spawn_interval_ticks:
            self.enemy_spawn_timer %= self.enemy_spawn_interval_ticks
            if len(self.enemies) < INITIAL_ENEMIES * 6:
                self.spawn_enemy()
        self.prey_spawn_timer += current_multiplier
        if self.prey_spawn_timer >= self.prey_spawn_interval_ticks:
            self.prey_spawn_timer %= self.prey_spawn_interval_ticks
            max_prey = INITIAL_PREY * 3
            if len(self.prey) < max_prey:
                self.spawn_prey()

    def _spawn_hatched_ant(self, caste: AntCaste, pupa_pos_int: tuple):
        """Tries to spawn a newly hatched ant near the pupa's location."""
        if self.add_ant(pupa_pos_int, caste):
            return True
        neighbors = get_neighbors(pupa_pos_int)
        random.shuffle(neighbors)
        for pos_int in neighbors:
            if self.add_ant(pos_int, caste):
                return True
        if self.queen:
            base_pos = self.queen.pos
            for _ in range(15):
                ox = rnd(-(NEST_RADIUS - 1), NEST_RADIUS - 1)
                oy = rnd(-(NEST_RADIUS - 1), NEST_RADIUS - 1)
                pos_int = (base_pos[0] + ox, base_pos[1] + oy)
                if self.add_ant(pos_int, caste):
                    return True
        return False

    # --- Drawing Methods ---
    def draw_debug_info(self):
        """Renders and draws the debug information overlay."""
        if not self.debug_font:
            return
        ant_c = len(self.ants)
        enemy_c = len(self.enemies)
        brood_c = len(self.brood)
        prey_c = len(self.prey)
        food_s = self.colony_food_storage_sugar
        food_p = self.colony_food_storage_protein
        tick_display = int(self.ticks)
        fps = self.clock.get_fps()
        w_c = sum(1 for a in self.ants if a.caste == AntCaste.WORKER)
        s_c = ant_c - w_c
        e_c = sum(1 for b in self.brood if b.stage == BroodStage.EGG)
        l_c = sum(1 for b in self.brood if b.stage == BroodStage.LARVA)
        p_c = brood_c - e_c - l_c
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]
        speed_text = (
            "Speed: Paused"
            if current_multiplier == 0.0
            else f"Speed: {current_multiplier:.1f}x".replace(".0x", "x")
        )

        texts = [
            f"Kolonie: {self.colony_generation}",
            f"Tick: {tick_display} FPS: {fps:.0f}",
            speed_text,
            f"Ants: {ant_c} (W:{w_c} S:{s_c})",
            f"Brood: {brood_c} (E:{e_c} L:{l_c} P:{p_c})",
            f"Enemies: {enemy_c}",
            f"Prey: {prey_c}",
            f"Food S:{food_s:.1f} P:{food_p:.1f}",
        ]
        y_start = 5
        line_height = self.debug_font.get_height() + 1
        text_color = (240, 240, 240)

        for i, txt in enumerate(texts):
            try:
                surf = self.debug_font.render(txt, True, text_color)
                self.screen.blit(surf, (5, y_start + i * line_height))
            except Exception as e:
                print(f"Debug Font render err: {e}")

        # Mouse Hover Info
        try:
            mx, my = pygame.mouse.get_pos()
            gx, gy = mx // CELL_SIZE, my // CELL_SIZE
            pos_i = (gx, gy)
            if is_valid(pos_i):
                hover_lines = []
                entity = (
                    self.get_ant_at(pos_i)
                    or self.get_enemy_at(pos_i)
                    or self.get_prey_at(pos_i)
                )
                if entity:
                    entity_pos_int = entity.pos
                    if isinstance(entity, Queen):
                        hover_lines.extend(
                            [
                                f"QUEEN @{entity_pos_int}",
                                f"HP:{entity.hp:.0f}/{entity.max_hp}",
                            ]
                        )
                    elif isinstance(entity, Ant):
                        hover_lines.extend(
                            [
                                f"{entity.caste.name} @{entity_pos_int}",
                                f"S:{entity.state.name} HP:{entity.hp:.0f}",
                                f"C:{entity.carry_amount:.1f}({entity.carry_type.name if entity.carry_type else '-'})",
                                f"Age:{entity.age:.0f}/{entity.max_age_ticks}",
                                f"Mv:{entity.last_move_info[:28]}",
                            ]
                        )
                    elif isinstance(entity, Enemy):
                        hover_lines.extend(
                            [
                                f"ENEMY @{entity_pos_int}",
                                f"HP:{entity.hp:.0f}/{entity.max_hp}",
                            ]
                        )
                    elif isinstance(entity, Prey):
                        hover_lines.extend(
                            [
                                f"PREY @{entity_pos_int}",
                                f"HP:{entity.hp:.0f}/{entity.max_hp}",
                            ]
                        )

                brood_at_pos = [b for b in self.brood if b.pos == pos_i]
                if brood_at_pos:
                    hover_lines.append(f"Brood:{len(brood_at_pos)} @{pos_i}")
                    for b in brood_at_pos[:3]:
                        hover_lines.append(
                            f"-{b.stage.name}({b.caste.name}) {int(b.progress_timer)}/{b.duration}"
                        )

                is_obs = self.grid.is_obstacle(pos_i)
                obs_txt = " OBSTACLE" if is_obs else ""
                hover_lines.append(f"Cell:{pos_i}{obs_txt}")
                if not is_obs:
                    try:
                        foods = self.grid.food[pos_i[0], pos_i[1]]
                        food_txt = f"Food S:{foods[0]:.1f} P:{foods[1]:.1f}"
                        ph_home = self.grid.get_pheromone(pos_i, "home")
                        ph_food_s = self.grid.get_pheromone(
                            pos_i, "food", FoodType.SUGAR
                        )
                        ph_food_p = self.grid.get_pheromone(
                            pos_i, "food", FoodType.PROTEIN
                        )
                        ph_alarm = self.grid.get_pheromone(pos_i, "alarm")
                        ph_neg = self.grid.get_pheromone(pos_i, "negative")
                        ph_rec = self.grid.get_pheromone(pos_i, "recruitment")
                        ph1 = f"Ph H:{ph_home:.0f} FS:{ph_food_s:.0f} FP:{ph_food_p:.0f}"
                        ph2 = f"Ph A:{ph_alarm:.0f} N:{ph_neg:.0f} R:{ph_rec:.0f}"
                        hover_lines.extend([food_txt, ph1, ph2])
                    except IndexError:
                        hover_lines.append("Error reading cell data")

                hover_color = (255, 255, 0)
                hover_y_start = HEIGHT - (len(hover_lines) * line_height) - 5
                for i, line in enumerate(hover_lines):
                    surf = self.debug_font.render(line, True, hover_color)
                    self.screen.blit(surf, (5, hover_y_start + i * line_height))
        except Exception:
            pass

    def draw(self):
        """Draws all simulation elements."""
        self._draw_grid()
        self._draw_brood()
        self._draw_queen()
        self._draw_entities()
        self._draw_prey()
        if self.show_debug_info:
            self.draw_debug_info()
        self._draw_buttons()
        pygame.display.flip()

    def _draw_grid(self):
        """Draws the grid background, obstacles, pheromones, and food."""
        cs = CELL_SIZE
        bg_surface = pygame.Surface((WIDTH, HEIGHT))
        bg_surface.fill(MAP_BG_COLOR)
        obstacle_coords = np.argwhere(self.grid.obstacles)
        for x, y in obstacle_coords:
            pygame.draw.rect(bg_surface, OBSTACLE_COLOR, (x * cs, y * cs, cs, cs))
        self.screen.blit(bg_surface, (0, 0))

        ph_types = [
            "home",
            "food_sugar",
            "food_protein",
            "alarm",
            "negative",
            "recruitment",
        ]
        ph_colors = {
            "home": PHEROMONE_HOME_COLOR,
            "food_sugar": PHEROMONE_FOOD_SUGAR_COLOR,
            "food_protein": PHEROMONE_FOOD_PROTEIN_COLOR,
            "alarm": PHEROMONE_ALARM_COLOR,
            "negative": PHEROMONE_NEGATIVE_COLOR,
            "recruitment": PHEROMONE_RECRUITMENT_COLOR,
        }
        ph_arrays = {
            "home": self.grid.pheromones_home,
            "food_sugar": self.grid.pheromones_food_sugar,
            "food_protein": self.grid.pheromones_food_protein,
            "alarm": self.grid.pheromones_alarm,
            "negative": self.grid.pheromones_negative,
            "recruitment": self.grid.pheromones_recruitment,
        }
        min_draw_ph = 0.5

        for ph_type in ph_types:
            ph_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            base_col = ph_colors[ph_type]
            arr = ph_arrays[ph_type]
            current_max = (
                RECRUITMENT_PHEROMONE_MAX
                if ph_type == "recruitment"
                else PHEROMONE_MAX
            )
            norm_divisor = max(current_max / 2.5, 1.0)
            nz_coords = np.argwhere(arr > min_draw_ph)

            for x, y in nz_coords:
                val = arr[x, y]
                norm_val = normalize(val, norm_divisor)
                alpha = min(max(int(norm_val * base_col[3]), 0), 255)
                if alpha > 4:
                    color = (*base_col[:3], alpha)
                    pygame.draw.rect(ph_surf, color, (x * cs, y * cs, cs, cs))
            self.screen.blit(ph_surf, (0, 0))

        min_draw_food = 0.1
        food_totals = np.sum(self.grid.food, axis=2)
        food_nz_coords = np.argwhere(food_totals > min_draw_food)
        for x, y in food_nz_coords:
            try:
                foods = self.grid.food[x, y]
                s = foods[FoodType.SUGAR.value]
                p = foods[FoodType.PROTEIN.value]
                total = s + p
                if total > 0.01:
                    sr = s / total
                    pr = p / total
                    s_col = FOOD_COLORS[FoodType.SUGAR]
                    p_col = FOOD_COLORS[FoodType.PROTEIN]
                    color = (
                        int(s_col[0] * sr + p_col[0] * pr),
                        int(s_col[1] * sr + p_col[1] * pr),
                        int(s_col[2] * sr + p_col[2] * pr),
                    )
                    pygame.draw.rect(self.screen, color, (x * cs, y * cs, cs, cs))
            except IndexError:
                continue

        r = NEST_RADIUS
        nx, ny = tuple(map(int, NEST_POS))
        nest_rect_coords = (
            (nx - r) * cs,
            (ny - r) * cs,
            (r * 2 + 1) * cs,
            (r * 2 + 1) * cs,
        )
        try:
            rect = pygame.Rect(nest_rect_coords)
            nest_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            nest_surf.fill((100, 100, 100, 35))
            self.screen.blit(nest_surf, rect.topleft)
        except ValueError:
            pass

    def _draw_brood(self):
        """Draws all brood items."""
        for item in list(self.brood):
            if item in self.brood and is_valid(item.pos):
                item.draw(self.screen)

    def _draw_queen(self):
        """Draws the queen ant."""
        if not self.queen or not is_valid(self.queen.pos):
            return
        pos_px = (
            int(self.queen.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.queen.pos[1] * CELL_SIZE + CELL_SIZE / 2),
        )
        radius = int(CELL_SIZE / 1.4)
        pygame.draw.circle(self.screen, self.queen.color, pos_px, radius)
        pygame.draw.circle(self.screen, (255, 255, 255), pos_px, radius, 1)

    def _draw_entities(self):
        """Draws all ants and enemies."""
        cs_half = CELL_SIZE / 2
        for a in list(self.ants):
            if a not in self.ants or not is_valid(a.pos):
                continue
            pos_px = (
                int(a.pos[0] * CELL_SIZE + cs_half),
                int(a.pos[1] * CELL_SIZE + cs_half),
            )
            radius = int(CELL_SIZE / a.size_factor)
            color = a.search_color
            if a.state == AntState.RETURNING_TO_NEST:
                color = a.return_color
            elif a.state == AntState.ESCAPING:
                color = WORKER_ESCAPE_COLOR
            elif a.state == AntState.DEFENDING:
                color = (255, 100, 0)
            elif a.state == AntState.HUNTING:
                color = (0, 200, 150)

            pygame.draw.circle(self.screen, color, pos_px, radius)
            if a.carry_amount > 0 and a.carry_type:
                food_color = FOOD_COLORS.get(a.carry_type, FOOD_COLOR_MIX)
                pygame.draw.circle(
                    self.screen, food_color, pos_px, int(radius * 0.55)
                )
        for e in list(self.enemies):
            if e not in self.enemies or not is_valid(e.pos):
                continue
            pos_px = (
                int(e.pos[0] * CELL_SIZE + cs_half),
                int(e.pos[1] * CELL_SIZE + cs_half),
            )
            radius = int(CELL_SIZE / 2.3)
            pygame.draw.circle(self.screen, e.color, pos_px, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), pos_px, radius, 1)

    def _draw_prey(self):
        """Draws all prey items."""
        for p in list(self.prey):
            if p in self.prey and is_valid(p.pos):
                p.draw(self.screen)

    def _draw_buttons(self):
        """Draws the UI buttons."""
        if not self.font:
            return
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            rect = button["rect"]
            text = button["text"]
            color = (
                BUTTON_HOVER_COLOR
                if rect.collidepoint(mouse_pos)
                else BUTTON_COLOR
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            try:
                text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)
            except Exception as e:
                print(f"Button font render error ('{text}'): {e}")

    # --- Event Handling ---
    def handle_events(self):
        """Processes Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.simulation_running = False
                self.app_running = False
                self.end_game_reason = "Fenster geschlossen"
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.simulation_running = False
                    self.end_game_reason = "ESC gedrückt"
                    return "sim_stop"
                if event.key == pygame.K_d:
                    self.show_debug_info = not self.show_debug_info
                if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self._handle_button_click("speed_down")
                    return "speed_change"
                if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    self._handle_button_click("speed_up")
                    return "speed_change"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self.simulation_running:
                        for button in self.buttons:
                            if button["rect"].collidepoint(event.pos):
                                self._handle_button_click(button["action"])
                                return "speed_change"
        return None

    def _handle_button_click(self, action):
        """Updates simulation speed index."""
        current_index = self.simulation_speed_index
        max_index = len(SPEED_MULTIPLIERS) - 1
        new_index = current_index
        if action == "speed_down":
            new_index = max(0, current_index - 1)
        elif action == "speed_up":
            new_index = min(max_index, current_index + 1)
        else:
            print(f"Warn: Unknown button action '{action}'")
            return
        if new_index != self.simulation_speed_index:
            self.simulation_speed_index = new_index
            self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
            new_speed = SPEED_MULTIPLIERS[self.simulation_speed_index]
            speed_str = "Paused" if new_speed == 0.0 else f"{new_speed:.1f}x"
            # print(f"Speed changed: {speed_str}") # Debug

    # --- End Game Dialog ---
    def _show_end_game_dialog(self):
        """Displays the 'Restart' or 'Quit' dialog."""
        if not self.font:
            return "quit"
        dialog_w, dialog_h = 300, 150
        dialog_x = (WIDTH - dialog_w) // 2
        dialog_y = (HEIGHT - dialog_h) // 2
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill(END_DIALOG_BG_COLOR)
        btn_w, btn_h = 100, 35
        btn_margin = 20
        btn_y = dialog_y + dialog_h - btn_h - 25
        total_btn_width = btn_w * 2 + btn_margin
        btn_restart_x = dialog_x + (dialog_w - total_btn_width) // 2
        btn_quit_x = btn_restart_x + btn_w + btn_margin
        restart_rect = pygame.Rect(btn_restart_x, btn_y, btn_w, btn_h)
        quit_rect = pygame.Rect(btn_quit_x, btn_y, btn_w, btn_h)
        text_color = (240, 240, 240)
        title_text = f"Kolonie {self.colony_generation} Ende"
        reason_text = f"Grund: {self.end_game_reason}"
        waiting_for_choice = True

        while waiting_for_choice and self.app_running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.app_running = False
                    waiting_for_choice = False
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.app_running = False
                    waiting_for_choice = False
                    return "quit"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if restart_rect.collidepoint(mouse_pos):
                        return "restart"
                    if quit_rect.collidepoint(mouse_pos):
                        self.app_running = False
                        return "quit"

            self.screen.blit(overlay, (0, 0))
            pygame.draw.rect(
                self.screen,
                (40, 40, 80),
                (dialog_x, dialog_y, dialog_w, dialog_h),
                border_radius=6,
            )
            try:
                title_surf = self.font.render(title_text, True, text_color)
                title_rect = title_surf.get_rect(
                    center=(dialog_x + dialog_w // 2, dialog_y + 35)
                )
                self.screen.blit(title_surf, title_rect)
                reason_surf = self.font.render(reason_text, True, text_color)
                reason_rect = reason_surf.get_rect(
                    center=(dialog_x + dialog_w // 2, dialog_y + 70)
                )
                self.screen.blit(reason_surf, reason_rect)
            except Exception:
                pass # Ignore font errors in dialog
            r_color = (
                BUTTON_HOVER_COLOR
                if restart_rect.collidepoint(mouse_pos)
                else BUTTON_COLOR
            )
            pygame.draw.rect(self.screen, r_color, restart_rect, border_radius=4)
            try:
                r_text_surf = self.font.render(
                    "Neu starten", True, BUTTON_TEXT_COLOR
                )
                r_text_rect = r_text_surf.get_rect(center=restart_rect.center)
                self.screen.blit(r_text_surf, r_text_rect)
            except Exception:
                pass
            q_color = (
                BUTTON_HOVER_COLOR
                if quit_rect.collidepoint(mouse_pos)
                else BUTTON_COLOR
            )
            pygame.draw.rect(self.screen, q_color, quit_rect, border_radius=4)
            try:
                q_text_surf = self.font.render(
                    "Beenden", True, BUTTON_TEXT_COLOR
                )
                q_text_rect = q_text_surf.get_rect(center=quit_rect.center)
                self.screen.blit(q_text_surf, q_text_rect)
            except Exception:
                pass
            pygame.display.flip()
            self.clock.tick(30)
        return "quit"

    # --- Main Loop (run) ---
    def run(self):
        """Main application loop."""
        print("Starting Ant Simulation...")
        print("Controls: D=Debug | ESC=End Run | +/- = Speed")
        while self.app_running:
            while self.simulation_running and self.app_running:
                action = self.handle_events()
                if not self.app_running:
                    break
                if action == "sim_stop":
                    break
                self.update()
                self.draw()
                self.clock.tick(self.current_target_fps)

            if not self.app_running:
                break
            if not self.end_game_reason:
                self.end_game_reason = "Simulation beendet"
            choice = self._show_end_game_dialog()
            if choice == "restart":
                self._reset_simulation()
            elif choice == "quit":
                self.app_running = False

        print("Exiting application.")
        try:
            pygame.quit()
            print("Pygame shut down.")
        except Exception as e:
            print(f"Error during Pygame quit: {e}")


# --- Start Simulation ---
if __name__ == "__main__":
    print("Initializing simulation...")
    try:
        import numpy

        print(f"NumPy version: {numpy.__version__}")
    except ImportError:
        print("FATAL: NumPy required.")
        input("Press Enter to Exit.")
        exit()
    try:
        import pygame

        print(f"Pygame version: {pygame.version.ver}")
    except ImportError as e:
        print(f"FATAL: Pygame import failed: {e}")
        input("Press Enter to Exit.")
        exit()
    except Exception as e:
        print(f"FATAL: Pygame import error: {e}")
        input("Press Enter to Exit.")
        exit()

    initialization_success = False
    try:
        pygame.init()
        if not pygame.display.get_init():
            raise RuntimeError("Display module failed")
        if not pygame.font.get_init():
            raise RuntimeError("Font module failed")
        print("Pygame initialized successfully.")
        initialization_success = True
    except Exception as e:
        print(f"FATAL: Pygame initialization failed: {e}")
        try:
            pygame.quit()
        except Exception:
            pass
        input("Press Enter to Exit.")
        exit()

    if initialization_success:
        simulation_instance = None
        try:
            simulation_instance = AntSimulation()
            if simulation_instance.app_running:
                simulation_instance.run()
            else:
                print("App cannot start due to init errors (fonts?).")
        except Exception as e:
            print("\n--- CRITICAL UNHANDLED EXCEPTION CAUGHT ---")
            traceback.print_exc()
            print("---------------------------------------------")
            print("Attempting to exit gracefully.")
            try:
                if simulation_instance:
                    simulation_instance.app_running = False
                pygame.quit()
            except Exception:
                pass
            input("Press Enter to Exit.")

    print("Simulation process finished.")

