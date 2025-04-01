# -*- coding: utf-8 -*-

# --- START OF FILE antsim.py ---
# Version mit Performance-Optimierungen (Dict Lookups, Static BG) und PEP8

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
MIN_PHEROMONE_DRAW_THRESHOLD = 0.5  # Optimization: Don't draw tiny amounts

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
W_AVOID_HISTORY = -1000.0  # Strong penalty for revisiting

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
P_FOOD_SEARCHING = 0.0  # Placeholder/Not used directly
P_FOOD_AT_NEST = 0.0  # Placeholder/Not used directly

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
    # Optimisation: Direct tuple access assumed after first check
    if not isinstance(pos, tuple) or len(pos) != 2:
        return False
    x, y = pos
    # Removed check for float/isfinite, assuming integer coords mostly
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def get_neighbors(pos, include_center=False):
    """Get valid integer neighbor coordinates for a given position."""
    # Assuming pos is generally valid when called internally
    x_int, y_int = int(pos[0]), int(pos[1])
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0 and not include_center:
                continue
            n_pos = (x_int + dx, y_int + dy)
            # Inline is_valid check for slight speedup
            if 0 <= n_pos[0] < GRID_WIDTH and 0 <= n_pos[1] < GRID_HEIGHT:
                neighbors.append(n_pos)
    return neighbors


def distance_sq(pos1, pos2):
    """Calculate squared Euclidean distance between two integer points."""
    # Assuming pos1 and pos2 are valid integer tuples when called internally
    try:
        x1, y1 = pos1
        x2, y2 = pos2
        return (x1 - x2) ** 2 + (y1 - y2) ** 2
    except (TypeError, ValueError, IndexError):
        # Fallback if assumptions fail, but should be rare in inner loops
        return float("inf")


def normalize(value, max_val):
    """Normalize a value to the range [0, 1], clamped."""
    if max_val <= 0:
        return 0.0
    # Use float() explicitly if inputs might be int
    norm_val = float(value) / float(max_val)
    # Slightly faster clamping? Benchmark needed if critical.
    return min(1.0, max(0.0, norm_val))


# --- Brood Class ---
class BroodItem:
    """Represents an item of brood (egg, larva, pupa) in the nest."""

    def __init__(
        self, stage: BroodStage, caste: AntCaste, position: tuple, current_tick: int
    ):
        self.stage = stage
        self.caste = caste
        self.pos = tuple(map(int, position)) # Ensure integer tuple
        self.creation_tick = current_tick
        self.progress_timer = 0.0
        self.last_feed_check = current_tick

        # Cache attributes based on stage
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
            # Should not happen, but provide safe defaults
            self.duration = 0
            self.color = (0, 0, 0, 0)
            self.radius = 0

    def update(self, current_tick, simulation):
        """Update progress, handle feeding, and check for stage transition."""
        current_multiplier = SPEED_MULTIPLIERS[simulation.simulation_speed_index]
        # Early exit if paused
        if current_multiplier == 0.0:
            return None

        self.progress_timer += current_multiplier

        # Larva feeding logic
        if self.stage == BroodStage.LARVA:
            if current_tick - self.last_feed_check >= LARVA_FEED_INTERVAL:
                self.last_feed_check = current_tick
                needed_p = LARVA_FOOD_CONSUMPTION_PROTEIN
                needed_s = LARVA_FOOD_CONSUMPTION_SUGAR
                # Direct access to simulation storage
                has_p = simulation.colony_food_storage_protein >= needed_p
                has_s = simulation.colony_food_storage_sugar >= needed_s

                if has_p and has_s:
                    simulation.colony_food_storage_protein -= needed_p
                    simulation.colony_food_storage_sugar -= needed_s
                else:
                    # Reverse progress if starved
                    self.progress_timer = max(
                        0.0, self.progress_timer - current_multiplier
                    )

        # Stage transition check
        if self.progress_timer >= self.duration:
            if self.stage == BroodStage.EGG:
                self.stage = BroodStage.LARVA
                self.progress_timer = 0.0
                self.duration = LARVA_DURATION
                self.color = LARVA_COLOR
                self.radius = CELL_SIZE // 4
                self.last_feed_check = current_tick # Reset feed check timer
                return None # Not hatched yet
            elif self.stage == BroodStage.LARVA:
                self.stage = BroodStage.PUPA
                self.progress_timer = 0.0
                self.duration = PUPA_DURATION
                self.color = PUPA_COLOR
                self.radius = int(CELL_SIZE / 3.5)
                return None # Not hatched yet
            elif self.stage == BroodStage.PUPA:
                return self # Signal hatching
        return None # No transition or hatching this tick

    def draw(self, surface):
        """Draw the brood item statically centered in its cell."""
        # Check validity upfront
        if not is_valid(self.pos) or self.radius <= 0:
            return
        # Calculate pixel position once
        center_x = int(self.pos[0] * CELL_SIZE + CELL_SIZE // 2)
        center_y = int(self.pos[1] * CELL_SIZE + CELL_SIZE // 2)
        draw_pos = (center_x, center_y)

        pygame.draw.circle(surface, self.color, draw_pos, self.radius)
        # Draw outline for pupa to indicate caste
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
        # Use float32 for memory efficiency if precision allows
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
        # Cache obstacle coordinates for faster drawing (if needed, now done in Sim)
        # self.obstacle_coords = []

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
        # Update obstacle cache if used here
        # self.obstacle_coords = np.argwhere(self.obstacles)

    def place_food_clusters(self):
        """Place initial food clusters of alternating types."""
        nest_pos_int = tuple(map(int, NEST_POS))
        min_dist_sq = MIN_FOOD_DIST_FROM_NEST**2

        for i in range(INITIAL_FOOD_CLUSTERS):
            food_type_index = i % NUM_FOOD_TYPES
            attempts = 0
            cx, cy = 0, 0
            found_spot = False
            # Try finding a suitable spot away from the nest
            while attempts < 150 and not found_spot:
                cx = rnd(0, GRID_WIDTH - 1)
                cy = rnd(0, GRID_HEIGHT - 1)
                # Check distance and obstacle in one go
                if (
                    not self.obstacles[cx, cy]
                    and distance_sq((cx, cy), nest_pos_int) > min_dist_sq
                ):
                    found_spot = True
                attempts += 1

            # Fallback 1: Any non-obstacle spot
            if not found_spot:
                attempts = 0
                while attempts < 200:
                    cx = rnd(0, GRID_WIDTH - 1)
                    cy = rnd(0, GRID_HEIGHT - 1)
                    if not self.obstacles[cx, cy]:
                        found_spot = True
                        break
                    attempts += 1

            # Fallback 2: Any spot (overwrite potential obstacle food)
            if not found_spot:
                cx = rnd(0, GRID_WIDTH - 1)
                cy = rnd(0, GRID_HEIGHT - 1)
                # Optionally log a warning here if placement is suboptimal

            # Distribute food around the cluster center using Gaussian
            added_amount = 0.0
            target_food_amount = FOOD_PER_CLUSTER
            max_placement_attempts = int(target_food_amount * 2.5) # Limit loop

            for _ in range(max_placement_attempts):
                if added_amount >= target_food_amount:
                    break
                # Use integer Gaussian distribution directly
                fx = cx + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                fy = cy + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))

                # Check bounds and obstacle before accessing array
                if 0 <= fx < GRID_WIDTH and 0 <= fy < GRID_HEIGHT and not self.obstacles[fx, fy]:
                    amount_to_add = rnd_uniform(0.5, 1.0) * (
                        MAX_FOOD_PER_CELL / 8
                    )
                    current_amount = self.food[fx, fy, food_type_index]
                    # Calculate actual added amount after clamping
                    new_amount = min(
                        MAX_FOOD_PER_CELL, current_amount + amount_to_add
                    )
                    actual_added = new_amount - current_amount
                    if actual_added > 0:
                        self.food[fx, fy, food_type_index] = new_amount
                        added_amount += actual_added

    def place_obstacles(self):
        """Place rectangular obstacles, avoiding the immediate nest area."""
        nest_area = set()
        nest_radius_buffer = NEST_RADIUS + 3
        nest_center_int = tuple(map(int, NEST_POS))
        # Calculate bounds for checking nest area
        min_x = max(0, nest_center_int[0] - nest_radius_buffer)
        max_x = min(GRID_WIDTH - 1, nest_center_int[0] + nest_radius_buffer)
        min_y = max(0, nest_center_int[1] - nest_radius_buffer)
        max_y = min(GRID_HEIGHT - 1, nest_center_int[1] + nest_radius_buffer)

        # Populate the set of coordinates to avoid near the nest
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if distance_sq((x, y), nest_center_int) <= nest_radius_buffer**2:
                    nest_area.add((x, y))

        placed_count = 0
        max_placement_attempts = NUM_OBSTACLES * 10 # Limit overall attempts

        for _ in range(max_placement_attempts):
            if placed_count >= NUM_OBSTACLES:
                break

            # Try placing one obstacle
            attempts_per_obstacle = 0
            placed_this_obstacle = False
            while attempts_per_obstacle < 25 and not placed_this_obstacle:
                w = rnd(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                h = rnd(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                # Ensure obstacle fits within bounds
                x = rnd(0, GRID_WIDTH - w)
                y = rnd(0, GRID_HEIGHT - h)

                # Check for overlap with nest area
                overlaps_nest = False
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        if (i, j) in nest_area:
                            overlaps_nest = True
                            break
                    if overlaps_nest:
                        break

                # If no overlap, place the obstacle using slicing
                if not overlaps_nest:
                    # Check bounds again just in case (should be correct due to rnd range)
                    if x + w <= GRID_WIDTH and y + h <= GRID_HEIGHT:
                        self.obstacles[x : x + w, y : y + h] = True
                        placed_this_obstacle = True
                        placed_count += 1
                attempts_per_obstacle += 1

        if placed_count < NUM_OBSTACLES:
            print(f"Warning: Placed only {placed_count}/{NUM_OBSTACLES} obstacles.")


    def is_obstacle(self, pos):
        """Check if a given position corresponds to an obstacle cell."""
        # Assume pos is usually a valid tuple from internal calls
        try:
            x, y = pos
            # Combine bounds check and array access
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                return self.obstacles[x, y]
            else:
                return True # Out of bounds is treated as obstacle
        except (IndexError, TypeError, ValueError):
            return True # Invalid input treated as obstacle

    def get_pheromone(self, pos, ph_type="home", food_type: FoodType = None):
        """Get the pheromone value at a specific integer position."""
        # Assume pos is usually a valid tuple from internal calls
        try:
            x, y = pos
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                return 0.0
        except (ValueError, TypeError, IndexError):
            return 0.0

        # Use a dictionary for faster array lookup based on type string
        pheromone_map = {
            "home": self.pheromones_home,
            "alarm": self.pheromones_alarm,
            "negative": self.pheromones_negative,
            "recruitment": self.pheromones_recruitment,
            # Handle food types within the 'food' key check
        }

        try:
            if ph_type == "food":
                if food_type == FoodType.SUGAR:
                    return self.pheromones_food_sugar[x, y]
                elif food_type == FoodType.PROTEIN:
                    return self.pheromones_food_protein[x, y]
                else:
                    return 0.0 # Unknown food type
            elif ph_type in pheromone_map:
                return pheromone_map[ph_type][x, y]
            else:
                return 0.0 # Unknown pheromone type
        except IndexError:
            # Should ideally not happen if bounds check passed, but safeguard
            return 0.0

    def add_pheromone(
        self, pos, amount, ph_type="home", food_type: FoodType = None
    ):
        """Add pheromone to a specific integer position, clamping."""
        # Early exit for invalid amount or obstacle
        if amount <= 0 or self.is_obstacle(pos):
            return
        # Assume pos is usually a valid tuple from internal calls
        try:
            x, y = pos
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                return
        except (ValueError, TypeError, IndexError):
            return

        target_array = None
        max_value = PHEROMONE_MAX # Default max

        # Select target array and max value
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
                return # Invalid food type
        else:
            return # Invalid pheromone type

        # Add amount and clamp
        if target_array is not None:
            try:
                current_val = target_array[x, y]
                target_array[x, y] = min(current_val + amount, max_value)
            except IndexError:
                pass # Should not happen after bounds check

    def update_pheromones(self, speed_multiplier):
        """Update pheromones: apply decay and diffusion using NumPy."""
        effective_multiplier = max(0.0, speed_multiplier)
        if effective_multiplier == 0.0:
            return # No change if paused

        # Calculate effective decay factors (clamped to avoid zeroing too fast)
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

        # Apply decay directly using broadcasting
        self.pheromones_home *= decay_factor_common
        self.pheromones_alarm *= decay_factor_common
        self.pheromones_negative *= decay_factor_neg
        self.pheromones_recruitment *= decay_factor_rec
        self.pheromones_food_sugar *= decay_factor_common
        self.pheromones_food_protein *= decay_factor_common

        # Calculate effective diffusion rates
        diffusion_rate_common = PHEROMONE_DIFFUSION_RATE * effective_multiplier
        diffusion_rate_neg = (
            NEGATIVE_PHEROMONE_DIFFUSION_RATE * effective_multiplier
        )
        diffusion_rate_rec = (
            RECRUITMENT_PHEROMONE_DIFFUSION_RATE * effective_multiplier
        )

        # Clamp diffusion rates to prevent excessive spreading/instability
        max_diffusion = 0.124 # Max proportion diffusing to neighbours (8 * 0.125 = 1)
        diffusion_rate_common = min(max_diffusion, max(0.0, diffusion_rate_common))
        diffusion_rate_neg = min(max_diffusion, max(0.0, diffusion_rate_neg))
        diffusion_rate_rec = min(max_diffusion, max(0.0, diffusion_rate_rec))

        # Create mask for non-obstacle cells ONCE
        obstacle_mask = ~self.obstacles

        # List of arrays and their diffusion rates for iteration
        arrays_rates = [
            (self.pheromones_home, diffusion_rate_common),
            (self.pheromones_food_sugar, diffusion_rate_common),
            (self.pheromones_food_protein, diffusion_rate_common),
            (self.pheromones_alarm, diffusion_rate_common),
            (self.pheromones_negative, diffusion_rate_neg),
            (self.pheromones_recruitment, diffusion_rate_rec),
        ]

        # Kernel for average diffusion (8 neighbours)
        # Using explicit sum is often clearer and potentially faster than convolution for simple kernels
        diffusion_kernel_divisor = 8.0

        for arr, rate in arrays_rates:
            if rate > 0:
                # Apply obstacle mask: Pheromones on obstacles don't diffuse out
                masked_arr = arr * obstacle_mask
                # Pad the array to handle boundaries smoothly
                pad = np.pad(masked_arr, 1, mode='constant')

                # Calculate sum of 8 neighbors using array slicing
                neighbors_sum = (
                    pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +  # Top row
                    pad[1:-1, :-2] +               pad[1:-1, 2:] +  # Middle row (sides)
                    pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]     # Bottom row
                )

                # Calculate diffused value: (1-rate)*current + rate*(average_of_neighbors)
                diffused = masked_arr * (1.0 - rate) + (neighbors_sum / diffusion_kernel_divisor) * rate

                # Update the original array ONLY where there are no obstacles
                # Pheromones cannot diffuse INTO obstacle cells
                arr[:] = np.where(obstacle_mask, diffused, 0)

        # --- Post-processing ---
        # Clamp values to max and set very low values to zero
        min_pheromone_threshold = 0.01
        pheromone_arrays = [
            (self.pheromones_home, PHEROMONE_MAX),
            (self.pheromones_food_sugar, PHEROMONE_MAX),
            (self.pheromones_food_protein, PHEROMONE_MAX),
            (self.pheromones_alarm, PHEROMONE_MAX),
            (self.pheromones_negative, PHEROMONE_MAX), # Note: No separate max for negative? Ok.
            (self.pheromones_recruitment, RECRUITMENT_PHEROMONE_MAX),
        ]
        for arr, max_val in pheromone_arrays:
            np.clip(arr, 0, max_val, out=arr) # Clip in-place
            arr[arr < min_pheromone_threshold] = 0 # Zero out tiny values


# --- Prey Class ---
class Prey:
    """Represents a small creature that ants can hunt for protein."""

    def __init__(self, pos, sim):
        self.pos = tuple(map(int, pos)) # Ensure integer tuple
        self.simulation = sim # Reference to main simulation
        self.hp = float(PREY_HP)
        self.max_hp = float(PREY_HP)
        self.move_delay_base = PREY_MOVE_DELAY
        self.move_delay_timer = rnd_uniform(0, self.move_delay_base)
        self.color = PREY_COLOR
        # self.fleeing_ant = None # Removed, logic checks nearby ants directly

    def update(self, speed_multiplier):
        """Update prey state: check for threats, flee or wander."""
        if speed_multiplier == 0.0:
            return # Paused

        grid = self.simulation.grid
        pos_int = self.pos # Cache current position

        # --- Fleeing Behavior ---
        nearest_ant_pos = None # Store position instead of object ref
        min_dist_sq = PREY_FLEE_RADIUS_SQ

        # Check for nearby ants efficiently using the simulation's lookup
        check_radius = int(PREY_FLEE_RADIUS_SQ**0.5) + 1
        min_x = max(0, pos_int[0] - check_radius)
        max_x = min(GRID_WIDTH - 1, pos_int[0] + check_radius)
        min_y = max(0, pos_int[1] - check_radius)
        max_y = min(GRID_HEIGHT - 1, pos_int[1] + check_radius)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                check_pos = (x, y)
                if check_pos == pos_int: # Don't check self
                    continue
                # Use optimized lookup
                ant = self.simulation.get_ant_at(check_pos)
                if ant: # Found an ant
                    d_sq = distance_sq(pos_int, check_pos) # Use ant's pos
                    if d_sq < min_dist_sq:
                        min_dist_sq = d_sq
                        nearest_ant_pos = check_pos # Store position

        # --- Movement Delay ---
        self.move_delay_timer -= speed_multiplier
        if self.move_delay_timer > 0:
            return # Waiting to move
        # Reset timer for next move
        self.move_delay_timer += self.move_delay_base

        # --- Movement Logic ---
        possible_moves = get_neighbors(pos_int)
        # Filter for valid moves (no obstacles, enemies, other prey, or ants)
        valid_moves = [
            m for m in possible_moves
            if not grid.is_obstacle(m)
            and not self.simulation.is_enemy_at(m)
            # Pass self to exclude checking against own current position
            and not self.simulation.is_prey_at(m, exclude_self=self)
            and not self.simulation.is_ant_at(m) # Ants block prey movement
        ]

        if not valid_moves:
            return # Cannot move

        chosen_move = None
        if nearest_ant_pos: # Fleeing logic takes priority
            # Calculate ideal flee direction (away from nearest ant)
            flee_dx = pos_int[0] - nearest_ant_pos[0]
            flee_dy = pos_int[1] - nearest_ant_pos[1]

            best_flee_move = None
            max_flee_score = -float("inf")

            # Score each valid move based on how well it aligns with flee direction
            for move in valid_moves:
                move_dx = move[0] - pos_int[0]
                move_dy = move[1] - pos_int[1]
                # Dot product alignment + bonus for increasing distance slightly
                # Normalize delta values roughly to avoid large distances dominating
                dist_approx = max(1, abs(flee_dx) + abs(flee_dy))
                norm_flee_dx = flee_dx / dist_approx
                norm_flee_dy = flee_dy / dist_approx
                alignment_score = move_dx * norm_flee_dx + move_dy * norm_flee_dy
                distance_score = distance_sq(move, nearest_ant_pos) * 0.05 # Smaller factor

                score = alignment_score + distance_score

                if score > max_flee_score:
                    max_flee_score = score
                    best_flee_move = move

            # Choose the best flee move, or a random one if none was clearly best
            chosen_move = best_flee_move if best_flee_move else random.choice(valid_moves)

        else: # Wander randomly if no threat nearby
            chosen_move = random.choice(valid_moves)

        # Execute the move
        if chosen_move and chosen_move != self.pos:
            old_pos = self.pos
            self.pos = chosen_move
            # IMPORTANT: Update position in simulation's lookup dictionary
            self.simulation.update_entity_position(self, old_pos, self.pos)


    def take_damage(self, amount, attacker):
        """Process damage taken by the prey."""
        if self.hp <= 0:
            return # Already dead
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            # Death is handled by the simulation's cleanup phase

    def draw(self, surface):
        """Draw the prey."""
        if not is_valid(self.pos):
            return
        # Calculate pixel position once
        pos_px = (
            int(self.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos[1] * CELL_SIZE + CELL_SIZE / 2),
        )
        radius = int(CELL_SIZE / 2.8) # Use integer division or int()
        pygame.draw.circle(surface, self.color, pos_px, radius)
        # Optional: Draw HP bar?
        # hp_ratio = self.hp / self.max_hp
        # hp_bar_width = int(radius * 1.8)
        # hp_bar_height = 3
        # hp_bar_x = pos_px[0] - hp_bar_width // 2
        # hp_bar_y = pos_px[1] + radius + 1
        # pygame.draw.rect(surface, (50, 50, 50), (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
        # pygame.draw.rect(surface, (0, 255, 0), (hp_bar_x, hp_bar_y, int(hp_bar_width * hp_ratio), hp_bar_height))


# --- Ant Class ---
class Ant:
    """Represents a worker or soldier ant."""

    def __init__(self, pos, simulation, caste: AntCaste):
        self.pos = tuple(map(int, pos)) # Ensure integer tuple
        self.simulation = simulation # Reference to main simulation
        self.caste = caste
        # Load attributes from dictionary
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
        # State and carrying
        self.state = AntState.SEARCHING
        self.carry_amount = 0.0
        self.carry_type: FoodType | None = None # Type hint
        # Age and lifespan
        self.age = 0.0
        self.max_age_ticks = int(
            rnd_gauss(WORKER_MAX_AGE_MEAN, WORKER_MAX_AGE_STDDEV)
        )
        # Movement and state tracking
        self.path_history = [] # Stores recent integer positions
        self.history_timestamps = [] # Corresponding timestamps
        self.move_delay_timer = 0
        self.last_move_direction = (0, 0) # For persistence bonus
        self.stuck_timer = 0
        self.escape_timer = 0.0
        self.last_move_info = "Born" # For debugging hover
        self.just_picked_food = False # Flag for pheromone logic
        self.food_consumption_timer = rnd_uniform(
            0, WORKER_FOOD_CONSUMPTION_INTERVAL
        )
        self.last_known_alarm_pos = None # For DEFENDING state
        self.target_prey = None # For HUNTING state

    def _update_path_history(self, new_pos_int):
        """Adds integer position to history if different, trims old entries."""
        current_sim_ticks = self.simulation.ticks # Get current time
        # Add to history only if it's a new position
        if not self.path_history or self.path_history[-1] != new_pos_int:
            self.path_history.append(new_pos_int)
            self.history_timestamps.append(current_sim_ticks)

            # Trim old history based on time length
            cutoff_time = current_sim_ticks - WORKER_PATH_HISTORY_LENGTH
            cutoff_index = 0
            # Find the first index that is NOT too old
            while (
                cutoff_index < len(self.history_timestamps)
                and self.history_timestamps[cutoff_index] < cutoff_time
            ):
                cutoff_index += 1
            # Slice lists to keep only recent history
            self.path_history = self.path_history[cutoff_index:]
            self.history_timestamps = self.history_timestamps[cutoff_index:]

    def _is_in_history(self, pos_int):
        """Check if an integer position is in the recent path history."""
        # Optimised check using 'in' on the list
        return pos_int in self.path_history

    def _clear_path_history(self):
        """Clears the path history."""
        self.path_history.clear()
        self.history_timestamps.clear()

    def _filter_valid_moves(
        self, potential_neighbors_int, ignore_history_near_nest=False
    ):
        """Filter potential integer moves: obstacles, history, queen, ants."""
        valid_moves_int = []
        # Cache queen position if she exists
        q_pos_int = self.simulation.queen.pos if self.simulation.queen else None
        pos_int = self.pos # Cache current position
        nest_pos_int = tuple(map(int, NEST_POS)) # Cache nest position

        # Determine if currently near the nest for history ignoring logic
        is_near_nest_now = distance_sq(pos_int, nest_pos_int) <= (
            NEST_RADIUS + 2 # Use a slightly larger radius for check
        ) ** 2
        check_history_flag = not (ignore_history_near_nest and is_near_nest_now)

        # Iterate through potential neighbors
        for n_pos_int in potential_neighbors_int:
            # Check history blocking first (potentially common)
            history_block = False
            if check_history_flag and self._is_in_history(n_pos_int):
                history_block = True

            if not history_block:
                # Check other blocking conditions using optimized lookups
                is_queen_pos = n_pos_int == q_pos_int
                is_obstacle_pos = self.simulation.grid.is_obstacle(n_pos_int)
                # Use optimized simulation lookup, excluding self
                is_ant_pos = self.simulation.is_ant_at(n_pos_int, exclude_self=self)

                if not is_queen_pos and not is_obstacle_pos and not is_ant_pos:
                    valid_moves_int.append(n_pos_int)

        return valid_moves_int

    def _choose_move(self):
        """Determine the next integer move based on state, goals, environment."""
        potential_neighbors_int = get_neighbors(self.pos)
        if not potential_neighbors_int:
            self.last_move_info = "No neighbors"
            return None # Cannot move if no neighbors exist

        # Determine if history should be ignored near nest (only when returning)
        ignore_hist_near_nest = self.state == AntState.RETURNING_TO_NEST
        valid_neighbors_int = self._filter_valid_moves(
            potential_neighbors_int, ignore_hist_near_nest
        )

        # --- Handle being blocked ---
        if not valid_neighbors_int:
            self.last_move_info = "Blocked"
            # Fallback: Consider moves even if in history (but not obstacles/queen/ants)
            fallback_neighbors_int = []
            q_pos_int = self.simulation.queen.pos if self.simulation.queen else None
            for n_pos_int in potential_neighbors_int:
                # Check only essential blocks
                if (
                    n_pos_int != q_pos_int
                    and not self.simulation.grid.is_obstacle(n_pos_int)
                    and not self.simulation.is_ant_at(n_pos_int, exclude_self=self)
                ):
                    fallback_neighbors_int.append(n_pos_int)

            # If fallback options exist, try moving to the oldest visited one
            if fallback_neighbors_int:
                # Sort by index in history (older is better), -1 if not in history
                fallback_neighbors_int.sort(
                    key=lambda p: self.path_history.index(p)
                    if p in self.path_history
                    else -1
                )
                return fallback_neighbors_int[0] # Choose the least recently visited
            return None # Completely blocked

        # --- Standard State-Based Movement ---
        # Special handling for ESCAPING state (prioritize unvisited cells)
        if self.state == AntState.ESCAPING:
            # Prefer moves not in recent history
            escape_moves_int = [
                p for p in valid_neighbors_int if not self._is_in_history(p)
            ]
            if escape_moves_int:
                self.last_move_info = "Esc->Unhist"
                return random.choice(escape_moves_int)
            else:
                # If all valid moves are in history, pick any valid one
                self.last_move_info = "Esc->Hist"
                return random.choice(valid_neighbors_int)

        # Score moves based on current state
        move_scores = {}
        # Use a dictionary dispatch for state scoring functions
        scoring_functions = {
            AntState.RETURNING_TO_NEST: self._score_moves_returning,
            AntState.SEARCHING: self._score_moves_searching,
            AntState.PATROLLING: self._score_moves_patrolling,
            AntState.DEFENDING: self._score_moves_defending,
            AntState.HUNTING: self._score_moves_hunting,
        }
        # Get the appropriate scoring function or default to searching
        score_func = scoring_functions.get(self.state, self._score_moves_searching)
        # Pass just_picked_food only to returning scorer
        if self.state == AntState.RETURNING_TO_NEST:
             move_scores = score_func(valid_neighbors_int, self.just_picked_food)
        else:
             move_scores = score_func(valid_neighbors_int)


        if not move_scores:
            # Should not happen if valid_neighbors_int is not empty, but safeguard
            self.last_move_info = f"No scores({self.state.name})"
            return random.choice(valid_neighbors_int)

        # Select move based on scores and state logic
        selected_move_int = None
        # Use different selection strategies based on state
        if self.state == AntState.RETURNING_TO_NEST:
            selected_move_int = self._select_best_move_returning(
                move_scores, valid_neighbors_int, self.just_picked_food
            )
        # Defending/Hunting uses deterministic best choice
        elif self.state in [AntState.DEFENDING, AntState.HUNTING]:
            selected_move_int = self._select_best_move(
                move_scores, valid_neighbors_int
            )
        # Other states use probabilistic choice
        else:
            selected_move_int = self._select_probabilistic_move(
                move_scores, valid_neighbors_int
            )

        # Final fallback if selection somehow fails
        return selected_move_int if selected_move_int else random.choice(valid_neighbors_int)


    def _score_moves_base(self, neighbor_pos_int):
        """Calculates a base score (persistence, randomness) for a move."""
        score = 0.0
        # Persistence bonus: Check if move continues in the same direction
        move_dir = (
            neighbor_pos_int[0] - self.pos[0],
            neighbor_pos_int[1] - self.pos[1],
        )
        if move_dir == self.last_move_direction and move_dir != (0, 0):
            score += W_PERSISTENCE
        # Add random noise for exploration
        score += rnd_uniform(-W_RANDOM_NOISE, W_RANDOM_NOISE)
        return score

    def _score_moves_returning(self, valid_neighbors_int, just_picked):
        """Scores potential integer moves for returning to the nest."""
        scores = {}
        pos_int = self.pos # Cache current position
        nest_pos_int = tuple(map(int, NEST_POS)) # Cache nest position
        # Cache grid reference
        grid = self.simulation.grid
        # Calculate current distance squared once
        dist_sq_now = distance_sq(pos_int, nest_pos_int)

        for n_pos_int in valid_neighbors_int:
            # Base score (randomness, persistence)
            score = self._score_moves_base(n_pos_int)

            # Pheromone influences
            home_ph = grid.get_pheromone(n_pos_int, "home")
            # sugar_ph = grid.get_pheromone(n_pos_int, "food", FoodType.SUGAR) # Less relevant when returning?
            # protein_ph = grid.get_pheromone(n_pos_int, "food", FoodType.PROTEIN)
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            neg_ph = grid.get_pheromone(n_pos_int, "negative")

            # Strong pull towards home pheromone and nest direction
            score += home_ph * W_HOME_PHEROMONE_RETURN
            # Bonus for getting closer to the nest
            if distance_sq(n_pos_int, nest_pos_int) < dist_sq_now:
                score += W_NEST_DIRECTION_RETURN

            # Avoidance of negative signals (scaled down when returning)
            score += alarm_ph * W_ALARM_PHEROMONE * 0.3
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.4

            # If just picked food, slightly avoid own food trail type
            # to prevent immediate U-turn back to depleted source
            # if just_picked and self.carry_type:
            #     if self.carry_type == FoodType.SUGAR:
            #         score -= grid.get_pheromone(n_pos_int, "food", FoodType.SUGAR) * W_FOOD_PHEROMONE_SEARCH_BASE * 0.1 # Small factor
            #     elif self.carry_type == FoodType.PROTEIN:
            #         score -= grid.get_pheromone(n_pos_int, "food", FoodType.PROTEIN) * W_FOOD_PHEROMONE_SEARCH_BASE * 0.1

            # Add penalty for history (applied via filter, could add weight here too if needed)
            # if self._is_in_history(n_pos_int): score += W_AVOID_HISTORY

            scores[n_pos_int] = score
        return scores

    def _score_moves_searching(self, valid_neighbors_int):
        """Scores potential integer moves for searching based on food needs."""
        scores = {}
        grid = self.simulation.grid # Cache grid ref
        sim = self.simulation # Cache sim ref
        nest_pos_int = tuple(map(int, NEST_POS)) # Cache nest pos

        # --- Determine Food Need Weights ---
        sugar_needed = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD
        protein_needed = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD
        # Base weights depend on colony food levels relative to critical threshold
        w_sugar = W_FOOD_PHEROMONE_SEARCH_LOW_NEED
        w_protein = W_FOOD_PHEROMONE_SEARCH_LOW_NEED

        if sugar_needed and not protein_needed: # Need Sugar badly
            w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE
            w_protein = W_FOOD_PHEROMONE_SEARCH_AVOID # Avoid protein trails
        elif protein_needed and not sugar_needed: # Need Protein badly
            w_protein = W_FOOD_PHEROMONE_SEARCH_BASE
            w_sugar = W_FOOD_PHEROMONE_SEARCH_AVOID # Avoid sugar trails
        elif sugar_needed and protein_needed: # Need Both badly
            # Slightly prioritize the lower one
            if sim.colony_food_storage_sugar <= sim.colony_food_storage_protein:
                w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 1.1
                w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 0.9
            else:
                w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 1.1
                w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 0.9
        else: # Neither is critical, moderate search based on ratio
            if sim.colony_food_storage_sugar <= sim.colony_food_storage_protein * 1.5:
                w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 0.6
            else:
                w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 0.6

        # Soldiers are less interested in food trails
        if self.caste == AntCaste.SOLDIER:
            w_sugar *= 0.1
            w_protein *= 0.1

        # --- Score Each Neighbor ---
        for n_pos_int in valid_neighbors_int:
            # Base score
            score = self._score_moves_base(n_pos_int)

            # Pheromone influences
            home_ph = grid.get_pheromone(n_pos_int, "home") # Usually low weight when searching
            sugar_ph = grid.get_pheromone(n_pos_int, "food", FoodType.SUGAR)
            protein_ph = grid.get_pheromone(n_pos_int, "food", FoodType.PROTEIN)
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            # Apply food weights
            score += sugar_ph * w_sugar
            score += protein_ph * w_protein

            # Recruitment pheromone (strong pull, stronger for soldiers)
            recruit_w = W_RECRUITMENT_PHEROMONE * (1.2 if self.caste == AntCaste.SOLDIER else 1.0)
            score += recr_ph * recruit_w

            # Apply other pheromone weights
            score += neg_ph * W_NEGATIVE_PHEROMONE
            score += alarm_ph * W_ALARM_PHEROMONE # Avoid danger
            score += home_ph * W_HOME_PHEROMONE_SEARCH # Usually 0 or small penalty

            # Avoid lingering near the nest center while searching
            if distance_sq(n_pos_int, nest_pos_int) <= (NEST_RADIUS * 1.8) ** 2:
                score += W_AVOID_NEST_SEARCHING

            # History penalty applied by filter, could add weight here if needed

            scores[n_pos_int] = score
        return scores

    def _score_moves_patrolling(self, valid_neighbors_int):
        """Scores potential integer moves for patrolling (soldiers only)."""
        scores = {}
        grid = self.simulation.grid # Cache refs
        pos_int = self.pos
        nest_pos_int = tuple(map(int, NEST_POS))
        # Calculate current distance squared once
        dist_sq_current = distance_sq(pos_int, nest_pos_int)
        patrol_radius_sq = SOLDIER_PATROL_RADIUS_SQ # Cache constant

        for n_pos_int in valid_neighbors_int:
            # Base score
            score = self._score_moves_base(n_pos_int)

            # Pheromone influences (less sensitive than searching)
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            score += recr_ph * W_RECRUITMENT_PHEROMONE * 0.7 # Respond to recruitment
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.5 # Mildly avoid negative
            score += alarm_ph * W_ALARM_PHEROMONE * 0.5 # Mildly avoid alarm

            # Patrolling behavior: stay near nest, but not too close/far
            dist_sq_next = distance_sq(n_pos_int, nest_pos_int)

            # Tendency to move away from nest center if inside patrol radius
            if dist_sq_current <= patrol_radius_sq:
                if dist_sq_next > dist_sq_current:
                    score -= W_NEST_DIRECTION_PATROL # W_NEST_DIRECTION_PATROL is negative

            # Strong penalty for moving too far from patrol zone
            if dist_sq_next > (patrol_radius_sq * 1.4): # Use a buffer
                score -= 8000 # Large penalty

            # History penalty applied by filter

            scores[n_pos_int] = score
        return scores

    def _score_moves_defending(self, valid_neighbors_int):
        """Scores potential integer moves for defending (moving towards threats)."""
        scores = {}
        grid = self.simulation.grid # Cache refs
        pos_int = self.pos

        # --- Update Threat Location ---
        # Periodically rescan for strongest signal or if no target known
        if self.last_known_alarm_pos is None or random.random() < 0.2:
            best_signal_pos = None
            max_signal_strength = -1.0
            search_radius_sq = 6 * 6 # Check a local area
            x0, y0 = pos_int
            # Calculate scan bounds once
            min_scan_x = max(0, x0 - int(search_radius_sq**0.5))
            max_scan_x = min(GRID_WIDTH - 1, x0 + int(search_radius_sq**0.5))
            min_scan_y = max(0, y0 - int(search_radius_sq**0.5))
            max_scan_y = min(GRID_HEIGHT - 1, y0 + int(search_radius_sq**0.5))

            # Scan the area for highest combined signal (alarm, recruit, enemy presence)
            for i in range(min_scan_x, max_scan_x + 1):
                for j in range(min_scan_y, max_scan_y + 1):
                    p_int = (i, j)
                    # Check distance efficiently
                    if distance_sq(pos_int, p_int) <= search_radius_sq:
                        signal = (
                            grid.get_pheromone(p_int, "alarm") * 1.2
                            + grid.get_pheromone(p_int, "recruitment") * 0.8
                        )
                        # Big bonus if enemy is directly there
                        if self.simulation.get_enemy_at(p_int):
                            signal += 600
                        # Update best signal found so far
                        if signal > max_signal_strength:
                            max_signal_strength = signal
                            best_signal_pos = p_int
            # Update known position if a strong signal was found
            if max_signal_strength > 80.0: # Threshold to avoid noise
                self.last_known_alarm_pos = best_signal_pos
            else:
                self.last_known_alarm_pos = None # Lost signal

        # --- Score Moves ---
        target_pos = self.last_known_alarm_pos # Use the determined target
        dist_now_sq = distance_sq(pos_int, target_pos) if target_pos else float('inf')

        for n_pos_int in valid_neighbors_int:
            # Base score
            score = self._score_moves_base(n_pos_int)

            # Pheromone influences
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")
            # Use optimized enemy lookup
            enemy_at_n_pos = self.simulation.get_enemy_at(n_pos_int)

            # HUGE bonus for moving onto a cell with an enemy
            if enemy_at_n_pos:
                score += 15000

            # If a threat location is known, bonus for moving closer
            if target_pos:
                dist_next_sq = distance_sq(n_pos_int, target_pos)
                if dist_next_sq < dist_now_sq:
                    score += W_ALARM_SOURCE_DEFEND

            # Strong attraction to recruitment signals when defending
            score += recr_ph * W_RECRUITMENT_PHEROMONE * 1.1
            # Follow alarm pheromone trail (negative weight means move towards higher values)
            score += alarm_ph * W_ALARM_PHEROMONE * -1.0 # Note the sign change

            # History penalty applied by filter

            scores[n_pos_int] = score
        return scores

    def _score_moves_hunting(self, valid_neighbors_int):
        """Scores potential integer moves for hunting a specific prey target."""
        scores = {}
        pos_int = self.pos # Cache current pos
        # Get target position if target exists and is valid
        target_pos = self.target_prey.pos if (self.target_prey and hasattr(self.target_prey, 'pos')) else None

        # If no valid target, behave like searching (or patrolling for soldiers)
        if not target_pos:
            # Fallback scoring (e.g., simple base score or searching score)
            # Using just base score for simplicity here
            return {
                n_pos_int: self._score_moves_base(n_pos_int)
                for n_pos_int in valid_neighbors_int
            }

        # Calculate current distance to target
        dist_sq_now = distance_sq(pos_int, target_pos)
        grid = self.simulation.grid # Cache grid ref

        for n_pos_int in valid_neighbors_int:
            # Base score
            score = self._score_moves_base(n_pos_int)

            # Strong incentive to move closer to the prey
            dist_sq_next = distance_sq(n_pos_int, target_pos)
            if dist_sq_next < dist_sq_now:
                score += W_HUNTING_TARGET

            # Minor influence from other pheromones (avoid strong negative/alarm)
            score += (
                grid.get_pheromone(n_pos_int, "alarm") * W_ALARM_PHEROMONE * 0.1
            )
            score += (
                grid.get_pheromone(n_pos_int, "negative") * W_NEGATIVE_PHEROMONE * 0.2
            )

            # History penalty applied by filter

            scores[n_pos_int] = score
        return scores

    def _select_best_move(self, move_scores, valid_neighbors_int):
        """Selects the integer move with the highest score (deterministic)."""
        best_score = -float("inf")
        # Use a list to handle ties
        best_moves_int = []
        for pos_int, score in move_scores.items():
            if score > best_score:
                best_score = score
                best_moves_int = [pos_int] # Start new list
            elif score == best_score:
                best_moves_int.append(pos_int) # Add to ties

        # Choose randomly among the best moves if there's a tie
        if not best_moves_int:
            # Fallback if something went wrong (shouldn't happen if move_scores is populated)
            self.last_move_info += "(Best:Fallback!)"
            return random.choice(valid_neighbors_int)

        chosen_int = random.choice(best_moves_int)
        # Update debug info
        score = move_scores.get(chosen_int, -999)
        state_prefix = self.state.name[:4]
        self.last_move_info = f"{state_prefix} Best->{chosen_int} (S:{score:.1f})"
        return chosen_int

    def _select_best_move_returning(
        self, move_scores, valid_neighbors_int, just_picked
    ):
        """Selects the best move for returning, prioritizing getting closer."""
        best_score = -float("inf")
        best_moves_int = []
        pos_int = self.pos # Cache pos
        nest_pos_int = tuple(map(int, NEST_POS))
        dist_sq_now = distance_sq(pos_int, nest_pos_int)

        # Separate moves into those getting closer and others
        closer_moves = {}
        other_moves = {}
        for pos_int_cand, score in move_scores.items():
            if distance_sq(pos_int_cand, nest_pos_int) < dist_sq_now:
                closer_moves[pos_int_cand] = score
            else:
                other_moves[pos_int_cand] = score

        # Prioritize moves that get closer
        target_pool = {}
        selection_type = ""
        if closer_moves:
            target_pool = closer_moves
            selection_type = "Closer"
        elif other_moves: # Only consider other moves if none get closer
            target_pool = other_moves
            selection_type = "Other"
        else:
            # If somehow neither pool has moves (e.g., only one valid neighbor not closer)
            # Use all available scores as fallback
            target_pool = move_scores
            selection_type = "All(Fallback)"
            if not target_pool: # If move_scores was empty initially
                 self.last_move_info += "(R: No moves?)"
                 return random.choice(valid_neighbors_int) if valid_neighbors_int else None


        # Find the best score within the selected pool
        for pos_int_cand, score in target_pool.items():
            if score > best_score:
                best_score = score
                best_moves_int = [pos_int_cand]
            elif score == best_score:
                best_moves_int.append(pos_int_cand)

        # If no best move found in the pool (unlikely but possible)
        if not best_moves_int:
             self.last_move_info += f"(R: No best in {selection_type})"
             # Fallback: choose best from *all* moves if primary pool failed
             if target_pool is not move_scores:
                 best_score = -float('inf')
                 for pos_int_cand, score in move_scores.items():
                     if score > best_score:
                         best_score = score
                         best_moves_int = [pos_int_cand]
                     elif score == best_score:
                         best_moves_int.append(pos_int_cand)

             if not best_moves_int: # Final fallback
                 return random.choice(valid_neighbors_int) if valid_neighbors_int else None


        # --- Tie-breaking (if multiple best moves) ---
        chosen_int = None
        if len(best_moves_int) == 1:
            chosen_int = best_moves_int[0]
            self.last_move_info = f"R({selection_type})Best->{chosen_int} (S:{best_score:.1f})"
        else:
            # Tie-break based on home pheromone strength (strongest wins)
            grid = self.simulation.grid # Cache grid ref
            # Sort tied moves by home pheromone level, descending
            best_moves_int.sort(
                key=lambda p: grid.get_pheromone(p, "home"), reverse=True
            )
            # Get max pheromone value among tied moves
            max_ph = grid.get_pheromone(best_moves_int[0], "home")
            # Filter for moves that have this maximum value
            top_ph_moves = [
                p for p in best_moves_int if grid.get_pheromone(p, "home") == max_ph
            ]
            # Choose randomly among the top pheromone moves
            chosen_int = random.choice(top_ph_moves)
            self.last_move_info = f"R({selection_type})TieBrk->{chosen_int} (S:{best_score:.1f})"

        return chosen_int

    def _select_probabilistic_move(self, move_scores, valid_neighbors_int):
        """Selects an integer move probabilistically based on scores (softmax-like)."""
        if not move_scores or not valid_neighbors_int:
            # Fallback if inputs are empty
            return random.choice(valid_neighbors_int) if valid_neighbors_int else None

        # Prepare scores for probability calculation
        pop_int = list(move_scores.keys())
        scores = np.array(list(move_scores.values()), dtype=np.float64)

        # Check for edge cases (e.g., all scores identical or invalid)
        if len(pop_int) == 0: return None # No moves to choose from
        if len(pop_int) == 1: # Only one option
             self.last_move_info = f"{self.state.name[:3]} Prob->{pop_int[0]} (Only)"
             return pop_int[0]

        # --- Softmax Calculation ---
        # Shift scores to be non-negative for power calculation (add small epsilon)
        min_score = np.min(scores)
        shifted_scores = scores - min_score + 0.01 # Add epsilon to avoid zero base

        # Apply temperature (higher temp -> more randomness)
        # Clamp temperature to reasonable bounds
        temp = min(max(PROBABILISTIC_CHOICE_TEMP, 0.1), 5.0)
        # Calculate weights (score ^ temperature)
        # Use np.power for potential speedup, handle potential overflow carefully
        # Clip weights to avoid potential issues with extremely large scores
        weights = np.power(shifted_scores, temp)
        # Ensure weights are at least a minimum value to allow choice even for low scores
        weights = np.maximum(MIN_SCORE_FOR_PROB_CHOICE, weights)

        # Normalize weights to get probabilities
        total_weight = np.sum(weights)

        # --- Handle potential numerical issues ---
        if total_weight <= 1e-9 or not np.isfinite(total_weight):
            # If total weight is zero or infinite, fall back to best score
            self.last_move_info += f"({self.state.name[:3]}:InvW)"
            # Find max score manually for fallback
            best_s = -float("inf")
            best_p = None
            for p_int_idx, s in enumerate(scores):
                if s > best_s:
                    best_s = s
                    best_p = pop_int[p_int_idx]
            return best_p if best_p else random.choice(valid_neighbors_int) # Return best or random valid

        probabilities = weights / total_weight

        # Final check for probability validity (sum should be close to 1)
        if not np.isclose(np.sum(probabilities), 1.0):
             # Attempt renormalization if slightly off
             if np.sum(probabilities) > 1e-9 and np.all(np.isfinite(probabilities)):
                  probabilities /= np.sum(probabilities) # Renormalize
                  if not np.isclose(np.sum(probabilities), 1.0): # Check again
                      self.last_move_info += "(ProbReNormFail)"
                      # Fallback to best if renormalization failed
                      best_s = -float("inf"); best_p = None
                      for p_int_idx, s in enumerate(scores):
                          if s > best_s: best_s = s; best_p = pop_int[p_int_idx]
                      return best_p if best_p else random.choice(valid_neighbors_int)
             else: # Sum is too small or contains non-finite numbers
                 self.last_move_info += "(ProbBadSum)"
                 best_s = -float("inf"); best_p = None
                 for p_int_idx, s in enumerate(scores):
                     if s > best_s: best_s = s; best_p = pop_int[p_int_idx]
                 return best_p if best_p else random.choice(valid_neighbors_int)


        # --- Choose move based on calculated probabilities ---
        try:
            chosen_index = np.random.choice(len(pop_int), p=probabilities)
            chosen_int = pop_int[chosen_index]
            # Update debug info
            score = move_scores.get(chosen_int, -999)
            self.last_move_info = f"{self.state.name[:3]} Prob->{chosen_int} (S:{score:.1f})"
            return chosen_int
        except ValueError as e:
            # Catch potential errors in np.random.choice (e.g., if probabilities don't sum to 1)
            print(f"WARN: Probabilistic choice error ({self.state.name}): {e}. Sum={np.sum(probabilities)}")
            self.last_move_info += "(ProbValErr)"
            # Fallback to best score on error
            best_s = -float("inf"); best_p = None
            for p_int_idx, s in enumerate(scores):
                if s > best_s: best_s = s; best_p = pop_int[p_int_idx]
            return best_p if best_p else random.choice(valid_neighbors_int)


    def _switch_state(self, new_state: AntState, reason: str):
        """Helper to switch state and clear history/targets if needed."""
        if self.state != new_state:
            # print(f"Ant {id(self)}: {self.state.name} -> {new_state.name} ({reason})") # Debug log
            self.state = new_state
            # Update debug info immediately
            self.last_move_info = reason
            # Clear path history on state change to allow revisiting areas if needed
            self._clear_path_history()
            # Reset state-specific targets/info
            if new_state != AntState.HUNTING:
                self.target_prey = None
            if new_state != AntState.DEFENDING:
                self.last_known_alarm_pos = None
            # Reset stuck timer on state change
            self.stuck_timer = 0

    def _update_state(self):
        """Handle automatic state transitions based on environment and needs."""
        sim = self.simulation # Cache sim reference
        pos_int = self.pos # Cache current position
        nest_pos_int = tuple(map(int, NEST_POS)) # Cache nest position

        # --- Worker Hunting Logic ---
        # Only consider hunting if searching, not carrying, needs protein, and not already hunting
        if (self.caste == AntCaste.WORKER and
            self.state == AntState.SEARCHING and
            self.carry_amount == 0 and
            not self.target_prey and
            sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * 1.5):

            # Look for nearby prey
            nearby_prey = sim.find_nearby_prey(
                pos_int, PREY_FLEE_RADIUS_SQ * 2.5 # Search slightly larger radius
            )
            if nearby_prey:
                # Sort prey by distance, closest first
                nearby_prey.sort(key=lambda p: distance_sq(pos_int, p.pos))
                self.target_prey = nearby_prey[0] # Target the closest one
                self._switch_state(AntState.HUNTING, f"HuntPrey@{self.target_prey.pos}")
                return # State changed, exit update

        # --- Soldier State Logic ---
        # Only applies if soldier is not already escaping, returning, or hunting
        if (self.caste == AntCaste.SOLDIER and
            self.state not in [AntState.ESCAPING, AntState.RETURNING_TO_NEST, AntState.HUNTING]):

            # Scan local area for threats (alarm/recruitment)
            max_alarm = 0.0
            max_recruit = 0.0
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
                        max_recruit = max(max_recruit, grid.get_pheromone(p_int, "recruitment"))

            # Calculate combined threat signal
            threat_signal = max_alarm + max_recruit * 0.6
            # Check distance relative to patrol radius
            is_near_nest = distance_sq(pos_int, nest_pos_int) <= SOLDIER_PATROL_RADIUS_SQ

            # High threat -> DEFEND state
            if threat_signal > SOLDIER_DEFEND_ALARM_THRESHOLD:
                if self.state != AntState.DEFENDING:
                    self._switch_state(AntState.DEFENDING, f"ThreatHi({threat_signal:.0f})!")
                return # State changed

            # Opportunity hunting for soldiers if not defending
            if self.state != AntState.DEFENDING and not self.target_prey:
                nearby_prey = sim.find_nearby_prey(pos_int, PREY_FLEE_RADIUS_SQ * 2.0)
                if nearby_prey:
                    nearby_prey.sort(key=lambda p: distance_sq(pos_int, p.pos))
                    self.target_prey = nearby_prey[0]
                    self._switch_state(AntState.HUNTING, f"SHuntPrey@{self.target_prey.pos}")
                    return # State changed

            # Transition back from DEFENDING if threat is low
            if self.state == AntState.DEFENDING:
                self._switch_state(AntState.PATROLLING, f"ThreatLow({threat_signal:.0f})")
            # Ensure PATROLLING near nest, SEARCHING further away
            elif is_near_nest and self.state != AntState.PATROLLING:
                 self._switch_state(AntState.PATROLLING, "NearNest->Patrol")
            elif not is_near_nest and self.state == AntState.PATROLLING:
                 self._switch_state(AntState.SEARCHING, "PatrolFar->Search")
            # Handle edge case: If searching but ended up near nest, switch to patrol
            elif is_near_nest and self.state == AntState.SEARCHING:
                 self._switch_state(AntState.PATROLLING, "SearchNear->Patrol")


        # --- Check if HUNTING target is lost (for both castes) ---
        if self.state == AntState.HUNTING:
            lost_target = False
            if not self.target_prey:
                lost_target = True
            else:
                # Check if prey still exists in simulation list
                # Using direct check on simulation.prey list (requires prey list is current)
                if self.target_prey not in sim.prey:
                    lost_target = True
                    self.target_prey = None # Clear invalid reference
                # Check if prey moved too far away
                elif distance_sq(pos_int, self.target_prey.pos) > PREY_FLEE_RADIUS_SQ * 4:
                     lost_target = True

            if lost_target:
                self.target_prey = None # Ensure target is cleared
                # Default state after losing target
                default_state = (
                    AntState.PATROLLING
                    if self.caste == AntCaste.SOLDIER
                    else AntState.SEARCHING
                )
                self._switch_state(default_state, "LostPreyTarget")
                # No return needed here, rest of update loop can proceed


    def update(self, speed_multiplier):
        """Update ant's state, position, age, food, and interactions."""
        # --- Basic Updates (Age, Starvation) ---
        self.age += speed_multiplier
        if self.age >= self.max_age_ticks:
            self.hp = 0 # Mark for removal
            self.last_move_info = "Died of old age"
            return # Stop update

        # Food Consumption (less frequent check)
        self.food_consumption_timer += speed_multiplier
        if self.food_consumption_timer >= WORKER_FOOD_CONSUMPTION_INTERVAL:
            self.food_consumption_timer %= WORKER_FOOD_CONSUMPTION_INTERVAL # Reset timer part
            needed_s = self.food_consumption_sugar
            needed_p = self.food_consumption_protein
            sim = self.simulation # Cache sim ref
            # Check and consume colony food
            can_eat = (sim.colony_food_storage_sugar >= needed_s and
                       sim.colony_food_storage_protein >= needed_p)
            if can_eat:
                 sim.colony_food_storage_sugar -= needed_s
                 sim.colony_food_storage_protein -= needed_p
            else:
                 # Ant starves if colony cannot provide food
                 self.hp = 0 # Mark for removal
                 self.last_move_info = "Starved"
                 return # Stop update

        # --- State Management ---
        # Handle ESCAPING state timer
        if self.state == AntState.ESCAPING:
            self.escape_timer -= speed_multiplier
            if self.escape_timer <= 0:
                # Transition back to default state after escaping
                next_state = (
                    AntState.PATROLLING
                    if self.caste == AntCaste.SOLDIER
                    else AntState.SEARCHING
                )
                self._switch_state(next_state, "EscapeEnd")
                # Don't return, allow movement in the same tick after escape ends

        # Update state based on environment (calls _update_state)
        self._update_state()
        # Ensure HP check after potential state changes
        if self.hp <= 0: return

        # --- Combat / Interaction Checks ---
        pos_int = self.pos # Cache current position
        # Get neighbors (including center for potential entities on same cell)
        neighbors_int = get_neighbors(pos_int, include_center=True)
        sim = self.simulation # Cache sim ref

        # Check for Enemies first (higher priority)
        target_enemy = None
        for p_int in neighbors_int:
            enemy = sim.get_enemy_at(p_int) # Use optimized lookup
            if enemy and enemy.hp > 0:
                target_enemy = enemy
                break # Found one enemy, attack it

        if target_enemy:
            self.attack(target_enemy) # Perform attack
            # Drop alarm pheromone immediately
            sim.grid.add_pheromone(pos_int, P_ALARM_FIGHT, "alarm")
            # Reset stuck timer as fighting is progress
            self.stuck_timer = 0
            # Drop prey target if fighting enemy
            self.target_prey = None
            self.last_move_info = f"FightEnemy@{target_enemy.pos}"
            # Ensure correct state (DEFENDING)
            if self.state != AntState.DEFENDING:
                self._switch_state(AntState.DEFENDING, "EnemyContact!")
            return # End update cycle after attacking

        # Check for Prey if not fighting enemy
        target_prey_to_attack = None
        prey_in_range = [] # Collect nearby prey first
        for p_int in neighbors_int:
            prey = sim.get_prey_at(p_int) # Use optimized lookup
            if prey and prey.hp > 0:
                prey_in_range.append(prey)

        should_attack_prey = False
        if prey_in_range:
            # 1. If HUNTING state and *target* is adjacent/on cell
            if (self.state == AntState.HUNTING and
                self.target_prey and
                self.target_prey in prey_in_range):
                 # Check adjacency specifically (target might be in range but not next to us)
                 if self.target_prey.pos in neighbors_int:
                     should_attack_prey = True
                     target_prey_to_attack = self.target_prey

            # 2. Opportunistic attack if not returning/defending and need protein or soldier
            elif self.state not in [AntState.RETURNING_TO_NEST, AntState.DEFENDING]:
                 can_hunt_opportunistically = (
                     (self.caste == AntCaste.WORKER and sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * 2) or
                     (self.caste == AntCaste.SOLDIER) # Soldiers attack prey opportunistically
                 )
                 if can_hunt_opportunistically:
                     # Prioritize prey on adjacent cells over prey on the same cell
                     adjacent_prey = [p for p in prey_in_range if p.pos != pos_int]
                     prey_on_cell = [p for p in prey_in_range if p.pos == pos_int]
                     if adjacent_prey:
                         should_attack_prey = True
                         # Attack a random adjacent prey
                         target_prey_to_attack = random.choice(adjacent_prey)
                     elif prey_on_cell:
                         should_attack_prey = True
                         # Attack prey on current cell if no adjacent ones
                         target_prey_to_attack = random.choice(prey_on_cell)

        # Perform attack on chosen prey target
        if should_attack_prey and target_prey_to_attack:
            self.attack(target_prey_to_attack) # Perform attack
            self.stuck_timer = 0 # Reset stuck timer
            self.last_move_info = f"AtkPrey@{target_prey_to_attack.pos}"

            # Check if prey was killed by this attack
            if target_prey_to_attack.hp <= 0:
                killed_prey_pos = target_prey_to_attack.pos # Store pos before removal
                sim.kill_prey(target_prey_to_attack) # Let simulation handle removal and food drop

                # Drop pheromones at the kill site
                sim.grid.add_pheromone(killed_prey_pos, P_FOOD_AT_SOURCE, "food", FoodType.PROTEIN)
                sim.grid.add_pheromone(killed_prey_pos, P_RECRUIT_PREY, "recruitment")

                # Clear target if it was the one killed
                if self.target_prey == target_prey_to_attack:
                    self.target_prey = None

                # Switch back to default state after kill
                next_s = AntState.SEARCHING if self.caste == AntCaste.WORKER else AntState.PATROLLING
                self._switch_state(next_s, "PreyKilled")
            return # End update cycle after attacking prey

        # --- Movement Logic ---
        # Movement Delay Check
        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return # Waiting to move
        # Calculate and set next delay based on speed multiplier
        effective_delay_updates = 0
        if self.move_delay_base > 0:
            if speed_multiplier > 0:
                 # Estimate ticks per move based on multiplier
                 # Rounding might be needed for fractional delays
                 effective_delay_updates = max(0, int(round(self.move_delay_base / speed_multiplier)) -1)
            else: # Paused
                 effective_delay_updates = float('inf') # Effectively infinite delay
        self.move_delay_timer = effective_delay_updates

        # Choose and Execute Move
        old_pos_int = self.pos # Store position before moving
        local_just_picked = self.just_picked_food # Store flag before move
        self.just_picked_food = False # Reset flag for this cycle

        new_pos_int = self._choose_move() # Determine next position

        moved = False
        found_food_type = None
        food_amount = 0.0

        # Execute move if a valid new position was chosen
        if new_pos_int and new_pos_int != old_pos_int:
            self.pos = new_pos_int # Update position attribute
            # Update position in simulation's lookup dictionary
            sim.update_entity_position(self, old_pos_int, new_pos_int)

            # Update movement tracking
            self.last_move_direction = (
                new_pos_int[0] - old_pos_int[0],
                new_pos_int[1] - old_pos_int[1],
            )
            self._update_path_history(new_pos_int)
            self.stuck_timer = 0 # Reset stuck timer on successful move
            moved = True

            # Check for food at the new location
            # Inline food check for potential small optimization
            try:
                 foods = sim.grid.food[new_pos_int[0], new_pos_int[1]]
                 if foods[FoodType.SUGAR.value] > 0.1:
                     found_food_type = FoodType.SUGAR
                     food_amount = foods[FoodType.SUGAR.value]
                 elif foods[FoodType.PROTEIN.value] > 0.1:
                     found_food_type = FoodType.PROTEIN
                     food_amount = foods[FoodType.PROTEIN.value]
            except IndexError: # Should not happen with valid pos
                 pass

        elif new_pos_int == old_pos_int: # Choice resulted in staying put
            self.stuck_timer += 1
            self.last_move_info += "(Move->Same)"
            self.last_move_direction = (0, 0)
        else: # No valid move choice returned
            self.stuck_timer += 1
            self.last_move_info += "(NoChoice)"
            self.last_move_direction = (0, 0)

        # --- Post-Movement Actions ---
        pos_int = self.pos # Use potentially updated position
        nest_pos_int = tuple(map(int, NEST_POS)) # Cache nest pos
        is_near_nest = distance_sq(pos_int, nest_pos_int) <= NEST_RADIUS**2
        grid = sim.grid # Cache grid ref

        # Actions based on state AFTER moving
        # SEARCHING / HUNTING State Actions
        if self.state in [AntState.SEARCHING, AntState.HUNTING]:
            # Worker picks up food if found and not carrying anything
            if (self.caste == AntCaste.WORKER and
                found_food_type and
                self.carry_amount == 0):
                # Determine pickup amount (limited by capacity and availability)
                pickup_amount = min(self.max_capacity, food_amount)
                if pickup_amount > 0.01: # Only pick up significant amounts
                    self.carry_amount = pickup_amount
                    self.carry_type = found_food_type
                    food_idx = found_food_type.value
                    # Update grid food amount safely
                    try:
                        current_food = grid.food[pos_int[0], pos_int[1], food_idx]
                        grid.food[pos_int[0], pos_int[1], food_idx] = max(0, current_food - pickup_amount)
                    except IndexError: pass # Should not happen

                    # Drop pheromones at food source
                    grid.add_pheromone(pos_int, P_FOOD_AT_SOURCE, "food", food_type=found_food_type)
                    # Drop recruitment if source is rich
                    if food_amount >= RICH_FOOD_THRESHOLD:
                        grid.add_pheromone(pos_int, P_RECRUIT_FOOD, "recruitment")

                    # Switch state to returning
                    self._switch_state(
                        AntState.RETURNING_TO_NEST,
                        f"Picked {found_food_type.name[:1]}({pickup_amount:.1f})"
                    )
                    self.just_picked_food = True # Set flag for next move's scoring
                    self.target_prey = None # Stop hunting after picking food

            # If moved to an empty cell far from nest while searching, drop negative pheromone
            elif (moved and not found_food_type and self.state == AntState.SEARCHING and
                  distance_sq(pos_int, nest_pos_int) > (NEST_RADIUS + 3) ** 2):
                 # Drop at the *previous* location to mark it as unproductive
                 if is_valid(old_pos_int): # Ensure old pos was valid
                     grid.add_pheromone(old_pos_int, P_NEGATIVE_SEARCH, "negative")


        # RETURNING State Actions
        elif self.state == AntState.RETURNING_TO_NEST:
            # If reached the nest area
            if is_near_nest:
                dropped_amount = self.carry_amount
                type_dropped = self.carry_type
                # Drop food if carrying any
                if dropped_amount > 0 and type_dropped:
                    if type_dropped == FoodType.SUGAR:
                        sim.colony_food_storage_sugar += dropped_amount
                    elif type_dropped == FoodType.PROTEIN:
                        sim.colony_food_storage_protein += dropped_amount
                    # Clear carried amount
                    self.carry_amount = 0
                    self.carry_type = None

                # Decide next state after returning
                next_state = AntState.SEARCHING # Default for worker
                state_reason = "Dropped->" # Base reason

                if self.caste == AntCaste.WORKER:
                     # Check immediate needs after drop-off
                     sugar_crit = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD
                     protein_crit = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD
                     if sugar_crit or protein_crit:
                          state_reason += "SEARCH(Need!)"
                     else:
                          state_reason += "SEARCH"
                elif self.caste == AntCaste.SOLDIER:
                     next_state = AntState.PATROLLING
                     state_reason += "PATROL"

                self._switch_state(next_state, state_reason)
                # History cleared by _switch_state

            # If returning but not yet at nest, drop trail pheromones
            elif moved and not local_just_picked: # Don't drop trail on first step away from food
                # Drop at the *previous* location
                if is_valid(old_pos_int) and distance_sq(old_pos_int, nest_pos_int) > (NEST_RADIUS-1)**2: # Avoid dropping inside nest core
                    grid.add_pheromone(old_pos_int, P_HOME_RETURNING, "home")
                    # If carrying food, also drop food trail pheromone
                    if self.carry_amount > 0 and self.carry_type:
                        grid.add_pheromone(
                            old_pos_int,
                            P_FOOD_RETURNING_TRAIL,
                            "food",
                            food_type=self.carry_type,
                        )

        # --- Stuck Check ---
        # Check only if not already escaping
        if self.stuck_timer >= WORKER_STUCK_THRESHOLD and self.state != AntState.ESCAPING:
             # Check if stuck condition is 'legitimate' (fighting/hunting adjacent)
             is_fighting = False
             is_hunting_adjacent = False
             neighbors_int_stuck = get_neighbors(pos_int, True) # Re-check neighbors
             for p_int in neighbors_int_stuck:
                 if sim.get_enemy_at(p_int):
                     is_fighting = True
                     break
             if not is_fighting and self.state == AntState.HUNTING and self.target_prey:
                  # Check if target prey is still valid and adjacent
                  if self.target_prey in sim.prey and self.target_prey.pos in neighbors_int_stuck:
                       is_hunting_adjacent = True

             # If not fighting or hunting adjacent prey, initiate escape
             if not is_fighting and not is_hunting_adjacent:
                 self._switch_state(AntState.ESCAPING, "Stuck!")
                 self.escape_timer = WORKER_ESCAPE_DURATION # Set escape duration
                 self.stuck_timer = 0 # Reset stuck counter


    def attack(self, target):
        """Attack either an Enemy or Prey."""
        # Check target validity and call its take_damage method
        if isinstance(target, (Enemy, Prey)) and hasattr(target, 'take_damage'):
            target.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        """Process damage taken by the ant."""
        if self.hp <= 0:
            return # Already dead
        self.hp -= amount
        if self.hp > 0:
            # Drop pheromones when damaged
            grid = self.simulation.grid
            pos_int = self.pos # Cache pos
            grid.add_pheromone(pos_int, P_ALARM_FIGHT / 2, "alarm") # Lower amount than direct fight
            # Drop recruitment pheromone based on caste
            recruit_amount = (
                P_RECRUIT_DAMAGE_SOLDIER
                if self.caste == AntCaste.SOLDIER
                else P_RECRUIT_DAMAGE
            )
            grid.add_pheromone(pos_int, recruit_amount, "recruitment")
        else:
            self.hp = 0 # Mark for removal (handled by simulation)
            # Optionally set last_move_info here, e.g., "Killed by X"
            # This might be overwritten later in the simulation loop though


# --- Queen Class ---
class Queen:
    """Manages queen state and egg laying."""
    def __init__(self, pos, sim):
        self.pos = tuple(map(int, pos)) # Ensure integer tuple
        self.simulation = sim # Reference to main simulation
        self.hp = float(QUEEN_HP)
        self.max_hp = float(QUEEN_HP)
        self.age = 0.0
        # Egg laying timer and interval
        self.egg_lay_timer_progress = 0.0
        self.egg_lay_interval_ticks = QUEEN_EGG_LAY_RATE # Ticks between eggs
        self.color = QUEEN_COLOR
        # Queen attributes (less mobile/active)
        self.attack_power = 0 # Queen doesn't attack
        self.carry_amount = 0 # Queen doesn't carry

    def update(self, speed_multiplier):
        """Update Queen's age and handle egg laying based on speed."""
        if speed_multiplier == 0.0:
            return # Paused

        self.age += speed_multiplier # Queen ages

        # --- Egg Laying ---
        self.egg_lay_timer_progress += speed_multiplier
        if self.egg_lay_timer_progress >= self.egg_lay_interval_ticks:
            # Reset timer (handle potential overshoot)
            self.egg_lay_timer_progress %= self.egg_lay_interval_ticks

            # Check if colony has enough food to lay an egg
            needed_s = QUEEN_FOOD_PER_EGG_SUGAR
            needed_p = QUEEN_FOOD_PER_EGG_PROTEIN
            sim = self.simulation # Cache sim ref
            can_lay = (sim.colony_food_storage_sugar >= needed_s and
                       sim.colony_food_storage_protein >= needed_p)

            if can_lay:
                # Consume food from colony storage
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p

                # Decide caste and find position for the new egg
                caste = self._decide_caste()
                egg_pos = self._find_egg_position()

                # Create and add the egg brood item if a spot was found
                if egg_pos:
                    # Use integer ticks for creation time
                    sim.add_brood(
                        BroodItem(BroodStage.EGG, caste, egg_pos, int(sim.ticks))
                    )

    def _decide_caste(self):
        """Decide the caste of the next egg based on colony soldier ratio."""
        sim = self.simulation # Cache sim ref
        # Calculate current effective soldier ratio (including developing pupae)
        soldier_count = 0
        worker_count = 0 # Count workers too for total population

        # Count active ants
        for a in sim.ants:
            if a.caste == AntCaste.SOLDIER: soldier_count += 1
            else: worker_count +=1

        # Count developing brood (larvae and pupae)
        for b in sim.brood:
             if b.stage in [BroodStage.LARVA, BroodStage.PUPA]:
                  if b.caste == AntCaste.SOLDIER: soldier_count += 1
                  else: worker_count +=1

        total_population = soldier_count + worker_count
        current_ratio = 0.0
        if total_population > 0:
            current_ratio = soldier_count / total_population

        # Decision logic based on target ratio
        target_ratio = QUEEN_SOLDIER_RATIO_TARGET
        # Higher chance of soldier if below target
        if current_ratio < target_ratio:
            return AntCaste.SOLDIER if random.random() < 0.65 else AntCaste.WORKER
        # Small chance of soldier even if above target
        elif random.random() < 0.04:
            return AntCaste.SOLDIER
        # Default to worker
        return AntCaste.WORKER

    def _find_egg_position(self):
        """Find a valid integer position near the queen for a new egg."""
        possible_spots = get_neighbors(self.pos) # Get adjacent cells
        valid_spots = []
        sim = self.simulation # Cache sim ref

        # Filter for valid, non-obstacle spots
        for p in possible_spots:
            if not sim.grid.is_obstacle(p):
                 valid_spots.append(p)

        if not valid_spots:
            return None # No valid adjacent spots

        # Check which valid spots are currently occupied by other brood
        brood_positions = sim.get_brood_positions() # Get current brood locations
        free_valid_spots = [p for p in valid_spots if p not in brood_positions]

        # Prefer free spots, otherwise choose any valid spot (overwrite)
        if free_valid_spots:
            return random.choice(free_valid_spots)
        else:
            # Maybe log a warning if overwriting?
            return random.choice(valid_spots)


    def take_damage(self, amount, attacker):
        """Process damage taken by the queen."""
        if self.hp <= 0:
            return # Already dead
        self.hp -= amount
        if self.hp > 0:
            # Emit strong distress signals if damaged
            grid = self.simulation.grid
            pos_int = self.pos
            grid.add_pheromone(pos_int, P_ALARM_FIGHT * 4, "alarm") # High alarm
            grid.add_pheromone(pos_int, P_RECRUIT_DAMAGE * 4, "recruitment") # High recruitment
        else:
            self.hp = 0 # Mark for removal (handled by simulation)
            # Simulation checks queen HP and ends game if needed


# --- Enemy Class ---
class Enemy:
    """Represents an enemy entity that attacks ants."""
    def __init__(self, pos, sim):
        self.pos = tuple(map(int, pos)) # Ensure integer tuple
        self.simulation = sim # Reference to main simulation
        self.hp = float(ENEMY_HP)
        self.max_hp = float(ENEMY_HP)
        self.attack_power = ENEMY_ATTACK
        self.move_delay_base = ENEMY_MOVE_DELAY
        self.move_delay_timer = rnd_uniform(0, self.move_delay_base)
        self.color = ENEMY_COLOR

    def update(self, speed_multiplier):
        """Update enemy state: attack nearby ants or move."""
        if speed_multiplier == 0.0:
            return # Paused

        sim = self.simulation # Cache sim ref
        pos_int = self.pos # Cache current position

        # --- Attack Logic ---
        # Check adjacent cells (including own) for ants to attack
        neighbors_int = get_neighbors(pos_int, include_center=True)
        target_ant = None
        queen_target = None

        # Find potential targets
        for p_int in neighbors_int:
            ant = sim.get_ant_at(p_int) # Use optimized lookup
            if ant and ant.hp > 0:
                # Prioritize attacking the Queen if she's adjacent
                if isinstance(ant, Queen):
                    queen_target = ant
                    break # Attack queen immediately
                elif target_ant is None: # Store first worker/soldier found
                    target_ant = ant

        # Select target (Queen > Worker/Soldier)
        chosen_target = queen_target if queen_target else target_ant

        # Perform attack if a target was found
        if chosen_target:
            self.attack(chosen_target)
            # Attacking resets move timer (maybe?) - current logic doesn't reset
            return # End update cycle after attacking

        # --- Movement Logic (if no attack occurred) ---
        # Movement Delay Check
        self.move_delay_timer -= speed_multiplier
        if self.move_delay_timer > 0:
            return # Waiting to move
        # Reset timer for next move
        self.move_delay_timer += self.move_delay_base

        # Find valid moves
        possible_moves_int = get_neighbors(pos_int)
        valid_moves_int = []
        for m_int in possible_moves_int:
            # Check for obstacles, other enemies, ants, and prey
            if (
                not sim.grid.is_obstacle(m_int)
                and not sim.is_enemy_at(m_int, exclude_self=self) # Use optimized lookup
                and not sim.is_ant_at(m_int) # Use optimized lookup
                and not sim.is_prey_at(m_int) # Use optimized lookup
            ):
                valid_moves_int.append(m_int)

        if valid_moves_int:
            chosen_move_int = None
            nest_pos_int = tuple(map(int, NEST_POS)) # Cache nest pos

            # Small chance to move towards the nest
            if random.random() < ENEMY_NEST_ATTRACTION:
                best_nest_move = None
                # Find move that gets closest to the nest
                min_dist_sq = distance_sq(pos_int, nest_pos_int)
                for move in valid_moves_int:
                    d_sq = distance_sq(move, nest_pos_int)
                    if d_sq < min_dist_sq:
                        min_dist_sq = d_sq
                        best_nest_move = move
                # Choose the best nest move, or random valid if none closer
                chosen_move_int = (
                    best_nest_move
                    if best_nest_move
                    else random.choice(valid_moves_int)
                )
            else:
                # Otherwise, move randomly
                chosen_move_int = random.choice(valid_moves_int)

            # Execute the move
            if chosen_move_int and chosen_move_int != self.pos:
                old_pos = self.pos
                self.pos = chosen_move_int
                # IMPORTANT: Update position in simulation's lookup dictionary
                sim.update_entity_position(self, old_pos, self.pos)

    def attack(self, target_ant):
        """Attack a target ant."""
        # Check target validity and call its take_damage method
        if isinstance(target_ant, (Ant, Queen)) and hasattr(target_ant, 'take_damage'):
            target_ant.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        """Process damage taken by the enemy."""
        if self.hp <= 0:
            return # Already dead
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            # Death handled by simulation cleanup, which also drops food


# --- Main Simulation Class ---
class AntSimulation:
    """Manages the overall simulation state, entities, drawing, and UI."""

    def __init__(self):
        # Initialize Pygame screen and clock
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ant Simulation - Optimized")
        self.clock = pygame.time.Clock()

        # Fonts (initialized carefully)
        self.font = None
        self.debug_font = None
        self._init_fonts() # Call font initialization

        # Simulation components
        self.grid = WorldGrid()

        # Simulation state variables
        self.simulation_running = False
        self.app_running = True # Overall application state
        self.end_game_reason = ""
        self.colony_generation = 0
        self.ticks = 0.0 # Use float for fractional ticks with speed multiplier

        # Entity lists and lookup dictionaries
        self.ants = []
        self.enemies = []
        self.brood = [] # Brood items list
        self.prey = []
        self.queen = None

        # Position lookup dictionaries (for performance)
        self.ant_positions = {} # (x, y) -> Ant object (workers/soldiers)
        self.enemy_positions = {} # (x, y) -> Enemy object
        self.prey_positions = {} # (x, y) -> Prey object
        # Brood lookup is less critical as they don't move, but can be useful
        self.brood_positions = {} # (x, y) -> list of BroodItem objects

        # Colony resources
        self.colony_food_storage_sugar = 0.0
        self.colony_food_storage_protein = 0.0

        # Spawning timers
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_interval_ticks = ENEMY_SPAWN_RATE
        self.prey_spawn_timer = 0.0
        self.prey_spawn_interval_ticks = PREY_SPAWN_RATE

        # UI / Debugging state
        self.show_debug_info = False # Default to off for performance
        self.simulation_speed_index = DEFAULT_SPEED_INDEX
        self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
        self.buttons = self._create_buttons()

        # Static background surface (for performance)
        self.static_background_surface = pygame.Surface((WIDTH, HEIGHT))

        # Start simulation if initialization was successful
        if self.app_running:
            self._reset_simulation()

    def _init_fonts(self):
        """Initialize fonts, handling potential errors."""
        try:
            pygame.font.init() # Ensure font module is initialized
            # Try loading preferred system fonts
            try:
                self.font = pygame.font.SysFont("sans", 16)
                self.debug_font = pygame.font.SysFont("monospace", 14)
                print("Using system 'sans' and 'monospace' fonts.")
            except Exception: # Fallback to default Pygame font
                print("System fonts not found or failed. Trying default font.")
                self.font = pygame.font.Font(None, 20) # Default font, adjust size
                self.debug_font = pygame.font.Font(None, 16)
                print("Using Pygame default font.")
            # Final check if any font loaded
            if not self.font or not self.debug_font:
                 raise RuntimeError("Both system and default fonts failed to load.")

        except Exception as e:
            print(f"FATAL: Font initialization failed: {e}. Cannot render text.")
            self.font = None
            self.debug_font = None
            self.app_running = False # Cannot run without fonts

    def _reset_simulation(self):
        """Resets the simulation state for a new game generation."""
        print(f"Resetting simulation for Kolonie {self.colony_generation + 1}...")
        self.ticks = 0.0
        # Clear entity lists
        self.ants.clear()
        self.enemies.clear()
        self.brood.clear()
        self.prey.clear()
        # Clear lookup dictionaries
        self.ant_positions.clear()
        self.enemy_positions.clear()
        self.prey_positions.clear()
        self.brood_positions.clear()

        self.queen = None # Reset queen reference
        # Reset resources and timers
        self.colony_food_storage_sugar = INITIAL_COLONY_FOOD_SUGAR
        self.colony_food_storage_protein = INITIAL_COLONY_FOOD_PROTEIN
        self.enemy_spawn_timer = 0.0
        self.prey_spawn_timer = 0.0
        self.end_game_reason = ""
        self.colony_generation += 1

        # Reset grid and pre-draw static elements
        self.grid.reset()
        self._prepare_static_background() # Draw obstacles etc.

        # Spawn initial entities
        if not self._spawn_initial_entities():
            print("CRITICAL ERROR during simulation reset. Cannot continue.")
            self.simulation_running = False
            self.app_running = False
            self.end_game_reason = "Initialisierungsfehler"
            return

        # Reset speed and running state
        self.simulation_speed_index = DEFAULT_SPEED_INDEX
        self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
        self.simulation_running = True
        print(
            f"Kolonie {self.colony_generation} gestartet at "
            f"{SPEED_MULTIPLIERS[self.simulation_speed_index]:.1f}x speed."
        )

    def _prepare_static_background(self):
        """Draws non-changing elements like background color and obstacles."""
        self.static_background_surface.fill(MAP_BG_COLOR)
        cs = CELL_SIZE
        # Find obstacle coordinates once
        obstacle_coords = np.argwhere(self.grid.obstacles)
        for x, y in obstacle_coords:
            pygame.draw.rect(
                self.static_background_surface,
                OBSTACLE_COLOR,
                (x * cs, y * cs, cs, cs)
            )
        print(f"Prepared static background with {len(obstacle_coords)} obstacles.")


    def _create_buttons(self):
        """Creates data structures for UI speed control buttons (+/-)."""
        buttons = []
        button_h = 20
        button_w = 30
        margin = 5
        # Position buttons relative to screen width
        btn_plus_x = WIDTH - button_w - margin
        btn_minus_x = btn_plus_x - button_w - margin
        # Create Rect objects for collision detection
        rect_minus = pygame.Rect(btn_minus_x, margin, button_w, button_h)
        buttons.append({"rect": rect_minus, "text": "-", "action": "speed_down"})
        rect_plus = pygame.Rect(btn_plus_x, margin, button_w, button_h)
        buttons.append({"rect": rect_plus, "text": "+", "action": "speed_up"})
        return buttons

    def _spawn_initial_entities(self):
        """Spawns the queen, initial ants, enemies, and prey."""
        # Spawn Queen first
        queen_pos = self._find_valid_queen_pos()
        if queen_pos:
            self.queen = Queen(queen_pos, self)
            # Note: Queen is not added to ant_positions for now
            print(f"Queen placed at {queen_pos}")
        else:
            print("CRITICAL: Cannot place Queen. Aborting simulation setup.")
            return False # Cannot proceed without a queen

        # Spawn initial Ants around the Queen
        spawned_ants = 0
        attempts = 0
        max_att = INITIAL_ANTS * 25 # Increase attempts slightly
        # Use queen's position as the center for spawning
        queen_pos_int = self.queen.pos

        while spawned_ants < INITIAL_ANTS and attempts < max_att:
            # Spawn within a radius around the queen
            radius = NEST_RADIUS + 1 # Spawn closer to queen
            # Generate random offset
            angle = rnd_uniform(0, 2 * math.pi)
            dist = rnd_uniform(0, radius)
            ox = int(dist * math.cos(angle))
            oy = int(dist * math.sin(angle))
            # Calculate potential position
            pos = (queen_pos_int[0] + ox, queen_pos_int[1] + oy)

            # Decide caste based on initial ratio
            caste = (
                AntCaste.SOLDIER if random.random() < QUEEN_SOLDIER_RATIO_TARGET
                else AntCaste.WORKER
            )
            # Attempt to add the ant (checks validity and updates lists/dicts)
            if self.add_ant(pos, caste):
                spawned_ants += 1
            attempts += 1

        if spawned_ants < INITIAL_ANTS:
            print(f"Warning: Spawned only {spawned_ants}/{INITIAL_ANTS} initial ants.")

        # Spawn initial Enemies
        enemies_spawned = 0
        for _ in range(INITIAL_ENEMIES):
            if self.spawn_enemy():
                enemies_spawned += 1
        if enemies_spawned < INITIAL_ENEMIES:
             print(f"Warning: Spawned only {enemies_spawned}/{INITIAL_ENEMIES} initial enemies.")
        else:
             print(f"Spawned {enemies_spawned} initial enemies.")


        # Spawn initial Prey
        prey_spawned = 0
        for _ in range(INITIAL_PREY):
            if self.spawn_prey():
                prey_spawned += 1
        if prey_spawned < INITIAL_PREY:
             print(f"Warning: Spawned only {prey_spawned}/{INITIAL_PREY} initial prey.")
        else:
             print(f"Spawned {prey_spawned} initial prey.")

        return True # Spawning successful (even if counts are low)

    def _find_valid_queen_pos(self):
        """Finds a valid, non-obstacle integer position near nest center."""
        base_int = tuple(map(int, NEST_POS)) # Ensure integer coords

        # Check center first
        if is_valid(base_int) and not self.grid.is_obstacle(base_int):
            return base_int

        # Check immediate neighbors randomly
        neighbors = get_neighbors(base_int)
        random.shuffle(neighbors)
        for p_int in neighbors:
            if not self.grid.is_obstacle(p_int):
                return p_int

        # Check slightly further out if immediate neighbors failed
        for r in range(2, 5): # Check radius 2, 3, 4
            perimeter = []
            # Iterate over the bounding box of the radius
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # Check if it's on the perimeter of this radius
                    if abs(dx) == r or abs(dy) == r:
                        p_int = (base_int[0] + dx, base_int[1] + dy)
                        # Check validity and obstacle status
                        if is_valid(p_int) and not self.grid.is_obstacle(p_int):
                            perimeter.append(p_int)
            # If valid spots found on this perimeter, choose one randomly
            if perimeter:
                return random.choice(perimeter)

        # Critical failure if no spot found
        print("CRITICAL: Could not find any valid spot for Queen near NEST_POS.")
        return None

    # --- Entity Management Methods ---

    def add_entity(self, entity, entity_list, position_dict):
        """Adds an entity to the simulation list and position dictionary."""
        pos_int = entity.pos
        # Check if position is valid and not occupied by other *types*
        if (is_valid(pos_int) and
            not self.grid.is_obstacle(pos_int) and
            not self.is_ant_at(pos_int) and # Check ants/queen
            not self.is_enemy_at(pos_int) and # Check enemies
            not self.is_prey_at(pos_int) and # Check prey
            pos_int not in position_dict): # Ensure no entity of same type exists there

            entity_list.append(entity)
            position_dict[pos_int] = entity
            return True
        return False

    def add_ant(self, pos, caste: AntCaste):
        """Creates and adds a worker/soldier ant."""
        # Ensure pos is int tuple before creating Ant object
        pos_int = tuple(map(int, pos))
        if not is_valid(pos_int): return False # Quick invalid check

        # Check validity using combined checks before creating object
        if (not self.grid.is_obstacle(pos_int) and
            not self.is_ant_at(pos_int) and
            not self.is_enemy_at(pos_int) and
            not self.is_prey_at(pos_int)):
             ant = Ant(pos_int, self, caste)
             self.ants.append(ant)
             self.ant_positions[pos_int] = ant
             return True
        return False

    def add_brood(self, brood_item: BroodItem):
        """Adds a brood item to the simulation."""
        pos_int = brood_item.pos
        if is_valid(pos_int) and not self.grid.is_obstacle(pos_int):
            self.brood.append(brood_item)
            # Add to brood position lookup (can be multiple per cell)
            if pos_int not in self.brood_positions:
                self.brood_positions[pos_int] = []
            self.brood_positions[pos_int].append(brood_item)
            return True
        return False

    def remove_entity(self, entity, entity_list, position_dict):
        """Removes an entity from the list and position dictionary."""
        try:
            entity_list.remove(entity)
            pos = entity.pos
            # Remove from lookup dictionary safely
            if pos in position_dict and position_dict[pos] == entity:
                del position_dict[pos]
            # For brood, need to remove the specific item from the list at that pos
            elif isinstance(entity, BroodItem) and pos in self.brood_positions:
                 if entity in self.brood_positions[pos]:
                      self.brood_positions[pos].remove(entity)
                      if not self.brood_positions[pos]: # Remove key if list is empty
                           del self.brood_positions[pos]

        except ValueError:
            # Entity might have been removed already, log warning if needed
            # print(f"Warning: Attempted to remove entity not in list: {entity}")
            pass

    def update_entity_position(self, entity, old_pos, new_pos):
        """Updates the position of an entity in the lookup dictionaries."""
        pos_dict = None
        if isinstance(entity, Ant):       pos_dict = self.ant_positions
        elif isinstance(entity, Enemy):   pos_dict = self.enemy_positions
        elif isinstance(entity, Prey):    pos_dict = self.prey_positions
        # Brood and Queen don't move, so no update needed for them here

        if pos_dict is not None:
            # Remove from old position safely
            if old_pos in pos_dict and pos_dict[old_pos] == entity:
                del pos_dict[old_pos]
            # Add to new position
            # Important: Check if new_pos is already occupied by *another* entity of the same type
            # This check should ideally happen *before* the move in the entity's logic,
            # but we add a safeguard here. If occupied, the entity might be stuck or lost.
            # For simplicity now, we overwrite. Consider logging if overwrite happens.
            # existing_entity = pos_dict.get(new_pos)
            # if existing_entity and existing_entity != entity:
            #     print(f"Warning: Entity {entity} overwriting {existing_entity} at {new_pos}")
            pos_dict[new_pos] = entity


    def spawn_enemy(self):
        """Spawns a new enemy at a valid random integer location."""
        tries = 0
        # Use Queen's position if available, otherwise nest center
        nest_pos_int = self.queen.pos if self.queen else tuple(map(int, NEST_POS))
        min_dist_sq = (MIN_FOOD_DIST_FROM_NEST + 5) ** 2 # Spawn further away

        while tries < 80: # Increased attempts
            # Generate random position
            pos_i = (rnd(0, GRID_WIDTH - 1), rnd(0, GRID_HEIGHT - 1))
            # Check validity (distance, obstacle, existing entities)
            if (
                distance_sq(pos_i, nest_pos_int) > min_dist_sq
                and not self.grid.is_obstacle(pos_i)
                and not self.is_enemy_at(pos_i) # Use optimized check
                and not self.is_ant_at(pos_i)   # Use optimized check
                and not self.is_prey_at(pos_i)  # Use optimized check
            ):
                enemy = Enemy(pos_i, self)
                self.enemies.append(enemy)
                self.enemy_positions[pos_i] = enemy
                return True # Successfully spawned
            tries += 1
        return False # Failed to find a spot

    def spawn_prey(self):
        """Spawns a new prey item at a valid random location."""
        tries = 0
        nest_pos_int = self.queen.pos if self.queen else tuple(map(int, NEST_POS))
        min_dist_sq = (MIN_FOOD_DIST_FROM_NEST - 10) ** 2 # Can spawn closer than enemies

        while tries < 70: # Increased attempts
            pos_i = (rnd(0, GRID_WIDTH - 1), rnd(0, GRID_HEIGHT - 1))
            # Check validity
            if (
                distance_sq(pos_i, nest_pos_int) > min_dist_sq
                and not self.grid.is_obstacle(pos_i)
                and not self.is_enemy_at(pos_i)
                and not self.is_ant_at(pos_i)
                and not self.is_prey_at(pos_i, exclude_self=None) # Exclude self needed? No, spawning new.
            ):
                prey_item = Prey(pos_i, self)
                self.prey.append(prey_item)
                self.prey_positions[pos_i] = prey_item
                return True
            tries += 1
        return False

    def kill_ant(self, ant_to_remove, reason="unknown"):
        """Removes an ant from the simulation."""
        self.remove_entity(ant_to_remove, self.ants, self.ant_positions)
        # Optionally log reason for death, etc.

    def kill_enemy(self, enemy_to_remove):
        """Removes an enemy and drops food resources."""
        pos_int = enemy_to_remove.pos # Get position before removal
        self.remove_entity(enemy_to_remove, self.enemies, self.enemy_positions)

        # Drop food at the enemy's last valid position
        if is_valid(pos_int) and not self.grid.is_obstacle(pos_int):
            fx, fy = pos_int
            grid = self.grid
            s_idx = FoodType.SUGAR.value
            p_idx = FoodType.PROTEIN.value
            try:
                # Add sugar, clamp to max
                grid.food[fx, fy, s_idx] = min(
                    MAX_FOOD_PER_CELL,
                    grid.food[fx, fy, s_idx] + ENEMY_TO_FOOD_ON_DEATH_SUGAR,
                )
                # Add protein, clamp to max
                grid.food[fx, fy, p_idx] = min(
                    MAX_FOOD_PER_CELL,
                    grid.food[fx, fy, p_idx] + ENEMY_TO_FOOD_ON_DEATH_PROTEIN,
                )
            except IndexError: pass # Safeguard

    def kill_prey(self, prey_to_remove):
        """Removes prey and adds protein food to the grid."""
        pos_int = prey_to_remove.pos # Get position before removal
        self.remove_entity(prey_to_remove, self.prey, self.prey_positions)

        # Drop protein at the prey's last valid position
        if is_valid(pos_int) and not self.grid.is_obstacle(pos_int):
            fx, fy = pos_int
            grid = self.grid
            p_idx = FoodType.PROTEIN.value
            try:
                # Add protein, clamp to max
                grid.food[fx, fy, p_idx] = min(
                    MAX_FOOD_PER_CELL,
                    grid.food[fx, fy, p_idx] + PROTEIN_ON_DEATH,
                )
            except IndexError: pass # Safeguard

    def kill_queen(self, queen_to_remove):
        """Handles the death of the queen, ending the simulation run."""
        if self.queen == queen_to_remove:
            print(
                f"\n--- QUEEN DIED (Tick {int(self.ticks)}, Kolonie {self.colony_generation}) ---"
            )
            print(
                f"    Food S:{self.colony_food_storage_sugar:.1f} "
                f"P:{self.colony_food_storage_protein:.1f}"
            )
            print(f"    Ants:{len(self.ants)}, Brood:{len(self.brood)}")
            self.queen = None # Set queen reference to None
            self.simulation_running = False # Stop the current simulation run
            self.end_game_reason = "Königin gestorben" # Set reason for end dialog

    # --- Optimized Lookup Methods ---

    def is_ant_at(self, pos_int, exclude_self=None):
        """Checks if a worker/soldier ant or the queen is at an integer position."""
        # Check Queen first (handled separately)
        if self.queen and self.queen.pos == pos_int and exclude_self != self.queen:
            return True
        # Check worker/soldier dictionary
        ant = self.ant_positions.get(pos_int)
        return ant is not None and ant is not exclude_self

    def get_ant_at(self, pos_int):
        """Returns the ant/queen object at an integer position, or None."""
        # Check Queen first
        if self.queen and self.queen.pos == pos_int:
            return self.queen
        # Check worker/soldier dictionary
        return self.ant_positions.get(pos_int)

    def is_enemy_at(self, pos_int, exclude_self=None):
        """Checks if an enemy is at an integer position using dictionary lookup."""
        enemy = self.enemy_positions.get(pos_int)
        return enemy is not None and enemy is not exclude_self

    def get_enemy_at(self, pos_int):
        """Returns the enemy object at an integer position, or None."""
        return self.enemy_positions.get(pos_int)

    def is_prey_at(self, pos_int, exclude_self=None):
        """Checks if a prey item is at an integer position using dictionary lookup."""
        prey_item = self.prey_positions.get(pos_int)
        return prey_item is not None and prey_item is not exclude_self

    def get_prey_at(self, pos_int):
        """Returns the prey object at an integer position, or None."""
        return self.prey_positions.get(pos_int)

    def get_brood_positions(self):
         """Returns a set of positions occupied by brood items."""
         # This can be slow if called frequently with large brood numbers
         # Consider optimizing if profiling shows it as a bottleneck
         # return {b.pos for b in self.brood}
         # Using the dictionary keys is faster:
         return set(self.brood_positions.keys())


    def find_nearby_prey(self, pos_int, radius_sq):
        """Finds prey within a certain squared radius of a position."""
        # This still requires iterating through all prey.
        # If this becomes a bottleneck, spatial hashing (e.g., a grid overlay)
        # could be implemented, but adds complexity.
        # For now, list comprehension is relatively clean.
        nearby = []
        for p in self.prey:
             # Check distance and HP > 0
             if p.hp > 0 and distance_sq(pos_int, p.pos) <= radius_sq:
                  nearby.append(p)
        return nearby
        # List comprehension version:
        # return [
        #     p for p in self.prey
        #     if p.hp > 0 and distance_sq(pos_int, p.pos) <= radius_sq
        # ]

    # --- Main Simulation Loop Step ---

    def update(self):
        """Runs one simulation tick, updating all entities and the environment."""
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]
        # Skip updates if paused, but still allow drawing and event handling
        if current_multiplier == 0.0:
            self.ticks += 0.001 # Increment ticks slightly even when paused
            return

        # Increment simulation time
        self.ticks += current_multiplier

        # --- Entity Updates ---
        # Update Queen first
        if self.queen:
            self.queen.update(current_multiplier)
        # Check if queen died during her update
        if not self.simulation_running: return # Queen death stops simulation

        # Update Brood and handle hatching
        hatched_pupae = [] # Store pupae that finished developing
        # Iterate over a copy in case brood list is modified during update
        brood_copy = list(self.brood)
        for item in brood_copy:
            # Check if item still exists (might have been removed if invalid?)
            if item in self.brood:
                hatch_signal = item.update(self.ticks, self)
                if hatch_signal: # hatch_signal is the BroodItem itself if ready
                    hatched_pupae.append(hatch_signal)

        # Process hatched pupae (spawn new ants)
        for pupa in hatched_pupae:
            if pupa in self.brood: # Ensure it wasn't removed elsewhere
                self.remove_entity(pupa, self.brood, self.brood_positions) # Remove brood item
                self._spawn_hatched_ant(pupa.caste, pupa.pos) # Spawn ant

        # Update mobile entities (Ants, Enemies, Prey)
        # Process copies and shuffle for fairness
        ants_copy = list(self.ants)
        random.shuffle(ants_copy)
        enemies_copy = list(self.enemies)
        random.shuffle(enemies_copy)
        prey_copy = list(self.prey)
        random.shuffle(prey_copy)

        # Update Ants
        for a in ants_copy:
            if a in self.ants and a.hp > 0: # Check if still alive and in list
                a.update(current_multiplier)

        # Update Enemies
        for e in enemies_copy:
            if e in self.enemies and e.hp > 0: # Check if still alive and in list
                e.update(current_multiplier)

        # Update Prey
        for p in prey_copy:
            if p in self.prey and p.hp > 0: # Check if still alive and in list
                p.update(current_multiplier)


        # --- Cleanup Phase (Remove dead/invalid entities) ---
        # Use list comprehensions for cleaner removal setup
        ants_to_remove = [a for a in self.ants if a.hp <= 0 or self.grid.is_obstacle(a.pos)]
        enemies_to_remove = [e for e in self.enemies if e.hp <= 0 or self.grid.is_obstacle(e.pos)]
        prey_to_remove = [p for p in self.prey if p.hp <= 0 or self.grid.is_obstacle(p.pos)]
        # Note: Brood removal happens during hatching or if simulation adds logic for brood death

        for a in ants_to_remove: self.kill_ant(a, "cleanup")
        for e in enemies_to_remove: self.kill_enemy(e) # Drops food
        for p in prey_to_remove: self.kill_prey(p) # Drops food

        # Final check for Queen's health after all updates and cleanup
        if self.queen and (self.queen.hp <= 0 or self.grid.is_obstacle(self.queen.pos)):
            self.kill_queen(self.queen) # This will set simulation_running to False

        # Check again if simulation should continue after cleanup
        if not self.simulation_running: return

        # --- Environment Updates ---
        # Update Pheromones (decay and diffusion)
        self.grid.update_pheromones(current_multiplier)

        # Spawn new Enemies periodically
        self.enemy_spawn_timer += current_multiplier
        if self.enemy_spawn_timer >= self.enemy_spawn_interval_ticks:
            self.enemy_spawn_timer %= self.enemy_spawn_interval_ticks # Reset timer part
            # Limit total number of enemies
            if len(self.enemies) < INITIAL_ENEMIES * 6:
                self.spawn_enemy() # Attempt to spawn one

        # Spawn new Prey periodically
        self.prey_spawn_timer += current_multiplier
        if self.prey_spawn_timer >= self.prey_spawn_interval_ticks:
            self.prey_spawn_timer %= self.prey_spawn_interval_ticks # Reset timer part
            # Limit total number of prey
            max_prey = INITIAL_PREY * 3
            if len(self.prey) < max_prey:
                self.spawn_prey() # Attempt to spawn one


    def _spawn_hatched_ant(self, caste: AntCaste, pupa_pos_int: tuple):
        """Tries to spawn a newly hatched ant at/near the pupa's location."""
        # Try spawning at the exact pupa location first
        if self.add_ant(pupa_pos_int, caste):
            return True

        # If exact spot is blocked, try adjacent neighbors randomly
        neighbors = get_neighbors(pupa_pos_int)
        random.shuffle(neighbors)
        for pos_int in neighbors:
            if self.add_ant(pos_int, caste):
                return True

        # If neighbors also failed, try a random spot near the nest center
        if self.queen: # Use queen pos if available
            base_pos = self.queen.pos
            for _ in range(15): # Limit attempts
                # Random offset within nest radius
                ox = rnd(-(NEST_RADIUS - 1), NEST_RADIUS - 1)
                oy = rnd(-(NEST_RADIUS - 1), NEST_RADIUS - 1)
                pos_int = (base_pos[0] + ox, base_pos[1] + oy)
                if self.add_ant(pos_int, caste):
                    return True

        # Log failure if ant couldn't be placed
        # print(f"Warning: Could not place hatched {caste.name} ant near {pupa_pos_int}.")
        return False

    # --- Drawing Methods ---
    def draw_debug_info(self):
        """Renders and draws the debug information overlay."""
        if not self.debug_font: return # Cannot draw without font
        # Gather data for display
        ant_c = len(self.ants)
        enemy_c = len(self.enemies)
        brood_c = len(self.brood)
        prey_c = len(self.prey)
        food_s = self.colony_food_storage_sugar
        food_p = self.colony_food_storage_protein
        tick_display = int(self.ticks)
        fps = self.clock.get_fps()
        # Count castes efficiently
        w_c = sum(1 for a in self.ants if a.caste == AntCaste.WORKER)
        s_c = ant_c - w_c # Soldier count is the remainder
        # Count brood stages
        e_c = sum(1 for b in self.brood if b.stage == BroodStage.EGG)
        l_c = sum(1 for b in self.brood if b.stage == BroodStage.LARVA)
        p_c = brood_c - e_c - l_c # Pupa count is remainder
        # Format speed text
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]
        speed_text = (
            "Speed: Paused"
            if current_multiplier == 0.0
            else f"Speed: {current_multiplier:.1f}x".replace(".0x", "x")
        )

        # Prepare text lines
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
        # Render and blit lines
        y_start = 5
        line_height = self.debug_font.get_height() + 1
        text_color = (240, 240, 240)
        for i, txt in enumerate(texts):
            try:
                surf = self.debug_font.render(txt, True, text_color)
                self.screen.blit(surf, (5, y_start + i * line_height))
            except Exception as e:
                # Log font rendering errors, but don't crash
                print(f"Debug Font render error (line '{txt}'): {e}")

        # --- Mouse Hover Info ---
        # This part can be slow due to grid lookups on mouse move.
        # Consider adding a flag to enable/disable hover info separately.
        try:
            mx, my = pygame.mouse.get_pos()
            gx, gy = mx // CELL_SIZE, my // CELL_SIZE
            pos_i = (gx, gy) # Integer grid position under mouse

            if is_valid(pos_i):
                hover_lines = []
                # Check for entities at the hovered cell using optimized lookups
                entity = (
                    self.get_ant_at(pos_i) # Checks queen and ants
                    or self.get_enemy_at(pos_i)
                    or self.get_prey_at(pos_i)
                )

                # Display info about the entity found
                if entity:
                    entity_pos_int = entity.pos # Use entity's actual position
                    if isinstance(entity, Queen):
                        hover_lines.extend([
                                f"QUEEN @{entity_pos_int}",
                                f"HP:{entity.hp:.0f}/{entity.max_hp} Age:{entity.age:.0f}",])
                    elif isinstance(entity, Ant):
                        hover_lines.extend([
                            f"{entity.caste.name} @{entity_pos_int}",
                            f"S:{entity.state.name} HP:{entity.hp:.0f}",
                            f"C:{entity.carry_amount:.1f}({entity.carry_type.name if entity.carry_type else '-'})",
                            f"Age:{entity.age:.0f}/{entity.max_age_ticks}",
                            f"Mv:{entity.last_move_info[:28]}", # Truncate long move info
                        ])
                    elif isinstance(entity, Enemy):
                        hover_lines.extend([
                            f"ENEMY @{entity_pos_int}", f"HP:{entity.hp:.0f}/{entity.max_hp}",])
                    elif isinstance(entity, Prey):
                        hover_lines.extend([
                            f"PREY @{entity_pos_int}", f"HP:{entity.hp:.0f}/{entity.max_hp}",])

                # Display info about brood at the hovered cell
                # Use brood_positions dict for faster lookup
                brood_at_pos = self.brood_positions.get(pos_i, [])
                if brood_at_pos:
                    hover_lines.append(f"Brood:{len(brood_at_pos)} @{pos_i}")
                    # Show details for first few brood items
                    for b in brood_at_pos[:3]:
                        hover_lines.append(
                            f"-{b.stage.name[:1]}({b.caste.name[:1]}) " # Abbreviate
                            f"{int(b.progress_timer)}/{b.duration}"
                        )

                # Display cell info (obstacle, food, pheromones)
                is_obs = self.grid.is_obstacle(pos_i)
                obs_txt = " OBSTACLE" if is_obs else ""
                hover_lines.append(f"Cell:{pos_i}{obs_txt}")

                if not is_obs:
                    try:
                        # Food info
                        foods = self.grid.food[pos_i[0], pos_i[1]]
                        food_txt = f"Food S:{foods[0]:.1f} P:{foods[1]:.1f}"
                        # Pheromone info (using optimized getter)
                        ph_home = self.grid.get_pheromone(pos_i, "home")
                        ph_food_s = self.grid.get_pheromone(pos_i, "food", FoodType.SUGAR)
                        ph_food_p = self.grid.get_pheromone(pos_i, "food", FoodType.PROTEIN)
                        ph_alarm = self.grid.get_pheromone(pos_i, "alarm")
                        ph_neg = self.grid.get_pheromone(pos_i, "negative")
                        ph_rec = self.grid.get_pheromone(pos_i, "recruitment")
                        # Format pheromone lines
                        ph1 = f"Ph H:{ph_home:.0f} FS:{ph_food_s:.0f} FP:{ph_food_p:.0f}"
                        ph2 = f"Ph A:{ph_alarm:.0f} N:{ph_neg:.0f} R:{ph_rec:.0f}"
                        hover_lines.extend([food_txt, ph1, ph2])
                    except IndexError:
                        hover_lines.append("Error reading cell data") # Should not happen

                # Render and blit hover lines at the bottom
                hover_color = (255, 255, 0) # Yellow
                hover_y_start = HEIGHT - (len(hover_lines) * line_height) - 5
                for i, line in enumerate(hover_lines):
                    surf = self.debug_font.render(line, True, hover_color)
                    self.screen.blit(surf, (5, hover_y_start + i * line_height))
        except Exception as e:
            # Catch potential errors during hover info generation
            # print(f"Hover info error: {e}") # Optional logging
            pass


    def draw(self):
        """Draws all simulation elements onto the screen."""
        # 1. Draw the grid (background, obstacles, pheromones, food)
        self._draw_grid()
        # 2. Draw brood items
        self._draw_brood()
        # 3. Draw the queen
        self._draw_queen()
        # 4. Draw ants and enemies
        self._draw_entities()
        # 5. Draw prey
        self._draw_prey()
        # 6. Draw debug overlay if enabled
        if self.show_debug_info:
            self.draw_debug_info()
        # 7. Draw UI buttons
        self._draw_buttons()
        # 8. Update the display
        pygame.display.flip()

    def _draw_grid(self):
        """Draws the grid: static background, pheromones, food, nest area."""
        cs = CELL_SIZE # Cache cell size

        # --- Draw Static Background (Color + Obstacles) ---
        # This surface was prepared in _prepare_static_background()
        self.screen.blit(self.static_background_surface, (0, 0))

        # --- Draw Pheromones ---
        # Define pheromone types, colors, arrays, and max values
        ph_info = {
            "home": (PHEROMONE_HOME_COLOR, self.grid.pheromones_home, PHEROMONE_MAX),
            "food_sugar": (PHEROMONE_FOOD_SUGAR_COLOR, self.grid.pheromones_food_sugar, PHEROMONE_MAX),
            "food_protein": (PHEROMONE_FOOD_PROTEIN_COLOR, self.grid.pheromones_food_protein, PHEROMONE_MAX),
            "alarm": (PHEROMONE_ALARM_COLOR, self.grid.pheromones_alarm, PHEROMONE_MAX),
            "negative": (PHEROMONE_NEGATIVE_COLOR, self.grid.pheromones_negative, PHEROMONE_MAX),
            "recruitment": (PHEROMONE_RECRUITMENT_COLOR, self.grid.pheromones_recruitment, RECRUITMENT_PHEROMONE_MAX),
        }
        min_alpha_for_draw = 5 # Don't draw extremely faint pheromones

        # Iterate through pheromone types and draw them
        for ph_type, (base_col, arr, current_max) in ph_info.items():
            # Create a temporary surface with alpha for blending
            ph_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            # Normalization factor (adjust for visual intensity)
            norm_divisor = max(current_max / 2.5, 1.0) # Avoid division by zero
            # Get coordinates where pheromone > threshold
            nz_coords = np.argwhere(arr > MIN_PHEROMONE_DRAW_THRESHOLD)

            # Draw rectangles for each pheromone cell above threshold
            for x, y in nz_coords:
                val = arr[x, y]
                norm_val = normalize(val, norm_divisor) # Normalize 0-1
                # Calculate alpha based on normalized value
                alpha = min(max(int(norm_val * base_col[3]), 0), 255) # Use alpha from color tuple

                if alpha >= min_alpha_for_draw: # Only draw if sufficiently visible
                    color = (*base_col[:3], alpha) # Create color tuple with calculated alpha
                    # Draw rectangle directly onto the pheromone surface
                    pygame.draw.rect(ph_surf, color, (x * cs, y * cs, cs, cs))

            # Blit the pheromone layer onto the main screen
            self.screen.blit(ph_surf, (0, 0))

        # --- Draw Food ---
        min_draw_food = 0.1 # Threshold for drawing food
        # Calculate total food per cell to find non-empty cells efficiently
        food_totals = np.sum(self.grid.food, axis=2)
        food_nz_coords = np.argwhere(food_totals > min_draw_food)

        s_idx = FoodType.SUGAR.value
        p_idx = FoodType.PROTEIN.value
        s_col = FOOD_COLORS[FoodType.SUGAR]
        p_col = FOOD_COLORS[FoodType.PROTEIN]

        # Draw food rectangles, mixing colors based on ratio
        for x, y in food_nz_coords:
            try:
                foods = self.grid.food[x, y]
                s = foods[s_idx]
                p = foods[p_idx]
                total = s + p # Already know total > min_draw_food
                # Calculate ratios (avoid division by zero, though total should be > 0 here)
                sr = s / total if total > 0 else 0.5
                pr = 1.0 - sr # Protein ratio is the complement

                # Mix colors based on ratios
                color = (
                    int(s_col[0] * sr + p_col[0] * pr),
                    int(s_col[1] * sr + p_col[1] * pr),
                    int(s_col[2] * sr + p_col[2] * pr),
                )
                # Draw the food rectangle
                pygame.draw.rect(self.screen, color, (x * cs, y * cs, cs, cs))
            except IndexError:
                continue # Skip if coords somehow invalid

        # --- Draw Nest Area Outline ---
        r = NEST_RADIUS
        nx, ny = tuple(map(int, NEST_POS))
        # Calculate pixel coordinates for the nest bounding box
        nest_rect_coords = (
            (nx - r) * cs, (ny - r) * cs, # Top-left corner
            (r * 2 + 1) * cs, (r * 2 + 1) * cs # Width and height
        )
        try:
            rect = pygame.Rect(nest_rect_coords)
            # Create a semi-transparent surface for the overlay
            nest_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            nest_surf.fill((100, 100, 100, 35)) # Light grey, low alpha
            # Blit the overlay onto the screen
            self.screen.blit(nest_surf, rect.topleft)
        except ValueError:
            # Handle potential errors if rect coords are invalid
            pass

    def _draw_brood(self):
        """Draws all brood items."""
        # Iterate over a copy in case list changes during drawing (less likely here)
        for item in list(self.brood):
             # Check if item still exists and is valid before drawing
             if item in self.brood and is_valid(item.pos):
                 item.draw(self.screen) # Call brood item's own draw method

    def _draw_queen(self):
        """Draws the queen ant."""
        if not self.queen or not is_valid(self.queen.pos):
            return # No queen or invalid position
        # Calculate center pixel position
        pos_px = (
            int(self.queen.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.queen.pos[1] * CELL_SIZE + CELL_SIZE / 2),
        )
        radius = int(CELL_SIZE / 1.4) # Queen is larger
        # Draw queen body
        pygame.draw.circle(self.screen, self.queen.color, pos_px, radius)
        # Draw outline
        pygame.draw.circle(self.screen, (255, 255, 255), pos_px, radius, 1) # White outline


    def _draw_entities(self):
        """Draws all worker/soldier ants and enemies."""
        cs_half = CELL_SIZE / 2 # Cache half cell size

        # --- Draw Ants ---
        # Iterate over a copy of the list
        for a in list(self.ants):
            # Check validity and existence
            if a not in self.ants or not is_valid(a.pos):
                continue
            # Calculate pixel center
            pos_px = (
                int(a.pos[0] * CELL_SIZE + cs_half),
                int(a.pos[1] * CELL_SIZE + cs_half),
            )
            radius = int(CELL_SIZE / a.size_factor) # Radius based on caste attribute

            # Determine color based on state
            color = a.search_color # Default color
            if a.state == AntState.RETURNING_TO_NEST: color = a.return_color
            elif a.state == AntState.ESCAPING: color = WORKER_ESCAPE_COLOR
            elif a.state == AntState.DEFENDING: color = (255, 100, 0) # Orange/Red
            elif a.state == AntState.HUNTING: color = (0, 200, 150) # Teal/Cyan

            # Draw ant body
            pygame.draw.circle(self.screen, color, pos_px, radius)

            # Draw carried food indicator if carrying
            if a.carry_amount > 0 and a.carry_type:
                food_color = FOOD_COLORS.get(a.carry_type, FOOD_COLOR_MIX) # Get color for type
                # Draw smaller circle inside representing food
                pygame.draw.circle(
                    self.screen, food_color, pos_px, int(radius * 0.55)
                )

        # --- Draw Enemies ---
        # Iterate over a copy of the list
        for e in list(self.enemies):
             # Check validity and existence
             if e not in self.enemies or not is_valid(e.pos):
                 continue
             # Calculate pixel center
             pos_px = (
                 int(e.pos[0] * CELL_SIZE + cs_half),
                 int(e.pos[1] * CELL_SIZE + cs_half),
             )
             radius = int(CELL_SIZE / 2.3) # Enemy size
             # Draw enemy body
             pygame.draw.circle(self.screen, e.color, pos_px, radius)
             # Draw outline
             pygame.draw.circle(self.screen, (0, 0, 0), pos_px, radius, 1) # Black outline

    def _draw_prey(self):
        """Draws all prey items."""
        # Iterate over a copy of the list
        for p in list(self.prey):
             # Check validity and existence
             if p in self.prey and is_valid(p.pos):
                 p.draw(self.screen) # Call prey's own draw method

    def _draw_buttons(self):
        """Draws the UI buttons for speed control."""
        if not self.font: return # Need font to draw text
        mouse_pos = pygame.mouse.get_pos() # Get current mouse position

        for button in self.buttons:
            rect = button["rect"]
            text = button["text"]
            # Determine color based on hover state
            color = (
                BUTTON_HOVER_COLOR
                if rect.collidepoint(mouse_pos)
                else BUTTON_COLOR
            )
            # Draw button background
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            # Render and draw button text centered
            try:
                text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)
            except Exception as e:
                # Log font errors but continue
                print(f"Button font render error ('{text}'): {e}")

    # --- Event Handling ---
    def handle_events(self):
        """Processes Pygame events for user input."""
        for event in pygame.event.get():
            # Quit event
            if event.type == pygame.QUIT:
                self.simulation_running = False
                self.app_running = False
                self.end_game_reason = "Fenster geschlossen"
                return "quit" # Signal app quit

            # Key press events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.simulation_running = False # Stop current run
                    self.end_game_reason = "ESC gedrückt"
                    return "sim_stop" # Signal sim stop, show dialog
                if event.key == pygame.K_d:
                    self.show_debug_info = not self.show_debug_info # Toggle debug
                # Speed control keys
                if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self._handle_button_click("speed_down")
                    return "speed_change" # Signal speed change
                if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    self._handle_button_click("speed_up")
                    return "speed_change" # Signal speed change

            # Mouse click events
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Handle button clicks only if simulation is running
                    if self.simulation_running:
                        for button in self.buttons:
                            if button["rect"].collidepoint(event.pos):
                                self._handle_button_click(button["action"])
                                return "speed_change" # Signal speed change
        return None # No significant action taken

    def _handle_button_click(self, action):
        """Updates simulation speed index based on button action."""
        current_index = self.simulation_speed_index
        max_index = len(SPEED_MULTIPLIERS) - 1
        new_index = current_index # Default to no change

        # Adjust index based on action
        if action == "speed_down":
            new_index = max(0, current_index - 1) # Decrease, clamp at 0
        elif action == "speed_up":
            new_index = min(max_index, current_index + 1) # Increase, clamp at max
        else:
            print(f"Warning: Unknown button action '{action}'")
            return # Ignore unknown actions

        # Update speed if index changed
        if new_index != self.simulation_speed_index:
            self.simulation_speed_index = new_index
            # Update target FPS based on new speed index
            self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
            # Optional: Log speed change
            # new_speed = SPEED_MULTIPLIERS[self.simulation_speed_index]
            # speed_str = "Paused" if new_speed == 0.0 else f"{new_speed:.1f}x"
            # print(f"Speed changed: {speed_str}")

    # --- End Game Dialog ---
    def _show_end_game_dialog(self):
        """Displays the 'Restart' or 'Quit' dialog."""
        if not self.font: # Cannot show dialog without font
            print("Error: Cannot display end game dialog - font not loaded.")
            return "quit" # Force quit if no font

        # Dialog dimensions and positioning
        dialog_w, dialog_h = 300, 150
        dialog_x = (WIDTH - dialog_w) // 2
        dialog_y = (HEIGHT - dialog_h) // 2

        # Button dimensions and positioning within dialog
        btn_w, btn_h = 100, 35
        btn_margin = 20
        btn_y = dialog_y + dialog_h - btn_h - 25 # Position near bottom
        total_btn_width = btn_w * 2 + btn_margin
        btn_restart_x = dialog_x + (dialog_w - total_btn_width) // 2
        btn_quit_x = btn_restart_x + btn_w + btn_margin

        # Create Rects for buttons
        restart_rect = pygame.Rect(btn_restart_x, btn_y, btn_w, btn_h)
        quit_rect = pygame.Rect(btn_quit_x, btn_y, btn_w, btn_h)

        # Text and colors
        text_color = (240, 240, 240)
        title_text = f"Kolonie {self.colony_generation} Ende"
        reason_text = f"Grund: {self.end_game_reason}"

        # Semi-transparent overlay for background dimming
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill(END_DIALOG_BG_COLOR)

        waiting_for_choice = True
        while waiting_for_choice and self.app_running:
            mouse_pos = pygame.mouse.get_pos() # Get mouse pos every frame

            # Event handling within the dialog loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.app_running = False
                    waiting_for_choice = False
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.app_running = False # Allow ESC to quit from dialog
                    waiting_for_choice = False
                    return "quit"
                # Check for mouse clicks on buttons
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if restart_rect.collidepoint(mouse_pos):
                        return "restart" # Signal restart
                    if quit_rect.collidepoint(mouse_pos):
                        self.app_running = False # Signal quit
                        return "quit"

            # --- Drawing the Dialog ---
            # Draw the dimming overlay
            self.screen.blit(overlay, (0, 0))
            # Draw dialog background box
            pygame.draw.rect(
                self.screen, (40, 40, 80), # Dark blue background
                (dialog_x, dialog_y, dialog_w, dialog_h),
                border_radius=6,
            )

            # Draw text elements (title, reason)
            try:
                title_surf = self.font.render(title_text, True, text_color)
                title_rect = title_surf.get_rect(
                    center=(dialog_x + dialog_w // 2, dialog_y + 35) # Centered
                )
                self.screen.blit(title_surf, title_rect)

                reason_surf = self.font.render(reason_text, True, text_color)
                reason_rect = reason_surf.get_rect(
                    center=(dialog_x + dialog_w // 2, dialog_y + 70) # Centered below title
                )
                self.screen.blit(reason_surf, reason_rect)
            except Exception: pass # Ignore font errors in dialog

            # Draw Restart button
            r_color = BUTTON_HOVER_COLOR if restart_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, r_color, restart_rect, border_radius=4)
            try:
                r_text_surf = self.font.render("Neu starten", True, BUTTON_TEXT_COLOR)
                r_text_rect = r_text_surf.get_rect(center=restart_rect.center)
                self.screen.blit(r_text_surf, r_text_rect)
            except Exception: pass

            # Draw Quit button
            q_color = BUTTON_HOVER_COLOR if quit_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, q_color, quit_rect, border_radius=4)
            try:
                q_text_surf = self.font.render("Beenden", True, BUTTON_TEXT_COLOR)
                q_text_rect = q_text_surf.get_rect(center=quit_rect.center)
                self.screen.blit(q_text_surf, q_text_rect)
            except Exception: pass

            # Update display and control frame rate
            pygame.display.flip()
            self.clock.tick(30) # Lower FPS for dialog is fine

        # Return "quit" if loop exited without making a choice (e.g., app_running became false)
        return "quit"

    # --- Main Application Loop ---
    def run(self):
        """Main application loop: handles simulation runs and end dialog."""
        print("Starting Ant Simulation...")
        print("Controls: D=Debug | ESC=End Run | +/- = Speed")

        while self.app_running:
            # Inner loop: Run the simulation until it stops or app quits
            while self.simulation_running and self.app_running:
                # Handle user input and window events
                action = self.handle_events()
                if not self.app_running: break # Exit outer loop if app quit signal received
                if action == "sim_stop": break # Exit inner loop to show dialog

                # Update simulation state
                self.update()
                # Draw the current state
                self.draw()
                # Control frame rate based on current speed setting
                self.clock.tick(self.current_target_fps)

            # --- End of Simulation Run ---
            if not self.app_running: break # Exit if app quit during simulation

            # Ensure there's a reason set if simulation stopped unexpectedly
            if not self.end_game_reason:
                self.end_game_reason = "Simulation beendet" # Default reason

            # Show the end game dialog and get user choice
            choice = self._show_end_game_dialog()

            # Handle user choice
            if choice == "restart":
                self._reset_simulation() # Reset and start new simulation run
            elif choice == "quit":
                self.app_running = False # Set flag to exit outer loop

        # --- Application Exit ---
        print("Exiting application.")
        try:
            pygame.quit() # Clean up Pygame resources
            print("Pygame shut down.")
        except Exception as e:
            print(f"Error during Pygame quit: {e}")


# --- Start Simulation ---
if __name__ == "__main__":
    print("Initializing simulation environment...")
    # Check for required libraries
    try:
        import numpy
        print(f"NumPy version: {numpy.__version__}")
    except ImportError:
        print("FATAL: NumPy is required but not found.")
        input("Press Enter to Exit.")
        exit()
    try:
        import pygame
        print(f"Pygame version: {pygame.version.ver}")
    except ImportError as e:
        print(f"FATAL: Pygame import failed: {e}")
        input("Press Enter to Exit.")
        exit()
    except Exception as e: # Catch other potential pygame import errors
        print(f"FATAL: Pygame import error: {e}")
        input("Press Enter to Exit.")
        exit()

    # Initialize Pygame modules
    initialization_success = False
    try:
        pygame.init()
        # Explicitly check critical modules
        if not pygame.display.get_init(): raise RuntimeError("Display module failed")
        # Font check is now inside AntSimulation constructor, but double-check init status
        if not pygame.font.get_init(): raise RuntimeError("Font module failed to init (basic)")
        print("Pygame initialized successfully.")
        initialization_success = True
    except Exception as e:
        print(f"FATAL: Pygame initialization failed: {e}")
        try: pygame.quit() # Attempt cleanup
        except Exception: pass
        input("Press Enter to Exit.")
        exit()

    # Run the simulation if initialization succeeded
    if initialization_success:
        simulation_instance = None # Keep reference for potential error handling
        try:
            simulation_instance = AntSimulation()
            # Check if AntSimulation itself failed (e.g., font loading)
            if simulation_instance.app_running:
                simulation_instance.run() # Start the main loop
            else:
                print("Application cannot start due to initialization errors (fonts?).")
        except Exception as e:
            # Catch unexpected errors during simulation run
            print("\n--- CRITICAL UNHANDLED EXCEPTION CAUGHT ---")
            traceback.print_exc() # Print detailed traceback
            print("---------------------------------------------")
            print("Attempting to exit gracefully...")
            try:
                # Try to signal stop and quit pygame
                if simulation_instance: simulation_instance.app_running = False
                pygame.quit()
            except Exception: pass # Ignore errors during emergency exit
            input("Press Enter to Exit.")

    print("Simulation process finished.")