# -*- coding: utf-8 -*-

# Standard Library Imports
import random
import math
import time
from enum import Enum, auto
import io  # Keep for potential future web streaming integration

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
    TENDING_BROOD = auto() # Placeholder


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
    SUGAR = 0   # Index for array access
    PROTEIN = 1 # Index for array access


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
MAX_FOOD_PER_CELL = 100.0 # Max per type per cell
INITIAL_COLONY_FOOD_SUGAR = 80.0
INITIAL_COLONY_FOOD_PROTEIN = 80.0
RICH_FOOD_THRESHOLD = 50.0 # Food amount per type to trigger recruitment

# Obstacles
NUM_OBSTACLES = 10
MIN_OBSTACLE_SIZE = 3
MAX_OBSTACLE_SIZE = 10
OBSTACLE_COLOR = (100, 100, 100)

# Pheromones
PHEROMONE_MAX = 1000.0
PHEROMONE_DECAY = 0.9985 # Slower decay
PHEROMONE_DIFFUSION_RATE = 0.04 # Slower diffusion
NEGATIVE_PHEROMONE_DECAY = 0.992
NEGATIVE_PHEROMONE_DIFFUSION_RATE = 0.06
RECRUITMENT_PHEROMONE_DECAY = 0.96
RECRUITMENT_PHEROMONE_DIFFUSION_RATE = 0.12
RECRUITMENT_PHEROMONE_MAX = 500.0

# Weights (Influence on ant decision-making)
W_HOME_PHEROMONE_RETURN = 45.0
W_FOOD_PHEROMONE_SEARCH = 40.0
W_HOME_PHEROMONE_SEARCH = 0.0     # No attraction to home when searching
W_ALARM_PHEROMONE = -35.0         # General avoidance of alarm
W_NEST_DIRECTION_RETURN = 85.0    # Attraction to nest when returning
W_NEST_DIRECTION_PATROL = -10.0   # Push away when patrolling inside radius
W_ALARM_SOURCE_DEFEND = 150.0     # Strong pull to alarm source for defend state
W_PERSISTENCE = 1.5
W_AVOID_HISTORY = -1000.0         # Implicitly handled by not choosing history
W_RANDOM_NOISE = 0.2
W_NEGATIVE_PHEROMONE = -50.0
W_RECRUITMENT_PHEROMONE = 200.0
W_AVOID_NEST_SEARCHING = -80.0    # Penalty for searching near nest center

# Probabilistic Choice Parameters
PROBABILISTIC_CHOICE_TEMP = 1.0
MIN_SCORE_FOR_PROB_CHOICE = 0.01

# Pheromone Drop Amounts
P_FOOD_SEARCHING = 0.0
P_HOME_RETURNING = 100.0
P_FOOD_RETURNING_TRAIL = 60.0
P_FOOD_AT_SOURCE = 500.0
P_FOOD_AT_NEST = 0.0
P_ALARM_FIGHT = 100.0
P_NEGATIVE_SEARCH = 10.0 # Reduced based on user finding
P_RECRUIT_FOOD = 400.0
P_RECRUIT_DAMAGE = 250.0
P_RECRUIT_DAMAGE_SOLDIER = 400.0

# Ant Parameters
INITIAL_ANTS = 10
QUEEN_HP = 600
WORKER_MAX_AGE_MEAN = 12000
WORKER_MAX_AGE_STDDEV = 2000
WORKER_PATH_HISTORY_LENGTH = 8 # Slightly shorter history
WORKER_STUCK_THRESHOLD = 60
WORKER_ESCAPE_DURATION = 30
WORKER_FOOD_CONSUMPTION_INTERVAL = 100
SOLDIER_PATROL_RADIUS_SQ = (NEST_RADIUS * 1)**2
SOLDIER_DEFEND_ALARM_THRESHOLD = 300.0

# Ant Caste Attributes
ANT_ATTRIBUTES = {
    AntCaste.WORKER: {
        "hp": 50, "attack": 3, "capacity": 1.5, "speed_delay": 0,
        "color": (0, 150, 255), "return_color": (0, 255, 100),
        "food_consumption_sugar": 0.02, "food_consumption_protein": 0.005,
        "description": "Worker", "size_factor": 2.5
    },
    AntCaste.SOLDIER: {
        "hp": 90, "attack": 10, "capacity": 0.2, "speed_delay": 1,
        "color": (0, 100, 255), "return_color": (255, 150, 50),
        "food_consumption_sugar": 0.025, "food_consumption_protein": 0.01,
        "description": "Soldier", "size_factor": 2.0
    }
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
INITIAL_ENEMIES = 3
ENEMY_HP = 50
ENEMY_ATTACK = 10
ENEMY_MOVE_DELAY = 4
ENEMY_SPAWN_RATE = 500
ENEMY_TO_FOOD_ON_DEATH_SUGAR = 10.0
ENEMY_TO_FOOD_ON_DEATH_PROTEIN = 50.0
ENEMY_NEST_ATTRACTION = 0.3

# --- Simulation Speed Control --- NEW Structure ---
BASE_FPS = 40  # Target FPS for 1.0x speed
# Define the available speed multipliers (0.0x represents Pause)
SPEED_MULTIPLIERS = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0]
# Calculate corresponding target FPS for each multiplier
# Use a minimum FPS (e.g., 10) when paused (0.0x) for UI responsiveness
TARGET_FPS_LIST = [10] + [max(1, int(m * BASE_FPS)) for m in SPEED_MULTIPLIERS[1:]]
DEFAULT_SPEED_INDEX = SPEED_MULTIPLIERS.index(1.0) # Default to 1.0x speed
# --- End Speed Control ---

# Colors
QUEEN_COLOR = (255, 0, 255)
WORKER_ESCAPE_COLOR = (255, 165, 0)
ENEMY_COLOR = (200, 0, 0)
FOOD_COLORS = {
    FoodType.SUGAR: (200, 200, 255),
    FoodType.PROTEIN: (255, 180, 180)
}
FOOD_COLOR_MIX = (230, 200, 230)
PHEROMONE_HOME_COLOR = (0, 0, 255, 150)
PHEROMONE_FOOD_COLOR = (0, 255, 0, 150)
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
# BUTTON_ACTIVE_COLOR = (120, 120, 220) # No longer needed for speed buttons
END_DIALOG_BG_COLOR = (0, 0, 0, 180) # Semi-transparent black for end dialog

# Define aliases for random functions for brevity (optional)
rnd = random.randint
rnd_gauss = random.gauss
rnd_uniform = random.uniform


# --- Helper Functions ---

def is_valid(pos):
    """Check if a position (x, y) is within the grid boundaries."""
    if not isinstance(pos, (tuple, list)) or len(pos) != 2:
        return False
    x, y = pos
    if not all(isinstance(coord, (int, float)) and math.isfinite(coord)
               for coord in [x, y]):
        return False
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def get_neighbors(pos, include_center=False):
    """Get valid neighbor coordinates for a given position."""
    if not is_valid(pos):
        return []
    x_int, y_int = int(pos[0]), int(pos[1])
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0 and not include_center:
                continue
            # Use integer coordinates for neighbors
            n_pos = (x_int + dx, y_int + dy)
            if is_valid(n_pos):
                neighbors.append(n_pos)
    return neighbors


def distance_sq(pos1, pos2):
    """Calculate the squared Euclidean distance between two points."""
    if not pos1 or not pos2 or not is_valid(pos1) or not is_valid(pos2):
        return float('inf')
    try:
        # Ensure integer positions for distance calculation if needed, or handle potential floats
        x1, y1 = int(pos1[0]), int(pos1[1])
        x2, y2 = int(pos2[0]), int(pos2[1])
        return (x1 - x2)**2 + (y1 - y2)**2
    except (TypeError, IndexError, ValueError):
        return float('inf')


def normalize(value, max_val):
    """Normalize a value to the range [0, 1], clamped."""
    if max_val <= 0:
        return 0.0
    norm_val = float(value) / float(max_val)
    return min(1.0, max(0.0, norm_val))


# --- Brood Class ---
class BroodItem:
    """Represents an item of brood (egg, larva, pupa) in the nest."""

    def __init__(self, stage: BroodStage, caste: AntCaste, position: tuple,
                 current_tick: int):
        self.stage = stage
        self.caste = caste
        # Store position as integer tuple for stable drawing
        self.pos = tuple(map(int, position))
        self.creation_tick = current_tick
        self.progress_timer = 0
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
        # --- Simulation Speed Influence on Brood Development ---
        # Get the current speed multiplier (avoiding 0x which means pause)
        current_multiplier = SPEED_MULTIPLIERS[simulation.simulation_speed_index]
        update_factor = current_multiplier if current_multiplier > 0 else 0

        # Apply speed factor to progress timer increment
        self.progress_timer += update_factor

        if self.stage == BroodStage.LARVA:
            # Check feed interval based on ticks, not affected by speed multiplier directly
            # The *rate* of checks increases with speed, but the interval is fixed in ticks
            if current_tick - self.last_feed_check >= LARVA_FEED_INTERVAL:
                self.last_feed_check = current_tick
                # Consumption amount could potentially scale with speed, but let's keep it fixed for now
                needed_p = LARVA_FOOD_CONSUMPTION_PROTEIN
                needed_s = LARVA_FOOD_CONSUMPTION_SUGAR
                has_p = simulation.colony_food_storage_protein >= needed_p
                has_s = simulation.colony_food_storage_sugar >= needed_s

                if has_p and has_s:
                    simulation.colony_food_storage_protein -= needed_p
                    simulation.colony_food_storage_sugar -= needed_s
                else:
                    # If starving, pause progress by *not* adding the update_factor again
                    # effectively negating the increment from the start of the function
                     self.progress_timer -= update_factor

        # Check against duration (duration is fixed in "standard ticks")
        if self.progress_timer >= self.duration:
            if self.stage == BroodStage.EGG:
                self.stage = BroodStage.LARVA
                self.progress_timer = 0 # Reset progress
                self.duration = LARVA_DURATION
                self.color = LARVA_COLOR
                self.radius = CELL_SIZE // 4
                return None
            elif self.stage == BroodStage.LARVA:
                self.stage = BroodStage.PUPA
                self.progress_timer = 0 # Reset progress
                self.duration = PUPA_DURATION
                self.color = PUPA_COLOR
                self.radius = int(CELL_SIZE / 3.5)
                return None
            elif self.stage == BroodStage.PUPA:
                return self # Signal hatching
        return None

    def draw(self, surface):
        """Draw the brood item statically centered in its cell."""
        if not is_valid(self.pos) or self.radius <= 0:
            return

        # Calculate the center of the cell
        center_x = int(self.pos[0]) * CELL_SIZE + CELL_SIZE // 2
        center_y = int(self.pos[1]) * CELL_SIZE + CELL_SIZE // 2
        draw_pos = (center_x, center_y)

        # Draw the main circle
        pygame.draw.circle(surface, self.color, draw_pos, self.radius)

        # Draw outline for pupa
        if self.stage == BroodStage.PUPA:
            o_col = ((50, 50, 50) if self.caste == AntCaste.WORKER
                     else (100, 0, 0))
            pygame.draw.circle(surface, o_col, draw_pos, self.radius, 1)


# --- Grid Class ---
class WorldGrid:
    """Manages the simulation grid (food, pheromones, obstacles)."""

    def __init__(self):
        self.food = np.zeros((GRID_WIDTH, GRID_HEIGHT, NUM_FOOD_TYPES),
                             dtype=float)
        self.pheromones_home = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=float)
        self.pheromones_food = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=float)
        self.pheromones_alarm = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=float)
        self.pheromones_negative = np.zeros((GRID_WIDTH, GRID_HEIGHT),
                                            dtype=float)
        self.pheromones_recruitment = np.zeros((GRID_WIDTH, GRID_HEIGHT),
                                               dtype=float)
        self.obstacles = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=bool)

    def reset(self):
        """Resets the grid state for a new simulation."""
        self.food.fill(0)
        self.pheromones_home.fill(0)
        self.pheromones_food.fill(0)
        self.pheromones_alarm.fill(0)
        self.pheromones_negative.fill(0)
        self.pheromones_recruitment.fill(0)
        self.obstacles.fill(0)
        # Repopulate obstacles and food
        self.place_obstacles()
        self.place_food_clusters()


    def place_food_clusters(self):
        """Place initial food clusters of alternating types."""
        for i in range(INITIAL_FOOD_CLUSTERS):
            food_type_index = i % NUM_FOOD_TYPES
            food_type = FoodType(food_type_index)
            attempts = 0; cx = cy = 0; found_spot = False
            while attempts < 100 and not found_spot:
                cx = rnd(0, GRID_WIDTH - 1); cy = rnd(0, GRID_HEIGHT - 1)
                dist_ok = distance_sq((cx, cy), NEST_POS) > MIN_FOOD_DIST_FROM_NEST**2
                if dist_ok and not self.obstacles[cx, cy]: found_spot = True
                attempts += 1
            if not found_spot:
                while attempts < 200:
                    cx = rnd(0, GRID_WIDTH - 1); cy = rnd(0, GRID_HEIGHT - 1)
                    if not self.obstacles[cx, cy]:
                        found_spot = True; break
                    attempts += 1
                if not found_spot:
                    # Fallback: place anywhere not obstacle if far spot failed
                    cx_f, cy_f = -1, -1
                    for _ in range(50): # Try 50 times for any non-obstacle
                        cx_try = rnd(0, GRID_WIDTH - 1)
                        cy_try = rnd(0, GRID_HEIGHT - 1)
                        if not self.obstacles[cx_try, cy_try]:
                             cx_f, cy_f = cx_try, cy_try
                             break
                    if cx_f != -1:
                        cx, cy = cx_f, cy_f
                    else: # Absolute fallback: place at random even if obstacle (should be rare)
                         cx = rnd(0, GRID_WIDTH - 1); cy = rnd(0, GRID_HEIGHT - 1)


            added = 0.0
            for _ in range(int(FOOD_PER_CLUSTER * 1.5)):
                if added >= FOOD_PER_CLUSTER: break
                fx = cx + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                fy = cy + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                if is_valid((fx, fy)) and not self.obstacles[fx, fy]:
                    amount = rnd_uniform(0.5, 1.0) * (MAX_FOOD_PER_CELL / 10)
                    fx_int, fy_int = int(fx), int(fy)
                    curr = self.food[fx_int, fy_int, food_type_index]
                    self.food[fx_int, fy_int, food_type_index] = min(MAX_FOOD_PER_CELL, curr + amount)
                    added += amount

    def place_obstacles(self):
        nest_area=set(); r=NEST_RADIUS+3
        nest_center_int = tuple(map(int, NEST_POS))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                p=(nest_center_int[0] + dx, nest_center_int[1] + dy)
                if is_valid(p): nest_area.add(p) # Add integer positions

        placed_count = 0
        for _ in range(NUM_OBSTACLES * 3): # Increase attempts to ensure obstacles are placed
             if placed_count >= NUM_OBSTACLES: break
             attempts=0; placed=False
             while attempts<20 and not placed: # Fewer attempts per obstacle, more overall tries
                w=rnd(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                h=rnd(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                x=rnd(0, GRID_WIDTH - w - 1); y=rnd(0, GRID_HEIGHT - h - 1)
                overlaps=False
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        # Use integer coordinates for check
                        if (int(i), int(j)) in nest_area: overlaps=True; break
                    if overlaps: break
                if not overlaps:
                     # Ensure placement doesn't overwrite edge cases
                     if x+w < GRID_WIDTH and y+h < GRID_HEIGHT:
                         self.obstacles[x : x + w, y : y + h] = True
                         placed = True
                         placed_count += 1
                attempts += 1
        # print(f"Placed {placed_count}/{NUM_OBSTACLES} obstacles.") # Optional debug

    def is_obstacle(self, pos):
        if not is_valid(pos): return True
        try:
            x = int(pos[0]); y = int(pos[1])
            # Ensure indices are within bounds after int conversion
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                return self.obstacles[x, y]
            else: return True # Outside bounds
        except (IndexError, TypeError, ValueError): return True

    def get_pheromone(self, pos, ph_type='home'):
        if not is_valid(pos): return 0.0
        try:
            x = int(pos[0]); y = int(pos[1])
            # Ensure indices are within bounds after int conversion
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                return 0.0
        except (ValueError, TypeError): return 0.0
        try:
            if ph_type == 'home': return self.pheromones_home[x, y]
            elif ph_type == 'food': return self.pheromones_food[x, y]
            elif ph_type == 'alarm': return self.pheromones_alarm[x, y]
            elif ph_type == 'negative': return self.pheromones_negative[x, y]
            elif ph_type == 'recruitment': return self.pheromones_recruitment[x, y]
            else: return 0.0
        except IndexError: return 0.0 # Should not happen with bounds check

    def add_pheromone(self, pos, amount, ph_type='home'):
        if not is_valid(pos) or amount <= 0 or self.is_obstacle(pos): return
        try:
            x = int(pos[0]); y = int(pos[1])
            # Ensure indices are within bounds after int conversion
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                return
        except (ValueError, TypeError): return
        try:
            target = None; max_val = PHEROMONE_MAX
            if ph_type == 'home': target = self.pheromones_home
            elif ph_type == 'food': target = self.pheromones_food
            elif ph_type == 'alarm': target = self.pheromones_alarm
            elif ph_type == 'negative': target = self.pheromones_negative
            elif ph_type == 'recruitment':
                 target = self.pheromones_recruitment; max_val = RECRUITMENT_PHEROMONE_MAX
            if target is not None:
                 current_val = target[x, y]
                 # Consider scaling deposit amount by simulation speed? Maybe not intuitive.
                 target[x, y] = min(current_val + amount, max_val)
        except IndexError: pass # Should not happen with bounds check

    def update_pheromones(self, speed_multiplier):
        """Update pheromones, considering simulation speed for decay/diffusion."""
        # Adjust decay and diffusion rates based on speed multiplier
        # If multiplier is 0, no update should happen (handled in main update loop)
        effective_multiplier = max(0.0, speed_multiplier) # Ensure non-negative

        # Decay: Apply decay more strongly if speed is higher (more time passes effectively)
        # decay_factor = PHEROMONE_DECAY ** effective_multiplier # Exponential decay scaling
        # Linear scaling might be more stable/predictable:
        decay_factor_home_food_alarm = 1.0 - (1.0 - PHEROMONE_DECAY) * effective_multiplier
        decay_factor_neg = 1.0 - (1.0 - NEGATIVE_PHEROMONE_DECAY) * effective_multiplier
        decay_factor_recruit = 1.0 - (1.0 - RECRUITMENT_PHEROMONE_DECAY) * effective_multiplier

        # Clamp decay factors to avoid negative values or > 1
        decay_factor_home_food_alarm = min(1.0, max(0.0, decay_factor_home_food_alarm))
        decay_factor_neg = min(1.0, max(0.0, decay_factor_neg))
        decay_factor_recruit = min(1.0, max(0.0, decay_factor_recruit))

        self.pheromones_home *= decay_factor_home_food_alarm
        self.pheromones_food *= decay_factor_home_food_alarm
        self.pheromones_alarm *= decay_factor_home_food_alarm
        self.pheromones_negative *= decay_factor_neg
        self.pheromones_recruitment *= decay_factor_recruit

        # Diffusion: Apply diffusion more strongly if speed is higher
        diffusion_rate_home_food_alarm = PHEROMONE_DIFFUSION_RATE * effective_multiplier
        diffusion_rate_neg = NEGATIVE_PHEROMONE_DIFFUSION_RATE * effective_multiplier
        diffusion_rate_recruit = RECRUITMENT_PHEROMONE_DIFFUSION_RATE * effective_multiplier

        # Clamp diffusion rates (e.g., max 0.5 or 1.0 / 8.0 = 0.125 per step?)
        max_diffusion = 0.12 # Avoid instability
        diffusion_rate_home_food_alarm = min(max_diffusion, max(0.0, diffusion_rate_home_food_alarm))
        diffusion_rate_neg = min(max_diffusion, max(0.0, diffusion_rate_neg))
        diffusion_rate_recruit = min(max_diffusion, max(0.0, diffusion_rate_recruit))

        mask = ~self.obstacles
        arrays_rates = [
            (self.pheromones_home, diffusion_rate_home_food_alarm),
            (self.pheromones_food, diffusion_rate_home_food_alarm),
            (self.pheromones_alarm, diffusion_rate_home_food_alarm),
            (self.pheromones_negative, diffusion_rate_neg),
            (self.pheromones_recruitment, diffusion_rate_recruit)
        ]
        for arr, rate in arrays_rates:
            if rate > 0:
                # Optimized diffusion using numpy operations
                masked = arr * mask # Apply obstacle mask
                pad = np.pad(masked, 1, mode='constant') # Pad with zeros for neighbor calculation

                # Calculate sum of 8 neighbors efficiently
                neighbors_sum = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                                 pad[1:-1, :-2]                + pad[1:-1, 2:] +
                                 pad[2:, :-2]   + pad[2:, 1:-1]   + pad[2:, 2:])

                # Update step: current value * (1-rate) + avg_neighbors * rate
                # where avg_neighbors = neighbors_sum / 8.0
                diffused = masked * (1.0 - rate) + (neighbors_sum / 8.0) * rate

                # Apply result back, ensuring obstacles remain 0
                arr[:] = np.where(mask, diffused, 0)

        # Clipping & Zeroing
        min_ph = 0.01 # Threshold below which pheromones are set to 0
        all_arrays = [
            self.pheromones_home, self.pheromones_food, self.pheromones_alarm,
            self.pheromones_negative, self.pheromones_recruitment
        ]
        for arr in all_arrays:
            max_val = RECRUITMENT_PHEROMONE_MAX if arr is self.pheromones_recruitment else PHEROMONE_MAX
            np.clip(arr, 0, max_val, out=arr) # Ensure values stay within [0, max_val]
            arr[arr < min_ph] = 0 # Remove negligible amounts
            # arr[self.obstacles] = 0 # Ensure obstacles are zero (redundant with mask logic but safe)


# --- Entity Classes ---
class Ant:
    """Represents a worker or soldier ant."""

    def __init__(self, pos, simulation, caste: AntCaste):
        # Ensure position is integer tuple
        self.pos = tuple(map(int, pos)); self.simulation = simulation; self.caste = caste
        attrs = ANT_ATTRIBUTES[caste]
        self.hp = attrs["hp"]; self.max_hp = attrs["hp"]
        self.attack_power = attrs["attack"]; self.max_capacity = attrs["capacity"]
        self.move_delay_base = attrs["speed_delay"] # Base delay
        self.search_color = attrs["color"]
        self.return_color = attrs["return_color"]
        self.food_consumption_sugar = attrs["food_consumption_sugar"]
        self.food_consumption_protein = attrs["food_consumption_protein"]
        self.size_factor = attrs["size_factor"]

        self.state = AntState.SEARCHING; self.carry_amount = 0.0; self.carry_type = None
        self.age = 0; self.max_age_ticks = int(rnd_gauss(WORKER_MAX_AGE_MEAN, WORKER_MAX_AGE_STDDEV)) # Max age in standard ticks
        # Use integer tuples for path history
        self.path_history = []; self.history_timestamps = [];
        self.move_delay_timer = 0 # Counts down frames/updates
        self.last_move_direction = (0, 0); self.stuck_timer = 0; self.escape_timer = 0
        self.last_move_info = "Born"; self.just_picked_food = False
        self.food_consumption_timer = rnd(0, WORKER_FOOD_CONSUMPTION_INTERVAL) # Counts ticks
        self.last_known_alarm_pos = None
        # Internal timer scaled by speed
        self.age_progress_timer = 0.0


    def _update_path_history(self, new_pos):
        t = self.simulation.ticks # Use simulation ticks for timestamp
        # Ensure position is integer tuple
        int_pos = tuple(map(int, new_pos))
        # Avoid adding the same position multiple times consecutively
        if not self.path_history or self.path_history[-1] != int_pos:
            self.path_history.append(int_pos)
            self.history_timestamps.append(t)

        # Limit history length by time (WORKER_PATH_HISTORY_LENGTH represents ticks)
        cutoff = t - WORKER_PATH_HISTORY_LENGTH
        idx = 0
        while (idx < len(self.history_timestamps) and
               self.history_timestamps[idx] < cutoff):
            idx += 1
        self.path_history = self.path_history[idx:]
        self.history_timestamps = self.history_timestamps[idx:]


    def _is_in_history(self, pos):
        # Ensure comparison with integer tuple
        return tuple(map(int, pos)) in self.path_history

    def _clear_path_history(self):
        self.path_history = []
        self.history_timestamps = []

    def _filter_valid_moves(self, potential_neighbors, ignore_history_near_nest=False):
        """Filter potential moves for obstacles, history, queen, and other ants."""
        valid = []
        q_pos_int = tuple(map(int, self.simulation.queen.pos)) if self.simulation.queen else None
        # Use integer position for calculations
        pos_int = tuple(map(int, self.pos))
        nest_pos_int = tuple(map(int, NEST_POS))
        is_near_nest_now = distance_sq(pos_int, nest_pos_int) <= (NEST_RADIUS + 1)**2

        for n_pos_float in potential_neighbors:
            # Work with integer coordinates for checks
            n_pos = tuple(map(int, n_pos_float))

            history_block = False
            # Check history only if not ignoring it or if far from nest when ignoring near nest
            check_hist = not ignore_history_near_nest or (ignore_history_near_nest and not is_near_nest_now)
            if check_hist and self._is_in_history(n_pos):
                 history_block = True

            is_queen = (n_pos == q_pos_int)
            is_obs = self.simulation.grid.is_obstacle(n_pos)
            # Check against other ants' integer positions
            is_blocked_ant = self.simulation.is_ant_at(n_pos, exclude_self=self)

            if not history_block and not is_queen and not is_obs and not is_blocked_ant:
                # Return the original float/tuple position if valid
                valid.append(n_pos_float)
        return valid

    def _choose_move(self):
        """Determine the next move based on state, goals, and environment."""
        potential_neighbors = get_neighbors(self.pos) # Gets integer neighbors
        if not potential_neighbors:
            self.last_move_info = "No neighbors"
            return None

        ignore_hist = (self.state == AntState.RETURNING_TO_NEST)
        valid_neighbors = self._filter_valid_moves(potential_neighbors,
                                                   ignore_history_near_nest=ignore_hist)

        if not valid_neighbors:
            self.last_move_info = "Blocked"
            # Improved fallback logic: check neighbors again, ignoring only obstacles/queen/self
            fallback_neighbors = []
            q_pos_int = tuple(map(int, self.simulation.queen.pos)) if self.simulation.queen else None
            for n_pos_int in potential_neighbors: # Iterate through original integer neighbors
                 if (n_pos_int != q_pos_int and
                     not self.simulation.grid.is_obstacle(n_pos_int) and
                     not self.simulation.is_ant_at(n_pos_int, exclude_self=self)):
                     # Check if this int pos is NOT in history
                     if not self._is_in_history(n_pos_int):
                          fallback_neighbors.append(n_pos_int)

            if fallback_neighbors:
                # Prefer non-history fallback moves
                return random.choice(fallback_neighbors)
            else:
                 # If all valid fallbacks are in history, allow moving back (less strict fallback)
                 fallback_neighbors_incl_history = []
                 for n_pos_int in potential_neighbors:
                     if (n_pos_int != q_pos_int and
                         not self.simulation.grid.is_obstacle(n_pos_int) and
                         not self.simulation.is_ant_at(n_pos_int, exclude_self=self)):
                         fallback_neighbors_incl_history.append(n_pos_int)
                 if fallback_neighbors_incl_history:
                      # Try moving to the least recently visited position
                      fallback_neighbors_incl_history.sort(key=lambda p: self.path_history.index(p) if p in self.path_history else -1)
                      return fallback_neighbors_incl_history[0]

            return None # Truly stuck if no non-obstacle/non-queen/non-ant neighbor exists

        if self.state == AntState.ESCAPING:
            # Prioritize moving away from the current position, ignore history strongly
            escape_moves = []
            pos_int = tuple(map(int, self.pos))
            for n_pos in valid_neighbors: # valid_neighbors contains original tuples
                n_pos_int = tuple(map(int, n_pos))
                if n_pos_int != pos_int and not self._is_in_history(n_pos_int):
                    escape_moves.append(n_pos)
            if escape_moves:
                 return random.choice(escape_moves)
            else: # Fallback: any valid move if no non-history move away exists
                 return random.choice(valid_neighbors)


        # Score moves based on state
        move_scores = {}
        if self.state == AntState.RETURNING_TO_NEST:
            move_scores = self._score_moves_returning(valid_neighbors, self.just_picked_food)
        elif self.state == AntState.SEARCHING:
            move_scores = self._score_moves_searching(valid_neighbors)
        elif self.state == AntState.PATROLLING:
            move_scores = self._score_moves_patrolling(valid_neighbors)
        elif self.state == AntState.DEFENDING:
            move_scores = self._score_moves_defending(valid_neighbors)
        else: # Fallback
            move_scores = self._score_moves_searching(valid_neighbors)

        if not move_scores:
            self.last_move_info = f"No scores({self.state})"
            # Fallback: random valid move if scoring fails
            return random.choice(valid_neighbors) if valid_neighbors else None


        # Select move based on strategy
        if self.state == AntState.RETURNING_TO_NEST:
            return self._select_best_move_returning(move_scores, valid_neighbors, self.just_picked_food)
        elif self.state == AntState.DEFENDING:
            return self._select_best_move(move_scores, valid_neighbors)
        else: # SEARCHING, PATROLLING
            return self._select_probabilistic_move(move_scores, valid_neighbors)

    def _score_moves_base(self, neighbor_pos):
        score = 0.0
        # Use integer positions for direction calculation
        pos_int = tuple(map(int, self.pos))
        n_pos_int = tuple(map(int, neighbor_pos))
        move_dir = (n_pos_int[0] - pos_int[0], n_pos_int[1] - pos_int[1])
        # Encourage moving in the same direction slightly
        if move_dir == self.last_move_direction and move_dir != (0,0):
            score += W_PERSISTENCE
        # Add random noise for exploration
        score += rnd_uniform(-W_RANDOM_NOISE, W_RANDOM_NOISE)
        return score

    def _score_moves_returning(self, valid_neighbors, just_picked):
        scores = {}
        # Use integer position for distance calculation
        pos_int = tuple(map(int, self.pos))
        nest_pos_int = tuple(map(int, NEST_POS))
        dist_sq_now = distance_sq(pos_int, nest_pos_int)
        grid = self.simulation.grid
        for n_pos in valid_neighbors: # n_pos is original tuple
            score = 0.0
            n_pos_int = tuple(map(int, n_pos)) # Use int for grid access/distance
            home_ph = grid.get_pheromone(n_pos_int, 'home')
            food_ph = grid.get_pheromone(n_pos_int, 'food')
            alarm_ph = grid.get_pheromone(n_pos_int, 'alarm')
            neg_ph = grid.get_pheromone(n_pos_int, 'negative') # Consider negative trails

            # Strong pull towards home pheromone and nest direction
            score += home_ph * W_HOME_PHEROMONE_RETURN
            if distance_sq(n_pos_int, nest_pos_int) < dist_sq_now:
                score += W_NEST_DIRECTION_RETURN

            # Avoid alarm/negative pheromones (less penalty than searching)
            score += alarm_ph * W_ALARM_PHEROMONE * 0.2
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.3

            # If just picked food, slightly avoid the food source pheromone trail immediately
            if just_picked:
                score -= food_ph * W_FOOD_PHEROMONE_SEARCH * 0.3

            score += self._score_moves_base(n_pos) # Add base score (persistence, noise)
            scores[n_pos] = score
        return scores

    def _score_moves_searching(self, valid_neighbors):
        scores = {}
        grid = self.simulation.grid
        pos_int = tuple(map(int, self.pos)) # Current int position
        nest_pos_int = tuple(map(int, NEST_POS))

        for n_pos in valid_neighbors:
            score = 0.0
            n_pos_int = tuple(map(int, n_pos)) # Use int for grid access/distance
            home_ph = grid.get_pheromone(n_pos_int, 'home') # Read home pheromone
            food_ph = grid.get_pheromone(n_pos_int, 'food')
            neg_ph = grid.get_pheromone(n_pos_int, 'negative')
            alarm_ph = grid.get_pheromone(n_pos_int, 'alarm')
            recr_ph = grid.get_pheromone(n_pos_int, 'recruitment')

            # Worker ants strongly follow food/recruitment pheromones
            food_weight = W_FOOD_PHEROMONE_SEARCH
            recruit_weight = W_RECRUITMENT_PHEROMONE
            # Soldiers are less interested in food, more in recruitment/alarm
            if self.caste == AntCaste.SOLDIER:
                 food_weight *= 0.1
                 recruit_weight *= 1.2 # Slightly more responsive to recruitment

            score += food_ph * food_weight
            score += recr_ph * recruit_weight

            # Avoid negative, alarm, and home pheromones when searching
            score += neg_ph * W_NEGATIVE_PHEROMONE
            score += alarm_ph * W_ALARM_PHEROMONE
            score += home_ph * W_HOME_PHEROMONE_SEARCH # Currently 0, but could be negative

            # Penalty for being too close to the nest center while searching
            if distance_sq(n_pos_int, nest_pos_int) <= (NEST_RADIUS * 1.5)**2:
                 score += W_AVOID_NEST_SEARCHING # Negative weight

            score += self._score_moves_base(n_pos) # Add base score (persistence, noise)
            scores[n_pos] = score
        return scores

    def _score_moves_patrolling(self, valid_neighbors):
        scores = {}
        grid = self.simulation.grid
        pos_int = tuple(map(int, self.pos)) # Use int for distance/grid
        nest_pos_int = tuple(map(int, NEST_POS))
        dist_sq_current = distance_sq(pos_int, nest_pos_int)

        for n_pos in valid_neighbors:
            score = 0.0
            n_pos_int = tuple(map(int, n_pos)) # Use int for grid/distance
            neg_ph = grid.get_pheromone(n_pos_int, 'negative')
            alarm_ph = grid.get_pheromone(n_pos_int, 'alarm')
            recr_ph = grid.get_pheromone(n_pos_int, 'recruitment') # Follow recruitment if nearby

            # Follow recruitment, avoid negative/alarm slightly less than searching
            score += recr_ph * W_RECRUITMENT_PHEROMONE * 0.8
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.6
            score += alarm_ph * W_ALARM_PHEROMONE * 0.4

            dist_sq_next = distance_sq(n_pos_int, nest_pos_int)

            # Encourage moving away from nest center if inside patrol radius
            if dist_sq_current <= SOLDIER_PATROL_RADIUS_SQ:
                 if dist_sq_next > dist_sq_current:
                     score -= W_NEST_DIRECTION_PATROL # W is negative -> push away

            # Heavy penalty for moving outside the maximum patrol radius
            if dist_sq_next > (SOLDIER_PATROL_RADIUS_SQ * 1.5): # Allow slight overshoot
                 score -= 5000 # Strong penalty

            score += self._score_moves_base(n_pos) # Add base score (persistence, noise)
            scores[n_pos] = score
        return scores

    def _score_moves_defending(self, valid_neighbors):
        scores = {}
        grid = self.simulation.grid
        pos_int = tuple(map(int, self.pos)) # Use int for calcs

        # --- Target Acquisition / Update ---
        # If no target or randomly (to re-evaluate), find strongest alarm/recruit nearby
        if self.last_known_alarm_pos is None or random.random() < 0.15:
            best_pos = None
            max_signal = -1
            radius_sq = 5*5 # Search radius squared
            x0, y0 = pos_int # Search around current int pos

            potential_targets = []
            for i in range(x0 - 2, x0 + 3):
                 for j in range(y0 - 2, y0 + 3):
                      p = (i, j)
                      # Check distance and validity
                      if distance_sq(pos_int, p) <= radius_sq and is_valid(p):
                           # Combine alarm and recruitment signals for target score
                           signal = (grid.get_pheromone(p, 'alarm') +
                                     grid.get_pheromone(p, 'recruitment') * 0.5) # Prioritize alarm
                           # Check if an enemy is also at this location
                           enemy_here = self.simulation.get_enemy_at(p)
                           if enemy_here:
                                signal += 500 # Heavily prioritize moving towards enemies

                           if signal > max_signal:
                               max_signal = signal
                               best_pos = p # Store int pos p

            # Update target only if a significant signal was found
            if max_signal > 50.0: # Threshold to avoid chasing noise
                 self.last_known_alarm_pos = best_pos # Store int pos
            else:
                 self.last_known_alarm_pos = None # Lose target if signal weak

        # --- Score Moves based on Target ---
        for n_pos in valid_neighbors:
            score = 0.0
            n_pos_int = tuple(map(int, n_pos)) # Use int pos
            alarm_ph = grid.get_pheromone(n_pos_int, 'alarm')
            recr_ph = grid.get_pheromone(n_pos_int, 'recruitment')
            enemy_at_n_pos = self.simulation.get_enemy_at(n_pos_int)

            # Very high score for moving onto a cell with an enemy
            if enemy_at_n_pos:
                 score += 10000

            # If has a target position, score moving closer higher
            if self.last_known_alarm_pos: # This is an int tuple
                 dist_now_sq = distance_sq(pos_int, self.last_known_alarm_pos)
                 dist_next_sq = distance_sq(n_pos_int, self.last_known_alarm_pos)
                 if dist_next_sq < dist_now_sq:
                     score += W_ALARM_SOURCE_DEFEND # Strong pull towards target

            # General attraction to alarm and recruitment pheromones
            score += alarm_ph * W_ALARM_PHEROMONE * -0.8 # Follow alarm strongly (W_ALARM is negative)
            score += recr_ph * W_RECRUITMENT_PHEROMONE * 1.2 # Follow recruitment strongly

            score += self._score_moves_base(n_pos) # Add base score (persistence, noise)
            scores[n_pos] = score
        return scores


    def _select_best_move(self, move_scores, valid_neighbors):
        """Selects the move with the highest score (for DEFEND)."""
        best_score = -float('inf'); best_moves = []
        for pos, score in move_scores.items(): # pos is original tuple
            if score > best_score: best_score = score; best_moves = [pos]
            elif score == best_score: best_moves.append(pos)

        if not best_moves:
            self.last_move_info += "(Def:No best?)"
            return random.choice(valid_neighbors) if valid_neighbors else None

        chosen = random.choice(best_moves) # Choose randomly among the best
        score = move_scores.get(chosen, -999)
        chosen_int = tuple(map(int, chosen))
        self.last_move_info = f"Def Best->{chosen_int} (S:{score:.1f})"
        return chosen # Return original tuple

    def _select_best_move_returning(self, move_scores, valid_neighbors, just_picked):
        """Selects the best move for returning, prioritizing nest direction."""
        best_score = -float('inf'); best_moves = []
        pos_int = tuple(map(int, self.pos)) # Use int for distance
        nest_pos_int = tuple(map(int, NEST_POS))
        dist_sq_now = distance_sq(pos_int, nest_pos_int)
        closer_moves = {}; other_moves = {}

        # Separate moves into those getting closer to the nest and others
        for pos, score in move_scores.items(): # pos is original tuple
            # Use int for distance check
            if distance_sq(tuple(map(int, pos)), nest_pos_int) < dist_sq_now:
                closer_moves[pos] = score
            else:
                other_moves[pos] = score

        target_pool = {}; selection_type = ""
        # Prioritize moves that get closer
        if closer_moves: target_pool = closer_moves; selection_type = "Closer"
        elif other_moves: target_pool = other_moves; selection_type = "Other"
        else:
            # If no moves possible (shouldn't happen if valid_neighbors exists), fallback
            self.last_move_info += "(R: No moves?)"
            return random.choice(valid_neighbors) if valid_neighbors else None

        # Find the best score within the prioritized pool
        for pos, score in target_pool.items():
            if score > best_score: best_score = score; best_moves = [pos]
            elif score == best_score: best_moves.append(pos)

        # Handle cases where the prioritized pool yields no best move (unlikely)
        if not best_moves:
            self.last_move_info += f"(R: No best in {selection_type})"
            # Fallback to considering all original valid moves if priority pool failed
            target_pool = move_scores
            best_score = -float('inf'); best_moves = []
            for pos, score in target_pool.items():
                 if score > best_score: best_score = score; best_moves = [pos]
                 elif score == best_score: best_moves.append(pos)
            # If still no best move, choose randomly from valid neighbors
            if not best_moves: return random.choice(valid_neighbors) if valid_neighbors else None

        # Select the final move
        if len(best_moves) == 1:
            chosen = best_moves[0]
            chosen_int = tuple(map(int, chosen))
            self.last_move_info = f"R({selection_type})Best->{chosen_int} (S:{best_score:.1f})"
        else:
            # Tie-break: Use home pheromone concentration
            best_moves.sort(key=lambda p: self.simulation.grid.get_pheromone(tuple(map(int,p)),'home'), reverse=True)
            # Get max pheromone value among tied moves
            max_ph = self.simulation.grid.get_pheromone(tuple(map(int,best_moves[0])),'home')
            # Select randomly from those with the max pheromone value
            top_ph_moves = [p for p in best_moves if self.simulation.grid.get_pheromone(tuple(map(int,p)),'home') == max_ph]
            chosen = random.choice(top_ph_moves)
            chosen_int = tuple(map(int, chosen))
            self.last_move_info = f"R({selection_type})TieBrk->{chosen_int} (S:{best_score:.1f})"

        return chosen # Return original tuple


    def _select_probabilistic_move(self, move_scores, valid_neighbors):
        """Selects a move probabilistically based on scores."""
        if not move_scores: # Handle empty scores dictionary
             return random.choice(valid_neighbors) if valid_neighbors else None

        pop = list(move_scores.keys()) # Original tuples
        scores = np.array(list(move_scores.values()))

        if len(pop) == 0: return None

        min_s = np.min(scores) if scores.size > 0 else 0
        # Shift scores to be non-negative, add epsilon for stability
        shifted_scores = scores - min_s + 0.01

        # Apply temperature scaling (clamped for stability)
        temp = min(max(PROBABILISTIC_CHOICE_TEMP, 0.1), 5.0)
        weights = np.power(shifted_scores, temp)

        # Ensure weights are not too small (helps avoid division by zero)
        weights = np.maximum(MIN_SCORE_FOR_PROB_CHOICE, weights)

        total_weight = np.sum(weights)

        # Handle invalid total weight (zero, NaN, Inf)
        if total_weight <= 1e-9 or not np.isfinite(total_weight):
            self.last_move_info += f"({self.state.name[:3]}:InvW)"
            # Fallback: Choose best score deterministically if weights failed
            best_s = -float('inf'); best_p = None
            for p, s in move_scores.items():
                if s > best_s: best_s = s; best_p = p
            if best_p: return best_p
            # If even that fails, random choice
            return random.choice(valid_neighbors) if valid_neighbors else None

        # Calculate probabilities
        probabilities = weights / total_weight

        # Renormalize if sum is not close to 1.0 (due to float precision)
        sum_probs = np.sum(probabilities)
        if not np.isclose(sum_probs, 1.0):
            if sum_probs > 1e-9:
                probabilities /= sum_probs
            else: # If sum is still ~zero, probabilities are ill-defined
                self.last_move_info += "(ProbSumLow)"
                best_s = -float('inf'); best_p = None
                for p, s in move_scores.items():
                    if s > best_s: best_s = s; best_p = p
                if best_p: return best_p
                return random.choice(valid_neighbors) if valid_neighbors else None

        # Final check for valid probabilities (sum ~ 1.0, no NaN/Inf)
        if not np.all(np.isfinite(probabilities)) or not np.isclose(np.sum(probabilities), 1.0):
             self.last_move_info += "(ProbErrFinal)"
             best_s = -float('inf'); best_p = None
             for p, s in move_scores.items():
                 if s > best_s: best_s = s; best_p = p
             if best_p: return best_p
             return random.choice(valid_neighbors) if valid_neighbors else None


        try:
            # Choose index based on calculated probabilities
            chosen_index = np.random.choice(len(pop), p=probabilities)
            chosen = pop[chosen_index] # Get the chosen original tuple
            score = move_scores.get(chosen, -999)
            chosen_int = tuple(map(int, chosen))
            self.last_move_info = f"{self.state.name[:3]} Prob->{chosen_int} (S:{score:.1f})"
            return chosen # Return original tuple
        except ValueError as e:
            # This error (often "probabilities do not sum to 1") can still occur with tiny float differences
            print(f"WARN choices ({self.state}): {e}. Sum={np.sum(probabilities)}")
            self.last_move_info += "(ProbValErr)"
            # Fallback: Choose the move with the highest score deterministically
            best_s = -float('inf'); best_p = None
            for p, s in move_scores.items():
                if s > best_s: best_s = s; best_p = p
            if best_p: return best_p
            # If that fails, choose randomly
            return random.choice(valid_neighbors) if valid_neighbors else None


    def update(self, speed_multiplier):
        """Update ant's state, position, age, food, and interactions."""

        # --- Aging ---
        # Increment internal age timer by the speed multiplier
        self.age_progress_timer += speed_multiplier
        # Increment actual age only when the progress timer crosses an integer threshold
        if self.age_progress_timer >= 1.0:
            self.age += int(self.age_progress_timer) # Add whole ticks passed
            self.age_progress_timer %= 1.0 # Keep the fractional part

        # Death from old age check (based on standard ticks)
        if self.age >= self.max_age_ticks:
            self.hp = 0
            self.last_move_info = "Died of old age"
            return # Dies in pre-update check next cycle


        # --- Food Consumption ---
        # Timer counts standard ticks
        self.food_consumption_timer += speed_multiplier
        if self.food_consumption_timer >= WORKER_FOOD_CONSUMPTION_INTERVAL:
            self.food_consumption_timer %= WORKER_FOOD_CONSUMPTION_INTERVAL # Reset timer keeping remainder
            needed_s = self.food_consumption_sugar
            needed_p = self.food_consumption_protein
            sim = self.simulation
            # Check if enough food is available in colony storage
            has_s = sim.colony_food_storage_sugar >= needed_s
            has_p = sim.colony_food_storage_protein >= needed_p

            if has_s and has_p:
                # Consume food from colony storage
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p
            else:
                # Starvation: Set HP to 0, will be removed in next pre-update check
                self.hp = 0
                self.last_move_info = "Starved"
                return # Skip rest of update for this tick


        # --- State Updates (Escape Timer) ---
        if self.state == AntState.ESCAPING:
            # Escape timer decreases based on speed multiplier
            self.escape_timer -= speed_multiplier
            if self.escape_timer <= 0:
                # Transition back to normal state (Patrolling for Soldier, Searching for Worker)
                next_state = (AntState.PATROLLING if self.caste == AntCaste.SOLDIER
                              else AntState.SEARCHING)
                self.state = next_state
                self.last_move_info = f"EscapeEnd->{next_state.name[:3]}"
                # History is already cleared when escape starts

        # Update state based on environment (Patrolling/Defending for Soldiers)
        # This happens regardless of escape timer state (e.g., can switch from P->D)
        self._update_state()


        # --- Combat Check ---
        pos_int = tuple(map(int, self.pos)) # Use int pos
        # Check integer neighbors for enemies
        neighbors_int = get_neighbors(pos_int, True) # Include center
        enemies_in_range = [e for p_int in neighbors_int
                            if (e := self.simulation.get_enemy_at(p_int)) and e.hp > 0]

        if enemies_in_range:
            target_enemy = random.choice(enemies_in_range)
            self.attack(target_enemy)
            # Add alarm pheromone at integer position
            self.simulation.grid.add_pheromone(pos_int, P_ALARM_FIGHT, 'alarm')
            self.stuck_timer = 0 # Reset stuck timer when fighting
            target_pos_int = tuple(map(int, target_enemy.pos)) # Use int pos
            self.last_move_info = f"Fighting {self.caste.name} vs {target_pos_int}"
            # No movement occurs this update cycle if fighting
            return


        # --- Movement ---
        # Movement Delay: Timer counts down each update cycle
        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return # Skip movement if delay active

        # Calculate actual delay based on base delay and speed multiplier
        # Higher speed means less delay between moves (minimum 1 update cycle per move)
        # Inverse relationship: delay_updates = base_delay / speed_multiplier
        # But simpler: move every `max(1, round(base_delay / speed_multiplier))` updates?
        # Or: Apply delay only if base_delay > 0
        effective_delay = 0
        if self.move_delay_base > 0 and speed_multiplier > 0:
             # Calculate how many update cycles constitute the base delay at current speed
             effective_delay = max(0, int(round(self.move_delay_base / speed_multiplier)) -1) # -1 because 1 cycle is the current move
        elif self.move_delay_base > 0 and speed_multiplier == 0:
             effective_delay = float('inf') # Infinite delay if paused


        # Set the timer for the *next* move delay
        self.move_delay_timer = effective_delay


        # --- Choose and Execute Move ---
        old_pos = self.pos # Store original (potentially float) tuple
        local_just_picked = self.just_picked_food # Store state before move decision
        self.just_picked_food = False # Reset flag before decision

        new_pos_cand = self._choose_move() # Returns original tuple type (or None)
        moved = False
        found_food_type = None
        food_amount = 0.0

        if new_pos_cand:
            target = new_pos_cand # Keep original type
            target_int = tuple(map(int, target))
            old_pos_int = tuple(map(int, old_pos))

            # Check if the integer position actually changes
            if target_int != old_pos_int:
                self.pos = target # Update position
                # Calculate direction based on integer positions
                self.last_move_direction = (target_int[0] - old_pos_int[0], target_int[1] - old_pos_int[1])
                self._update_path_history(target_int) # Use int pos for history
                self.stuck_timer = 0 # Reset stuck timer on successful move
                moved = True

                # Check for food at the new integer position
                try:
                    x_int, y_int = target_int
                    # Ensure coords are valid before accessing grid
                    if 0 <= x_int < GRID_WIDTH and 0 <= y_int < GRID_HEIGHT:
                        foods = self.simulation.grid.food[x_int, y_int]
                        if foods[FoodType.SUGAR.value] > 0.1:
                            found_food_type = FoodType.SUGAR
                            food_amount = foods[FoodType.SUGAR.value]
                        elif foods[FoodType.PROTEIN.value] > 0.1:
                            found_food_type = FoodType.PROTEIN
                            food_amount = foods[FoodType.PROTEIN.value]
                    else: found_food_type = None # Invalid coords
                except (IndexError, TypeError):
                    found_food_type = None # Safety net

            else: # Candidate position didn't change integer cell
                 self.stuck_timer += 1
                 self.last_move_info += "(Move->SameCell)"
                 # Clear last move direction if stuck moving to same cell
                 self.last_move_direction = (0,0)

        else: # No move candidate was chosen
            self.stuck_timer += 1
            self.last_move_info += "(NoChoice)"
            self.last_move_direction = (0,0) # Clear direction if no choice

        # --- Post-Movement Actions ---
        pos_int = tuple(map(int, self.pos)) # Use current int pos
        nest_pos_int = tuple(map(int, NEST_POS))
        is_near_nest = distance_sq(pos_int, nest_pos_int) <= NEST_RADIUS**2
        grid = self.simulation.grid
        sim = self.simulation

        # Actions based on state after potential move
        if self.state == AntState.SEARCHING:
            # If Worker finds food and isn't carrying anything
            if (self.caste == AntCaste.WORKER and found_food_type and
                    self.carry_amount == 0):
                pickup_amount = min(self.max_capacity, food_amount) # Amount to pick up
                # Only pick up a meaningful amount
                if pickup_amount > 0.01:
                    self.carry_amount = pickup_amount
                    self.carry_type = found_food_type
                    food_idx = found_food_type.value
                    try:
                        x, y = pos_int # Use int pos for grid access
                        # Update food amount in the grid cell
                        grid.food[x, y, food_idx] -= pickup_amount
                        grid.food[x, y, food_idx] = max(0, grid.food[x, y, food_idx]) # Ensure not negative
                        # Drop pheromones at the food source
                        grid.add_pheromone(pos_int, P_FOOD_AT_SOURCE, 'food')
                        # Drop recruitment pheromone if the source is rich
                        if food_amount >= RICH_FOOD_THRESHOLD:
                            grid.add_pheromone(pos_int, P_RECRUIT_FOOD, 'recruitment')

                        # Change state and clear history for return journey
                        self.state = AntState.RETURNING_TO_NEST
                        self._clear_path_history()
                        self.last_move_info = f"Picked {found_food_type.name}({pickup_amount:.1f})"
                        self.just_picked_food = True # Flag set AFTER state change
                    except (IndexError, TypeError):
                        # Safety: Reset carry state if grid access failed
                        self.carry_amount = 0; self.carry_type = None
                else: # Found food, but too little to pick up
                     self.last_move_info += "(FoodTraceFound)"


            # If moved, found no food, and not too close to nest, drop negative pheromone
            elif (moved and not found_food_type and
                  distance_sq(pos_int, nest_pos_int) > (NEST_RADIUS + 2)**2):
                 old_pos_int = tuple(map(int, old_pos)) # Previous integer position
                 # Check if old_pos_int is valid before adding pheromone
                 if is_valid(old_pos_int):
                     grid.add_pheromone(old_pos_int, P_NEGATIVE_SEARCH, 'negative')


        elif self.state == AntState.RETURNING_TO_NEST:
            # If ant reaches the nest area
            if is_near_nest:
                dropped_amount = self.carry_amount
                type_dropped = self.carry_type
                # If carrying food, drop it in the colony storage
                if dropped_amount > 0 and type_dropped:
                    if type_dropped == FoodType.SUGAR: sim.colony_food_storage_sugar += dropped_amount
                    elif type_dropped == FoodType.PROTEIN: sim.colony_food_storage_protein += dropped_amount
                    self.carry_amount = 0; self.carry_type = None # Reset carry state

                # Transition to next state (Patrolling for Soldier, Searching for Worker)
                next_state = (AntState.PATROLLING if self.caste == AntCaste.SOLDIER
                              else AntState.SEARCHING)
                self.state = next_state
                self._clear_path_history() # Clear path for new task
                type_str = f" {type_dropped.name}" if type_dropped else ""
                next_s_str = next_state.name[:3]
                self.last_move_info = (f"Dropped{type_str}({dropped_amount:.1f})->{next_s_str}"
                                       if dropped_amount > 0 else f"NestEmpty->{next_s_str}")
            # If moved while returning (and outside nest), drop pheromone trails
            elif moved and not local_just_picked: # Don't drop trails immediately after pickup
                 old_pos_int = tuple(map(int, old_pos))
                 # Check if old position is valid and outside nest radius
                 if is_valid(old_pos_int) and distance_sq(old_pos_int, nest_pos_int) > NEST_RADIUS**2:
                    # Drop home trail pheromone
                    grid.add_pheromone(old_pos_int, P_HOME_RETURNING, 'home')
                    # If carrying food, also drop food trail pheromone
                    if self.carry_amount > 0:
                        grid.add_pheromone(old_pos_int, P_FOOD_RETURNING_TRAIL, 'food')


        # --- Stuck Check ---
        # Check if stuck timer exceeds threshold and not already escaping
        if (self.stuck_timer >= WORKER_STUCK_THRESHOLD and
                self.state != AntState.ESCAPING):
            # Check if currently fighting (neighbors include self cell)
            neighbors_int = get_neighbors(pos_int, True)
            is_fighting = any(sim.get_enemy_at(p_int) for p_int in neighbors_int)

            # If not fighting, initiate escape behavior
            if not is_fighting:
                self.state = AntState.ESCAPING
                # Escape duration could potentially scale with speed? Let's keep fixed for now.
                self.escape_timer = WORKER_ESCAPE_DURATION # Set escape duration (in ticks/updates)
                self.stuck_timer = 0 # Reset stuck timer
                self._clear_path_history() # Clear history to allow moving back
                self.last_move_info = "Stuck->Escaping"


    def _update_state(self):
        """Handle automatic state transitions (mainly for Soldiers)."""
        # Only applicable to Soldiers not currently escaping or returning
        if (self.caste != AntCaste.SOLDIER or
                self.state in [AntState.ESCAPING, AntState.RETURNING_TO_NEST]):
            return

        pos_int = tuple(map(int, self.pos)) # Use int pos
        nest_pos_int = tuple(map(int, NEST_POS))
        max_alarm = 0; max_recruit = 0; radius_sq = 5*5 # Check radius
        grid = self.simulation.grid
        x0, y0 = pos_int

        # Sense local alarm and recruitment levels using integer coordinates
        for i in range(x0 - 2, x0 + 3):
            for j in range(y0 - 2, y0 + 3):
                p = (i, j)
                if distance_sq(pos_int, p) <= radius_sq and is_valid(p):
                    max_alarm = max(max_alarm, grid.get_pheromone(p, 'alarm'))
                    max_recruit = max(max_recruit, grid.get_pheromone(p, 'recruitment')) # Check recruitment too


        is_near_nest = distance_sq(pos_int, nest_pos_int) <= SOLDIER_PATROL_RADIUS_SQ
        # Combine signals for state transition trigger
        threat_signal = max_alarm + max_recruit * 0.5 # Prioritize alarm

        # --- State transition logic ---
        # If high threat detected -> DEFENDING
        if threat_signal > SOLDIER_DEFEND_ALARM_THRESHOLD:
            if self.state != AntState.DEFENDING:
                self.state = AntState.DEFENDING
                self._clear_path_history()
                self.last_known_alarm_pos = None # Reset target when entering defend state
                self.last_move_info += " ->DEFEND(Threat!)"
        # If currently DEFENDING but threat subsided -> PATROLLING
        elif self.state == AntState.DEFENDING:
            self.state = AntState.PATROLLING
            self.last_move_info += " ->PATROL(ThreatLow)"
        # If near nest, no threat, and not already PATROLLING -> PATROLLING
        elif is_near_nest and self.state != AntState.PATROLLING:
            self.state = AntState.PATROLLING
            self.last_move_info += " ->PATROL(NearNest)"
        # If PATROLLING but moved too far from nest -> SEARCHING (explore)
        elif not is_near_nest and self.state == AntState.PATROLLING:
            self.state = AntState.SEARCHING
            self.last_move_info += " ->SEARCH(PatrolFar)"
        # If currently SEARCHING but back near nest -> PATROLLING
        elif is_near_nest and self.state == AntState.SEARCHING:
             self.state = AntState.PATROLLING
             self.last_move_info += " ->PATROL(SearchNear)"


    def attack(self, target_enemy):
        # Basic attack action
        target_enemy.take_damage(self.attack_power, self)
        # Maybe add recoil or brief pause after attacking? (optional)


    def take_damage(self, amount, attacker):
        """Process damage taken by the ant."""
        if self.hp <= 0: return # Already dead
        self.hp -= amount
        if self.hp > 0:
            # If still alive, release alarm/recruitment pheromones
            grid = self.simulation.grid
            pos_int = tuple(map(int, self.pos)) # Use int pos
            grid.add_pheromone(pos_int, P_ALARM_FIGHT / 2, 'alarm') # Add standard alarm
            # Add recruitment pheromone based on caste
            recruit_amount = (P_RECRUIT_DAMAGE_SOLDIER if self.caste == AntCaste.SOLDIER
                              else P_RECRUIT_DAMAGE)
            grid.add_pheromone(pos_int, recruit_amount, 'recruitment')
            # Potential state change on taking damage? (e.g., SEARCHING -> DEFENDING if soldier?)
            # Handled by _update_state sensing the new pheromones.
        # else: # HP <= 0
             # self.last_move_info = "Killed in combat" # Set reason for removal


# --- Queen Class ---
class Queen:
    """Manages queen state and egg laying based on food types."""
    def __init__(self, pos, sim):
        # Ensure position is integer tuple
        self.pos = tuple(map(int, pos))
        self.simulation = sim
        self.hp = QUEEN_HP; self.max_hp = QUEEN_HP
        self.age = 0; self.max_age = float('inf') # Effectively immortal unless killed
        self.egg_lay_timer_progress = 0.0 # Internal timer scaled by speed
        self.egg_lay_interval_ticks = QUEEN_EGG_LAY_RATE # Interval in standard ticks
        self.color = QUEEN_COLOR
        self.state = None # Queen doesn't really have states like workers
        self.attack_power = 0 # Queen doesn't attack
        self.carry_amount = 0 # Queen doesn't carry

    def update(self, speed_multiplier):
        """Update Queen's age and handle egg laying based on speed."""
        # --- Aging ---
        # Queen ages like other ants, based on speed multiplier
        self.age += speed_multiplier # Simple age increment scaled by speed

        # --- Egg Laying ---
        self.egg_lay_timer_progress += speed_multiplier
        # Check if enough time has passed (scaled by speed) to lay an egg
        if self.egg_lay_timer_progress >= self.egg_lay_interval_ticks:
            self.egg_lay_timer_progress %= self.egg_lay_interval_ticks # Reset timer keeping remainder

            # Check for sufficient food resources
            needed_s = QUEEN_FOOD_PER_EGG_SUGAR
            needed_p = QUEEN_FOOD_PER_EGG_PROTEIN
            sim = self.simulation
            has_food = (sim.colony_food_storage_sugar >= needed_s and
                        sim.colony_food_storage_protein >= needed_p)

            if has_food:
                # Consume food
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p

                # Decide caste for the new egg
                caste = self._decide_caste()
                # Find a valid position for the egg (returns int pos or None)
                egg_pos = self._find_egg_position()

                if egg_pos: # Ensure a valid position was found
                    # Create new brood item
                    new_brood = BroodItem(BroodStage.EGG, caste, egg_pos, sim.ticks)
                    sim.brood.append(new_brood)
            # else: # Not enough food, skip laying egg this cycle


    def _decide_caste(self):
        """Decide the caste of the next egg based on colony needs."""
        ratio = 0.0; ants = self.simulation.ants; brood = self.simulation.brood
        # Calculate current soldier ratio (considering active ants and developing brood)
        # Only count larvae and pupae for caste ratio, as eggs are undecided/fragile
        developing_brood = [b for b in brood if b.stage in [BroodStage.LARVA, BroodStage.PUPA]]
        total_population = len(ants) + len(developing_brood)

        if total_population > 0:
            soldier_count = (sum(1 for a in ants if a.caste == AntCaste.SOLDIER) +
                             sum(1 for b in developing_brood if b.caste == AntCaste.SOLDIER))
            ratio = soldier_count / total_population

        # --- Decision Logic ---
        # If soldier ratio is below target, higher chance to lay a soldier egg
        if ratio < QUEEN_SOLDIER_RATIO_TARGET:
            return AntCaste.SOLDIER if random.random() < 0.6 else AntCaste.WORKER
        # If ratio is met, low chance to lay a soldier (maintain population)
        elif random.random() < 0.05: # Reduced chance compared to previous version
            return AntCaste.SOLDIER
        # Default: Lay a worker egg
        return AntCaste.WORKER

    def _find_egg_position(self):
        """Find a valid integer position near the queen for a new egg."""
        # Get integer neighbors of the queen's integer position
        possible_spots = get_neighbors(self.pos) # get_neighbors returns int
        valid_spots = [p for p in possible_spots
                       if not self.simulation.grid.is_obstacle(p)]

        # Prefer positions without existing brood items
        brood_positions = {b.pos for b in self.simulation.brood} # Set of int positions
        free_valid_spots = [p for p in valid_spots if p not in brood_positions]

        if free_valid_spots:
            return random.choice(free_valid_spots)
        elif valid_spots:
            # If no free spots, place on top of existing brood (less ideal)
            return random.choice(valid_spots)
        else:
             # Very unlikely fallback: if no valid neighbor, place at queen's pos
             if not self.simulation.grid.is_obstacle(self.pos):
                  return self.pos
             else:
                  return None # Cannot place egg


    def take_damage(self, amount, attacker):
        """Process damage taken by the queen."""
        if self.hp <= 0: return # Already dead
        self.hp -= amount
        if self.hp > 0:
            # If still alive, release strong alarm/recruitment pheromones at her integer position
            grid = self.simulation.grid
            grid.add_pheromone(self.pos, P_ALARM_FIGHT * 3, 'alarm') # More alarm
            grid.add_pheromone(self.pos, P_RECRUIT_DAMAGE * 3, 'recruitment') # More recruitment


# --- Enemy Class ---
class Enemy:
    """Represents an enemy entity."""
    def __init__(self, pos, sim):
         # Ensure position is integer tuple
        self.pos = tuple(map(int, pos)); self.simulation = sim; self.hp = ENEMY_HP
        self.max_hp = ENEMY_HP; self.attack_power = ENEMY_ATTACK
        self.move_delay_base = ENEMY_MOVE_DELAY # Base delay in standard ticks
        self.move_delay_timer = rnd(0, self.move_delay_base) # Initial random delay timer
        self.color = ENEMY_COLOR

    def update(self, speed_multiplier):
        """Update enemy state, potentially moving or attacking."""
        pos_int = self.pos # Position is already int

        # --- Combat Check ---
        # Check integer neighbors (including current cell) for ants (worker, soldier, queen)
        neighbors_int = get_neighbors(pos_int, True)
        target_ants = [a for p_int in neighbors_int
                       if (a := self.simulation.get_ant_at(p_int)) and a.hp > 0]

        if target_ants:
            # Prioritize attacking queen if she's adjacent
            queen_target = next((a for a in target_ants if isinstance(a, Queen)), None)
            target = queen_target if queen_target else random.choice(target_ants)

            self.attack(target)
            # Add small amount of negative pheromone when attacking? Optional.
            # self.simulation.grid.add_pheromone(pos_int, 5.0, 'negative')
            return # Don't move if fighting


        # --- Movement Delay ---
        # Timer counts down scaled by speed
        self.move_delay_timer -= speed_multiplier
        if self.move_delay_timer > 0:
            return # Skip movement if delay active

        # Reset timer for next move, possibly adding random variation
        self.move_delay_timer = self.move_delay_base + rnd_uniform(-0.5, 0.5)


        # --- Movement Logic ---
        # Get valid integer neighbor cells for movement
        possible_moves = get_neighbors(pos_int) # Gets int neighbors
        valid_moves = []
        for m_int in possible_moves: # Neighbors are already int
            # Check for obstacles, other enemies, and ants at the potential destination
            if (not self.simulation.grid.is_obstacle(m_int) and
                not self.simulation.is_enemy_at(m_int, self) and # Avoid colliding with other enemies
                not self.simulation.is_ant_at(m_int)): # Avoid walking onto ants directly
                valid_moves.append(m_int) # Add valid integer position

        if valid_moves:
            chosen_move = None
            nest_pos_int = tuple(map(int, NEST_POS))

            # Small chance to specifically target nest direction
            if random.random() < ENEMY_NEST_ATTRACTION:
                best_nest_move = None
                min_dist_sq = distance_sq(pos_int, nest_pos_int)
                # Find valid move that gets closer to the nest
                for move in valid_moves:
                     d_sq = distance_sq(move, nest_pos_int)
                     if d_sq < min_dist_sq:
                         min_dist_sq = d_sq
                         best_nest_move = move
                # If a closer move exists, choose it
                if best_nest_move:
                     chosen_move = best_nest_move
                else: # If no move gets closer, choose randomly from valid moves
                     chosen_move = random.choice(valid_moves)
            else:
                # Default: Choose a random valid move
                chosen_move = random.choice(valid_moves)

            # Execute the chosen move
            if chosen_move: # chosen_move is an integer tuple
                 self.pos = chosen_move # Update position with the chosen int tuple


    def attack(self, target_ant):
        """Attack a target ant."""
        target_ant.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        """Process damage taken by the enemy."""
        self.hp -= amount
        # Optional: Add reaction like fleeing at low HP?


# --- Main Simulation Class ---
class AntSimulation:
    """Manages the overall simulation state, entities, and drawing."""

    def __init__(self):
        """Initialize Pygame, Grid, Entities, and Simulation state."""
        # Pygame Init moved to main block for safety
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ant Simulation - Complex Dynamics")
        self.clock = pygame.time.Clock()
        self.font = None # Initialize font later
        self.debug_font = None
        self._init_fonts()

        self.grid = WorldGrid()
        self.simulation_running = False # Controls the main simulation update loop
        self.app_running = True       # Controls the overall application loop (incl. menus)
        self.end_game_reason = ""     # Store why the game ended

        # Colony Generation Counter
        self.colony_generation = 0 # Start at 0, reset will increment to 1 first time

        # Defer initialization of simulation state to _reset_simulation
        self.ticks = 0 # Represents standard time units passed
        self.ants = []
        self.enemies = []
        self.brood = []
        self.queen = None
        self.colony_food_storage_sugar = 0.0
        self.colony_food_storage_protein = 0.0
        self.enemy_spawn_timer = 0.0 # Use float timer scaled by speed
        self.enemy_spawn_interval_ticks = ENEMY_SPAWN_RATE # Interval in standard ticks

        self.show_debug_info = True

        # --- NEW Speed Control State ---
        self.simulation_speed_index = DEFAULT_SPEED_INDEX
        self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
        # --- End Speed Control State ---

        self.buttons = self._create_buttons() # Create UI buttons

        # Call reset to perform initial setup and start first generation
        self._reset_simulation()

    def _init_fonts(self):
        """Initialize fonts, handling potential errors."""
        try:
            # Use a common, often available font first
            self.font = pygame.font.SysFont("sans", 16) # Main UI font
            self.debug_font = pygame.font.SysFont("monospace", 14) # Debug overlay font
            print("Using system 'sans' and 'monospace' fonts.")
        except Exception as e1:
            print(f"System font error: {e1}. Trying default font.")
            try:
                self.font = pygame.font.Font(None, 20) # Default pygame font
                self.debug_font = pygame.font.Font(None, 16)
                print("Using Pygame default font.")
            except Exception as e2:
                print(f"FATAL: Default font error: {e2}. Cannot render text.")
                self.font = None # Ensure it's None if failed
                self.debug_font = None
                # Stop the application if fonts cannot be loaded
                self.app_running = False

    def _reset_simulation(self):
        """Resets the simulation state for a new game."""
        print(f"Resetting simulation for Kolonie {self.colony_generation + 1}...")
        self.ticks = 0
        self.ants.clear()
        self.enemies.clear()
        self.brood.clear()
        self.queen = None # Ensure queen is cleared before potentially failing to place new one
        self.colony_food_storage_sugar = INITIAL_COLONY_FOOD_SUGAR
        self.colony_food_storage_protein = INITIAL_COLONY_FOOD_PROTEIN
        self.enemy_spawn_timer = 0.0 # Reset scaled timer
        self.end_game_reason = ""

        # INCREMENT Colony Counter
        self.colony_generation += 1

        # Reset grid (places obstacles and food)
        self.grid.reset()

        # Spawn initial entities
        if not self._spawn_initial_entities():
             print("CRITICAL ERROR during simulation reset. Cannot continue.")
             self.simulation_running = False
             self.app_running = False # Stop the whole app if reset fails critically
             self.end_game_reason = "Initialisierungsfehler"
             return

        # Set simulation state to running and reset speed to default
        self.simulation_speed_index = DEFAULT_SPEED_INDEX
        self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
        self.simulation_running = True
        print(f"Kolonie {self.colony_generation} gestartet at {SPEED_MULTIPLIERS[self.simulation_speed_index]:.1f}x speed.")


    def _create_buttons(self):
        """Creates data structures for UI buttons (now +/- speed)."""
        buttons = []
        button_h = 20
        button_w = 30 # Make +/- buttons narrower
        margin = 5
        # Position buttons on the right side
        btn_plus_x = WIDTH - button_w - margin
        btn_minus_x = btn_plus_x - button_w - margin

        # Minus Button
        rect_minus = pygame.Rect(btn_minus_x, margin, button_w, button_h)
        buttons.append({'rect': rect_minus, 'text': '-', 'action': 'speed_down'})

        # Plus Button
        rect_plus = pygame.Rect(btn_plus_x, margin, button_w, button_h)
        buttons.append({'rect': rect_plus, 'text': '+', 'action': 'speed_up'})

        return buttons

    def _spawn_initial_entities(self):
        """Spawn the queen and initial set of ants and enemies. Returns False on critical failure."""
        queen_pos = self._find_valid_queen_pos()
        if queen_pos:
             self.queen = Queen(queen_pos, self) # Queen pos is already int
             print(f"Queen placed at {queen_pos}")
        else:
             print("CRITICAL: Cannot place Queen. Obstacles might block nest center.")
             return False # Indicate critical failure

        spawned=0; attempts=0; max_att=INITIAL_ANTS*20
        # Use integer queen position for spawning radius
        queen_pos_int = self.queen.pos
        while spawned<INITIAL_ANTS and attempts<max_att:
            r=NEST_RADIUS+2; ox=rnd(-r,r); oy=rnd(-r,r);
            # Calculate spawn position as integer tuple
            pos=(queen_pos_int[0]+ox, queen_pos_int[1]+oy)
            caste = AntCaste.SOLDIER if random.random()<0.2 else AntCaste.WORKER
            if self.add_ant(pos, caste): # add_ant handles validation
                spawned+=1
            attempts+=1
        if spawned<INITIAL_ANTS: print(f"Warn: Spawned only {spawned}/{INITIAL_ANTS} initial ants.")

        enemies_spawned = 0
        for _ in range(INITIAL_ENEMIES):
            if self.spawn_enemy(): enemies_spawned += 1
        print(f"Spawned {enemies_spawned}/{INITIAL_ENEMIES} initial enemies.")
        return True # Success

    def _find_valid_queen_pos(self):
        """Find a valid, non-obstacle integer position for the queen near NEST_POS."""
        base = tuple(map(int, NEST_POS))
        if is_valid(base) and not self.grid.is_obstacle(base): return base

        # Check immediate neighbors (already integer positions)
        neighbors = get_neighbors(base)
        random.shuffle(neighbors) # Check in random order
        for p in neighbors:
             if not self.grid.is_obstacle(p): return p # Returns int pos

        # Check slightly further out if immediate neighbors fail
        for r in range(2, 5): # Increase search radius slightly
             perimeter = []
             for dx in range(-r, r + 1):
                 for dy in range(-r, r + 1):
                     # Check only the perimeter of the radius r box
                     if abs(dx) == r or abs(dy) == r:
                         p = (base[0] + dx, base[1] + dy)
                         if is_valid(p) and not self.grid.is_obstacle(p):
                              perimeter.append(p)
             if perimeter:
                 return random.choice(perimeter) # Return random valid spot on perimeter

        print("CRITICAL: Could not find ANY valid spot near nest center for Queen.")
        return None # Indicate failure

    def add_ant(self, pos, caste: AntCaste):
        """Create and add a new ant of a specific caste if position is valid (expects int pos)."""
        pos_int = tuple(map(int, pos)) # Ensure integer tuple
        if not is_valid(pos_int): return False
        # Check obstacle, existing ants/enemies, and queen position
        if (not self.grid.is_obstacle(pos_int) and
            not self.is_ant_at(pos_int) and
            not self.is_enemy_at(pos_int) and
            (not self.queen or pos_int != self.queen.pos)):
            self.ants.append(Ant(pos_int, self, caste)); return True # Pass int pos
        return False

    def spawn_enemy(self):
        """Spawn a new enemy at a valid random integer location."""
        tries = 0
        while tries < 50:
            # Generate integer position directly
            pos_i = (rnd(0,GRID_WIDTH-1), rnd(0,GRID_HEIGHT-1))
            q_pos_int = self.queen.pos if self.queen else tuple(map(int, NEST_POS))
            # Ensure enemy spawns sufficiently far from the nest
            dist_ok = distance_sq(pos_i, q_pos_int) > (MIN_FOOD_DIST_FROM_NEST)**2

            # Check validity: not obstacle, far enough, no other entity present
            if (dist_ok and not self.grid.is_obstacle(pos_i) and
                not self.is_enemy_at(pos_i) and not self.is_ant_at(pos_i)):
                self.enemies.append(Enemy(pos_i, self)); return True # Pass int pos
            tries += 1
        return False # Failed to find suitable spawn location

    def kill_ant(self, ant_to_remove, reason="unknown"):
        """Remove an ant from the simulation."""
        if ant_to_remove in self.ants:
             self.ants.remove(ant_to_remove)
        # else: print(f"Warn: Tried to remove non-existent ant ({reason}).") # Optional debug

    def kill_enemy(self, enemy_to_remove):
        """Remove an enemy and potentially drop food."""
        if enemy_to_remove in self.enemies:
            pos_int = enemy_to_remove.pos # Position is already int
            # Drop food only if the position is valid and not an obstacle
            if is_valid(pos_int) and not self.grid.is_obstacle(pos_int):
                fx, fy = pos_int; grid = self.grid
                s_idx = FoodType.SUGAR.value; p_idx = FoodType.PROTEIN.value
                # Add food resources to the grid cell
                try:
                    grid.food[fx, fy, s_idx] = min(MAX_FOOD_PER_CELL, grid.food[fx, fy, s_idx] + ENEMY_TO_FOOD_ON_DEATH_SUGAR)
                    grid.food[fx, fy, p_idx] = min(MAX_FOOD_PER_CELL, grid.food[fx, fy, p_idx] + ENEMY_TO_FOOD_ON_DEATH_PROTEIN)
                except IndexError:
                     print(f"WARN: IndexError accessing grid food at {pos_int} during enemy kill.")
            self.enemies.remove(enemy_to_remove)
        # else: print("Warn: Tried to remove non-existent enemy.") # Optional debug

    def kill_queen(self, queen_to_remove):
        """Handle the death of the queen, stopping the current simulation run."""
        if self.queen == queen_to_remove:
            print(f"\n--- QUEEN DIED (Tick {self.ticks}, Kolonie {self.colony_generation}) ---")
            print(f"    Food S:{self.colony_food_storage_sugar:.1f} P:{self.colony_food_storage_protein:.1f}")
            print(f"    Ants:{len(self.ants)}, Brood:{len(self.brood)}")
            self.queen = None
            # Stop the simulation loop, trigger end game dialog
            self.simulation_running = False
            self.end_game_reason = "Königin gestorben"
        else:
            print("Warn: Attempted Kill inactive/non-existent queen.")

    def is_ant_at(self, pos, exclude_self=None):
        """Check if an ant (worker, soldier, or queen) is at an integer position."""
        pos_i = tuple(map(int, pos)) # Ensure integer comparison
        q = self.queen
        # Check queen (position is already int)
        if (q and q.pos == pos_i and exclude_self != q): return True
        # Check worker/soldier ants (position is already int)
        for a in self.ants:
            if a is exclude_self: continue
            if a.pos == pos_i: return True
        return False

    def get_ant_at(self, pos):
        """Return the ant object (worker, soldier, or queen) at an integer position."""
        pos_i = tuple(map(int, pos)) # Ensure integer comparison
        q = self.queen
        if q and q.pos == pos_i: return q # Queen's pos is int
        for a in self.ants:
             if a.pos == pos_i: return a # Ant's pos is int
        return None

    def is_enemy_at(self, pos, exclude_self=None):
        """Check if an enemy is at an integer position."""
        pos_i = tuple(map(int, pos)) # Ensure integer comparison
        for e in self.enemies:
             if e is exclude_self: continue
             if e.pos == pos_i: return True # Enemy's pos is int
        return False

    def get_enemy_at(self, pos):
        """Return the enemy object at an integer position."""
        pos_i = tuple(map(int, pos)) # Ensure integer comparison
        for e in self.enemies:
             if e.pos == pos_i: return e # Enemy's pos is int
        return None

    def update(self):
        """Run one simulation tick. Assumes simulation_running is True."""

        # Get the current speed multiplier (0.0x if paused)
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]

        # If paused, do nothing except increment ticks minimally for display
        if current_multiplier == 0.0:
            self.ticks += 0.01 # Increment very slowly if paused
            return

        # --- Simulation Tick Increment ---
        # Increment ticks based on the speed multiplier
        self.ticks += current_multiplier


        # --- Pre-Update Checks (Removal of dead/invalid entities) ---
        ants_to_remove = []
        for a in self.ants:
            pos_int = a.pos # Already int
            reason = ""
            if a.hp <= 0: reason = "hp <= 0"
            elif a.age >= a.max_age_ticks: reason = f"aged out ({a.age:.0f}/{a.max_age_ticks})"
            elif self.grid.is_obstacle(pos_int): reason = f"in obstacle {pos_int}"
            if reason:
                 # Store ant and reason for removal after iteration
                 ants_to_remove.append((a, reason))

        enemies_to_remove = [e for e in self.enemies if e.hp <= 0 or self.grid.is_obstacle(e.pos)]
        queen_remove = None
        if self.queen:
            pos_int = self.queen.pos # Already int
            reason = ""
            if self.queen.hp <= 0: reason = "hp <= 0"
            elif self.grid.is_obstacle(pos_int): reason = f"in obstacle {pos_int}"
            if reason:
                 queen_remove = self.queen

        # Perform removals
        for ant, reason in ants_to_remove: self.kill_ant(ant, reason)
        for enemy in enemies_to_remove: self.kill_enemy(enemy)
        if queen_remove:
             self.kill_queen(queen_remove) # This might set simulation_running to False

        # If queen died, stop further updates for this tick
        if not self.simulation_running: return


        # --- Update Entities --- Pass the speed multiplier ---
        if self.queen: self.queen.update(current_multiplier)
        if not self.simulation_running: return # Check if queen died during update

        # Update Brood Items
        hatched=[]; brood_copy=list(self.brood)
        for item in brood_copy:
             if item in self.brood: # Check if not already removed
                 hatch_signal = item.update(self.ticks, self) # Pass simulation obj for food access
                 if hatch_signal and hatch_signal in self.brood:
                     hatched.append(hatch_signal) # Should be the item itself

        # Spawn hatched ants
        for pupa in hatched:
            if pupa in self.brood: # Check again before removing/spawning
                 self.brood.remove(pupa)
                 self._spawn_hatched_ant(pupa.caste, pupa.pos) # Pass caste and position

        # Update Ants and Enemies (pass speed multiplier)
        ants_copy=list(self.ants); enemies_copy=list(self.enemies)
        # Shuffle update order slightly to break synchronicity
        random.shuffle(ants_copy)
        random.shuffle(enemies_copy)

        for a in ants_copy:
             if a in self.ants: a.update(current_multiplier)
        for e in enemies_copy:
            if e in self.enemies: e.update(current_multiplier)


        # --- Post-Update Checks (Catch deaths during update) ---
        final_ants_invalid = [a for a in self.ants if a.hp <= 0]
        final_enemies_invalid = [e for e in self.enemies if e.hp <= 0]
        for a in final_ants_invalid: self.kill_ant(a, "post-update")
        for e in final_enemies_invalid: self.kill_enemy(e)
        if self.queen and self.queen.hp <= 0: self.kill_queen(self.queen)

        if not self.simulation_running: return # Exit if queen died


        # --- Update Environment ---
        # Update Pheromones (pass speed multiplier for decay/diffusion scaling)
        self.grid.update_pheromones(current_multiplier)

        # Update Enemy Spawner
        self.enemy_spawn_timer += current_multiplier
        if self.enemy_spawn_timer >= self.enemy_spawn_interval_ticks:
            self.enemy_spawn_timer %= self.enemy_spawn_interval_ticks # Reset timer
            # Limit total number of enemies
            if len(self.enemies) < INITIAL_ENEMIES * 5:
                 self.spawn_enemy()


    def _spawn_hatched_ant(self, caste: AntCaste, pupa_pos: tuple):
        """Tries to spawn a hatched ant near the pupa's location."""
        # Try spawning exactly at the pupa's (integer) location first
        if self.add_ant(pupa_pos, caste): return True

        # If blocked, try neighbors of the pupa's location
        neighbors = get_neighbors(pupa_pos) # Gets int neighbors
        random.shuffle(neighbors) # Try neighbors in random order
        for pos in neighbors:
             if self.add_ant(pos, caste): return True

        # Fallback: Try spawning near queen (less ideal)
        if self.queen:
            base = self.queen.pos
            for attempts in range(10): # Limit attempts
                ox = rnd(-NEST_RADIUS + 1, NEST_RADIUS - 1) # Spawn closer inside nest
                oy = rnd(-NEST_RADIUS + 1, NEST_RADIUS - 1)
                pos = (base[0] + ox, base[1] + oy) # int pos
                if self.add_ant(pos, caste): return True

        # print(f"Warn: Failed hatch spawn {caste.name} near {pupa_pos}") # Optional Debug
        return False


    def draw_debug_info(self):
        if not self.debug_font: return
        ant_c=len(self.ants); enemy_c=len(self.enemies); brood_c=len(self.brood)
        food_s=self.colony_food_storage_sugar; food_p=self.colony_food_storage_protein
        tick_display = int(self.ticks) # Show integer ticks passed
        fps=self.clock.get_fps()
        w_c=sum(1 for a in self.ants if a.caste==AntCaste.WORKER); s_c=sum(1 for a in self.ants if a.caste==AntCaste.SOLDIER)
        e_c=sum(1 for b in self.brood if b.stage==BroodStage.EGG); l_c=sum(1 for b in self.brood if b.stage==BroodStage.LARVA); p_c=sum(1 for b in self.brood if b.stage==BroodStage.PUPA)

        # --- Get Current Speed Multiplier ---
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]
        if current_multiplier == 0.0:
            speed_text = "Speed: Paused"
        else:
            # Format with one decimal place, unless it's an integer
            speed_text = f"Speed: {current_multiplier:.1f}x".replace('.0x','x')

        # --- Assemble Debug Text Lines ---
        texts = [
            f"Kolonie: {self.colony_generation}",
            f"Tick: {tick_display} FPS: {fps:.0f}",
             speed_text, # Display current speed
            f"Ants: {ant_c} (W:{w_c} S:{s_c})",
            f"Brood: {brood_c} (E:{e_c} L:{l_c} P:{p_c})",
            f"Enemies: {enemy_c}",
            f"Food S:{food_s:.1f} P:{food_p:.1f}"
        ]
        y=5; col=(255,255,255); line_h = self.debug_font.get_height() + 1

        # Render standard debug lines
        for i, txt in enumerate(texts):
            try:
                surf=self.debug_font.render(txt,True,col); self.screen.blit(surf,(5, y+i*line_h))
            except Exception as e: print(f"Debug Font render err: {e}")

        # --- Mouse Hover Info ---
        try:
            mx,my=pygame.mouse.get_pos(); gx,gy=mx//CELL_SIZE,my//CELL_SIZE
            pos_i=(gx,gy) # Integer position for grid access and entity checks

            if is_valid(pos_i):
                lines=[];
                entity = self.get_ant_at(pos_i) or self.get_enemy_at(pos_i) # Use int pos

                # Entity Info (Ant, Queen, Enemy)
                if entity:
                    entity_pos_int = entity.pos # Already int
                    if isinstance(entity,Queen): lines.extend([f"QUEEN @{entity_pos_int}", f"HP:{entity.hp:.0f}/{entity.max_hp}"])
                    elif isinstance(entity,Ant): lines.extend([f"{entity.caste.name}@{entity_pos_int}", f"S:{entity.state.name} HP:{entity.hp:.0f}", f"C:{entity.carry_amount:.1f}({entity.carry_type.name if entity.carry_type else '-'})", f"Age:{entity.age:.0f}/{entity.max_age_ticks}", f"Mv:{entity.last_move_info[:25]}"])
                    elif isinstance(entity,Enemy): lines.extend([f"ENEMY @{entity_pos_int}", f"HP:{entity.hp:.0f}/{entity.max_hp}"])

                # Brood Info
                brood_at_pos=[b for b in self.brood if b.pos == pos_i]
                if brood_at_pos: lines.append(f"Brood:{len(brood_at_pos)} @{pos_i}");
                for b in brood_at_pos[:2]: lines.append(f"-{b.stage.name}({b.caste.name}) {int(b.progress_timer)}/{b.duration}") # Show integer progress

                # Cell Info (Obstacle, Food, Pheromones)
                obs=self.grid.is_obstacle(pos_i); obs_txt=" OBSTACLE" if obs else ""
                lines.append(f"Cell:{pos_i}{obs_txt}")
                if not obs:
                    try:
                        foods=self.grid.food[pos_i[0],pos_i[1]]; food_txt=f"Food S:{foods[0]:.1f} P:{foods[1]:.1f}"
                        ph={t:self.grid.get_pheromone(pos_i,t) for t in ['home','food','alarm','negative','recruitment']}
                        ph1=f"Ph H:{ph['home']:.0f} F:{ph['food']:.0f}"; ph2=f"Ph A:{ph['alarm']:.0f} N:{ph['negative']:.0f} R:{ph['recruitment']:.0f}"
                        lines.extend([food_txt, ph1, ph2])
                    except IndexError: lines.append("Error reading cell data")

                # Render hover info at the bottom
                hover_col=(255,255,0); y_off=HEIGHT-(len(lines)*line_h)-5
                for i, line in enumerate(lines):
                     surf=self.debug_font.render(line,True,hover_col); self.screen.blit(surf,(5,y_off+i*line_h))
        except Exception as e:
            import traceback
            print(f"Debug draw err (mouse @ {pygame.mouse.get_pos()}): {e}")
            # traceback.print_exc() # Uncomment for full traceback

    def draw(self):
        """Draw all simulation elements."""
        self._draw_grid()
        self._draw_brood()
        self._draw_queen()
        self._draw_entities()
        if self.show_debug_info: self.draw_debug_info()
        self._draw_buttons() # Draw speed +/- buttons
        pygame.display.flip()

    def _draw_grid(self):
        # 1. BG & Obstacles
        bg=pygame.Surface((WIDTH,HEIGHT)); bg.fill(MAP_BG_COLOR)
        obstacle_coords = np.argwhere(self.grid.obstacles)
        cs = CELL_SIZE
        for x, y in obstacle_coords:
            pygame.draw.rect(bg, OBSTACLE_COLOR, (x * cs, y * cs, cs, cs))
        self.screen.blit(bg,(0,0))

        # 2. Pheromones
        ph_types=['home','food','alarm','negative','recruitment']
        ph_colors={'home':PHEROMONE_HOME_COLOR, 'food':PHEROMONE_FOOD_COLOR, 'alarm':PHEROMONE_ALARM_COLOR, 'negative':PHEROMONE_NEGATIVE_COLOR, 'recruitment':PHEROMONE_RECRUITMENT_COLOR}
        ph_arrays={'home':self.grid.pheromones_home, 'food':self.grid.pheromones_food, 'alarm':self.grid.pheromones_alarm, 'negative':self.grid.pheromones_negative, 'recruitment':self.grid.pheromones_recruitment}
        min_draw_ph=0.5

        for ph_type in ph_types:
            ph_surf=pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
            base_col=ph_colors[ph_type]; arr=ph_arrays[ph_type]
            cur_max = RECRUITMENT_PHEROMONE_MAX if ph_type=='recruitment' else PHEROMONE_MAX
            norm_divisor = max(cur_max / 3.0, 1.0)
            nz_coords = np.argwhere(arr > min_draw_ph)

            for x,y in nz_coords:
                val = arr[x, y]
                norm_val = normalize(val, norm_divisor)
                alpha = int(norm_val * base_col[3])
                alpha = min(max(alpha, 0), 255)
                if alpha > 3:
                    color = (*base_col[:3], alpha)
                    pygame.draw.rect(ph_surf, color, (x * cs, y * cs, cs, cs))
            self.screen.blit(ph_surf, (0, 0))


        # 3. Food
        min_draw_food=0.1
        food_totals = np.sum(self.grid.food, axis=2)
        food_nz_coords = np.argwhere(food_totals > min_draw_food)

        for x,y in food_nz_coords:
            try:
                foods = self.grid.food[x, y]
                s = foods[FoodType.SUGAR.value]; p = foods[FoodType.PROTEIN.value]; total = s + p
                color = MAP_BG_COLOR
                if total > 0.01:
                     sr = s / total; pr = p / total;
                     s_col = FOOD_COLORS[FoodType.SUGAR]; p_col = FOOD_COLORS[FoodType.PROTEIN]
                     r = int(s_col[0] * sr + p_col[0] * pr)
                     g = int(s_col[1] * sr + p_col[1] * pr)
                     b = int(s_col[2] * sr + p_col[2] * pr)
                     color = (r, g, b)
                rect = (x * cs, y * cs, cs, cs); pygame.draw.rect(self.screen, color, rect)
            except IndexError: continue

        # 4. Nest Area Highlight
        r = NEST_RADIUS; nx, ny = tuple(map(int, NEST_POS));
        nest_rect_coords = ((nx - r) * cs, (ny - r) * cs, (r * 2 + 1) * cs, (r * 2 + 1) * cs)
        try:
            rect = pygame.Rect(nest_rect_coords)
            nest_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            nest_surf.fill((100, 100, 100, 30))
            self.screen.blit(nest_surf, rect.topleft)
        except ValueError as e:
            print(f"Error creating nest rect surface {nest_rect_coords}: {e}")


    def _draw_brood(self):
        brood_copy=list(self.brood);
        for item in brood_copy:
             if item in self.brood and is_valid(item.pos):
                 item.draw(self.screen)

    def _draw_queen(self):
        if not self.queen or not is_valid(self.queen.pos): return
        pos_px = (int(self.queen.pos[0] * CELL_SIZE + CELL_SIZE / 2),
                  int(self.queen.pos[1] * CELL_SIZE + CELL_SIZE / 2))
        radius = int(CELL_SIZE / 1.5);
        pygame.draw.circle(self.screen, self.queen.color, pos_px, radius);
        pygame.draw.circle(self.screen, (255, 255, 255), pos_px, radius, 1)

    def _draw_entities(self):
        cs_half = CELL_SIZE / 2
        # Ants
        ants_copy = list(self.ants)
        for a in ants_copy:
             if a not in self.ants or not is_valid(a.pos): continue
             pos_px = (int(a.pos[0] * CELL_SIZE + cs_half), int(a.pos[1] * CELL_SIZE + cs_half))
             radius = int(CELL_SIZE / a.size_factor)
             color = a.search_color if a.state in [AntState.SEARCHING, AntState.PATROLLING, AntState.DEFENDING] else a.return_color
             if a.state == AntState.ESCAPING: color = WORKER_ESCAPE_COLOR
             pygame.draw.circle(self.screen, color, pos_px, radius)
             if a.carry_amount > 0:
                 food_color = FOOD_COLORS.get(a.carry_type, FOOD_COLOR_MIX)
                 pygame.draw.circle(self.screen, food_color, pos_px, int(radius * 0.6))

        # Enemies
        enemies_copy = list(self.enemies)
        for e in enemies_copy:
             if e not in self.enemies or not is_valid(e.pos): continue
             pos_px = (int(e.pos[0] * CELL_SIZE + cs_half), int(e.pos[1] * CELL_SIZE + cs_half))
             radius = int(CELL_SIZE / 2.2)
             pygame.draw.circle(self.screen, e.color, pos_px, radius);
             pygame.draw.circle(self.screen, (0, 0, 0), pos_px, radius, 1)


    def _draw_buttons(self):
        """Draws the +/- speed control buttons."""
        if not self.font: return
        mouse_pos = pygame.mouse.get_pos()

        for button in self.buttons:
            rect = button['rect']
            text = button['text']

            # Determine button color (hover or default)
            color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else BUTTON_COLOR

            # Draw the button rectangle
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

            # Render and draw the button text (+ or -)
            try:
                text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)
            except Exception as e:
                print(f"Button font render error ({text}): {e}")


    def handle_events(self):
        """Process Pygame events (Quit, Keyboard, Mouse Clicks). Returns action if needed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.simulation_running = False
                self.app_running = False # Signal to exit the main application loop
                self.end_game_reason = "Fenster geschlossen"
                return 'quit'

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.simulation_running = False # Stop current simulation run
                    self.end_game_reason = "ESC gedrückt"
                    return 'sim_stop' # Trigger end game dialog
                if event.key == pygame.K_d:
                    self.show_debug_info = not self.show_debug_info
                # Optional: Keyboard speed controls
                if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                     self._handle_button_click('speed_down')
                     return 'speed_change'
                if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                     self._handle_button_click('speed_up')
                     return 'speed_change'


            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    # Check speed +/- buttons only if simulation is running
                    if self.simulation_running:
                        for button in self.buttons:
                            if button['rect'].collidepoint(event.pos):
                                self._handle_button_click(button['action'])
                                return 'speed_change' # Indicate action handled
        return None # No quit or simulation stop action triggered

    def _handle_button_click(self, action):
        """Updates simulation speed based on button action (+/-)."""
        current_index = self.simulation_speed_index
        max_index = len(SPEED_MULTIPLIERS) - 1

        if action == 'speed_down':
            new_index = max(0, current_index - 1) # Decrease index, min 0
        elif action == 'speed_up':
            new_index = min(max_index, current_index + 1) # Increase index, max max_index
        else:
             print(f"Warn: Unknown button action '{action}'")
             return # Do nothing for unknown actions

        # Update index and target FPS only if the index changed
        if new_index != self.simulation_speed_index:
            self.simulation_speed_index = new_index
            self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
            # Debug print the new speed
            new_speed = SPEED_MULTIPLIERS[self.simulation_speed_index]
            speed_str = "Paused" if new_speed == 0.0 else f"{new_speed:.1f}x"
            print(f"Speed changed to: {speed_str} (Index: {self.simulation_speed_index}, Target FPS: {self.current_target_fps})")


    # --- End Game Dialog --- (Unchanged from previous version)
    def _show_end_game_dialog(self):
        """Displays the 'Restart' or 'Quit' dialog and handles input."""
        if not self.font:
             print("Error: Cannot show end dialog - font not loaded.")
             return 'quit' # Force quit if font is missing

        dialog_w = 300
        dialog_h = 150
        dialog_x = (WIDTH - dialog_w) // 2
        dialog_y = (HEIGHT - dialog_h) // 2

        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill(END_DIALOG_BG_COLOR) # Semi-transparent black background

        # Button properties
        btn_w, btn_h = 100, 30
        btn_margin = 20
        btn_y = dialog_y + dialog_h - btn_h - 20
        btn_restart_x = dialog_x + (dialog_w // 2) - btn_w - (btn_margin // 2)
        btn_quit_x = dialog_x + (dialog_w // 2) + (btn_margin // 2)

        restart_rect = pygame.Rect(btn_restart_x, btn_y, btn_w, btn_h)
        quit_rect = pygame.Rect(btn_quit_x, btn_y, btn_w, btn_h)

        # Text properties
        text_color = (240, 240, 240)
        title_text = f"Kolonie {self.colony_generation} Ende"
        reason_text = f"Grund: {self.end_game_reason}"

        waiting_for_choice = True
        while waiting_for_choice and self.app_running: # Check app_running too
            mouse_pos = pygame.mouse.get_pos()

            # --- Event Handling within Dialog ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.app_running = False # Exit main loop
                    waiting_for_choice = False
                    return 'quit'
                if event.type == pygame.KEYDOWN:
                     if event.key == pygame.K_ESCAPE: # Allow Esc to quit from dialog too
                         self.app_running = False
                         waiting_for_choice = False
                         return 'quit'
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        if restart_rect.collidepoint(mouse_pos):
                            waiting_for_choice = False
                            return 'restart'
                        if quit_rect.collidepoint(mouse_pos):
                            self.app_running = False # Exit main loop
                            waiting_for_choice = False
                            return 'quit'

            # --- Drawing the Dialog ---
            self.screen.blit(overlay, (0, 0)) # Draw overlay first
            pygame.draw.rect(self.screen, (40, 40, 80), (dialog_x, dialog_y, dialog_w, dialog_h), border_radius=5)

            try: # Render and draw text
                title_surf = self.font.render(title_text, True, text_color)
                title_rect = title_surf.get_rect(center=(dialog_x + dialog_w // 2, dialog_y + 30))
                self.screen.blit(title_surf, title_rect)
                reason_surf = self.font.render(reason_text, True, text_color)
                reason_rect = reason_surf.get_rect(center=(dialog_x + dialog_w // 2, dialog_y + 60))
                self.screen.blit(reason_surf, reason_rect)
            except Exception as e: print(f"Dialog text render error: {e}")

            # Draw Restart Button
            r_color = BUTTON_HOVER_COLOR if restart_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, r_color, restart_rect, border_radius=3)
            try:
                 r_text_surf = self.font.render("Neu starten", True, BUTTON_TEXT_COLOR)
                 r_text_rect = r_text_surf.get_rect(center=restart_rect.center)
                 self.screen.blit(r_text_surf, r_text_rect)
            except Exception as e: print(f"Restart Button render error: {e}")

            # Draw Quit Button
            q_color = BUTTON_HOVER_COLOR if quit_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, q_color, quit_rect, border_radius=3)
            try:
                 q_text_surf = self.font.render("Beenden", True, BUTTON_TEXT_COLOR)
                 q_text_rect = q_text_surf.get_rect(center=quit_rect.center)
                 self.screen.blit(q_text_surf, q_text_rect)
            except Exception as e: print(f"Quit Button render error: {e}")

            pygame.display.flip()
            self.clock.tick(30) # Lower FPS for menu

        return 'quit' # Default if loop exits unexpectedly


    def run(self):
        """Main application loop, handles simulation and end-game dialog."""
        print("Starting Ant Simulation - Complex Dynamics...")
        print("Press 'D' to toggle debug info overlay.")
        print("Press 'ESC' during simulation to end current run.")
        print("Use +/- buttons or keyboard +/- for speed control.")

        while self.app_running:
            # --- Simulation Phase ---
            while self.simulation_running and self.app_running:
                action = self.handle_events()

                if not self.app_running: break
                if action == 'sim_stop': break

                # --- Update simulation state if not paused ---
                current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]
                if current_multiplier > 0.0:
                    self.update() # update() now handles internal scaling

                # --- Draw current state ---
                self.draw()

                # --- Control frame rate ---
                # Use the target FPS based on the current speed index
                self.clock.tick(self.current_target_fps)

            # --- End Game / Dialog Phase ---
            if not self.app_running: break

            if not self.end_game_reason: self.end_game_reason = "Unbekannt"
            choice = self._show_end_game_dialog()

            if choice == 'restart':
                self._reset_simulation()
            elif choice == 'quit':
                self.app_running = False

        # --- Cleanup ---
        print("Exiting application.")
        try:
            pygame.quit()
            print("Pygame shut down.")
        except Exception as e:
            print(f"Error during Pygame quit: {e}")


# --- Start Simulation ---
if __name__ == '__main__':
    try: import numpy; print(f"NumPy version: {numpy.__version__}")
    except ImportError: print("FATAL: NumPy required."); input("Exit."); exit()
    try: import pygame; print(f"Pygame version: {pygame.version.ver}")
    except ImportError as e: print(f"FATAL: Pygame import failed: {e}"); input("Exit."); exit()
    except Exception as e: print(f"FATAL: Pygame import error: {e}"); input("Exit."); exit()

    try:
        pygame.init()
        if not pygame.display.get_init(): raise RuntimeError("Display module failed")
        if not pygame.font.get_init(): raise RuntimeError("Font module failed")
        print("Pygame initialized successfully.")
    except Exception as e:
         print(f"FATAL: Pygame initialization failed: {e}"); pygame.quit(); input("Exit."); exit()

    try:
        simulation = AntSimulation()
        if simulation.app_running: # Check if font loading succeeded in init
            simulation.run()
        else:
             print("Application cannot start due to initialization errors (e.g., fonts).")
    except Exception as e:
        print("\n--- UNHANDLED EXCEPTION CAUGHT ---")
        import traceback
        traceback.print_exc()
        print("------------------------------------")
        print("An critical error occurred during simulation execution.")
        pygame.quit(); input("Press Enter to Exit.")

    print("Simulation process finished.")
