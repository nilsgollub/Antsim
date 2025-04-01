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

# Simulation Speed Control
SPEED_LEVEL_NORMAL = 1
SPEED_LEVELS = {
    0: 1,     # Paused (run very slowly to keep UI responsive)
    1: 40,    # Normal speed
    2: 80,    # Fast (2x)
    3: 160    # Very Fast (4x)
}
MAX_SPEED_LEVEL = max(SPEED_LEVELS.keys())

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
BUTTON_ACTIVE_COLOR = (120, 120, 220)
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
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
    except (TypeError, IndexError):
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
        self.progress_timer += 1

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
                    self.progress_timer -= 1 # Pause progress

        if self.progress_timer >= self.duration:
            if self.stage == BroodStage.EGG:
                self.stage = BroodStage.LARVA
                self.progress_timer = 0
                self.duration = LARVA_DURATION
                self.color = LARVA_COLOR
                self.radius = CELL_SIZE // 4
                return None
            elif self.stage == BroodStage.LARVA:
                self.stage = BroodStage.PUPA
                self.progress_timer = 0
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

        # --- MODIFICATION: Remove random offset for static drawing ---
        # Calculate the center of the cell
        center_x = int(self.pos[0]) * CELL_SIZE + CELL_SIZE // 2
        center_y = int(self.pos[1]) * CELL_SIZE + CELL_SIZE // 2

        # Use a surface slightly larger than the circle for anti-aliasing effect if needed
        # But for simplicity, draw directly onto the main surface
        draw_pos = (center_x, center_y)

        # Draw the main circle
        pygame.draw.circle(surface, self.color, draw_pos, self.radius)

        # Draw outline for pupa
        if self.stage == BroodStage.PUPA:
            o_col = ((50, 50, 50) if self.caste == AntCaste.WORKER
                     else (100, 0, 0))
            pygame.draw.circle(surface, o_col, draw_pos, self.radius, 1)
        # --- END MODIFICATION ---


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
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                p=(NEST_POS[0] + dx, NEST_POS[1] + dy)
                if is_valid(p): nest_area.add(p)
        for _ in range(NUM_OBSTACLES):
            attempts=0; placed=False
            while attempts<50 and not placed:
                w=rnd(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                h=rnd(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                x=rnd(0, GRID_WIDTH - w - 1); y=rnd(0, GRID_HEIGHT - h - 1)
                overlaps=False
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        # Use integer coordinates for check
                        if (int(i), int(j)) in nest_area: overlaps=True; break
                    if overlaps: break
                if not overlaps: self.obstacles[x : x + w, y : y + h] = True; placed = True
                attempts += 1

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
                 target[x, y] = min(target[x, y] + amount, max_val)
        except IndexError: pass # Should not happen with bounds check

    def update_pheromones(self):
        # Decay
        self.pheromones_home *= PHEROMONE_DECAY
        self.pheromones_food *= PHEROMONE_DECAY
        self.pheromones_alarm *= PHEROMONE_DECAY
        self.pheromones_negative *= NEGATIVE_PHEROMONE_DECAY
        self.pheromones_recruitment *= RECRUITMENT_PHEROMONE_DECAY
        # Diffusion
        mask = ~self.obstacles
        arrays_rates = [
            (self.pheromones_home, PHEROMONE_DIFFUSION_RATE),
            (self.pheromones_food, PHEROMONE_DIFFUSION_RATE),
            (self.pheromones_alarm, PHEROMONE_DIFFUSION_RATE),
            (self.pheromones_negative, NEGATIVE_PHEROMONE_DIFFUSION_RATE),
            (self.pheromones_recruitment, RECRUITMENT_PHEROMONE_DIFFUSION_RATE)
        ]
        for arr, rate in arrays_rates:
            if rate > 0:
                masked = arr * mask; pad = np.pad(masked, 1, mode='constant')
                neighbors = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                             pad[1:-1, :-2] + pad[1:-1, 2:] +
                             pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:])
                diffused = masked * (1 - rate) + (neighbors / 8.0) * rate
                arr[:] = np.where(mask, diffused, 0)
        # Clipping & Zeroing
        min_ph = 0.01
        all_arrays = [
            self.pheromones_home, self.pheromones_food, self.pheromones_alarm,
            self.pheromones_negative, self.pheromones_recruitment
        ]
        for arr in all_arrays:
            max_val = RECRUITMENT_PHEROMONE_MAX if arr is self.pheromones_recruitment else PHEROMONE_MAX
            np.clip(arr, 0, max_val, out=arr)
            arr[arr < min_ph] = 0
            arr[self.obstacles] = 0


# --- Entity Classes ---
class Ant:
    """Represents a worker or soldier ant."""

    def __init__(self, pos, simulation, caste: AntCaste):
        # Ensure position is integer tuple
        self.pos = tuple(map(int, pos)); self.simulation = simulation; self.caste = caste
        attrs = ANT_ATTRIBUTES[caste]
        self.hp = attrs["hp"]; self.max_hp = attrs["hp"]
        self.attack_power = attrs["attack"]; self.max_capacity = attrs["capacity"]
        self.move_delay = attrs["speed_delay"]; self.search_color = attrs["color"]
        self.return_color = attrs["return_color"]
        self.food_consumption_sugar = attrs["food_consumption_sugar"]
        self.food_consumption_protein = attrs["food_consumption_protein"]
        self.size_factor = attrs["size_factor"]

        self.state = AntState.SEARCHING; self.carry_amount = 0.0; self.carry_type = None
        self.age = 0; self.max_age = int(rnd_gauss(WORKER_MAX_AGE_MEAN, WORKER_MAX_AGE_STDDEV))
        # Use integer tuples for path history
        self.path_history = []; self.history_timestamps = []; self.move_delay_timer = 0
        self.last_move_direction = (0, 0); self.stuck_timer = 0; self.escape_timer = 0
        self.last_move_info = "Born"; self.just_picked_food = False
        self.food_consumption_timer = rnd(0, WORKER_FOOD_CONSUMPTION_INTERVAL)
        self.last_known_alarm_pos = None

    def _update_path_history(self, new_pos):
        t = self.simulation.ticks
        # Ensure position is integer tuple
        self.path_history.append(tuple(map(int, new_pos)))
        self.history_timestamps.append(t)
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
        is_near_nest_now = distance_sq(pos_int, NEST_POS) <= (NEST_RADIUS + 1)**2

        for n_pos_float in potential_neighbors:
            # Work with integer coordinates for checks
            n_pos = tuple(map(int, n_pos_float))

            history_block = (
                (not ignore_history_near_nest and self._is_in_history(n_pos)) or
                (ignore_history_near_nest and is_near_nest_now and
                 self._is_in_history(n_pos) and
                 distance_sq(n_pos, NEST_POS) > NEST_RADIUS**2)
            )
            is_queen = (n_pos == q_pos_int)
            is_obs = self.simulation.grid.is_obstacle(n_pos)
            is_blocked_ant = self.simulation.is_ant_at(n_pos, exclude_self=self)

            if not history_block and not is_queen and not is_obs and not is_blocked_ant:
                # Return the original float/tuple position if valid
                valid.append(n_pos_float)
        return valid

    def _choose_move(self):
        """Determine the next move based on state, goals, and environment."""
        potential_neighbors = get_neighbors(self.pos)
        if not potential_neighbors:
            self.last_move_info = "No neighbors"
            return None

        ignore_hist = (self.state == AntState.RETURNING_TO_NEST)
        valid_neighbors = self._filter_valid_moves(potential_neighbors,
                                                   ignore_history_near_nest=ignore_hist)

        if not valid_neighbors:
            self.last_move_info = "Blocked"
            # Improved fallback logic
            fallback_neighbors = []
            q_pos_int = tuple(map(int, self.simulation.queen.pos)) if self.simulation.queen else None
            for n_pos_float in potential_neighbors:
                 n_pos = tuple(map(int, n_pos_float)) # Check with ints
                 if (n_pos != q_pos_int and
                     not self.simulation.grid.is_obstacle(n_pos) and
                     not self.simulation.is_ant_at(n_pos, exclude_self=self)):
                     fallback_neighbors.append(n_pos_float) # Add original if valid

            if fallback_neighbors:
                # Check history using integer tuples
                history_fallback = [p for p in fallback_neighbors
                                    if tuple(map(int, p)) in self.path_history]
                if history_fallback:
                     # Sort based on index in integer history
                     history_fallback.sort(key=lambda p: self.path_history.index(tuple(map(int,p))))
                     if is_valid(history_fallback[0]): return history_fallback[0]

                non_hist_fallback = [p for p in fallback_neighbors
                                     if tuple(map(int, p)) not in self.path_history]
                if non_hist_fallback: return random.choice(non_hist_fallback)
                elif fallback_neighbors: return random.choice(fallback_neighbors)
            return None # Truly stuck

        if self.state == AntState.ESCAPING:
            return random.choice(valid_neighbors) if valid_neighbors else None

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
        if move_dir == self.last_move_direction:
            score += W_PERSISTENCE
        score += rnd_uniform(-W_RANDOM_NOISE, W_RANDOM_NOISE)
        return score

    def _score_moves_returning(self, valid_neighbors, just_picked):
        scores = {}
        # Use integer position for distance calculation
        pos_int = tuple(map(int, self.pos))
        dist_sq_now = distance_sq(pos_int, NEST_POS)
        grid = self.simulation.grid
        for n_pos in valid_neighbors: # n_pos is original tuple (can be float)
            score = 0.0
            n_pos_int = tuple(map(int, n_pos)) # Use int for grid access
            home_ph = grid.get_pheromone(n_pos_int, 'home')
            food_ph = grid.get_pheromone(n_pos_int, 'food')
            alarm_ph = grid.get_pheromone(n_pos_int, 'alarm')

            score += home_ph * W_HOME_PHEROMONE_RETURN
            # Use int for distance check
            if distance_sq(n_pos_int, NEST_POS) < dist_sq_now:
                score += W_NEST_DIRECTION_RETURN

            if just_picked:
                score -= food_ph * W_FOOD_PHEROMONE_SEARCH * 0.5 # Avoid food source
                score += alarm_ph * W_ALARM_PHEROMONE * 0.3 # Less penalty
            else:
                score += alarm_ph * W_ALARM_PHEROMONE * 0.1 # Normal reduced penalty

            score += self._score_moves_base(n_pos) # Pass original tuple
            scores[n_pos] = score
        return scores

    def _score_moves_searching(self, valid_neighbors):
        scores = {}
        grid = self.simulation.grid
        for n_pos in valid_neighbors:
            score = 0.0
            n_pos_int = tuple(map(int, n_pos)) # Use int for grid access
            home_ph = grid.get_pheromone(n_pos_int, 'home')
            food_ph = grid.get_pheromone(n_pos_int, 'food')
            neg_ph = grid.get_pheromone(n_pos_int, 'negative')
            alarm_ph = grid.get_pheromone(n_pos_int, 'alarm')
            recr_ph = grid.get_pheromone(n_pos_int, 'recruitment')

            food_w = (W_FOOD_PHEROMONE_SEARCH if self.caste == AntCaste.WORKER
                      else W_FOOD_PHEROMONE_SEARCH * 0.2)
            score += food_ph * food_w
            # score += home_ph * W_HOME_PHEROMONE_SEARCH # W is 0
            score += neg_ph * W_NEGATIVE_PHEROMONE
            score += alarm_ph * W_ALARM_PHEROMONE
            score += recr_ph * W_RECRUITMENT_PHEROMONE

            # Use int for distance check
            if distance_sq(n_pos_int, NEST_POS) <= (NEST_RADIUS * 1.5)**2:
                 score += W_AVOID_NEST_SEARCHING # Stronger penalty

            score += self._score_moves_base(n_pos) # Pass original tuple
            scores[n_pos] = score
        return scores

    def _score_moves_patrolling(self, valid_neighbors):
        scores = {}
        grid = self.simulation.grid
        pos_int = tuple(map(int, self.pos)) # Use int for distance
        dist_sq_current = distance_sq(pos_int, NEST_POS)
        for n_pos in valid_neighbors:
            score = 0.0
            n_pos_int = tuple(map(int, n_pos)) # Use int for grid/distance
            neg_ph = grid.get_pheromone(n_pos_int, 'negative')
            alarm_ph = grid.get_pheromone(n_pos_int, 'alarm')
            recr_ph = grid.get_pheromone(n_pos_int, 'recruitment')

            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.5
            score += alarm_ph * W_ALARM_PHEROMONE * 0.5
            score += recr_ph * W_RECRUITMENT_PHEROMONE

            dist_sq_next = distance_sq(n_pos_int, NEST_POS)
            if dist_sq_current <= (NEST_RADIUS * 2)**2:
                 if dist_sq_next > dist_sq_current:
                     score -= W_NEST_DIRECTION_PATROL # W is negative -> push away

            if dist_sq_next > SOLDIER_PATROL_RADIUS_SQ:
                 score -= 10000 # Heavy penalty

            score += self._score_moves_base(n_pos) # Pass original tuple
            scores[n_pos] = score
        return scores

    def _score_moves_defending(self, valid_neighbors):
        scores = {}
        grid = self.simulation.grid
        pos_int = tuple(map(int, self.pos)) # Use int for calcs

        if self.last_known_alarm_pos is None or random.random() < 0.1: # Update target
            best_pos = None; max_alarm = -1; r_sq = 5*5
            x0, y0 = pos_int # Use int pos
            for i in range(x0 - 2, x0 + 3):
                 for j in range(y0 - 2, y0 + 3):
                      p = (i, j)
                      if distance_sq(pos_int, p) <= r_sq and is_valid(p):
                           alarm = grid.get_pheromone(p, 'alarm') # Use int pos p
                           if alarm > max_alarm:
                               max_alarm = alarm; best_pos = p # Store int pos p
            self.last_known_alarm_pos = best_pos # Store int pos

        for n_pos in valid_neighbors:
            score = 0.0
            n_pos_int = tuple(map(int, n_pos)) # Use int pos
            alarm_ph = grid.get_pheromone(n_pos_int, 'alarm')
            recr_ph = grid.get_pheromone(n_pos_int, 'recruitment')

            if self.last_known_alarm_pos: # This is an int tuple
                 dist_now = distance_sq(pos_int, self.last_known_alarm_pos)
                 dist_next = distance_sq(n_pos_int, self.last_known_alarm_pos)
                 if dist_next < dist_now:
                     score += W_ALARM_SOURCE_DEFEND

            score += alarm_ph * W_ALARM_PHEROMONE * -0.5 # Follow alarm up
            score += recr_ph * W_RECRUITMENT_PHEROMONE * 1.5 # Follow recruitment

            score += self._score_moves_base(n_pos) # Pass original tuple
            scores[n_pos] = score
        return scores

    def _select_best_move(self, move_scores, valid_neighbors):
        """Selects the move with the highest score (for DEFEND)."""
        best_score = -float('inf'); best_moves = []
        for pos, score in move_scores.items():
            if score > best_score: best_score = score; best_moves = [pos]
            elif score == best_score: best_moves.append(pos)

        if not best_moves:
            self.last_move_info += "(No best?)"
            return random.choice(valid_neighbors) if valid_neighbors else None

        chosen = random.choice(best_moves)
        score = move_scores.get(chosen, -999)
        # Display chosen position (can be float)
        self.last_move_info = f"Best->({chosen[0]:.1f},{chosen[1]:.1f}) (S:{score:.1f})"
        return chosen

    def _select_best_move_returning(self, move_scores, valid_neighbors, just_picked):
        """Selects the best move for returning, prioritizing nest direction."""
        best_score = -float('inf'); best_moves = []
        pos_int = tuple(map(int, self.pos)) # Use int for distance
        dist_sq_now = distance_sq(pos_int, NEST_POS)
        closer_moves = {}; other_moves = {}
        for pos, score in move_scores.items():
            # Use int for distance check
            if distance_sq(tuple(map(int, pos)), NEST_POS) < dist_sq_now:
                closer_moves[pos] = score
            else:
                other_moves[pos] = score

        target_pool = {}; selection_type = ""
        if closer_moves: target_pool = closer_moves; selection_type = "Closer"
        elif other_moves: target_pool = other_moves; selection_type = "Other"
        else:
            self.last_move_info += "(R: No moves?)"
            return random.choice(valid_neighbors) if valid_neighbors else None

        for pos, score in target_pool.items():
            if score > best_score: best_score = score; best_moves = [pos]
            elif score == best_score: best_moves.append(pos)

        if not best_moves:
            self.last_move_info += f"(R: No best in {selection_type})"
            target_pool = move_scores # Fallback to all scores
            best_score = -float('inf'); best_moves = []
            for pos, score in target_pool.items():
                 if score > best_score: best_score = score; best_moves = [pos]
                 elif score == best_score: best_moves.append(pos)
            if not best_moves: return random.choice(valid_neighbors) if valid_neighbors else None

        if len(best_moves) == 1:
            chosen = best_moves[0]
            self.last_move_info = f"R({selection_type})Best->({chosen[0]:.1f},{chosen[1]:.1f}) (S:{best_score:.1f})"
        else:
            # Tie-break by home pheromone (use int pos for grid access)
            best_moves.sort(key=lambda p: self.simulation.grid.get_pheromone(tuple(map(int,p)),'home'), reverse=True)
            max_ph = self.simulation.grid.get_pheromone(tuple(map(int,best_moves[0])),'home')
            top_ph = [p for p in best_moves if self.simulation.grid.get_pheromone(tuple(map(int,p)),'home') == max_ph]
            chosen = random.choice(top_ph)
            self.last_move_info = f"R({selection_type})TieBrk->({chosen[0]:.1f},{chosen[1]:.1f}) (S:{best_score:.1f})"
        return chosen

    def _select_probabilistic_move(self, move_scores, valid_neighbors):
        """Selects a move probabilistically based on scores."""
        pop=list(move_scores.keys()); scores=np.array(list(move_scores.values()))
        if len(pop)==0: return None

        min_s = np.min(scores) if scores.size>0 else 0
        # Add a small epsilon to avoid issues with zero scores after shift
        shifted = scores - min_s + 0.01 # Changed epsilon slightly
        # Clamp temperature effect to avoid extreme weights
        weights = np.power(shifted, min(max(PROBABILISTIC_CHOICE_TEMP, 0.1), 5.0))
        weights = np.maximum(MIN_SCORE_FOR_PROB_CHOICE, weights)
        total = np.sum(weights)

        # Handle cases where total is zero or invalid
        if total <= 1e-9 or not np.isfinite(total):
            self.last_move_info += f"({self.state.name[:3]}:LowW/Inv)"
            return random.choice(valid_neighbors) if valid_neighbors else None

        probs = weights / total
        # Check for NaN/inf in probs and handle normalization issues
        if not np.all(np.isfinite(probs)):
            self.last_move_info += "(ProbNaN/Inf)"
            # Fallback: Equal probability
            if valid_neighbors: return random.choice(valid_neighbors)
            else: return None

        # Renormalize if needed due to potential floating point inaccuracies
        sum_probs = np.sum(probs)
        if abs(sum_probs - 1.0) > 1e-6:
            if sum_probs > 1e-9: probs /= sum_probs
            else: # If sum is still near zero, fallback to equal probability
                 self.last_move_info += "(ProbSumLow)"
                 if valid_neighbors: return random.choice(valid_neighbors)
                 else: return None

        # Final check on probability sum
        if abs(np.sum(probs) - 1.0) > 1e-6:
             self.last_move_info += "(ProbSumErrFinal)"
             if valid_neighbors: return random.choice(valid_neighbors)
             else: return None

        try:
            idx = np.random.choice(len(pop), p=probs)
            chosen = pop[idx]
            score = move_scores.get(chosen, -999)
            self.last_move_info = f"{self.state.name[:3]} Prob->({chosen[0]:.1f},{chosen[1]:.1f}) (S:{score:.1f})"
            return chosen
        except ValueError as e:
            # This often happens if probs don't sum to 1 *exactly* due to float issues
            print(f"Err choices ({self.state}): {e}. Sum={np.sum(probs)}, Probs={probs}")
            self.last_move_info += "(ProbValErr)"
            # Fallback to max score if possible, otherwise random
            max_s = -float('inf')
            best_choice = None
            for p, s in move_scores.items():
                if s > max_s: max_s=s; best_choice=p
            if best_choice: return best_choice
            elif valid_neighbors: return random.choice(valid_neighbors)
            else: return None

    def update(self):
        """Update ant's state, position, age, food, and interactions."""
        self.age += 1

        # Food Consumption
        self.food_consumption_timer += 1
        if self.food_consumption_timer >= WORKER_FOOD_CONSUMPTION_INTERVAL:
            self.food_consumption_timer = 0
            needed_s = self.food_consumption_sugar
            needed_p = self.food_consumption_protein
            sim = self.simulation
            if sim.colony_food_storage_sugar >= needed_s and sim.colony_food_storage_protein >= needed_p:
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p
            else:
                self.hp = 0
                self.last_move_info = "Starved"
                return # Dies next cycle

        # Handle escape state countdown
        if self.state == AntState.ESCAPING:
            self.escape_timer -= 1
            if self.escape_timer <= 0:
                next_state = (AntState.PATROLLING if self.caste == AntCaste.SOLDIER
                              else AntState.SEARCHING)
                self.state = next_state
                self.last_move_info = f"EscapeEnd->{next_state.name[:3]}"
                # Don't clear path history here, might be useful

        self._update_state() # Update state (Patrolling/Defending) - AFTER escape check

        # Combat Check
        pos_int = tuple(map(int, self.pos)) # Use int pos
        # Check neighbors using integer coordinates
        neighbors_int = get_neighbors(pos_int, True)
        enemies = [e for p_int in neighbors_int
                   if (e := self.simulation.get_enemy_at(p_int)) and e.hp > 0]
        if enemies:
            target = random.choice(enemies)
            self.attack(target)
            # Add pheromone at integer position
            self.simulation.grid.add_pheromone(pos_int, P_ALARM_FIGHT, 'alarm')
            self.stuck_timer = 0
            target_pos_int = tuple(map(int, target.pos)) # Use int pos
            self.last_move_info = f"Fighting {self.caste.name} vs {target_pos_int}"
            return # Don't move if fighting

        # Movement Delay & Execution
        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return
        self.move_delay_timer = self.move_delay

        old_pos = self.pos # Store original (float?) tuple
        local_just_picked = self.just_picked_food
        self.just_picked_food = False # Reset before decision

        new_pos_cand = self._choose_move() # Returns original tuple type
        moved = False; found_food_type = None; food_amount = 0.0

        if new_pos_cand and tuple(map(int, new_pos_cand)) != tuple(map(int, old_pos)):
            target = new_pos_cand # Keep original type
            self.pos = target
            # Calculate direction based on integer positions
            old_pos_int = tuple(map(int, old_pos))
            target_int = tuple(map(int, target))
            self.last_move_direction = (target_int[0] - old_pos_int[0], target_int[1] - old_pos_int[1])
            self._update_path_history(target_int) # Use int pos for history
            self.stuck_timer = 0
            moved = True
            try: # Check food at new integer position
                x_int, y_int = target_int
                foods = self.simulation.grid.food[x_int, y_int]
                if foods[FoodType.SUGAR.value] > 0.1:
                    found_food_type = FoodType.SUGAR
                    food_amount = foods[FoodType.SUGAR.value]
                elif foods[FoodType.PROTEIN.value] > 0.1:
                    found_food_type = FoodType.PROTEIN
                    food_amount = foods[FoodType.PROTEIN.value]
            except (IndexError, TypeError):
                found_food_type = None # Safety
        else:
            self.stuck_timer += 1
            if not new_pos_cand and not moved: self.last_move_info += "(NoChoice)"
            elif not moved: self.last_move_info += "(StayedPut)"

        # Post-Movement Actions
        pos_int = tuple(map(int, self.pos)) # Use int pos
        is_near_nest = distance_sq(pos_int, NEST_POS) <= NEST_RADIUS**2
        grid = self.simulation.grid
        sim = self.simulation

        if self.state == AntState.SEARCHING:
            if (self.caste == AntCaste.WORKER and found_food_type and
                    self.carry_amount == 0):
                pickup = min(self.max_capacity, food_amount)
                self.carry_amount = pickup
                self.carry_type = found_food_type
                food_idx = found_food_type.value
                try:
                    x, y = pos_int # Use int pos for grid access
                    grid.food[x, y, food_idx] -= pickup
                    grid.food[x, y, food_idx] = max(0, grid.food[x, y, food_idx])
                    grid.add_pheromone(pos_int, P_FOOD_AT_SOURCE, 'food')
                    if food_amount >= RICH_FOOD_THRESHOLD:
                        grid.add_pheromone(pos_int, P_RECRUIT_FOOD, 'recruitment')
                    self.state = AntState.RETURNING_TO_NEST
                    self._clear_path_history()
                    self.last_move_info = f"Picked {found_food_type.name}({pickup:.1f})"
                    self.just_picked_food = True # Set flag AFTER state change
                except (IndexError, TypeError):
                    self.carry_amount = 0; self.carry_type = None
            elif (moved and not found_food_type and
                  distance_sq(pos_int, NEST_POS) > (NEST_RADIUS + 2)**2):
                 # Drop negative pheromone at previous integer position
                 old_pos_int = tuple(map(int, old_pos))
                 grid.add_pheromone(old_pos_int, P_NEGATIVE_SEARCH, 'negative')

        elif self.state == AntState.RETURNING_TO_NEST:
            if is_near_nest: # Checked using int pos
                dropped = self.carry_amount; type_dropped = self.carry_type
                if dropped > 0 and type_dropped:
                    if type_dropped == FoodType.SUGAR: sim.colony_food_storage_sugar += dropped
                    elif type_dropped == FoodType.PROTEIN: sim.colony_food_storage_protein += dropped
                    self.carry_amount = 0; self.carry_type = None
                next_state = (AntState.PATROLLING if self.caste == AntCaste.SOLDIER
                              else AntState.SEARCHING)
                self.state = next_state
                self._clear_path_history()
                type_str = f" {type_dropped.name}" if type_dropped else ""
                next_s_str = next_state.name[:3]
                self.last_move_info = (f"Dropped{type_str}({dropped:.1f})->{next_s_str}"
                                       if dropped > 0 else f"NestEmpty->{next_s_str}")
            elif moved and not local_just_picked:
                 # Drop trails only outside nest radius (use int pos) and not immediately after pickup
                 old_pos_int = tuple(map(int, old_pos))
                 if distance_sq(old_pos_int, NEST_POS) > NEST_RADIUS**2:
                    grid.add_pheromone(old_pos_int, P_HOME_RETURNING, 'home')
                    if self.carry_amount > 0:
                        grid.add_pheromone(old_pos_int, P_FOOD_RETURNING_TRAIL, 'food')

        # Stuck Check
        if (self.stuck_timer >= WORKER_STUCK_THRESHOLD and
                self.state != AntState.ESCAPING):
            # Check neighbors using integer coordinates
            neighbors_int = get_neighbors(pos_int, True)
            is_fighting = any(sim.get_enemy_at(p_int) for p_int in neighbors_int)

            if not is_fighting:
                self.state = AntState.ESCAPING
                self.escape_timer = WORKER_ESCAPE_DURATION
                self.stuck_timer = 0
                self._clear_path_history() # Clear history when starting escape
                self.last_move_info = "Stuck->Escaping"

    def _update_state(self):
        """Handle automatic state transitions (mainly for Soldiers)."""
        if (self.caste != AntCaste.SOLDIER or
                self.state in [AntState.ESCAPING, AntState.RETURNING_TO_NEST]):
            return

        pos_int = tuple(map(int, self.pos)) # Use int pos
        max_alarm = 0; r_sq = 5*5; grid = self.simulation.grid
        x0, y0 = pos_int
        # Sense local alarm level using integer coordinates
        for i in range(x0 - 2, x0 + 3):
            for j in range(y0 - 2, y0 + 3):
                p = (i, j)
                if distance_sq(pos_int, p) <= r_sq and is_valid(p):
                    max_alarm = max(max_alarm, grid.get_pheromone(p, 'alarm'))

        is_near_nest = distance_sq(pos_int, NEST_POS) <= SOLDIER_PATROL_RADIUS_SQ

        # State transition logic
        if max_alarm > SOLDIER_DEFEND_ALARM_THRESHOLD:
            if self.state != AntState.DEFENDING:
                self.state = AntState.DEFENDING
                self._clear_path_history()
                self.last_known_alarm_pos = None # Reset target when entering defend
                self.last_move_info += " ->DEFEND"
        elif self.state == AntState.DEFENDING:
            # If alarm subsided, return to patrolling
            self.state = AntState.PATROLLING
            self.last_move_info += " ->PATROL"
        elif is_near_nest and self.state != AntState.PATROLLING:
            # If near nest and no major alarm, start patrolling
            self.state = AntState.PATROLLING
            self.last_move_info += " ->PATROL(Near)"
        elif not is_near_nest and self.state == AntState.PATROLLING:
            # If wandered too far while patrolling, switch to searching
            self.state = AntState.SEARCHING
            self.last_move_info += " ->SEARCH(Far)"


    def attack(self, target_enemy):
        target_enemy.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        if self.hp <= 0: return
        self.hp -= amount
        if self.hp > 0:
            grid = self.simulation.grid
            pos_int = tuple(map(int, self.pos)) # Use int pos
            grid.add_pheromone(pos_int, P_ALARM_FIGHT / 2, 'alarm')
            recruit = (P_RECRUIT_DAMAGE_SOLDIER if self.caste == AntCaste.SOLDIER
                       else P_RECRUIT_DAMAGE)
            grid.add_pheromone(pos_int, recruit, 'recruitment')


# --- Queen Class ---
class Queen:
    """Manages queen state and egg laying based on food types."""
    def __init__(self, pos, sim):
        # Ensure position is integer tuple
        self.pos = tuple(map(int, pos))
        self.simulation = sim
        self.hp = QUEEN_HP; self.max_hp = QUEEN_HP
        self.age = 0; self.max_age = float('inf')
        self.egg_lay_timer = rnd(0, QUEEN_EGG_LAY_RATE)
        self.color = QUEEN_COLOR
        self.state = None; self.attack_power = 0; self.carry_amount = 0

    def update(self):
        self.age += 1
        self.egg_lay_timer += 1
        if self.egg_lay_timer >= QUEEN_EGG_LAY_RATE:
            needed_s = QUEEN_FOOD_PER_EGG_SUGAR
            needed_p = QUEEN_FOOD_PER_EGG_PROTEIN
            sim = self.simulation
            if (sim.colony_food_storage_sugar >= needed_s and
                    sim.colony_food_storage_protein >= needed_p):
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p
                self.egg_lay_timer = 0
                caste = self._decide_caste()
                egg_pos = self._find_egg_position() # Returns int pos
                if egg_pos: # Ensure a valid position was found
                    new_brood = BroodItem(BroodStage.EGG, caste, egg_pos, sim.ticks)
                    sim.brood.append(new_brood)

    def _decide_caste(self):
        ratio = 0.0; ants = self.simulation.ants; brood = self.simulation.brood
        # Consider ants + larvae/pupae for ratio calculation
        total = len(ants) + sum(1 for b in brood if b.stage != BroodStage.EGG)
        if total > 0:
            soldiers = (sum(1 for a in ants if a.caste == AntCaste.SOLDIER) +
                        sum(1 for b in brood if b.caste == AntCaste.SOLDIER and
                            b.stage != BroodStage.EGG))
            ratio = soldiers / total
        # Decision logic
        if ratio < QUEEN_SOLDIER_RATIO_TARGET:
            return AntCaste.SOLDIER if random.random() < 0.6 else AntCaste.WORKER
        elif random.random() < 0.03:
            return AntCaste.SOLDIER
        return AntCaste.WORKER

    def _find_egg_position(self):
        # Get integer neighbors of integer queen position
        valid = [p for p in get_neighbors(self.pos)
                 if not self.simulation.grid.is_obstacle(p)] # get_neighbors returns int
        # Prefer positions without other brood items (compare int positions)
        free_valid = [p for p in valid
                      if not any(b.pos == p for b in self.simulation.brood)]
        if free_valid: return random.choice(free_valid)
        elif valid: return random.choice(valid)
        else: return self.pos # Fallback (already int)

    def take_damage(self, amount, attacker):
        if self.hp <= 0: return
        self.hp -= amount
        if self.hp > 0:
            grid = self.simulation.grid
            # Use integer position
            grid.add_pheromone(self.pos, P_ALARM_FIGHT * 2, 'alarm')
            grid.add_pheromone(self.pos, P_RECRUIT_DAMAGE * 2, 'recruitment')


# --- Enemy Class ---
class Enemy:
    """Represents an enemy entity."""
    def __init__(self, pos, sim):
         # Ensure position is integer tuple
        self.pos = tuple(map(int, pos)); self.simulation = sim; self.hp = ENEMY_HP
        self.max_hp = ENEMY_HP; self.attack_power = ENEMY_ATTACK
        self.move_delay_timer = rnd(0, ENEMY_MOVE_DELAY); self.color = ENEMY_COLOR

    def update(self):
        pos_int = self.pos # Position is already int
        # Combat Check - Check integer neighbors
        neighbors_int = get_neighbors(pos_int, True)
        targets = [a for p_int in neighbors_int
                   if (a := self.simulation.get_ant_at(p_int)) and a.hp > 0]
        if targets:
            target = random.choice(targets); self.attack(target); return # Don't move if fighting

        # Movement Delay & Execution
        if self.move_delay_timer > 0: self.move_delay_timer -= 1; return
        self.move_delay_timer = ENEMY_MOVE_DELAY

        # Get integer neighbors
        possible = get_neighbors(pos_int); valid = []
        for m_int in possible: # Neighbors are already int
            if (not self.simulation.grid.is_obstacle(m_int) and
                not self.simulation.is_enemy_at(m_int, self) and
                not self.simulation.is_ant_at(m_int)):
                valid.append(m_int) # Add valid integer position

        if valid:
            chosen = None
            # Use integer positions for distance calculation
            if random.random() < ENEMY_NEST_ATTRACTION:
                best = None; min_d = distance_sq(pos_int, NEST_POS)
                for move_int in valid:
                     d = distance_sq(move_int, NEST_POS)
                     if d < min_d: min_d = d; best = move_int
                chosen = best if best else random.choice(valid)
            else:
                chosen = random.choice(valid)

            if chosen: # chosen is an integer tuple
                 self.pos = chosen # Update position with the chosen int tuple

    def attack(self, target_ant):
        target_ant.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        self.hp -= amount


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

        # --- NEW: Colony Generation Counter ---
        self.colony_generation = 0 # Start at 0, reset will increment to 1 first time

        # Defer initialization of simulation state to _reset_simulation
        self.ticks = 0
        self.ants = []
        self.enemies = []
        self.brood = []
        self.queen = None
        self.colony_food_storage_sugar = 0.0
        self.colony_food_storage_protein = 0.0
        self.enemy_spawn_timer = 0
        self.show_debug_info = True
        self.simulation_speed_level = SPEED_LEVEL_NORMAL
        self.current_target_fps = SPEED_LEVELS[self.simulation_speed_level]
        self.buttons = self._create_buttons()

        # Call reset to perform initial setup and start first generation
        self._reset_simulation()

    def _init_fonts(self):
        """Initialize fonts, handling potential errors."""
        try:
            # Use a common, often available font first
            self.font = pygame.font.SysFont("sans", 16)
            self.debug_font = pygame.font.SysFont("monospace", 14) # Separate font for debug
            print("Using system 'sans' and 'monospace' fonts.")
        except Exception as e1:
            print(f"System font error: {e1}. Trying default font.")
            try:
                self.font = pygame.font.Font(None, 20) # Default pygame font, maybe larger
                self.debug_font = pygame.font.Font(None, 16)
                print("Using Pygame default font.")
            except Exception as e2:
                print(f"FATAL: Default font error: {e2}. Cannot render text.")
                self.font = None # Ensure it's None if failed
                self.debug_font = None
                # Optionally exit or handle the lack of font later
                # self.app_running = False

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
        self.enemy_spawn_timer = 0
        self.end_game_reason = ""

        # --- INCREMENT Colony Counter ---
        self.colony_generation += 1

        # Reset grid (places obstacles and food)
        self.grid.reset()

        # Spawn initial entities
        if not self._spawn_initial_entities():
             # Handle critical failure if entities can't spawn (e.g., queen pos invalid)
             print("CRITICAL ERROR during simulation reset. Cannot continue.")
             self.simulation_running = False
             self.app_running = False # Stop the whole app if reset fails critically
             self.end_game_reason = "Initialisierungsfehler"
             return

        # Set simulation state to running
        self.simulation_running = True
        print(f"Kolonie {self.colony_generation} gestartet.")


    def _create_buttons(self):
        """Creates data structures for UI buttons."""
        buttons = []
        button_h = 20; button_w = 60; margin = 5
        start_x = WIDTH - (button_w + margin) * 4
        actions = [('pause', 'Pause', 0), ('slow', 'Normal', 1), # Changed Slow label
                   ('fast', 'Fast', 2), ('faster', 'Faster', 3)] # Added Faster button action
        for i, (action, text, _) in enumerate(actions):
            rect = pygame.Rect(start_x + i * (button_w + margin), margin, button_w, button_h)
            buttons.append({'rect': rect, 'text': text, 'action': action})
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
        for p in get_neighbors(base):
             if not self.grid.is_obstacle(p): return p # Returns int pos

        # Check slightly further out if immediate neighbors fail
        for r in range(2, 5): # Increase search radius slightly
             for dx in range(-r, r + 1):
                 for dy in range(-r, r + 1):
                     # Check only the perimeter of the radius r box
                     if abs(dx) != r and abs(dy) != r: continue
                     p = (base[0] + dx, base[1] + dy)
                     if is_valid(p) and not self.grid.is_obstacle(p): return p # Returns int pos

        print("CRITICAL: Could not find ANY valid spot near nest center for Queen.")
        return None # Indicate failure

    def add_ant(self, pos, caste: AntCaste):
        """Create and add a new ant of a specific caste if position is valid (expects int pos)."""
        pos_int = tuple(map(int, pos)) # Ensure integer tuple
        if not is_valid(pos_int): return False
        if (not self.grid.is_obstacle(pos_int) and
            not self.is_ant_at(pos_int) and
            not self.is_enemy_at(pos_int) and
            (not self.queen or pos_int != self.queen.pos)): # Don't spawn on queen
            self.ants.append(Ant(pos_int, self, caste)); return True # Pass int pos
        return False

    def spawn_enemy(self):
        """Spawn a new enemy at a valid random integer location."""
        tries = 0
        while tries < 50:
            # Generate integer position directly
            pos_i = (rnd(0,GRID_WIDTH-1), rnd(0,GRID_HEIGHT-1))
            q_pos_int = self.queen.pos if self.queen else tuple(map(int, NEST_POS))
            dist_ok = distance_sq(pos_i, q_pos_int) > (MIN_FOOD_DIST_FROM_NEST)**2

            if (not self.grid.is_obstacle(pos_i) and dist_ok and
                not self.is_enemy_at(pos_i) and not self.is_ant_at(pos_i)):
                self.enemies.append(Enemy(pos_i, self)); return True # Pass int pos
            tries += 1
        return False

    def kill_ant(self, ant_to_remove, reason="unknown"):
        """Remove an ant from the simulation."""
        if ant_to_remove in self.ants: self.ants.remove(ant_to_remove)

    def kill_enemy(self, enemy_to_remove):
        """Remove an enemy and potentially drop food."""
        if enemy_to_remove in self.enemies:
            pos_int = enemy_to_remove.pos # Position is already int
            if not self.grid.is_obstacle(pos_int): # Use int pos
                fx, fy = pos_int; grid = self.grid
                s_idx = FoodType.SUGAR.value; p_idx = FoodType.PROTEIN.value
                # Ensure indices are valid before accessing food array
                if 0 <= fx < GRID_WIDTH and 0 <= fy < GRID_HEIGHT:
                    grid.food[fx, fy, s_idx] = min(MAX_FOOD_PER_CELL, grid.food[fx, fy, s_idx] + ENEMY_TO_FOOD_ON_DEATH_SUGAR)
                    grid.food[fx, fy, p_idx] = min(MAX_FOOD_PER_CELL, grid.food[fx, fy, p_idx] + ENEMY_TO_FOOD_ON_DEATH_PROTEIN)
            self.enemies.remove(enemy_to_remove)

    def kill_queen(self, queen_to_remove):
        """Handle the death of the queen, stopping the current simulation run."""
        if self.queen == queen_to_remove:
            print(f"\n--- QUEEN DIED (Tick {self.ticks}, Kolonie {self.colony_generation}) ---")
            print(f"    Food S:{self.colony_food_storage_sugar:.1f} P:{self.colony_food_storage_protein:.1f}")
            print(f"    Ants:{len(self.ants)}, Brood:{len(self.brood)}")
            self.queen = None
            # --- MODIFICATION: Stop simulation, don't exit app ---
            self.simulation_running = False
            self.end_game_reason = "Königin gestorben"
        else:
            print("Warn: Attempted Kill inactive queen.")

    def is_ant_at(self, pos, exclude_self=None):
        """Check if an ant (worker, soldier, or queen) is at an integer position."""
        pos_i = tuple(map(int, pos)) # Ensure integer comparison
        q = self.queen
        # Queen's position is already int
        if (q and q.pos == pos_i and exclude_self != q): return True
        for a in self.ants:
            if a is exclude_self: continue
            # Ant's position is already int
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
        """Run one simulation tick."""
        # This method only runs if simulation_running is True

        self.ticks += 1

        # --- Pre-Update Checks (use integer positions) ---
        ants_to_remove = []
        for a in self.ants:
            pos_int = a.pos # Already int
            reason = ""
            if a.hp <= 0: reason = "hp <= 0"
            elif a.age > a.max_age: reason = f"aged out ({a.age}/{a.max_age})"
            elif self.grid.is_obstacle(pos_int): reason = f"in obstacle {pos_int}"
            if reason:
                 ants_to_remove.append((a, reason))

        enemies_to_remove = []
        for e in self.enemies:
            pos_int = e.pos # Already int
            reason = ""
            if e.hp <= 0: reason = "hp <= 0"
            elif self.grid.is_obstacle(pos_int): reason = f"in obstacle {pos_int}"
            if reason:
                 enemies_to_remove.append(e) # No reason needed for enemy removal message

        queen_remove = None
        if self.queen:
            pos_int = self.queen.pos # Already int
            reason = ""
            if self.queen.hp <= 0: reason = "hp <= 0"
            elif self.grid.is_obstacle(pos_int): reason = f"in obstacle {pos_int}"
            if reason:
                 queen_remove = self.queen

        # Perform removals after iteration
        for ant, reason in ants_to_remove:
             # print(f"Removing ant {ant.caste} at {ant.pos} reason: {reason}") # Debug Optional
             self.kill_ant(ant, reason)
        for enemy in enemies_to_remove:
             self.kill_enemy(enemy)
        if queen_remove:
             self.kill_queen(queen_remove) # This might set simulation_running to False

        # If queen died, stop further updates for this tick
        if not self.simulation_running: return

        # --- Update Entities ---
        if self.queen: self.queen.update()
        # Check again if queen died during her update (e.g., starvation check if added)
        if not self.simulation_running: return

        hatched=[]; brood_copy=list(self.brood)
        for item in brood_copy:
             # Ensure item is still in the main list before updating
             if item in self.brood:
                 hatch_signal = item.update(self.ticks, self)
                 if hatch_signal and hatch_signal in self.brood: # Check if it wasn't removed already
                     hatched.append(hatch_signal) # Should be the item itself

        for pupa in hatched:
            if pupa in self.brood: # Check again before removing/spawning
                 self.brood.remove(pupa)
                 self._spawn_hatched_ant(pupa.caste, pupa.pos) # Pass caste and position

        # Use copies for safe iteration while entities might be removed
        ants_copy=list(self.ants); enemies_copy=list(self.enemies)

        # Shuffle lists slightly to vary update order? Optional.
        # random.shuffle(ants_copy)
        # random.shuffle(enemies_copy)

        for a in ants_copy:
             if a in self.ants: # Check if ant wasn't killed mid-update
                 a.update()
        for e in enemies_copy:
            if e in self.enemies: # Check if enemy wasn't killed mid-update
                e.update()

        # Check for dead ants/enemies *after* their updates (e.g., starvation, combat results)
        # This is slightly redundant with pre-update checks but ensures effects within the tick are handled
        final_ants_invalid = [a for a in self.ants if a.hp <= 0]
        final_enemies_invalid = [e for e in self.enemies if e.hp <= 0]
        for a in final_ants_invalid: self.kill_ant(a, "post-update")
        for e in final_enemies_invalid: self.kill_enemy(e)
        if self.queen and self.queen.hp <= 0: self.kill_queen(self.queen)

        # If simulation stopped during updates, exit early
        if not self.simulation_running: return

        # Update Pheromones
        self.grid.update_pheromones()

        # Spawn new enemies periodically
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= ENEMY_SPAWN_RATE:
            self.enemy_spawn_timer = 0
            if len(self.enemies) < INITIAL_ENEMIES * 4: # Limit total enemies somewhat
                 self.spawn_enemy()

    def _spawn_hatched_ant(self, caste: AntCaste, pupa_pos: tuple):
        """Tries to spawn a hatched ant near the pupa's location."""
        # Try spawning exactly at the pupa's (integer) location first
        if self.add_ant(pupa_pos, caste):
            return True

        # If blocked, try neighbors of the pupa's location
        attempts = 0
        neighbors = get_neighbors(pupa_pos) # Gets int neighbors
        random.shuffle(neighbors) # Try neighbors in random order
        for pos in neighbors:
             if self.add_ant(pos, caste):
                 return True
             attempts += 1
             if attempts >= 5: break # Limit attempts near pupa

        # Fallback: Try spawning near queen (less ideal)
        if self.queen:
            base = self.queen.pos
            attempts = 0
            while attempts < 10:
                ox = rnd(-NEST_RADIUS, NEST_RADIUS); oy = rnd(-NEST_RADIUS, NEST_RADIUS)
                pos = (base[0] + ox, base[1] + oy) # int pos
                if self.add_ant(pos, caste): return True
                attempts += 1

        print(f"Warn: Failed hatch spawn {caste.name} near {pupa_pos}") # Debug
        return False


    def draw_debug_info(self):
        if not self.debug_font: return # Use the dedicated debug font
        ant_c=len(self.ants); enemy_c=len(self.enemies); brood_c=len(self.brood)
        food_s=self.colony_food_storage_sugar; food_p=self.colony_food_storage_protein
        tick=self.ticks; fps=self.clock.get_fps()
        w_c=sum(1 for a in self.ants if a.caste==AntCaste.WORKER); s_c=sum(1 for a in self.ants if a.caste==AntCaste.SOLDIER)
        e_c=sum(1 for b in self.brood if b.stage==BroodStage.EGG); l_c=sum(1 for b in self.brood if b.stage==BroodStage.LARVA); p_c=sum(1 for b in self.brood if b.stage==BroodStage.PUPA)

        # --- MODIFICATION: Add Kolonie Counter ---
        texts = [
            f"Kolonie: {self.colony_generation}", # Display colony generation
            f"Tick: {tick} FPS: {fps:.0f}",
            f"Ants: {ant_c} (W:{w_c} S:{s_c})",
            f"Brood: {brood_c} (E:{e_c} L:{l_c} P:{p_c})",
            f"Enemies: {enemy_c}",
            f"Food S:{food_s:.1f} P:{food_p:.1f}"
        ]
        y=5; col=(255,255,255); line_h = self.debug_font.get_height() + 1

        for i, txt in enumerate(texts):
            try:
                surf=self.debug_font.render(txt,True,col); self.screen.blit(surf,(5, y+i*line_h))
            except Exception as e: print(f"Debug Font render err: {e}")

        # --- Mouse hover ---
        try:
            mx,my=pygame.mouse.get_pos(); gx,gy=mx//CELL_SIZE,my//CELL_SIZE
            pos_f=(gx,gy) # Keep float for potential future use, but use int for checks
            pos_i=(gx,gy) # Integer position for grid access and entity checks

            if is_valid(pos_i):
                lines=[];
                entity=self.get_ant_at(pos_i) or self.get_enemy_at(pos_i) # Use int pos
                if entity:
                    entity_pos_int = entity.pos # Already int
                    if isinstance(entity,Queen): lines.extend([f"QUEEN @{entity_pos_int}", f"HP:{entity.hp}/{entity.max_hp}"])
                    elif isinstance(entity,Ant): lines.extend([f"{entity.caste.name}@{entity_pos_int}", f"S:{entity.state.name} HP:{entity.hp:.0f}", f"C:{entity.carry_amount:.1f}({entity.carry_type.name if entity.carry_type else '-'})", f"Age:{entity.age}", f"Mv:{entity.last_move_info[:25]}"])
                    elif isinstance(entity,Enemy): lines.extend([f"ENEMY @{entity_pos_int}", f"HP:{entity.hp}/{entity.max_hp}"])

                # Check for brood at integer position
                brood_at_pos=[b for b in self.brood if b.pos == pos_i]
                if brood_at_pos: lines.append(f"Brood:{len(brood_at_pos)} @{pos_i}");
                for b in brood_at_pos[:2]: lines.append(f"-{b.stage.name}({b.caste.name}) {b.progress_timer}/{b.duration}")

                obs=self.grid.is_obstacle(pos_i); obs_txt=" OBSTACLE" if obs else "" # Use int pos
                lines.append(f"Cell:{pos_i}{obs_txt}")

                if not obs:
                    try:
                        # Access grid data using integer position
                        foods=self.grid.food[pos_i[0],pos_i[1]]; food_txt=f"Food S:{foods[0]:.1f} P:{foods[1]:.1f}"
                        ph={t:self.grid.get_pheromone(pos_i,t) for t in ['home','food','alarm','negative','recruitment']} # Use int pos
                        ph1=f"Ph H:{ph['home']:.0f} F:{ph['food']:.0f}"; ph2=f"Ph A:{ph['alarm']:.0f} N:{ph['negative']:.0f} R:{ph['recruitment']:.0f}"
                        lines.extend([food_txt, ph1, ph2])
                    except IndexError: lines.append("Error reading cell data")

                hover_col=(255,255,0); y_off=HEIGHT-(len(lines)*line_h)-5
                for i, line in enumerate(lines):
                     surf=self.debug_font.render(line,True,hover_col); self.screen.blit(surf,(5,y_off+i*line_h))
        except Exception as e:
            # Print more details on error
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
        self._draw_buttons() # Draw speed buttons
        # Note: End game dialog is drawn separately in its own loop
        pygame.display.flip()

    def _draw_grid(self):
        # 1. BG & Obstacles
        bg=pygame.Surface((WIDTH,HEIGHT)); bg.fill(MAP_BG_COLOR)
        # Draw obstacles based on the boolean array
        obstacle_coords = np.argwhere(self.grid.obstacles)
        cs = CELL_SIZE
        for x, y in obstacle_coords:
            pygame.draw.rect(bg, OBSTACLE_COLOR, (x * cs, y * cs, cs, cs))
        self.screen.blit(bg,(0,0))

        # 2. Pheromones
        ph_types=['home','food','alarm','negative','recruitment']
        ph_colors={'home':PHEROMONE_HOME_COLOR, 'food':PHEROMONE_FOOD_COLOR, 'alarm':PHEROMONE_ALARM_COLOR, 'negative':PHEROMONE_NEGATIVE_COLOR, 'recruitment':PHEROMONE_RECRUITMENT_COLOR}
        ph_arrays={'home':self.grid.pheromones_home, 'food':self.grid.pheromones_food, 'alarm':self.grid.pheromones_alarm, 'negative':self.grid.pheromones_negative, 'recruitment':self.grid.pheromones_recruitment}
        min_draw_ph=0.5 # Min pheromone value to draw

        for ph_type in ph_types:
            ph_surf=pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
            base_col=ph_colors[ph_type]; arr=ph_arrays[ph_type]
            cur_max = RECRUITMENT_PHEROMONE_MAX if ph_type=='recruitment' else PHEROMONE_MAX
            # Use a lower normalization value for better visibility of lower pheromone levels
            norm_divisor = max(cur_max / 3.0, 1.0) # Normalize against 1/3rd of max

            # Find coordinates where pheromone > min_draw_ph
            nz_coords = np.argwhere(arr > min_draw_ph)

            for x,y in nz_coords:
                val = arr[x, y]
                # Normalize value (clamped between 0 and 1)
                norm_val = normalize(val, norm_divisor)
                # Alpha depends on normalized value (more intense for stronger pheromones)
                alpha = int(norm_val * base_col[3]) # base_col[3] is the max alpha for this type
                alpha = min(max(alpha, 0), 255) # Clamp alpha

                if alpha > 3: # Only draw if somewhat visible
                    color = (*base_col[:3], alpha)
                    pygame.draw.rect(ph_surf, color, (x * cs, y * cs, cs, cs))

            self.screen.blit(ph_surf, (0, 0))


        # 3. Food
        min_draw_food=0.1 # Min total food to draw cell color
        # Find coordinates where total food > min_draw_food
        food_totals = np.sum(self.grid.food, axis=2)
        food_nz_coords = np.argwhere(food_totals > min_draw_food)

        for x,y in food_nz_coords:
            try:
                foods = self.grid.food[x, y]
                s = foods[FoodType.SUGAR.value]; p = foods[FoodType.PROTEIN.value]; total = s + p
                color = MAP_BG_COLOR # Default if total is somehow zero despite check

                if total > 0.01: # Recalculate ratio for safety
                     sr = s / total; pr = p / total;
                     s_col = FOOD_COLORS[FoodType.SUGAR]; p_col = FOOD_COLORS[FoodType.PROTEIN]
                     # Mix colors based on ratio
                     r = int(s_col[0] * sr + p_col[0] * pr)
                     g = int(s_col[1] * sr + p_col[1] * pr)
                     b = int(s_col[2] * sr + p_col[2] * pr)
                     color = (r, g, b)

                # Draw food cell
                rect = (x * cs, y * cs, cs, cs); pygame.draw.rect(self.screen, color, rect)
            except IndexError: continue # Skip if coords somehow invalid

        # 4. Nest Area Highlight
        r = NEST_RADIUS; nx, ny = tuple(map(int, NEST_POS));
        # Calculate top-left corner and size based on integer center and radius
        nest_rect_coords = ((nx - r) * cs, (ny - r) * cs, (r * 2 + 1) * cs, (r * 2 + 1) * cs)
        try:
            rect = pygame.Rect(nest_rect_coords)
            # Create a surface for the overlay
            nest_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            nest_surf.fill((100, 100, 100, 30)) # Semi-transparent gray
            self.screen.blit(nest_surf, rect.topleft)
        except ValueError as e:
            print(f"Error creating nest rect surface {nest_rect_coords}: {e}")


    def _draw_brood(self):
        # Use a copy in case list changes during drawing (less likely but safer)
        brood_copy=list(self.brood);
        for item in brood_copy:
             # Check if item still exists and has valid position
             if item in self.brood and is_valid(item.pos):
                 item.draw(self.screen) # Draw method now handles static drawing

    def _draw_queen(self):
        if not self.queen or not is_valid(self.queen.pos): return
        # Queen pos is already int
        pos_px = (int(self.queen.pos[0] * CELL_SIZE + CELL_SIZE / 2),
                  int(self.queen.pos[1] * CELL_SIZE + CELL_SIZE / 2))
        radius = int(CELL_SIZE / 1.5);
        pygame.draw.circle(self.screen, self.queen.color, pos_px, radius);
        pygame.draw.circle(self.screen, (255, 255, 255), pos_px, radius, 1) # White outline

    def _draw_entities(self):
        cs_half = CELL_SIZE / 2
        # Ants
        ants_copy = list(self.ants)
        for a in ants_copy:
             if a not in self.ants or not is_valid(a.pos): continue
             # Ant pos is already int
             pos_px = (int(a.pos[0] * CELL_SIZE + cs_half), int(a.pos[1] * CELL_SIZE + cs_half))
             radius = int(CELL_SIZE / a.size_factor)
             color = a.search_color if a.state in [AntState.SEARCHING, AntState.PATROLLING, AntState.DEFENDING] else a.return_color
             if a.state == AntState.ESCAPING: color = WORKER_ESCAPE_COLOR
             pygame.draw.circle(self.screen, color, pos_px, radius)
             # Draw carried food indicator
             if a.carry_amount > 0:
                 food_color = FOOD_COLORS.get(a.carry_type, FOOD_COLOR_MIX)
                 pygame.draw.circle(self.screen, food_color, pos_px, int(radius * 0.6))

        # Enemies
        enemies_copy = list(self.enemies)
        for e in enemies_copy:
             if e not in self.enemies or not is_valid(e.pos): continue
             # Enemy pos is already int
             pos_px = (int(e.pos[0] * CELL_SIZE + cs_half), int(e.pos[1] * CELL_SIZE + cs_half))
             radius = int(CELL_SIZE / 2.2)
             pygame.draw.circle(self.screen, e.color, pos_px, radius);
             pygame.draw.circle(self.screen, (0, 0, 0), pos_px, radius, 1) # Black outline


    def _draw_buttons(self):
        """Draws the speed control buttons."""
        if not self.font: return # Use main UI font
        mouse_pos = pygame.mouse.get_pos()

        # Map button action to corresponding speed level
        action_level_map = {'pause': 0, 'slow': 1, 'fast': 2, 'faster': 3}

        for button in self.buttons:
            rect = button['rect']
            text = button['text']
            action = button['action']
            level = action_level_map.get(action) # Get speed level for this button

            color = BUTTON_COLOR
            # Highlight button if it corresponds to the current speed level
            if level is not None and self.simulation_speed_level == level:
                color = BUTTON_ACTIVE_COLOR
            elif rect.collidepoint(mouse_pos):
                color = BUTTON_HOVER_COLOR # Highlight if mouse is over it

            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            try:
                text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)
            except Exception as e:
                print(f"Button font render error: {e}")

    def handle_events(self):
        """Process Pygame events (Quit, Keyboard, Mouse Clicks). Returns action if needed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 # --- MODIFICATION: Don't just set running=False, set app_running=False ---
                self.simulation_running = False
                self.app_running = False # Signal to exit the main application loop
                self.end_game_reason = "Fenster geschlossen"
                return 'quit' # Indicate quit action

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                     # --- MODIFICATION: Stop simulation, don't exit app ---
                    self.simulation_running = False
                    self.end_game_reason = "ESC gedrückt"
                    return 'sim_stop' # Indicate simulation stop
                if event.key == pygame.K_d:
                    self.show_debug_info = not self.show_debug_info

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    # Check speed buttons only if simulation is running
                    if self.simulation_running:
                        for button in self.buttons:
                            if button['rect'].collidepoint(event.pos):
                                self._handle_button_click(button['action'])
                                return 'speed_change' # Indicate action handled
        return None # No quit or simulation stop action triggered

    def _handle_button_click(self, action):
        """Updates simulation speed based on button action."""
        level_map = {'pause': 0, 'slow': 1, 'fast': 2, 'faster': 3}
        new_level = level_map.get(action)

        if new_level is not None:
            self.simulation_speed_level = new_level
            # Clamp level just in case (redundant if actions are fixed, but safe)
            self.simulation_speed_level = max(0, min(self.simulation_speed_level, MAX_SPEED_LEVEL))
            self.current_target_fps = SPEED_LEVELS.get(self.simulation_speed_level, SPEED_LEVELS[SPEED_LEVEL_NORMAL])
            print(f"Speed set to level: {self.simulation_speed_level} (Target FPS: {self.current_target_fps})") # Debug
        else:
             print(f"Warn: Unknown button action '{action}'")


    # --- NEW: End Game Dialog ---
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

            # Draw dialog box background (optional)
            pygame.draw.rect(self.screen, (40, 40, 80), (dialog_x, dialog_y, dialog_w, dialog_h), border_radius=5)

            # Render and draw text
            try:
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
            self.clock.tick(30) # Lower FPS for menu is fine

        # If loop exits without choice (e.g. app_running became false), default to quit
        return 'quit'


    def run(self):
        """Main application loop, handles simulation and end-game dialog."""
        print("Starting Ant Simulation - Complex Dynamics...")
        print("Press 'D' to toggle debug info overlay.")
        print("Press 'ESC' during simulation to end current run.")
        print("Use UI buttons for speed control.")

        # Outer loop controls the application lifetime (including restarts)
        while self.app_running:

            # --- Simulation Phase ---
            # simulation_running is set true by _reset_simulation
            while self.simulation_running and self.app_running:
                action = self.handle_events() # Process quit, ESC, clicks

                if not self.app_running: break # Exit outer loop if QUIT event occurred
                if action == 'sim_stop': break # Exit simulation loop if ESC pressed

                # Only update and draw if not paused AND simulation is running
                if self.simulation_speed_level > 0:
                    self.update()

                # Draw the current simulation state
                self.draw()

                # Control simulation speed (even when paused, tick minimally for UI responsiveness)
                target_fps = self.current_target_fps if self.simulation_speed_level > 0 else 10
                self.clock.tick(target_fps)

            # --- End Game / Dialog Phase ---
            if not self.app_running:
                 break # Exit main loop immediately if app should close

            # If simulation stopped (queen died, ESC), show the dialog
            # Make sure we have a reason, otherwise assume user quit/closed window
            if not self.end_game_reason: self.end_game_reason = "Unbekannt"

            choice = self._show_end_game_dialog()

            if choice == 'restart':
                self._reset_simulation() # Resets state and sets simulation_running = True
                # The outer loop will then re-enter the simulation phase
            elif choice == 'quit':
                self.app_running = False # Signal outer loop to terminate

        # --- Cleanup ---
        print("Exiting application.")
        try:
            pygame.quit()
            print("Pygame shut down.")
        except Exception as e: # Catch broader exceptions during quit
            print(f"Error during Pygame quit: {e}")


# --- Start Simulation ---
if __name__ == '__main__':
    # Dependency checks & Init
    try:
        import numpy
        print(f"NumPy version: {numpy.__version__}")
    except ImportError:
        print("FATAL: NumPy library is required but not found.")
        input("Press Enter to Exit.")
        exit()
    try:
        import pygame
        print(f"Pygame version: {pygame.version.ver}")
    except ImportError as e:
        print(f"FATAL: Pygame library failed to import: {e}")
        input("Press Enter to Exit.")
        exit()
    except Exception as e:
         print(f"FATAL: An unexpected error occurred during Pygame import: {e}")
         input("Press Enter to Exit.")
         exit()

    # Initialize Pygame modules safely *after* successful import
    try:
        pygame.init()
        # Font init is now handled within AntSimulation._init_fonts()
        if pygame.display.get_init() and pygame.font.get_init():
             print("Pygame and Font modules initialized successfully.")
        else:
             raise RuntimeError("Pygame display or font module failed to initialize.")

    except Exception as e:
         print(f"FATAL: Pygame initialization failed: {e}")
         # Attempt to quit pygame if partially initialized
         try: pygame.quit()
         except: pass
         input("Press Enter to Exit.")
         exit()

    # Create and run the simulation
    try:
        simulation = AntSimulation()
        simulation.run() # run() now handles the main app loop including restarts
    except Exception as e:
        print("\n--- UNHANDLED EXCEPTION CAUGHT ---")
        import traceback
        traceback.print_exc()
        print("------------------------------------")
        print("An critical error occurred during simulation execution.")
        # Attempt to quit pygame if simulation crashed
        try: pygame.quit()
        except: pass
        input("Press Enter to Exit.") # Keep window open to see error

    print("Simulation process finished.")