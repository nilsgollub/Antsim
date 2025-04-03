# -*- coding: utf-8 -*-

# --- START OF FILE antsim.py ---
# Version mit Performance-Optimierungen, PEP8, Netzwerk-Stream, Legende
# UND: Responsive Design Anpassungen (Größenberechnung bei Start)

# Standard Library Imports
import random
import math
import time
from enum import Enum, auto
import io  # Needed for streaming
import traceback  # For detailed error reporting
import threading  # Needed for streaming server
import sys  # For checking if running interactively (affects Flask reload)
import os  # For centering window

# Third-Party Imports
try:
    import pygame
except ImportError:
    print("FATAL: Pygame is required but not found. Install it: pip install pygame")
    exit()
try:
    import numpy as np
except ImportError:
    print("FATAL: NumPy is required but not found. Install it: pip install numpy")
    exit()
try:
    from flask import Flask, Response, render_template_string
except ImportError:
    # Flask is optional, only needed for network streaming
    Flask = None
    Response = None
    render_template_string = None
    print(
        "INFO: Flask not found. Network streaming disabled. "
        "Install it for streaming: pip install Flask"
    )

try:
    import scipy.ndimage
except ImportError:
    print("FATAL: SciPy is required but not found. Install it: pip install scipy")
    exit()

# --- Configuration Constants ---

# --- NEW: Screen/Grid Handling ---
# Option 1: Use Fullscreen
USE_FULLSCREEN = False
# Option 2: Use Specific Window Size (if not fullscreen)
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
# Option 3: Use a percentage of the detected screen size (if not fullscreen)
# Example: USE_SCREEN_PERCENT = 0.8 # Use 80% of the screen width/height
USE_SCREEN_PERCENT = 0.5 # Set to a float between 0.1 and 1.0 or None
#USE_SCREEN_PERCENT = None

# Base size for calculations (adjust for overall detail level)
CELL_SIZE = 12

# --- THESE ARE NOW CALCULATED IN AntSimulation.__init__ ---
# GRID_WIDTH = 150
# GRID_HEIGHT = 80
# WIDTH = GRID_WIDTH * CELL_SIZE
# HEIGHT = GRID_HEIGHT * CELL_SIZE
# NEST_POS = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
# ----------------------------------------------------------

MAP_BG_COLOR = (20, 20, 10)

# Nest (Radius is in grid cells)
NEST_RADIUS = 3 # Relative to grid center

# Food
NUM_FOOD_TYPES = 2 # len(FoodType) - Hardcoded for clarity below
INITIAL_FOOD_CLUSTERS = 6
FOOD_PER_CLUSTER = 250
FOOD_CLUSTER_RADIUS = 5 # In grid cells
MIN_FOOD_DIST_FROM_NEST = 30 # In grid cells (relative to nest center)
MAX_FOOD_PER_CELL = 100.0
INITIAL_COLONY_FOOD_SUGAR = 200.0
INITIAL_COLONY_FOOD_PROTEIN = 200.0
RICH_FOOD_THRESHOLD = 50.0
CRITICAL_FOOD_THRESHOLD = 25.0
FOOD_REPLENISH_RATE = 1200  # Intervall in Ticks
# Food drawing change - adjust how many dots per unit of food
FOOD_DOTS_PER_UNIT = 0.15 # Lower value = more dots per food unit. Adjust for density.
FOOD_MAX_DOTS_PER_CELL = 25 # Limit dots per cell for performance/visuals
FOOD_DOT_RADIUS = 1 # Pixel radius of food dots# Food drawing change - adjust how many dots per unit of food
FOOD_DOTS_PER_UNIT = 0.15 # Lower value = more dots per food unit. Adjust for density.
FOOD_MAX_DOTS_PER_CELL = 25 # Limit dots per cell for performance/visuals
FOOD_DOT_RADIUS = 1 # Pixel radius of food dots


# Obstacles
NUM_OBSTACLES = 10
MIN_OBSTACLE_SIZE = 3 # In grid cells (Used by old rectangle method)
MAX_OBSTACLE_SIZE = 10 # In grid cells (Used by old rectangle method)
OBSTACLE_COLOR = (100, 100, 100) # <--- HIER IST DIE DEFINITION
MIN_OBSTACLE_CLUMPS = 3 # Instead of size, number of clumps per obstacle 'area'
MAX_OBSTACLE_CLUMPS = 8
MIN_OBSTACLE_CLUMP_RADIUS = 1 # Min radius of a small circle within the obstacle
MAX_OBSTACLE_CLUMP_RADIUS = 4 # Max radius of a small circle
OBSTACLE_CLUSTER_SPREAD_RADIUS = 5 # How far clumps spread from the obstacle center
OBSTACLE_COLOR_VARIATION = 15 # +/- range for obstacle cell color variation

# Pheromones
PHEROMONE_MAX = 1000.0
PHEROMONE_DECAY = 0.9995  # Geänderter Wert
PHEROMONE_DIFFUSION_RATE = 0.02  # Geänderter Wert
NEGATIVE_PHEROMONE_DECAY = 0.995  # Geänderter Wert
NEGATIVE_PHEROMONE_DIFFUSION_RATE = 0.03  # Geänderter Wert
RECRUITMENT_PHEROMONE_DECAY = 0.98  # Geänderter Wert
RECRUITMENT_PHEROMONE_DIFFUSION_RATE = 0.03  # Geänderter Wert
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
W_ALARM_SOURCE_DEFEND = 500.0 # Increased
W_PERSISTENCE = 1.5
W_RANDOM_NOISE = 0.2
W_NEGATIVE_PHEROMONE = -50.0
W_RECRUITMENT_PHEROMONE = 200.0 # Increased
W_AVOID_NEST_SEARCHING = -150.0 # Penalty for searching near nest
W_HUNTING_TARGET = 300.0
W_AVOID_HISTORY = -1000.0  # Strong penalty for revisiting

# Probabilistic Choice Parameters
PROBABILISTIC_CHOICE_TEMP = 1.0
MIN_SCORE_FOR_PROB_CHOICE = 0.01

# Pheromone Drop Amounts
P_HOME_RETURNING = 100.0
P_FOOD_RETURNING_TRAIL = 60.0
P_FOOD_AT_SOURCE = 500.0
P_ALARM_FIGHT = 200.0 # Increased
P_NEGATIVE_SEARCH = 10.0
P_RECRUIT_FOOD = 400.0
P_RECRUIT_DAMAGE = 350.0 # Increased
P_RECRUIT_DAMAGE_SOLDIER = 500.0 # Increased
P_RECRUIT_PREY = 300.0
P_FOOD_SEARCHING = 0.0  # Placeholder/Not used directly
P_FOOD_AT_NEST = 0.0  # Placeholder/Not used directly

# Ant Parameters
INITIAL_ANTS = 50
MAX_ANTS = 200
QUEEN_HP = 1000
WORKER_MAX_AGE_MEAN = 12000
WORKER_MAX_AGE_STDDEV = 2000
WORKER_PATH_HISTORY_LENGTH = 8 # In ticks
WORKER_STUCK_THRESHOLD = 60 # In ticks
WORKER_ESCAPE_DURATION = 30 # In ticks
WORKER_FOOD_CONSUMPTION_INTERVAL = 100 # In ticks
SOLDIER_PATROL_RADIUS_MULTIPLIER = 0.2 # Multiplier for NEST_RADIUS
SOLDIER_DEFEND_ALARM_THRESHOLD = 100.0 # Combined alarm/recruit signal
ANT_SPEED_BOOST_MULTIPLIER = 0.7 # Multiplier for move delay (lower = faster)
ANT_SPEED_BOOST_DURATION = 10 # Ticks
ALARM_SEARCH_RADIUS_SIGNAL = 10  # Radius for searching the strongest alarm/recruitment signal
ALARM_SEARCH_RADIUS_RANDOM = 20  # Radius for random search around the alarm source



# Brood Cycle Parameters
QUEEN_EGG_LAY_RATE = 60 # Ticks per attempt
QUEEN_FOOD_PER_EGG_SUGAR = 1.0
QUEEN_FOOD_PER_EGG_PROTEIN = 1.5
QUEEN_SOLDIER_RATIO_TARGET = 0.15
EGG_DURATION = 500 # Ticks
LARVA_DURATION = 800 # Ticks
PUPA_DURATION = 600 # Ticks
LARVA_FOOD_CONSUMPTION_PROTEIN = 0.06
LARVA_FOOD_CONSUMPTION_SUGAR = 0.01
LARVA_FEED_INTERVAL = 50 # Ticks

# Enemy Parameters
INITIAL_ENEMIES = 10
ENEMY_HP = 60
ENEMY_ATTACK = 10
ENEMY_MOVE_DELAY = 4 # Ticks
ENEMY_SPAWN_RATE = 1000 # Ticks
ENEMY_TO_FOOD_ON_DEATH_SUGAR = 10.0
ENEMY_TO_FOOD_ON_DEATH_PROTEIN = 50.0
ENEMY_NEST_ATTRACTION = 0.05 # Probability to move towards nest

# Prey Parameters
INITIAL_PREY = 5
PREY_HP = 25
PREY_MOVE_DELAY = 2 # Ticks
PREY_SPAWN_RATE = 600 # Ticks
PROTEIN_ON_DEATH = 30.0
PREY_FLEE_RADIUS_SQ = 5 * 5 # Grid cells squared

# Simulation Speed Control
BASE_FPS = 40
SPEED_MULTIPLIERS = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0]
TARGET_FPS_LIST = [10] + [
    max(1, int(m * BASE_FPS)) for m in SPEED_MULTIPLIERS[1:]
]
DEFAULT_SPEED_INDEX = SPEED_MULTIPLIERS.index(1.0)

# --- NEW: Network Streaming ---
ENABLE_NETWORK_STREAM = False  # Set to True to enable web streaming
STREAMING_HOST = "0.0.0.0"  # Host for the streaming server (0.0.0.0 for external access)
STREAMING_PORT = 5000       # Port for the streaming server
STREAM_FRAME_QUALITY = 75   # JPEG quality for streaming (0-100)
STREAM_FPS_LIMIT = 15       # Limit FPS for streaming to reduce load


# --- Enums ---
class AntState(Enum):
    SEARCHING = auto()
    RETURNING_TO_NEST = auto()
    ESCAPING = auto()
    PATROLLING = auto()
    DEFENDING = auto()
    HUNTING = auto()
    TENDING_BROOD = auto()  # Placeholder

class BroodStage(Enum):
    EGG = auto()
    LARVA = auto()
    PUPA = auto()

class AntCaste(Enum):
    WORKER = auto()
    SOLDIER = auto()

class FoodType(Enum):
    SUGAR = 0
    PROTEIN = 1


# --- Ant Caste Attributes ---
ANT_ATTRIBUTES = {
    AntCaste.WORKER: {
        "hp": 50,
        "attack": 3,
        "capacity": 1.5,
        "speed_delay": 0,
        # New: Use gray base color and subtle state colors
        "color": (200, 200, 200),  # Light Blue for searching
        "return_color": (200, 255, 200),  # Light Green for returning
        "food_consumption_sugar": 0.02,
        "food_consumption_protein": 0.005,
        "description": "Worker",
        "size_factor": 2.5,
        "head_size_factor": 0.4, # New: Head size factor
    },
    AntCaste.SOLDIER: {
        "hp": 90,
        "attack": 10,
        "capacity": 0.2,
        "speed_delay": 1,
        # New: Use gray base color and subtle state colors
        "color": (230, 200, 200),  # Light Red for patrolling
        "return_color": (255, 230, 200),  # Light Orange for returning
        "food_consumption_sugar": 0.025,
        "food_consumption_protein": 0.01,
        "description": "Soldier",
        "size_factor": 1.8,  # Soldiers are larger
        "head_size_factor": 0.6, # New: Head size factor
    },
}

# --- Other Colors ---
ANT_BASE_COLOR = (200, 200, 200)  # Medium gray
QUEEN_COLOR = (0, 0, 255)  # Blue
WORKER_ESCAPE_COLOR = (255, 255, 0)  # Yellow
ANT_DEFEND_COLOR = (255, 0, 0)  # Red
ANT_HUNT_COLOR = (0, 255, 255)  # Cyan
ENEMY_COLOR = (200, 0, 0)  # Red
PREY_COLOR = (0, 100, 0)  # Dark green
FOOD_COLORS = {
    FoodType.SUGAR: (200, 200, 255),  # Light Blue/Purple
    FoodType.PROTEIN: (255, 180, 180),  # Light Red/Pink
}
FOOD_COLOR_MIX = (230, 200, 230)  # Mix color for UI/Legend
PHEROMONE_HOME_COLOR = (0, 0, 255, 150)  # Blue (alpha for intensity)
PHEROMONE_FOOD_SUGAR_COLOR = (180, 180, 255, 150)  # Lighter Blue/Purple (angepasst)
PHEROMONE_FOOD_PROTEIN_COLOR = (255, 160, 160, 150)  # Lighter Red/Pink (angepasst)
PHEROMONE_ALARM_COLOR = (255, 0, 0, 180)  # Red
PHEROMONE_NEGATIVE_COLOR = (150, 150, 150, 100)  # Grey
PHEROMONE_RECRUITMENT_COLOR = (255, 0, 255, 180)  # Magenta/Pink
EGG_COLOR = (255, 255, 255, 200)  # White (alpha for density)
LARVA_COLOR = (255, 255, 200, 220)  # Pale Yellow
PUPA_COLOR = (200, 180, 150, 220)  # Beige/Brown
ATTACK_INDICATOR_COLOR_ANT = (255, 255, 100, 180) # Yellowish flash for ant attacks
ATTACK_INDICATOR_COLOR_ENEMY = (255, 100, 100, 180) # Reddish flash for enemy attacks
ATTACK_INDICATOR_DURATION_TICKS = 6 # How many ticks the indicator lasts (adjust as needed)
OBSTACLE_COLOR_VARIATION = 15 # +/- range for obstacle cell color variation

# UI Colors
BUTTON_COLOR = (80, 80, 150)
BUTTON_HOVER_COLOR = (100, 100, 180)
BUTTON_TEXT_COLOR = (240, 240, 240)
END_DIALOG_BG_COLOR = (0, 0, 0, 180) # Semi-transparent black
LEGEND_BG_COLOR = (10, 10, 30, 180) # Semi-transparent dark blue BG for legend
LEGEND_TEXT_COLOR = (230, 230, 230)

# --- NEW: Font Size Scaling ---
# Base sizes (will be scaled based on screen height)
BASE_FONT_SIZE = 20
BASE_DEBUG_FONT_SIZE = 16
BASE_LEGEND_FONT_SIZE = 15
# Reference height for scaling (e.g., the default window height)
REFERENCE_HEIGHT_FOR_SCALING = DEFAULT_WINDOW_HEIGHT

# Define aliases for random functions
rnd = random.randint
rnd_gauss = random.gauss
rnd_uniform = random.uniform


# --- Network Streaming Setup ---
streaming_app = None
streaming_thread = None
latest_frame_lock = threading.Lock()
latest_frame_bytes = None
stop_streaming_event = threading.Event()

# Basic HTML page for viewing the stream
HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Ant Simulation Stream</title>
    <style>
      body {{ background-color: #333; margin: 0; padding: 0; }}
      img {{ display: block; margin: auto; padding-top: 20px; }}
    </style>
  </head>
  <body>
    <img src="{{ url_for('video_feed') }}" width="{{ width }}" height="{{ height }}">
  </body>
</html>
"""

def stream_frames():
    """Generator function to yield MJPEG frames."""
    global latest_frame_bytes
    last_yield_time = time.time()
    min_interval = 1.0 / STREAM_FPS_LIMIT if STREAM_FPS_LIMIT > 0 else 0

    while not stop_streaming_event.is_set():
        current_time = time.time()
        if min_interval > 0 and current_time - last_yield_time < min_interval:
            time.sleep(min_interval / 5) # Sleep briefly if too fast
            continue

        frame_data = None
        with latest_frame_lock:
            if latest_frame_bytes:
                frame_data = latest_frame_bytes
                # Optional: Clear latest_frame_bytes here if you only want to send new frames
                # latest_frame_bytes = None

        if frame_data:
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                last_yield_time = current_time
            except Exception as e:
                print(f"Streaming error: {e}")
                # Optionally break or handle the error
                time.sleep(0.5) # Avoid busy-looping on persistent errors
        else:
            # No new frame available, wait a bit
            sleep_duration = min_interval / 2 if min_interval > 0 else 0.05
            time.sleep(sleep_duration)


def run_server(app, host, port):
    """Runs the Flask server in a separate thread."""
    print(f" * Starting Flask server on http://{host}:{port}")
    try:
        # Check if running in an interactive environment where Flask's reloader might cause issues
        use_reloader = False # Disabling reloader for threaded mode stability
        # use_reloader = False if hasattr(sys, 'ps1') or not sys.stdout.isatty() else True
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=use_reloader)
    except Exception as e:
        print(f"FATAL: Could not start Flask server: {e}")
        # Optionally signal the main thread to stop
    finally:
        print(" * Flask server stopped.")


# --- Helper Functions ---
def is_valid_pos(pos, grid_width, grid_height):
    """Check if a position (x, y) is within the grid boundaries."""
    if not isinstance(pos, tuple) or len(pos) != 2: return False
    x, y = pos
    return 0 <= x < grid_width and 0 <= y < grid_height

def get_neighbors(pos, grid_width, grid_height, include_center=False):
    """Get valid integer neighbor coordinates for a given position."""
    x_int, y_int = int(pos[0]), int(pos[1])
    if not (0 <= x_int < grid_width and 0 <= y_int < grid_height): return [] # Return empty if center is invalid
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0 and not include_center: continue
            n_pos = (x_int + dx, y_int + dy)
            if 0 <= n_pos[0] < grid_width and 0 <= n_pos[1] < grid_height:
                neighbors.append(n_pos)
    return neighbors

def distance_sq(pos1, pos2):
    """Calculate squared Euclidean distance between two integer points."""
    # Returns float('inf') if input is invalid
    try:
        x1, y1 = pos1
        x2, y2 = pos2
        return (x1 - x2) ** 2 + (y1 - y2) ** 2
    except (TypeError, ValueError, IndexError):
        # print(f"Warning: Invalid input for distance_sq: {pos1}, {pos2}") # Optional debug
        return float("inf")

def normalize(value, max_val):
    """Normalize a value to the range [0, 1], clamped."""
    if max_val <= 0: return 0.0
    norm_val = float(value) / float(max_val)
    return min(1.0, max(0.0, norm_val))

# --- Brood Class ---
class BroodItem:
    """Represents an item of brood (egg, larva, pupa) in the nest."""
    def __init__(
        self, stage: BroodStage, caste: AntCaste, position: tuple, current_tick: int,
        simulation # Pass simulation reference for grid dimensions
    ):
        self.stage = stage
        self.caste = caste
        self.pos = tuple(map(int, position))
        self.creation_tick = current_tick
        self.progress_timer = 0.0
        self.last_feed_check = current_tick
        self.simulation = simulation # Store reference

        # Size relative to CELL_SIZE
        cell_size = self.simulation.cell_size
        if self.stage == BroodStage.EGG:
            self.duration = EGG_DURATION; self.color = EGG_COLOR; self.radius = max(1, cell_size // 5)
        elif self.stage == BroodStage.LARVA:
            self.duration = LARVA_DURATION; self.color = LARVA_COLOR; self.radius = max(1, cell_size // 4)
        elif self.stage == BroodStage.PUPA:
            self.duration = PUPA_DURATION; self.color = PUPA_COLOR; self.radius = max(1, int(cell_size / 3.5))
        else:
            self.duration = 0; self.color = (0, 0, 0, 0); self.radius = 0

    def update(self, current_tick):
        sim = self.simulation
        current_multiplier = SPEED_MULTIPLIERS[sim.simulation_speed_index]
        if current_multiplier == 0.0: return None
        self.progress_timer += current_multiplier

        if self.stage == BroodStage.LARVA:
            # Check food consumption only if enough time has passed
            if current_tick - self.last_feed_check >= LARVA_FEED_INTERVAL:
                self.last_feed_check = current_tick
                needed_p = LARVA_FOOD_CONSUMPTION_PROTEIN
                needed_s = LARVA_FOOD_CONSUMPTION_SUGAR
                has_p = sim.colony_food_storage_protein >= needed_p
                has_s = sim.colony_food_storage_sugar >= needed_s
                if has_p and has_s:
                    sim.colony_food_storage_protein -= needed_p
                    sim.colony_food_storage_sugar -= needed_s
                else:
                    # Larva doesn't grow if not fed, halt progress
                    self.progress_timer = max(0.0, self.progress_timer - current_multiplier)

        # Check for stage progression
        if self.progress_timer >= self.duration:
            if self.stage == BroodStage.EGG:
                self.stage = BroodStage.LARVA
                self.progress_timer = 0.0; self.duration = LARVA_DURATION; self.color = LARVA_COLOR; self.radius = max(1, sim.cell_size // 4)
                self.last_feed_check = current_tick # Start feed check timer for larva
                return None # Still brood
            elif self.stage == BroodStage.LARVA:
                self.stage = BroodStage.PUPA
                self.progress_timer = 0.0; self.duration = PUPA_DURATION; self.color = PUPA_COLOR; self.radius = max(1, int(sim.cell_size / 3.5))
                return None # Still brood
            elif self.stage == BroodStage.PUPA:
                return self # Signal hatching by returning self
        return None # Still brood

    def draw(self, surface):
        sim = self.simulation
        if not is_valid_pos(self.pos, sim.grid_width, sim.grid_height) or self.radius <= 0:
             return
        center_x = int(self.pos[0] * sim.cell_size + sim.cell_size // 2)
        center_y = int(self.pos[1] * sim.cell_size + sim.cell_size // 2)
        draw_pos = (center_x, center_y)
        pygame.draw.circle(surface, self.color, draw_pos, self.radius)
        # Add outline to pupa indicating caste
        if self.stage == BroodStage.PUPA:
            outline_col = (50, 50, 50) if self.caste == AntCaste.WORKER else (100, 0, 0)
            pygame.draw.circle(surface, outline_col, draw_pos, self.radius, 1)

# --- Grid Class ---
class WorldGrid:
    """Manages the simulation grid (food, obstacles, pheromones)."""
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        print(f"Initializing WorldGrid with dimensions: {grid_width}x{grid_height}")
        # Ensure dimensions are valid before creating arrays
        if not (grid_width > 0 and grid_height > 0):
             raise ValueError(f"Invalid grid dimensions for WorldGrid: {grid_width}x{grid_height}")

        self.food = np.zeros((grid_width, grid_height, NUM_FOOD_TYPES), dtype=np.float32)
        self.obstacles = np.zeros((grid_width, grid_height), dtype=bool)
        self.pheromones_home = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_alarm = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_negative = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_recruitment = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_food_sugar = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_food_protein = np.zeros((grid_width, grid_height), dtype=np.float32)

    def reset(self, nest_pos):
        """Resets grid state and places initial elements."""
        self.food.fill(0)
        self.obstacles.fill(0)
        self.pheromones_home.fill(0)
        self.pheromones_alarm.fill(0)
        self.pheromones_negative.fill(0)
        self.pheromones_recruitment.fill(0)
        self.pheromones_food_sugar.fill(0)
        self.pheromones_food_protein.fill(0)
        self.place_obstacles(nest_pos)
        self.place_food_clusters(nest_pos)

    def place_food_clusters(self, nest_pos):
        """Places initial food clusters relative to the nest."""
        nest_pos_int = tuple(map(int, nest_pos))
        min_dist_sq = MIN_FOOD_DIST_FROM_NEST ** 2
        for i in range(INITIAL_FOOD_CLUSTERS):
            food_type_index = i % NUM_FOOD_TYPES
            attempts = 0
            cx, cy = 0, 0
            found_spot = False
            # Try finding a suitable spot away from the nest
            while attempts < 150 and not found_spot:
                cx = rnd(0, self.grid_width - 1)
                cy = rnd(0, self.grid_height - 1)
                if (is_valid_pos((cx, cy), self.grid_width, self.grid_height) and
                        not self.obstacles[cx, cy] and
                        distance_sq((cx, cy), nest_pos_int) > min_dist_sq):
                    found_spot = True
                attempts += 1

            # Fallback 1: Any non-obstacle spot
            if not found_spot:
                attempts = 0
                while attempts < 200:
                    cx = rnd(0, self.grid_width - 1)
                    cy = rnd(0, self.grid_height - 1)
                    if is_valid_pos((cx, cy), self.grid_width, self.grid_height) and not self.obstacles[cx, cy]:
                        found_spot = True
                        break
                    attempts += 1

            # Fallback 2: Any spot (should rarely happen)
            if not found_spot:
                cx = rnd(0, self.grid_width - 1)
                cy = rnd(0, self.grid_height - 1)

            # Distribute food around the cluster center
            added_amount = 0.0
            target_food_amount = FOOD_PER_CLUSTER
            max_placement_attempts = int(target_food_amount * 2.5) # More attempts needed
            for _ in range(max_placement_attempts):
                if added_amount >= target_food_amount: break
                # Gaussian distribution around the center
                fx = cx + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                fy = cy + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))

                if (0 <= fx < self.grid_width and 0 <= fy < self.grid_height and
                        not self.obstacles[fx, fy]):
                    amount_to_add = rnd_uniform(0.5, 1.0) * (MAX_FOOD_PER_CELL / 8)
                    current_amount = self.food[fx, fy, food_type_index]
                    new_amount = min(MAX_FOOD_PER_CELL, current_amount + amount_to_add)
                    actual_added = new_amount - current_amount
                    if actual_added > 0:
                        self.food[fx, fy, food_type_index] = new_amount
                        added_amount += actual_added

        print(f"Placed {INITIAL_FOOD_CLUSTERS} initial food clusters around nest {nest_pos_int}.")

    def place_obstacles(self, nest_pos):
        """Places obstacles with more organic, rounded shapes."""
        nest_area = set()
        nest_radius_buffer = NEST_RADIUS + 4 # Increased buffer slightly for rounder shapes
        nest_center_int = tuple(map(int, nest_pos))

        min_x_nest = max(0, nest_center_int[0] - nest_radius_buffer)
        max_x_nest = min(self.grid_width - 1, nest_center_int[0] + nest_radius_buffer)
        min_y_nest = max(0, nest_center_int[1] - nest_radius_buffer)
        max_y_nest = min(self.grid_height - 1, nest_center_int[1] + nest_radius_buffer)

        for x in range(min_x_nest, max_x_nest + 1):
            for y in range(min_y_nest, max_y_nest + 1):
                if distance_sq((x, y), nest_center_int) <= nest_radius_buffer ** 2:
                    nest_area.add((x, y))

        placed_count = 0
        max_obstacle_attempts = NUM_OBSTACLES * 25 # More attempts for complex placement

        for _ in range(max_obstacle_attempts):
            if placed_count >= NUM_OBSTACLES: break

            attempts_per_obstacle = 0
            placed_this_obstacle = False
            while attempts_per_obstacle < 35 and not placed_this_obstacle:
                # 1. Find a potential center for the obstacle cluster
                cluster_cx = rnd(0, self.grid_width - 1)
                cluster_cy = rnd(0, self.grid_height - 1)

                # Rough check: Is the center too close to the nest?
                if (cluster_cx, cluster_cy) in nest_area:
                    attempts_per_obstacle += 1
                    continue

                # 2. Generate multiple overlapping clumps (circles)
                num_clumps = rnd(MIN_OBSTACLE_CLUMPS, MAX_OBSTACLE_CLUMPS)
                obstacle_cells_this_attempt = set()
                can_place_this = True

                for _ in range(num_clumps):
                    # Clump center relative to cluster center
                    clump_offset_x = int(rnd_gauss(0, OBSTACLE_CLUSTER_SPREAD_RADIUS * 0.5))
                    clump_offset_y = int(rnd_gauss(0, OBSTACLE_CLUSTER_SPREAD_RADIUS * 0.5))
                    clump_cx = cluster_cx + clump_offset_x
                    clump_cy = cluster_cy + clump_offset_y
                    clump_radius = rnd(MIN_OBSTACLE_CLUMP_RADIUS, MAX_OBSTACLE_CLUMP_RADIUS)
                    clump_radius_sq = clump_radius ** 2

                    # Iterate bounding box around the clump
                    min_x = max(0, clump_cx - clump_radius)
                    max_x = min(self.grid_width - 1, clump_cx + clump_radius)
                    min_y = max(0, clump_cy - clump_radius)
                    max_y = min(self.grid_height - 1, clump_cy + clump_radius)

                    for x in range(min_x, max_x + 1):
                        for y in range(min_y, max_y + 1):
                            pos_check = (x, y)
                            # Check distance and if it overlaps nest or existing temp obstacles
                            if (distance_sq(pos_check, (clump_cx, clump_cy)) <= clump_radius_sq):
                                if pos_check in nest_area:
                                     can_place_this = False
                                     break # This clump hit the nest, abandon this obstacle attempt
                                if is_valid_pos(pos_check, self.grid_width, self.grid_height):
                                     obstacle_cells_this_attempt.add(pos_check)
                        if not can_place_this: break
                    if not can_place_this: break # Break outer loop if inner failed

                # 3. If all clumps were valid (didn't hit nest), place them
                if can_place_this and obstacle_cells_this_attempt:
                    # Check against already placed permanent obstacles
                    is_clear = True
                    for cell in obstacle_cells_this_attempt:
                        # Need to check bounds again before accessing self.obstacles
                        if not (0 <= cell[0] < self.grid_width and 0 <= cell[1] < self.grid_height) or self.obstacles[cell[0], cell[1]]:
                            is_clear = False
                            break
                    # Place if the area is clear
                    if is_clear:
                         for x, y in obstacle_cells_this_attempt:
                              self.obstacles[x, y] = True
                         placed_this_obstacle = True
                         placed_count += 1

                attempts_per_obstacle += 1

        if placed_count < NUM_OBSTACLES:
            print(f"Warning: Placed only {placed_count}/{NUM_OBSTACLES} organic obstacles.")
        else:
            print(f"Placed {placed_count} organic obstacles.")

    def is_obstacle(self, pos):
        """Checks if a grid cell contains an obstacle."""
        try:
            x, y = pos
            # Important: Check bounds first
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                return self.obstacles[x, y]
            else:
                return True # Treat out-of-bounds as an obstacle
        except (IndexError, TypeError, ValueError):
            return True # Treat errors as obstacles

    def get_pheromone(self, pos, ph_type="home", food_type: FoodType = None):
        """Gets the pheromone value at a specific grid cell."""
        try:
            x, y = pos
            if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                 return 0.0 # Out of bounds
        except (ValueError, TypeError, IndexError):
            return 0.0 # Invalid input

        try:
            if ph_type == "home": return self.pheromones_home[x, y]
            if ph_type == "alarm": return self.pheromones_alarm[x, y]
            if ph_type == "negative": return self.pheromones_negative[x, y]
            if ph_type == "recruitment": return self.pheromones_recruitment[x, y]
            if ph_type == "food":
                if food_type == FoodType.SUGAR: return self.pheromones_food_sugar[x, y]
                if food_type == FoodType.PROTEIN: return self.pheromones_food_protein[x, y]
            return 0.0 # Unknown type or food type not specified
        except IndexError:
            # This should not happen if bounds check passed, but safety first
            return 0.0

    def add_pheromone(self, pos, amount, ph_type="home", food_type: FoodType = None):
        """Adds pheromone to a specific grid cell, respecting max values."""
        if amount <= 0: return # No effect if amount is zero or negative
        try:
            x, y = pos
            # Check bounds first, also check if it's an obstacle
            if not (0 <= x < self.grid_width and 0 <= y < self.grid_height) or self.obstacles[x, y]:
                return
        except (ValueError, TypeError, IndexError):
            return # Invalid input or error accessing obstacle map

        target_array = None
        max_value = PHEROMONE_MAX # Default max

        if ph_type == "home": target_array = self.pheromones_home
        elif ph_type == "alarm": target_array = self.pheromones_alarm
        elif ph_type == "negative": target_array = self.pheromones_negative
        elif ph_type == "recruitment":
             target_array = self.pheromones_recruitment
             max_value = RECRUITMENT_PHEROMONE_MAX # Special max for recruitment
        elif ph_type == "food":
            if food_type == FoodType.SUGAR: target_array = self.pheromones_food_sugar
            elif food_type == FoodType.PROTEIN: target_array = self.pheromones_food_protein
            else: return # Invalid food type
        else: return # Invalid pheromone type

        if target_array is not None:
            try:
                # Add amount and clamp to the maximum value
                target_array[x, y] = min(target_array[x, y] + amount, max_value)
            except IndexError:
                 # Safety check, should have been caught by bounds check
                 pass

    def update_pheromones(self, speed_multiplier):
        """Applies decay and diffusion to all pheromone maps."""
        effective_multiplier = max(0.0, speed_multiplier)
        if effective_multiplier == 0.0: return  # No update if paused

        # Calculate effective decay factors based on speed multiplier
        # Ensure decay doesn't become too extreme (e.g., pow(0.99, 1000))
        min_decay_factor = 0.1  # Prevent decay from making values vanish instantly
        decay_factor_common = max(min_decay_factor, PHEROMONE_DECAY ** effective_multiplier)
        decay_factor_neg = max(min_decay_factor, NEGATIVE_PHEROMONE_DECAY ** effective_multiplier)
        decay_factor_rec = max(min_decay_factor, RECRUITMENT_PHEROMONE_DECAY ** effective_multiplier)

        # Apply decay
        self.pheromones_home *= decay_factor_common
        self.pheromones_alarm *= decay_factor_common
        self.pheromones_food_sugar *= decay_factor_common
        self.pheromones_food_protein *= decay_factor_common
        self.pheromones_negative *= decay_factor_neg
        self.pheromones_recruitment *= decay_factor_rec

        # Calculate effective diffusion rates
        diffusion_rate_common = PHEROMONE_DIFFUSION_RATE * effective_multiplier
        diffusion_rate_neg = NEGATIVE_PHEROMONE_DIFFUSION_RATE * effective_multiplier
        diffusion_rate_rec = RECRUITMENT_PHEROMONE_DIFFUSION_RATE * effective_multiplier

        # Clamp diffusion rates to prevent instability (especially at high multipliers)
        max_diffusion = 0.124  # Max proportion that can diffuse out in one step
        diffusion_rate_common = min(max_diffusion, max(0.0, diffusion_rate_common))
        diffusion_rate_neg = min(max_diffusion, max(0.0, diffusion_rate_neg))
        diffusion_rate_rec = min(max_diffusion, max(0.0, diffusion_rate_rec))

        # --- Optimized Diffusion using Gauss Filter ---
        obstacle_mask = ~self.obstacles  # Mask where diffusion *can* occur

        arrays_rates = [
            (self.pheromones_home, diffusion_rate_common),
            (self.pheromones_food_sugar, diffusion_rate_common),
            (self.pheromones_food_protein, diffusion_rate_common),
            (self.pheromones_alarm, diffusion_rate_common),
            (self.pheromones_negative, diffusion_rate_neg),
            (self.pheromones_recruitment, diffusion_rate_rec)
        ]

        # Apply obstacle mask *before* diffusion calculation
        for arr, rate in arrays_rates:
            if rate > 0:
                # Apply obstacle mask *before* diffusion calculation
                arr *= obstacle_mask
                # Apply Gaussian filter
                diffused = scipy.ndimage.gaussian_filter(arr, sigma=0.32, mode='constant', cval=0.0)
                # Update the original array only where there are no obstacles
                arr[:] = diffused

        # Clamp values and remove negligible amounts
        min_pheromone_threshold = 0.01  # Values below this are set to 0
        pheromone_arrays = [
            (self.pheromones_home, PHEROMONE_MAX),
            (self.pheromones_food_sugar, PHEROMONE_MAX),
            (self.pheromones_food_protein, PHEROMONE_MAX),
            (self.pheromones_alarm, PHEROMONE_MAX),
            (self.pheromones_negative, PHEROMONE_MAX),  # Uses standard max
            (self.pheromones_recruitment, RECRUITMENT_PHEROMONE_MAX)  # Special max
        ]
        for arr, max_val in pheromone_arrays:
            np.clip(arr, 0, max_val, out=arr)  # Clamp between 0 and max_val
            arr[arr < min_pheromone_threshold] = 0  # Zero out tiny amounts

    def replenish_food(self, nest_pos):
        """Places new food clusters in the world, avoiding the nest."""
        nest_pos_int = tuple(map(int, nest_pos))
        min_dist_sq = MIN_FOOD_DIST_FROM_NEST ** 2

        # Replenish fewer clusters than initially placed
        num_clusters_to_add = max(1, INITIAL_FOOD_CLUSTERS // 3)

        for i in range(num_clusters_to_add):
            food_type_index = rnd(0, NUM_FOOD_TYPES - 1) # Random type for replenishment
            attempts = 0
            cx, cy = 0, 0
            found_spot = False
            # Try finding a suitable spot away from the nest
            while attempts < 150 and not found_spot:
                cx = rnd(0, self.grid_width - 1)
                cy = rnd(0, self.grid_height - 1)
                if (is_valid_pos((cx, cy), self.grid_width, self.grid_height) and
                        not self.obstacles[cx, cy] and
                        distance_sq((cx, cy), nest_pos_int) > min_dist_sq):
                    found_spot = True
                attempts += 1

            # Fallback 1: Any non-obstacle spot
            if not found_spot:
                attempts = 0
                while attempts < 200:
                    cx = rnd(0, self.grid_width - 1)
                    cy = rnd(0, self.grid_height - 1)
                    if is_valid_pos((cx, cy), self.grid_width, self.grid_height) and not self.obstacles[cx, cy]:
                        found_spot = True
                        break
                    attempts += 1

            # Fallback 2: Any spot
            if not found_spot:
                cx = rnd(0, self.grid_width - 1)
                cy = rnd(0, self.grid_height - 1)

            # Distribute smaller amount of food for replenishment
            added_amount = 0.0
            # Replenish about half the amount of an initial cluster
            target_food_amount = FOOD_PER_CLUSTER // 2
            max_placement_attempts = int(target_food_amount * 2.5)
            for _ in range(max_placement_attempts):
                if added_amount >= target_food_amount: break
                fx = cx + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                fy = cy + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))

                if (0 <= fx < self.grid_width and 0 <= fy < self.grid_height and
                        not self.obstacles[fx, fy]):
                    # Add a smaller chunk of food during replenishment
                    amount_to_add = rnd_uniform(0.3, 0.8) * (MAX_FOOD_PER_CELL / 10)
                    current_amount = self.food[fx, fy, food_type_index]
                    new_amount = min(MAX_FOOD_PER_CELL, current_amount + amount_to_add)
                    actual_added = new_amount - current_amount
                    if actual_added > 0:
                        self.food[fx, fy, food_type_index] = new_amount
                        added_amount += actual_added

class SpatialGrid:
    """Divides the game world into a grid for efficient collision detection."""

    def __init__(self, grid_width, grid_height, cell_size):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.grid = {}  # {(x, y): [entities]}

    def _get_cell_coords(self, pos):
        """Calculates the grid cell coordinates for a given position."""
        x, y = pos
        cell_x = x // self.cell_size
        cell_y = y // self.cell_size
        return cell_x, cell_y

    def add_entity(self, entity):
        """Adds an entity to the grid."""
        cell_coords = self._get_cell_coords(entity.pos)
        if cell_coords not in self.grid:
            self.grid[cell_coords] = []
        self.grid[cell_coords].append(entity)

    def remove_entity(self, entity):
        """Removes an entity from the grid."""
        cell_coords = self._get_cell_coords(entity.pos)
        if cell_coords in self.grid and entity in self.grid[cell_coords]:
            self.grid[cell_coords].remove(entity)

    def update_entity_position(self, entity, old_pos):
        """Updates an entity's position in the grid."""
        old_cell_coords = self._get_cell_coords(old_pos)
        new_cell_coords = self._get_cell_coords(entity.pos)

        # Only update if the cell changed
        if old_cell_coords != new_cell_coords:
            if old_cell_coords in self.grid and entity in self.grid[old_cell_coords]:
                self.grid[old_cell_coords].remove(entity)
            if new_cell_coords not in self.grid:
                self.grid[new_cell_coords] = []
            self.grid[new_cell_coords].append(entity)

    def get_nearby_entities(self, pos, entity_type=None):
        """Gets all entities near a given position (in the same and adjacent cells)."""
        cell_x, cell_y = self._get_cell_coords(pos)
        nearby_entities = []

        # Check current and adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell_x + dx, cell_y + dy)
                if check_cell in self.grid:
                    for entity in self.grid[check_cell]:
                        if entity_type is None or isinstance(entity, entity_type):
                            nearby_entities.append(entity)

        return nearby_entities

# --- Prey Class ---
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

    def update(self, speed_multiplier):
        sim = self.simulation
        if speed_multiplier == 0.0:
            return

        # Check for nearby ants to flee from
        grid = sim.grid
        pos_int = self.pos
        nearest_ant_pos = None
        min_dist_sq_found = PREY_FLEE_RADIUS_SQ  # Use configured flee radius

        # Define search area around prey
        check_radius = int(PREY_FLEE_RADIUS_SQ ** 0.5) + 1
        min_x = max(0, pos_int[0] - check_radius)
        max_x = min(sim.grid_width - 1, pos_int[0] + check_radius)
        min_y = max(0, pos_int[1] - check_radius)
        max_y = min(sim.grid_height - 1, pos_int[1] + check_radius)

        # Check cells within the radius for ants
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                check_pos = (x, y)
                # Skip self position
                if check_pos == pos_int:
                    continue
                # Check if an ant exists at this position using simulation's method
                ant = sim.get_ant_at(check_pos)
                if ant:
                    d_sq = distance_sq(pos_int, check_pos)
                    if d_sq < min_dist_sq_found:
                        min_dist_sq_found = d_sq
                        nearest_ant_pos = check_pos  # Store position of nearest ant

        # Update move timer
        self.move_delay_timer -= speed_multiplier
        if self.move_delay_timer > 0:
            return  # Not time to move yet
        # Reset timer (add base delay back)
        self.move_delay_timer += self.move_delay_base

        # Get possible moves (valid neighbors)
        possible_moves = get_neighbors(pos_int, sim.grid_width, sim.grid_height)
        valid_moves = [m for m in possible_moves if
                       not grid.is_obstacle(m) and
                       not sim.is_enemy_at(m) and
                       # Ensure no other prey is already there (pass self to exclude)
                       not sim.is_prey_at(m, exclude_self=self) and
                       not sim.is_ant_at(m)]  # Avoid moving onto ants

        if not valid_moves:
            return  # No valid place to move

        chosen_move = None
        # If fleeing from an ant
        if nearest_ant_pos:
            flee_dx = pos_int[0] - nearest_ant_pos[0]
            flee_dy = pos_int[1] - nearest_ant_pos[1]
            best_flee_move = None
            max_flee_score = -float("inf")

            # Score potential moves based on direction away from ant
            for move in valid_moves:
                move_dx = move[0] - pos_int[0]
                move_dy = move[1] - pos_int[1]
                # Approximate normalization (Manhattan distance)
                dist_approx = max(1, abs(flee_dx) + abs(flee_dy))
                norm_flee_dx = flee_dx / dist_approx
                norm_flee_dy = flee_dy / dist_approx
                # Score = Alignment with flee direction + distance gain (weighted)
                alignment_score = move_dx * norm_flee_dx + move_dy * norm_flee_dy
                distance_score = distance_sq(move, nearest_ant_pos) * 0.05  # Small bonus for increasing distance
                score = alignment_score + distance_score

                if score > max_flee_score:
                    max_flee_score = score
                    best_flee_move = move

            # Choose the best flee move, or a random one if no clear best direction
            chosen_move = best_flee_move if best_flee_move else random.choice(valid_moves)
        else:
            # Not fleeing, move randomly
            chosen_move = random.choice(valid_moves)

        # Execute the move if a valid move was chosen and it's different from current
        if chosen_move and chosen_move != self.pos:
            old_pos = self.pos
            self.pos = chosen_move
            # IMPORTANT: Update the simulation's position tracking
            sim.update_entity_position(self, old_pos, self.pos)

    def take_damage(self, amount, attacker):
        if self.hp <= 0:
            return  # Already dead
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            # print(f"Prey died at {self.pos}") # Optional debug

    def draw(self, surface):
        """Draws the prey (beetle-like)."""
        sim = self.simulation
        if not is_valid_pos(self.pos, sim.grid_width, sim.grid_height):
            return

        cs = sim.cell_size
        pos_px = (int(self.pos[0] * cs + cs / 2),
                  int(self.pos[1] * cs + cs / 2))

        # Body (oval/ellipse)
        body_width = max(3, int(cs / 1.8))
        body_height = max(2, int(cs / 2.4))
        body_rect = pygame.Rect(pos_px[0] - body_width // 2, pos_px[1] - body_height // 2, body_width, body_height)
        body_color = self.color
        pygame.draw.ellipse(surface, body_color, body_rect)

        # Shell line (elytra division)
        line_color = tuple(max(0, c - 40) for c in body_color) # Darker shade
        line_start = (pos_px[0], body_rect.top)
        line_end = (pos_px[0], body_rect.bottom)
        pygame.draw.line(surface, line_color, line_start, line_end, 1)

        # Optional: Tiny antennae
        antenna_length = body_width * 0.3
        antenna_angle = 0.3 # Radians from front
        antenna_origin_x = pos_px[0] # Center X
        antenna_origin_y = body_rect.top + body_height * 0.2 # Slightly back from front

        # Left Antenna
        angle_l = math.pi * 1.5 - antenna_angle # Pointing upwards-left
        end_lx = antenna_origin_x + antenna_length * math.cos(angle_l)
        end_ly = antenna_origin_y + antenna_length * math.sin(angle_l)
        pygame.draw.line(surface, line_color, (antenna_origin_x, antenna_origin_y), (int(end_lx), int(end_ly)), 1)

        # Right Antenna
        angle_r = math.pi * 1.5 + antenna_angle # Pointing upwards-right
        end_rx = antenna_origin_x + antenna_length * math.cos(angle_r)
        end_ry = antenna_origin_y + antenna_length * math.sin(angle_r)
        pygame.draw.line(surface, line_color, (antenna_origin_x, antenna_origin_y), (int(end_rx), int(end_ry)), 1)

        # Outline the body
        pygame.draw.ellipse(surface, (0, 0, 0), body_rect, 1) # Black outline

# --- Ant Class ---
class Ant:
    def __init__(self, pos, simulation, caste: AntCaste):
        """Initialisiert eine neue Ameise."""
        self.pos = tuple(map(int, pos))
        self.simulation = simulation
        self.caste = caste
        attrs = ANT_ATTRIBUTES[caste]

        # Core attributes from caste
        self.hp = float(attrs["hp"])
        self.max_hp = float(attrs["hp"])
        self.attack_power = attrs["attack"]
        self.max_capacity = attrs["capacity"]
        self.move_delay_base = attrs["speed_delay"] # Base ticks between moves
        self.search_color = attrs["color"]
        self.return_color = attrs["return_color"]
        self.food_consumption_sugar = attrs["food_consumption_sugar"]
        self.food_consumption_protein = attrs["food_consumption_protein"]
        self.size_factor = attrs["size_factor"] # Used for drawing radius
        self.head_size_factor = attrs["head_size_factor"]  # New: Head size factor

        # State variables
        self.state = AntState.SEARCHING
        self.carry_amount = 0.0
        self.carry_type: FoodType | None = None
        self.age = 0.0 # In ticks
        # Calculate max age with Gaussian distribution
        self.max_age_ticks = int(rnd_gauss(WORKER_MAX_AGE_MEAN, WORKER_MAX_AGE_STDDEV))

        # Movement and pathfinding
        self.path_history = [] # List of recently visited (x, y) tuples
        self.history_timestamps = [] # Simulation ticks corresponding to path_history entries
        self.move_delay_timer = 0 # Ticks remaining until next move allowed
        self.last_move_direction = (0, 0) # (dx, dy) of the last move
        self.stuck_timer = 0 # Ticks spent without moving
        self.escape_timer = 0.0 # Ticks remaining in ESCAPING state
        self.speed_boost_timer = 0.0  # New: Speed boost timer
        self.speed_boost_multiplier = 1.0  # New: Speed boost multiplier

        # Status/Debug info
        self.last_move_info = "Born" # Reason for the last action/decision
        self.just_picked_food = False # Flag for pheromone logic right after pickup

        # Timers
        # Random initial offset for consumption check
        self.food_consumption_timer = rnd_uniform(0, WORKER_FOOD_CONSUMPTION_INTERVAL)

        # Targeting/State specific
        self.last_known_alarm_pos = None # For DEFENDING state
        self.target_prey: Prey | None = None # For HUNTING state
        self.alarm_search_timer = 0  # Timer für die Suche nach Alarm
        self.alarm_search_radius = 5 # Radius für die zufällige Suche um die Alarmquelle
        self.initial_alarm_direction = None # Richtung, aus der das Alarmpheromon zuerst wahrgenommen wurde
        self.visual_range = 6 # Sichtradius in Zellen

    def _calculate_visual_score(self, neighbor_pos_int):
        """Berechnet einen Score basierend auf der Sichtbarkeit von Feinden (iteriert über Feinde)."""
        sim = self.simulation
        pos_int = self.pos
        score = 0.0

        # Durchlaufe die Liste der Feinde
        for enemy in sim.enemies:
            if enemy.hp <= 0: continue  # Überspringe tote Feinde

            enemy_pos = enemy.pos  # Position des Gegners

            # Berechne die quadratische Distanz zwischen dem Nachbarfeld und dem Feind
            dist_sq = distance_sq(neighbor_pos_int, enemy_pos)

            # Überprüfe, ob sich der Feind im Sichtfeld befindet (quadratische Distanzvergleich für Performance)
            if dist_sq <= self.visual_range ** 2:
                # Der Feind ist sichtbar!  Erhöhe den Score, gewichtet mit der Nähe
                # Höhere Punktzahl, wenn der Feind näher ist
                score += 1000.0 / (dist_sq + 1)  # Experimentiere mit der Gewichtungsfunktion
                # Optional: Füge Code hinzu, um die Sicht zu blockieren (z. B. Raycasting)

        return score

    def draw(self, surface):
        """Draws the ant onto the given surface."""
        sim = self.simulation
        cs = sim.cell_size
        # Calculate center pixel position
        pos_px = (int(self.pos[0] * cs + cs / 2),
                  int(self.pos[1] * cs + cs / 2))

        # --- Calculate Body Part Sizes ---
        # Overall size of the ant (adjust as needed)
        ant_size = max(2, int(cs / self.size_factor))
        # Calculate sizes of body parts relative to ant_size
        head_size = int(ant_size * self.head_size_factor) # Changed: Use head_size_factor
        thorax_size = int(ant_size * 0.2)
        abdomen_size = int(ant_size * 0.6)

        # --- Calculate Body Part Positions ---
        # Use last_move_direction to orient the ant
        move_dir_x, move_dir_y = self.last_move_direction
        # Normalize direction vector (if moving)
        move_dir_len = (move_dir_x ** 2 + move_dir_y ** 2) ** 0.5
        if move_dir_len > 0:
            move_dir_x /= move_dir_len
            move_dir_y /= move_dir_len

        # --- Draw Abdomen ---
        # Abdomen is behind the thorax
        abdomen_offset = int(ant_size * 0.6)
        abdomen_center = (pos_px[0] - int(move_dir_x * abdomen_offset),
                          pos_px[1] - int(move_dir_y * abdomen_offset))
        # Draw abdomen as ellipse
        abdomen_width = max(1, int(abdomen_size * 0.8))
        abdomen_height = max(1, int(abdomen_size * 1.2))
        abdomen_rect = pygame.Rect(abdomen_center[0] - abdomen_width // 2,
                                   abdomen_center[1] - abdomen_height // 2,
                                   abdomen_width, abdomen_height)
        pygame.draw.ellipse(surface, self.return_color, abdomen_rect)  # Changed

        # --- Draw Thorax ---
        # Thorax is the center of the ant
        thorax_width = max(1, int(thorax_size * 1.2))
        thorax_height = max(1, int(thorax_size * 0.8))
        thorax_rect = pygame.Rect(pos_px[0] - thorax_width // 2,
                                  pos_px[1] - thorax_height // 2,
                                  thorax_width, thorax_height)
        pygame.draw.ellipse(surface, self.search_color, thorax_rect)  # Changed

        # --- Draw Head ---
        # Head is in front of the thorax
        head_offset = int(ant_size * 0.6)
        head_center = (pos_px[0] + int(move_dir_x * head_offset),
                       pos_px[1] + int(move_dir_y * head_offset))
        # Draw head as circle
        head_radius = max(1, int(head_size / 2))
        head_color = tuple(max(0, c - 40) for c in self.search_color[:3])  # Darken color
        pygame.draw.circle(surface, head_color, head_center, head_radius)  # Changed

        # --- Draw Antennae ---
        # Antennae start at the head
        antenna_base_pos = head_center
        antenna_length = int(head_size * 1.2)
        antenna_angle_offset = 0.52  # Radians (about 17 degrees)

        # Calculate antenna end positions
        # Left antenna
        antenna_left_angle = math.atan2(move_dir_y, move_dir_x) + antenna_angle_offset
        antenna_left_end = (antenna_base_pos[0] + int(antenna_length * math.cos(antenna_left_angle)),
                            antenna_base_pos[1] + int(antenna_length * math.sin(antenna_left_angle)))
        # Right antenna
        antenna_right_angle = math.atan2(move_dir_y, move_dir_x) - antenna_angle_offset
        antenna_right_end = (antenna_base_pos[0] + int(antenna_length * math.cos(antenna_right_angle)),
                             antenna_base_pos[1] + int(antenna_length * math.sin(antenna_right_angle)))

        # Draw antennae lines
        antenna_color = head_color  # Dark gray
        pygame.draw.line(surface, antenna_color, antenna_base_pos, antenna_left_end, 1)  # Changed
        pygame.draw.line(surface, antenna_color, antenna_base_pos, antenna_right_end, 1)  # Changed

    def _update_state(self):
        """Checks conditions and potentially changes the ant's state."""
        sim = self.simulation
        pos_int = self.pos
        nest_pos_int = sim.nest_pos  # Use dynamic nest position

        # --- Worker: Opportunity Hunting ---
        # If searching, has no food, no target, and colony needs protein, check for nearby prey
        if (self.caste == AntCaste.WORKER and
                self.state == AntState.SEARCHING and
                self.carry_amount == 0 and not self.target_prey and
                sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * 1.5):

            # Look for prey in a slightly larger radius
            nearby_prey = sim.find_nearby_prey(pos_int, PREY_FLEE_RADIUS_SQ * 2.5)
            if nearby_prey:
                # Sort by distance to target the closest one
                nearby_prey.sort(key=lambda p: distance_sq(pos_int, p.pos))
                self.target_prey = nearby_prey[0]
                self._switch_state(AntState.HUNTING, f"HuntPrey@{self.target_prey.pos}")
                return  # State changed, skip other checks

        # --- Soldier: State Management (Patrol/Defend/Hunt) ---
        if (self.caste == AntCaste.SOLDIER and
                # Don't override these critical states
                self.state not in [AntState.ESCAPING, AntState.RETURNING_TO_NEST]):

            # Check local threat level (alarm/recruitment pheromones)
            max_alarm = 0.0
            max_recruit = 0.0
            search_radius_sq = 10 * 10  # Check nearby cells # Increased from 5*5
            grid = sim.grid
            x0, y0 = pos_int
            min_scan_x = max(0, x0 - int(search_radius_sq ** 0.5))
            max_scan_x = min(sim.grid_width - 1, x0 + int(search_radius_sq ** 0.5))
            min_scan_y = max(0, y0 - int(search_radius_sq ** 0.5))
            max_scan_y = min(sim.grid_height - 1, y0 + int(search_radius_sq ** 0.5))

            for i in range(min_scan_x, max_scan_x + 1):
                for j in range(min_scan_y, max_scan_y + 1):
                    p_int = (i, j)
                    if distance_sq(pos_int, p_int) <= search_radius_sq:
                        max_alarm = max(max_alarm, grid.get_pheromone(p_int, "alarm"))
                        max_recruit = max(max_recruit, grid.get_pheromone(p_int, "recruitment"))

            # Combine signals to estimate threat level
            threat_signal = max_alarm + max_recruit * 0.6

            is_near_nest = distance_sq(pos_int, nest_pos_int) <= sim.soldier_patrol_radius_sq  # Neue Zeile

            # High threat -> Switch to DEFENDING
            if threat_signal > SOLDIER_DEFEND_ALARM_THRESHOLD * 0.7:  # Reduced threshold
                if self.state != AntState.DEFENDING:
                    self._switch_state(AntState.DEFENDING, f"ThreatHi({threat_signal:.0f})!")
                    return  # State changed

            # Moderate threat / No active defense -> Check for prey (opportunity hunting for soldiers)
            # Also check if not already hunting
            if self.state != AntState.DEFENDING and not self.target_prey:
                nearby_prey = sim.find_nearby_prey(pos_int, PREY_FLEE_RADIUS_SQ * 2.0)
                if nearby_prey:
                    nearby_prey.sort(key=lambda p: distance_sq(pos_int, p.pos))
                    self.target_prey = nearby_prey[0]
                    self._switch_state(AntState.HUNTING, f"SHuntPrey@{self.target_prey.pos}")
                    # --- NEW: Speed Boost ---
                    self.speed_boost_timer = ANT_SPEED_BOOST_DURATION
                    self.speed_boost_multiplier = ANT_SPEED_BOOST_MULTIPLIER
                    return  # State changed

            # Low threat / Finished Defending -> Revert based on location
            if self.state == AntState.DEFENDING:
                # If threat dropped below threshold, stop defending
                self._switch_state(AntState.PATROLLING, f"ThreatLow({threat_signal:.0f})")
            elif is_near_nest and self.state != AntState.PATROLLING:
                # If near nest and not defending/hunting/returning, should patrol
                self._switch_state(AntState.PATROLLING, "NearNest->Patrol")
            elif not is_near_nest and self.state == AntState.PATROLLING:
                # If wandered too far while patrolling, switch to general searching
                self._switch_state(AntState.SEARCHING, "PatrolFar->Search")
            elif is_near_nest and self.state == AntState.SEARCHING:
                # If searching but wandered back near nest, switch to patrolling
                self._switch_state(AntState.PATROLLING, "SearchNear->Patrol")

    def _update_path_history(self, new_pos_int):
        """Adds a position to the history and removes old entries."""
        current_sim_ticks = self.simulation.ticks
        # Only add if it's a new position
        if not self.path_history or self.path_history[-1] != new_pos_int:
            self.path_history.append(new_pos_int)
            self.history_timestamps.append(current_sim_ticks)

            # Prune old history based on time, not just length
            cutoff_time = current_sim_ticks - WORKER_PATH_HISTORY_LENGTH # WORKER_PATH_HISTORY_LENGTH is time duration now
            cutoff_index = 0
            # Find the first index whose timestamp is *not* older than the cutoff
            while (cutoff_index < len(self.history_timestamps) and
                   self.history_timestamps[cutoff_index] < cutoff_time):
                cutoff_index += 1

            # Keep only the recent part of the history
            self.path_history = self.path_history[cutoff_index:]
            self.history_timestamps = self.history_timestamps[cutoff_index:]

    def _is_in_history(self, pos_int):
        """Checks if a position is in the recent path history."""
        return pos_int in self.path_history

    def _clear_path_history(self):
        """Clears the path history, e.g., when changing state."""
        self.path_history.clear()
        self.history_timestamps.clear()

    def _filter_valid_moves(self, potential_neighbors_int, ignore_history_near_nest=False):
        """Filters potential moves based on obstacles, other ants, and history."""
        sim = self.simulation
        valid_moves_int = []
        q_pos_int = sim.queen.pos if sim.queen else None
        pos_int = self.pos
        nest_pos_int = sim.nest_pos # Use dynamic nest position
        # Check if currently near the nest
        is_near_nest_now = distance_sq(pos_int, nest_pos_int) <= (NEST_RADIUS + 2) ** 2
        # Determine if path history should be checked for avoidance
        check_history_flag = not (ignore_history_near_nest and is_near_nest_now)

        for n_pos_int in potential_neighbors_int:
            # Check history avoidance first
            history_block = check_history_flag and self._is_in_history(n_pos_int)
            if not history_block:
                # Check other blocking factors
                is_queen_pos = n_pos_int == q_pos_int
                is_obstacle_pos = sim.grid.is_obstacle(n_pos_int)
                # Check if another ant (excluding self) is at the position
                is_ant_pos = sim.is_ant_at(n_pos_int, exclude_self=self)

                # If none of the blocking conditions are met, it's a valid move
                if not is_queen_pos and not is_obstacle_pos and not is_ant_pos:
                    valid_moves_int.append(n_pos_int)

        return valid_moves_int

    def _choose_move(self):
        """Determines the best neighboring cell to move to based on state."""
        sim = self.simulation
        # Get all 8 neighbours (or fewer if at edge)
        potential_neighbors_int = get_neighbors(self.pos, sim.grid_width, sim.grid_height)
        if not potential_neighbors_int:
            self.last_move_info = "No neighbors"
            return None # Cannot move

        # Filter valid moves (no obstacles, ants, history etc.)
        # Allow moving into history when returning near nest to avoid getting stuck
        ignore_hist_near_nest = self.state == AntState.RETURNING_TO_NEST
        valid_neighbors_int = self._filter_valid_moves(potential_neighbors_int, ignore_hist_near_nest)

        # If completely blocked by dynamic obstacles (ants) or history:
        if not valid_neighbors_int:
            self.last_move_info = "Blocked"
            # Try a fallback: allow moving into history, but still avoid obstacles/queen
            fallback_neighbors_int = []
            q_pos_int = sim.queen.pos if sim.queen else None
            for n_pos_int in potential_neighbors_int:
                 if (n_pos_int != q_pos_int and
                     not sim.grid.is_obstacle(n_pos_int) and
                     not sim.is_ant_at(n_pos_int, exclude_self=self)):
                     fallback_neighbors_int.append(n_pos_int)

            # If fallback options exist, prefer the oldest visited one
            if fallback_neighbors_int:
                # Sort by index in history (lower index = older), non-history first (-1)
                fallback_neighbors_int.sort(key=lambda p: self.path_history.index(p) if p in self.path_history else -1)
                # Choose the path visited longest ago
                return fallback_neighbors_int[0]
            return None # Truly blocked

        # --- State-Specific Move Logic ---

        # ESCAPING: Prioritize moving to unvisited cells
        if self.state == AntState.ESCAPING:
            # Filter valid moves to only those NOT in history
            escape_moves_int = [p for p in valid_neighbors_int if not self._is_in_history(p)]
            if escape_moves_int:
                self.last_move_info = "Esc->Unhist"
                return random.choice(escape_moves_int)
            else:
                # If all valid moves are in history, pick a random one (better than not moving)
                self.last_move_info = "Esc->Hist"
                return random.choice(valid_neighbors_int)

        # Standard states: Score moves based on current goal
        scoring_functions = {
            AntState.RETURNING_TO_NEST: self._score_moves_returning,
            AntState.SEARCHING: self._score_moves_searching,
            AntState.PATROLLING: self._score_moves_patrolling,
            AntState.DEFENDING: self._score_moves_defending,
            AntState.HUNTING: self._score_moves_hunting,
        }
        # Default to searching score if state is unknown (shouldn't happen)
        score_func = scoring_functions.get(self.state, self._score_moves_searching)

        # Calculate scores for all valid moves
        # Pass just_picked flag only for returning state scoring
        move_scores = score_func(valid_neighbors_int, self.just_picked_food) if self.state == AntState.RETURNING_TO_NEST else score_func(valid_neighbors_int)


        # If no scores calculated (e.g., error or no valid moves scored):
        if not move_scores:
            self.last_move_info = f"No scores({self.state.name})"
            # Fallback: choose randomly from the valid (but unscored) neighbours
            return random.choice(valid_neighbors_int)

        # Select the best move based on scores and state logic
        selected_move_int = None
        if self.state == AntState.RETURNING_TO_NEST:
            selected_move_int = self._select_best_move_returning(move_scores, valid_neighbors_int, self.just_picked_food)
        elif self.state in [AntState.DEFENDING, AntState.HUNTING]:
            # For urgent states, pick the absolute best score
            selected_move_int = self._select_best_move(move_scores, valid_neighbors_int)
        else: # SEARCHING, PATROLLING
            # Use probabilistic choice to allow exploration
            selected_move_int = self._select_probabilistic_move(move_scores, valid_neighbors_int)

        # Final fallback if selection failed
        return selected_move_int if selected_move_int else random.choice(valid_neighbors_int)

    def _score_moves_base(self, neighbor_pos_int):
        """Calculates base score components common to most states."""
        score = 0.0
        # Persistence bonus: encourage moving in the same direction
        move_dir = (neighbor_pos_int[0] - self.pos[0], neighbor_pos_int[1] - self.pos[1])
        if move_dir == self.last_move_direction and move_dir != (0, 0):
            score += W_PERSISTENCE
        # Random noise: Add small randomness to break ties and encourage exploration
        score += rnd_uniform(-W_RANDOM_NOISE, W_RANDOM_NOISE)
        # History penalty (applied strongly via filtering, but can add minor score penalty too)
        # if self._is_in_history(neighbor_pos_int):
        #     score += W_AVOID_HISTORY # Already filtered mostly, but could add minor negative score
        return score

    def _score_moves_returning(self, valid_neighbors_int, just_picked):
        """Scores moves for ants returning to the nest (carrying food or finished task)."""
        scores = {}
        sim = self.simulation
        pos_int = self.pos
        nest_pos_int = sim.nest_pos
        grid = sim.grid
        dist_sq_now = distance_sq(pos_int, nest_pos_int) # Current distance to nest

        for n_pos_int in valid_neighbors_int:
            score = self._score_moves_base(n_pos_int)

            # Pheromone influence
            home_ph = grid.get_pheromone(n_pos_int, "home")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            neg_ph = grid.get_pheromone(n_pos_int, "negative")

            score += home_ph * W_HOME_PHEROMONE_RETURN # Strong attraction to home trail

            # Directional bias towards nest
            if distance_sq(n_pos_int, nest_pos_int) < dist_sq_now:
                score += W_NEST_DIRECTION_RETURN # Bonus for getting closer

            # Avoidance (less strong than when searching)
            score += alarm_ph * W_ALARM_PHEROMONE * 0.3 # Avoid danger slightly
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.4 # Avoid negative areas

            scores[n_pos_int] = score
        return scores

    def _score_moves_searching(self, valid_neighbors_int):
        """Scores moves for ants searching for food."""
        scores = {}
        sim = self.simulation
        grid = sim.grid
        nest_pos_int = sim.nest_pos

        # Determine current food needs (weights change based on colony storage)
        sugar_needed = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD
        protein_needed = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD

        # Base weights for food pheromones
        w_sugar = W_FOOD_PHEROMONE_SEARCH_LOW_NEED
        w_protein = W_FOOD_PHEROMONE_SEARCH_LOW_NEED

        # Adjust weights based on specific needs
        if sugar_needed and not protein_needed:
             # Need sugar, avoid protein trails
             w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE
             w_protein = W_FOOD_PHEROMONE_SEARCH_AVOID
        elif protein_needed and not sugar_needed:
             # Need protein, avoid sugar trails
             w_protein = W_FOOD_PHEROMONE_SEARCH_BASE
             w_sugar = W_FOOD_PHEROMONE_SEARCH_AVOID
        elif sugar_needed and protein_needed:
             # Need both, slightly prefer the one that's lower relatively
             if sim.colony_food_storage_sugar <= sim.colony_food_storage_protein:
                  w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 1.1
                  w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 0.9
             else:
                  w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 1.1
                  w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 0.9
        else: # Neither critically low, follow general trails moderately
            if sim.colony_food_storage_sugar <= sim.colony_food_storage_protein * 1.5:
                 w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE * 0.6 # Slightly prefer sugar if balanced
            else:
                 w_protein = W_FOOD_PHEROMONE_SEARCH_BASE * 0.6 # Slightly prefer protein otherwise

        # Soldiers are less interested in food trails
        if self.caste == AntCaste.SOLDIER:
            w_sugar *= 0.1
            w_protein *= 0.1

        # Avoid searching inside/too close to the nest
        nest_search_avoid_radius_sq = (NEST_RADIUS * 1.8) ** 2

        for n_pos_int in valid_neighbors_int:
            score = self._score_moves_base(n_pos_int)

            # Get pheromone levels at the potential next cell
            home_ph = grid.get_pheromone(n_pos_int, "home")
            sugar_ph = grid.get_pheromone(n_pos_int, "food", FoodType.SUGAR)
            protein_ph = grid.get_pheromone(n_pos_int, "food", FoodType.PROTEIN)
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            # Apply weights to pheromones
            score += sugar_ph * w_sugar
            score += protein_ph * w_protein

            # Recruitment pheromone (strong signal for important events)
            # Soldiers might react more strongly to recruitment signals
            recruit_w = W_RECRUITMENT_PHEROMONE * (1.2 if self.caste == AntCaste.SOLDIER else 1.0)
            score += recr_ph * recruit_w

            # Avoidance pheromones
            score += neg_ph * W_NEGATIVE_PHEROMONE # Avoid explored/empty areas
            score += alarm_ph * W_ALARM_PHEROMONE # Avoid danger zones

            # Home pheromone influence (usually low or zero when searching)
            score += home_ph * W_HOME_PHEROMONE_SEARCH

            # Penalty for moving towards/staying near the nest while searching
            if distance_sq(n_pos_int, nest_pos_int) <= nest_search_avoid_radius_sq:
                score += W_AVOID_NEST_SEARCHING

            scores[n_pos_int] = score
        return scores

    def _score_moves_patrolling(self, valid_neighbors_int):
        """Scores moves for soldiers patrolling the nest area."""
        scores = {}
        sim = self.simulation
        grid = sim.grid
        pos_int = self.pos
        nest_pos_int = sim.nest_pos
        dist_sq_current = distance_sq(pos_int, nest_pos_int)
        # Use the configured patrol radius squared
        # patrol_radius_sq = SOLDIER_PATROL_RADIUS_SQ # Alte Zeile
        patrol_radius_sq = sim.soldier_patrol_radius_sq  # Neue Zeile
        # Define outer boundary slightly beyond patrol radius
        outer_boundary_sq = patrol_radius_sq * 1.4

        for n_pos_int in valid_neighbors_int:
            score = self._score_moves_base(n_pos_int)

            # Pheromone influence (less sensitive than searching)
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            score += recr_ph * W_RECRUITMENT_PHEROMONE * 0.7  # Follow recruitment moderately
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.5  # Slightly avoid negative
            score += alarm_ph * W_ALARM_PHEROMONE * 0.5  # Slightly avoid alarm

            # Directional control for patrolling
            dist_sq_next = distance_sq(n_pos_int, nest_pos_int)

            # Discourage moving further away if already inside patrol radius
            if dist_sq_current <= patrol_radius_sq and dist_sq_next > dist_sq_current:
                score += W_NEST_DIRECTION_PATROL  # Negative weight discourages moving away

            # Strong penalty for moving too far from the nest
            if dist_sq_next > outer_boundary_sq:
                score -= 8000  # Very strong discouragement

            scores[n_pos_int] = score
        return scores

    def _score_moves_defending(self, valid_neighbors_int):
        """Bewertet Züge für Ameisen, die sich gegen Bedrohungen verteidigen (folgt Alarm/Rekrutierung)."""
        scores = {}
        sim = self.simulation
        grid = sim.grid
        pos_int = self.pos

        # --- Update Target Location / Initialen Alarm zurücksetzen, damit der nicht immer wieder zurück will ----
        # Prüfe, ob der Timer abgelaufen ist oder die vorherige Quelle durch eine Stärkere ersetzt werden soll (Abfrage)
        if self.alarm_search_timer > 150:  # Timer ausgelaufen oder die Quelle ist zu alt
            self.last_known_alarm_pos = None  # Verwerfe die vorherige Alarmquelle
            self.initial_alarm_direction = None
            self.alarm_search_timer = 0  # Timer zurücksetzen

        # --- Update Target Location ---
        # Occasionally re-evaluate the strongest signal source nearby
        if self.last_known_alarm_pos is None or random.random() < 0.2:  # Re-scan periodically
            best_signal_pos = None
            max_signal_strength = -1.0
            # Scan a small radius around the ant
            search_radius_sq = ALARM_SEARCH_RADIUS_SIGNAL * ALARM_SEARCH_RADIUS_SIGNAL  # Increased from 6*6
            x0, y0 = pos_int
            min_scan_x = max(0, x0 - int(search_radius_sq ** 0.5))
            max_scan_x = min(sim.grid_width - 1, x0 + int(search_radius_sq ** 0.5))
            min_scan_y = max(0, y0 - int(search_radius_sq ** 0.5))
            max_scan_y = min(sim.grid_height - 1, y0 + int(search_radius_sq ** 0.5))

            for i in range(min_scan_x, max_scan_x + 1):
                for j in range(min_scan_y, max_scan_y + 1):
                    p_int = (i, j)
                    if distance_sq(pos_int, p_int) <= search_radius_sq:
                        # Combine alarm and recruitment signals, weighted
                        signal = (grid.get_pheromone(p_int, "alarm") * 1.2 +
                                  grid.get_pheromone(p_int, "recruitment") * 0.8)
                        # Add bonus if an enemy is directly visible at the location
                        if sim.get_enemy_at(p_int):
                            signal += 600  # Strong incentive to move towards visible enemy

                        if signal > max_signal_strength:
                            max_signal_strength = signal
                            best_signal_pos = p_int

            # Update the target if a strong signal was found
            if max_signal_strength > 80.0:  # Threshold to consider it a valid target
                self.last_known_alarm_pos = best_signal_pos
            else:
                # Signal faded or too weak, lose the target
                self.last_known_alarm_pos = None  # Will cause state change later

        # --- Score Moves Towards Target ---
        target_pos = self.last_known_alarm_pos
        dist_now_sq = distance_sq(pos_int, target_pos) if target_pos else float('inf')

        for n_pos_int in valid_neighbors_int:
            score = self._score_moves_base(n_pos_int)

            # Get pheromones at the potential next cell
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            # Very high bonus for moving onto a cell with an enemy
            enemy_at_n_pos = sim.get_enemy_at(n_pos_int)
            if enemy_at_n_pos:
                score += 15000  # Engage immediately

            # Wenn eine zufällige Suche gestartet wurde
            if target_pos:
                dist_next_sq = distance_sq(n_pos_int, target_pos)
                # Bonus fcore += W_ALARM_SOURCE_DEFEND * 1.5  # Increased
            else: # Zufälliges suchen im Umkreis
                random_x = random.randint(-ALARM_SEARCH_RADIUS_RANDOM, ALARM_SEARCH_RADIUS_RANDOM)
                random_y = random.randint(-ALARM_SEARCH_RADIUS_RANDOM, ALARM_SEARCH_RADIUS_RANDOM)
                search = (random_x + pos_int[0], random_y + pos_int[1])
                dis_rand_sq = distance_sq(n_pos_int, search)
                score += dis_rand_sq * 0.5

    def _score_moves_hunting(self, valid_neighbors_int):
        """Scores moves for ants hunting a specific prey target."""
        scores = {}
        sim = self.simulation
        grid = sim.grid
        pos_int = self.pos

        # Get the target prey's current position
        target_pos = self.target_prey.pos if (self.target_prey and hasattr(self.target_prey, 'pos')) else None

        # If no target, behave like searching (should be handled by state update, but safety)
        if not target_pos:
            return {n_pos_int: self._score_moves_base(n_pos_int) for n_pos_int in valid_neighbors_int}

        dist_sq_now = distance_sq(pos_int, target_pos)

        for n_pos_int in valid_neighbors_int:
            score = self._score_moves_base(n_pos_int)
            dist_sq_next = distance_sq(n_pos_int, target_pos)

            # Strong bonus for getting closer to the prey
            if dist_sq_next < dist_sq_now:
                score += W_HUNTING_TARGET

            # Minor influence from other pheromones (avoid negative/alarm slightly)
            score += grid.get_pheromone(n_pos_int, "alarm") * W_ALARM_PHEROMONE * 0.1
            score += grid.get_pheromone(n_pos_int, "negative") * W_NEGATIVE_PHEROMONE * 0.2

            scores[n_pos_int] = score
        return scores

    def _select_best_move(self, move_scores, valid_neighbors_int):
        """Selects the move with the absolute highest score (deterministic)."""
        best_score = -float("inf")
        best_moves_int = []

        # Find the highest score and all moves achieving it
        for pos_int, score in move_scores.items():
            if score > best_score:
                best_score = score
                best_moves_int = [pos_int] # New best, reset list
            elif score == best_score:
                best_moves_int.append(pos_int) # Tied score, add to list

        # If no best moves found (error?), fallback to random valid move
        if not best_moves_int:
            self.last_move_info += "(Best:Fallback!)"
            return random.choice(valid_neighbors_int)

        # If ties, choose randomly among the best
        chosen_int = random.choice(best_moves_int)

        # Debug info
        score = move_scores.get(chosen_int, -999)
        state_prefix = self.state.name[:4]
        self.last_move_info = f"{state_prefix} Best->{chosen_int} (S:{score:.1f})"

        return chosen_int

    def _select_best_move_returning(self, move_scores, valid_neighbors_int, just_picked):
        """Selects the best move for returning, prioritizing getting closer."""
        best_score = -float("inf")
        best_moves_int = []
        sim = self.simulation
        pos_int = self.pos
        nest_pos_int = sim.nest_pos
        dist_sq_now = distance_sq(pos_int, nest_pos_int)

        # Separate moves into those getting closer and others
        closer_moves = {}
        other_moves = {}
        for pos_int_cand, score in move_scores.items():
            if distance_sq(pos_int_cand, nest_pos_int) < dist_sq_now:
                closer_moves[pos_int_cand] = score
            else:
                other_moves[pos_int_cand] = score

        # Prioritize moves that get closer to the nest
        target_pool = {}
        selection_type = ""
        if closer_moves:
            target_pool = closer_moves
            selection_type = "Closer"
        elif other_moves: # If no closer moves, consider others
            target_pool = other_moves
            selection_type = "Other"
        else: # Should not happen if move_scores is not empty, but fallback
            target_pool = move_scores
            selection_type = "All(Fallback)"

        # If target pool is somehow empty, fallback
        if not target_pool:
            self.last_move_info += "(R: No moves?)"
            return random.choice(valid_neighbors_int) if valid_neighbors_int else None

        # Find the best score within the chosen pool (closer or other)
        for pos_int_cand, score in target_pool.items():
            if score > best_score:
                best_score = score
                best_moves_int = [pos_int_cand]
            elif score == best_score:
                best_moves_int.append(pos_int_cand)

        # If still no best move found (e.g., all scores were -inf in the pool)
        if not best_moves_int:
             self.last_move_info += f"(R: No best in {selection_type})"
             # Fallback: check the *original* full move_scores list if we filtered
             if target_pool is not move_scores:
                 best_score = -float('inf')
                 best_moves_int = []
                 for pos_int_cand, score in move_scores.items():
                     if score > best_score:
                          best_score = score
                          best_moves_int = [pos_int_cand]
                     elif score == best_score:
                          best_moves_int.append(pos_int_cand)

             # If *still* no best move, choose randomly from any valid neighbor
             if not best_moves_int:
                  return random.choice(valid_neighbors_int) if valid_neighbors_int else None

        # --- Tie-breaking for Returning Ants ---
        chosen_int = None
        if len(best_moves_int) == 1:
            # No tie, choose the single best move
            chosen_int = best_moves_int[0]
            self.last_move_info = f"R({selection_type})Best->{chosen_int} (S:{best_score:.1f})"
        else:
            # Tie! Use home pheromone level as a tie-breaker
            grid = sim.grid
            # Sort tied moves by home pheromone level (descending)
            best_moves_int.sort(key=lambda p: grid.get_pheromone(p, "home"), reverse=True)
            # Find the maximum pheromone level among the tied best moves
            max_ph = grid.get_pheromone(best_moves_int[0], "home")
            # Select randomly from all moves that have this maximum pheromone level
            top_ph_moves = [p for p in best_moves_int if grid.get_pheromone(p, "home") == max_ph]
            chosen_int = random.choice(top_ph_moves)
            self.last_move_info = f"R({selection_type})TieBrk->{chosen_int} (S:{best_score:.1f})"

        return chosen_int

    def _select_probabilistic_move(self, move_scores, valid_neighbors_int):
        """Selects a move probabilistically based on scores (for exploration)."""
        if not move_scores or not valid_neighbors_int:
             # Fallback if no scores or neighbors
             return random.choice(valid_neighbors_int) if valid_neighbors_int else None

        # Extract positions and scores
        pop_int = list(move_scores.keys())
        scores = np.array(list(move_scores.values()), dtype=np.float64) # Use float64 for precision

        # Handle edge cases: only one option
        if len(pop_int) == 0: return None
        if len(pop_int) == 1:
             self.last_move_info = f"{self.state.name[:3]} Prob->{pop_int[0]} (Only)"
             return pop_int[0]

        # Normalize scores to be non-negative for weighting
        min_score = np.min(scores)
        # Shift scores so the minimum is slightly positive (avoids issues with zero)
        shifted_scores = scores - min_score + 0.01

        # Apply temperature parameter for randomness tuning
        # Clamp temperature to prevent extreme values
        temp = min(max(PROBABILISTIC_CHOICE_TEMP, 0.1), 5.0)

        try:
             # Calculate weights = score ^ temperature
             # Use clipping to prevent overflow with very large scores/temps before power
             clipped_scores = np.clip(shifted_scores, 0, 1e6) # Limit base of power
             weights = np.power(clipped_scores, temp)
        except OverflowError:
             # Fallback if power calculation overflows (e.g., huge scores)
             print(f"Warning: Overflow in probabilistic weight calculation. Scores: {shifted_scores}, Temp: {temp}")
             # Use linear scaling or just max score as fallback
             weights = np.maximum(0.01, shifted_scores) # Linear fallback

        # Ensure weights are at least a minimum value
        weights = np.maximum(MIN_SCORE_FOR_PROB_CHOICE, weights)

        # Calculate total weight
        total_weight = np.sum(weights)

        # --- Sanity checks for weights and probabilities ---
        if total_weight <= 1e-9 or not np.isfinite(total_weight) or np.any(~np.isfinite(weights)):
            # Invalid weights (e.g., all zero, NaN, Inf)
            self.last_move_info += f"({self.state.name[:3]}:InvW)"
            # Fallback: Choose the move with the original highest score
            best_s = -float("inf")
            best_p = None
            # Iterate through original scores to find the best valid one
            for p_int_idx, s in enumerate(scores):
                if np.isfinite(s) and s > best_s:
                    best_s = s
                    best_p = pop_int[p_int_idx]
            return best_p if best_p else random.choice(valid_neighbors_int) # Final fallback

        # Calculate probabilities
        probabilities = weights / total_weight

        # Re-normalize probabilities just in case of floating point inaccuracies
        if not np.isclose(np.sum(probabilities), 1.0):
             if np.sum(probabilities) > 1e-9 and np.all(np.isfinite(probabilities)):
                  probabilities /= np.sum(probabilities) # Force normalization
                  # If it still fails after renormalization, something is very wrong
                  if not np.isclose(np.sum(probabilities), 1.0):
                      self.last_move_info += "(ProbReNormFail)"
                      # Fallback to best score method
                      best_s = -float('inf'); best_p = None
                      for p_int_idx, s in enumerate(scores):
                           if np.isfinite(s) and s > best_s: best_s = s; best_p = pop_int[p_int_idx]
                      return best_p if best_p else random.choice(valid_neighbors_int)
             else: # Sum is zero or contains NaN/Inf
                 self.last_move_info += "(ProbBadSum)"
                 # Fallback to best score method
                 best_s = -float('inf'); best_p = None
                 for p_int_idx, s in enumerate(scores):
                      if np.isfinite(s) and s > best_s: best_s = s; best_p = pop_int[p_int_idx]
                 return best_p if best_p else random.choice(valid_neighbors_int)

        # --- Perform the probabilistic choice ---
        try:
            # Use numpy's random.choice with the calculated probabilities
            chosen_index = np.random.choice(len(pop_int), p=probabilities)
            chosen_int = pop_int[chosen_index]

            # Debug info
            score = move_scores.get(chosen_int, -999)
            self.last_move_info = f"{self.state.name[:3]} Prob->{chosen_int} (S:{score:.1f})"
            return chosen_int
        except ValueError as e:
            # Catch errors like "probabilities do not sum to 1"
            print(f"WARN: Probabilistic choice error ({self.state.name}): {e}. Sum={np.sum(probabilities)}, Probs={probabilities}")
            self.last_move_info += "(ProbValErr)"
            # Fallback to best score method
            best_s = -float('inf'); best_p = None
            for p_int_idx, s in enumerate(scores):
                 if np.isfinite(s) and s > best_s: best_s = s; best_p = pop_int[p_int_idx]
            return best_p if best_p else random.choice(valid_neighbors_int) # Final fallback

    def _switch_state(self, new_state: AntState, reason: str):
        """Changes the ant's state and resets relevant variables."""
        if self.state != new_state:
            # print(f"Ant {id(self)}: {self.state.name} -> {new_state.name} ({reason}) at {self.pos}") # Debug
            self.state = new_state
            self.last_move_info = reason # Store reason for state change
            self._clear_path_history() # Clear history when changing major goals

            # Reset state-specific targets
            if new_state != AntState.HUNTING:
                self.target_prey = None
            if new_state != AntState.DEFENDING:
                self.last_known_alarm_pos = None

            # Reset stuck timer on state change
            self.stuck_timer = 0

    def update(self, speed_multiplier):
        """Main update logic for the ant."""
        sim = self.simulation
        attrs = ANT_ATTRIBUTES[self.caste]

        # --- Aging and Starvation ---
        self.age += speed_multiplier
        if self.age >= self.max_age_ticks:
            self.hp = 0  # Die of old age
            self.last_move_info = "Died of old age"
            return  # No further actions

        self.food_consumption_timer += speed_multiplier
        if self.food_consumption_timer >= WORKER_FOOD_CONSUMPTION_INTERVAL:
            self.food_consumption_timer %= WORKER_FOOD_CONSUMPTION_INTERVAL  # Reset timer preserving overshoot
            needed_s = self.food_consumption_sugar
            needed_p = self.food_consumption_protein

            # Check colony storage (ants eat from shared resources)
            can_eat = (sim.colony_food_storage_sugar >= needed_s and
                       sim.colony_food_storage_protein >= needed_p)

            if can_eat:
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p
            else:
                self.hp = 0  # Starved
                self.last_move_info = "Starved"
                return  # No further actions

        # --- Speed Boost ---
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= speed_multiplier
            # Apply boost to move delay (ensure it doesn't become negative if multiplier is huge)
            boosted_delay = attrs["speed_delay"] / self.speed_boost_multiplier
            self.move_delay_base = max(0, int(boosted_delay)) # Use max(0,...)
        else:
            self.speed_boost_multiplier = 1.0  # Reset multiplier
            self.move_delay_base = attrs["speed_delay"]  # Reset move delay

        # --- State Management ---
        # Handle escape timer countdown
        if self.state == AntState.ESCAPING:
            self.escape_timer -= speed_multiplier
            if self.escape_timer <= 0:
                # Escape finished, revert to default state
                next_state = AntState.PATROLLING if self.caste == AntCaste.SOLDIER else AntState.SEARCHING
                self._switch_state(next_state, "EscapeEnd")

        # Check for state transitions based on environment
        self._update_state()

        # Check if died during state update (e.g., starvation check moved there)
        if self.hp <= 0: return

        # --- Interaction Checks (Attack) ---
        pos_int = self.pos
        # Check immediate surroundings (including current cell) for enemies/prey
        neighbors_int = get_neighbors(pos_int, sim.grid_width, sim.grid_height, include_center=True)
        grid = sim.grid

        # --- Enemy Interaction ---
        target_enemy = None
        # --- DEBUGGING START ---
        found_enemy_nearby_debug = False # Flag für Debug-Ausgabe
        # --- DEBUGGING END ---
        for p_int in neighbors_int:
            enemy = sim.get_enemy_at(p_int)
            if enemy and enemy.hp > 0:
                # --- DEBUGGING START ---
                #print(f"DEBUG: Ant {id(self)} at {pos_int} DETECTED Enemy {id(enemy)} at {p_int} (HP: {enemy.hp:.1f})")
                found_enemy_nearby_debug = True
                # --- DEBUGGING END ---
                target_enemy = enemy
                break  # Attack the first enemy found

        # --- DEBUGGING START ---
        # Gib nur aus, wenn ein Feind in der Nähe gefunden wurde, um die Konsole nicht zu überfluten
        # if found_enemy_nearby_debug and target_enemy is None:
        #     print(f"WARN: Ant {id(self)} found enemy nearby but target_enemy is None!")
        # --- DEBUGGING END ---

        if target_enemy:
            # --- DEBUGGING START ---
            #print(f"DEBUG: Ant {id(self)} at {pos_int} is about to ATTACK Enemy {id(target_enemy)} at {target_enemy.pos}")
            # --- DEBUGGING END ---
            self.attack(target_enemy)
            grid.add_pheromone(pos_int, P_ALARM_FIGHT, "alarm")  # Signal danger
            self.stuck_timer = 0  # Reset stuck timer during fight
            self.target_prey = None  # Stop hunting if fighting
            self.last_move_info = f"FightEnemy@{target_enemy.pos}"
            # Ensure state is DEFENDING
            if self.state != AntState.DEFENDING:
                 # --- DEBUGGING START ---
                 # print(f"DEBUG: Ant {id(self)} switching to DEFENDING due to enemy contact.") # Optional: Noch mehr Details
                 # --- DEBUGGING END ---
                self._switch_state(AntState.DEFENDING, "EnemyContact!")
            return  # Attacked, skip movement this tick
        # --- DEBUGGING START ---
        # elif found_enemy_nearby_debug: # Wenn ein Feind gefunden wurde, aber target_enemy nicht gesetzt wurde (sollte nicht passieren) oder der if-Block nicht betreten wurde
        #     print(f"WARN: Ant {id(self)} detected enemy but did NOT enter attack block. State: {self.state.name}")
        # --- DEBUGGING END ---


        # --- Prey Interaction --- (Check if enemy attack already happened)
        target_prey_to_attack = None
        prey_in_range = []  # Find all prey in neighbor cells (use same neighbors_int)
        for p_int in neighbors_int:
            prey = sim.get_prey_at(p_int)
            if prey and prey.hp > 0:
                prey_in_range.append(prey)

        should_attack_prey = False
        if prey_in_range:
            # If currently HUNTING and the target is adjacent, attack it
            if (self.state == AntState.HUNTING and self.target_prey and
                    self.target_prey in prey_in_range):
                # Ensure the target is actually in a neighboring cell
                if self.target_prey.pos in neighbors_int:
                    should_attack_prey = True
                    target_prey_to_attack = self.target_prey
            # If not returning/defending, consider opportunistic attack
            elif self.state not in [AntState.RETURNING_TO_NEST, AntState.DEFENDING]:
                # Check if colony needs protein (worker) or if soldier (always hunts)
                can_hunt = ((self.caste == AntCaste.WORKER and sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * 2) or
                            (self.caste == AntCaste.SOLDIER))
                if can_hunt:
                    adjacent_prey = [p for p in prey_in_range if p.pos != pos_int]
                    prey_on_cell = [p for p in prey_in_range if p.pos == pos_int]

                    if adjacent_prey:
                        should_attack_prey = True
                        target_prey_to_attack = random.choice(adjacent_prey)
                    elif prey_on_cell:
                        should_attack_prey = True
                        target_prey_to_attack = random.choice(prey_on_cell)

        # Perform the attack if decided
        if should_attack_prey and target_prey_to_attack:
            # --- DEBUGGING ---
            # print(f"DEBUG: Ant {id(self)} at {pos_int} attacking Prey {id(target_prey_to_attack)} at {target_prey_to_attack.pos}")
            # ---
            self.attack(target_prey_to_attack)
            self.stuck_timer = 0  # Reset stuck timer
            self.last_move_info = f"AtkPrey@{target_prey_to_attack.pos}"

            if target_prey_to_attack.hp <= 0:
                killed_prey_pos = target_prey_to_attack.pos
                sim.kill_prey(target_prey_to_attack)
                grid.add_pheromone(killed_prey_pos, P_FOOD_AT_SOURCE, "food", FoodType.PROTEIN)
                grid.add_pheromone(killed_prey_pos, P_RECRUIT_PREY, "recruitment")
                if self.target_prey == target_prey_to_attack:
                    self.target_prey = None
                next_s = AntState.SEARCHING if self.caste == AntCaste.WORKER else AntState.PATROLLING
                self._switch_state(next_s, "PreyKilled")
            # Skip movement this tick since we attacked prey
            return

        # --- Movement ---
        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return  # Cannot move yet

        effective_delay_updates = 0
        if self.move_delay_base > 0:
            if speed_multiplier > 0:
                # Calculate effective # frames to wait. -1 because current frame is one step.
                effective_delay_updates = max(0, int(round(self.move_delay_base / speed_multiplier)) - 1)
            else:  # Paused
                effective_delay_updates = float('inf')
        self.move_delay_timer = effective_delay_updates

        # --- Choose and Execute Move ---
        old_pos_int = self.pos
        local_just_picked = self.just_picked_food
        self.just_picked_food = False

        new_pos_int = self._choose_move()

        moved = False
        found_food_type = None
        food_amount = 0.0

        if new_pos_int and new_pos_int != old_pos_int:
            self.pos = new_pos_int
            sim.update_entity_position(self, old_pos_int, new_pos_int)
            self.last_move_direction = (new_pos_int[0] - old_pos_int[0], new_pos_int[1] - old_pos_int[1])
            self._update_path_history(new_pos_int)
            self.stuck_timer = 0
            moved = True

            try:
                foods = grid.food[new_pos_int[0], new_pos_int[1]]
                if foods[FoodType.SUGAR.value] > 0.1:
                    found_food_type = FoodType.SUGAR
                    food_amount = foods[FoodType.SUGAR.value]
                elif foods[FoodType.PROTEIN.value] > 0.1:
                    found_food_type = FoodType.PROTEIN
                    food_amount = foods[FoodType.PROTEIN.value]
            except IndexError:
                pass

        elif new_pos_int == old_pos_int:
            self.stuck_timer += 1
            self.last_move_info += "(Move->Same)"
            self.last_move_direction = (0, 0)
        else:
            self.stuck_timer += 1
            self.last_move_info += "(NoChoice)"
            self.last_move_direction = (0, 0)

        # --- Post-Movement Actions (Pheromones, State Changes) ---
        pos_int = self.pos # Use potentially updated position
        nest_pos_int = sim.nest_pos
        is_near_nest = distance_sq(pos_int, nest_pos_int) <= NEST_RADIUS ** 2

        if self.state in [AntState.SEARCHING, AntState.HUNTING]:
            if (self.caste == AntCaste.WORKER and found_food_type and
                    self.carry_amount == 0):
                pickup_amount = min(self.max_capacity, food_amount)
                if pickup_amount > 0.01:
                    self.carry_amount = pickup_amount
                    self.carry_type = found_food_type
                    food_idx = found_food_type.value
                    try:
                        grid.food[pos_int[0], pos_int[1], food_idx] = max(0,
                                                                          grid.food[pos_int[0], pos_int[1],
                                                                          food_idx] - pickup_amount)
                    except IndexError: pass
                    grid.add_pheromone(pos_int, P_FOOD_AT_SOURCE, "food", food_type=found_food_type)
                    if food_amount >= RICH_FOOD_THRESHOLD:
                        grid.add_pheromone(pos_int, P_RECRUIT_FOOD, "recruitment")
                    self._switch_state(AntState.RETURNING_TO_NEST,
                                       f"Picked {found_food_type.name[:1]}({pickup_amount:.1f})")
                    self.just_picked_food = True
                    self.target_prey = None

            elif (moved and not found_food_type and self.state == AntState.SEARCHING and
                  distance_sq(pos_int, nest_pos_int) > (NEST_RADIUS + 3) ** 2):
                if is_valid_pos(old_pos_int, sim.grid_width, sim.grid_height):
                    grid.add_pheromone(old_pos_int, P_NEGATIVE_SEARCH, "negative")

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
                state_reason = "Dropped->"
                if self.caste == AntCaste.WORKER:
                    sugar_crit = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD
                    protein_crit = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD
                    if sugar_crit or protein_crit: state_reason += "SEARCH(Need!)"
                    else: state_reason += "SEARCH"
                elif self.caste == AntCaste.SOLDIER:
                    next_state = AntState.PATROLLING
                    state_reason += "PATROL"
                self._switch_state(next_state, state_reason)

            elif moved and not local_just_picked:
                if is_valid_pos(old_pos_int, sim.grid_width, sim.grid_height) and distance_sq(old_pos_int, nest_pos_int) > (NEST_RADIUS - 1) ** 2:
                    grid.add_pheromone(old_pos_int, P_HOME_RETURNING, "home")
                    if self.carry_amount > 0 and self.carry_type:
                        grid.add_pheromone(old_pos_int, P_FOOD_RETURNING_TRAIL, "food", food_type=self.carry_type)

        # --- Stuck Detection and Escape ---
        if self.stuck_timer >= WORKER_STUCK_THRESHOLD and self.state != AntState.ESCAPING:
            is_fighting = False
            is_hunting_adjacent = False
            # Re-check neighbors for stuck reason
            neighbors_int_stuck = get_neighbors(pos_int, sim.grid_width, sim.grid_height, True)
            for p_int in neighbors_int_stuck:
                if sim.get_enemy_at(p_int):
                    is_fighting = True
                    break
            if not is_fighting and self.state == AntState.HUNTING and self.target_prey:
                 # Check if the target prey still exists and is adjacent
                 if self.target_prey in sim.prey and self.target_prey.pos in neighbors_int_stuck:
                      is_hunting_adjacent = True

            # Only enter ESCAPING state if not stuck due to fighting/hunting
            if not is_fighting and not is_hunting_adjacent:
                # --- DEBUGGING ---
                # print(f"DEBUG: Ant {id(self)} entering ESCAPING state due to stuck timer.")
                # ---
                self._switch_state(AntState.ESCAPING, "Stuck!")
                self.escape_timer = WORKER_ESCAPE_DURATION
                self.stuck_timer = 0 # Reset stuck timer only when entering escape

    def attack(self, target):
        """Deals damage to a target (Enemy or Prey)."""
        sim = self.simulation # Get simulation instance
        # Check if target is valid and has take_damage method
        if isinstance(target, (Enemy, Prey)) and hasattr(target, 'take_damage'):
            target_pos = target.pos # Get target position before potential death
            target.take_damage(self.attack_power, self)
            # --- NEW: Record attack for visual indicator ---
            sim.add_attack_indicator(self.pos, target_pos, ATTACK_INDICATOR_COLOR_ANT)
            # --- END NEW ---
            # --- NEW: Speed Boost ---
            self.speed_boost_timer = ANT_SPEED_BOOST_DURATION
            self.speed_boost_multiplier = ANT_SPEED_BOOST_MULTIPLIER
        # --- NEW: Speed Boost ---
        self.speed_boost_timer = ANT_SPEED_BOOST_DURATION # This line seems duplicated, keep one.
        self.speed_boost_multiplier = ANT_SPEED_BOOST_MULTIPLIER # This line seems duplicated, keep one.

    def take_damage(self, amount, attacker):
        """Reduces HP and potentially drops pheromones upon being hit."""
        if self.hp <= 0: return  # Already dead

        # Get grid *before* checking HP and using it
        grid = self.simulation.grid
        pos_int = self.pos

        if self.hp > 0:
            # Queen taking damage is a major threat
            # Drop very strong alarm and recruitment signals
            grid.add_pheromone(pos_int, P_ALARM_FIGHT * 1.5, "alarm")  # Increased
            # Drop recruitment pheromone (stronger if soldier)
            recruit_amount = P_RECRUIT_DAMAGE_SOLDIER * 1.5 if self.caste == AntCaste.SOLDIER else P_RECRUIT_DAMAGE * 1.5  # Increased
            grid.add_pheromone(pos_int, recruit_amount, "recruitment")
        else:
            self.hp = 0  # Ensure HP doesn't go negative
            # Ant died, could add logic here (e.g., drop negative pheromone?)

# --- Queen Class ---
class Queen:
    """Manages queen state, egg laying, and represents the colony's core."""

    def __init__(self, pos, sim):
        self.pos = tuple(map(int, pos))
        self.simulation = sim
        self.hp = float(QUEEN_HP)
        self.max_hp = float(QUEEN_HP)
        self.age = 0.0  # In ticks
        self.egg_lay_timer_progress = 0.0  # Progress towards next egg lay attempt
        self.egg_lay_interval_ticks = QUEEN_EGG_LAY_RATE  # Ticks between attempts
        self.color = QUEEN_COLOR
        self.attack_power = 0  # Queen doesn't attack
        self.carry_amount = 0  # Queen doesn't carry food

    def update(self, speed_multiplier):
        """Updates queen's age and handles egg laying."""
        sim = self.simulation
        if speed_multiplier == 0.0: return  # Paused

        self.age += speed_multiplier

        # --- Egg Laying ---
        self.egg_lay_timer_progress += speed_multiplier
        if self.egg_lay_timer_progress >= self.egg_lay_interval_ticks:
            self.egg_lay_timer_progress %= self.egg_lay_interval_ticks  # Reset timer

            # --- NEU: Überprüfe, ob die maximale Ameisenanzahl erreicht ist ---
            if len(sim.ants) >= MAX_ANTS:
                # print("Maximale Ameisenanzahl erreicht. Königin legt keine Eier mehr.") # Optional: Debug-Ausgabe
                return  # Beende die Methode, ohne ein Ei zu legen

            # Check if colony has enough resources to lay an egg
            needed_s = QUEEN_FOOD_PER_EGG_SUGAR
            needed_p = QUEEN_FOOD_PER_EGG_PROTEIN
            can_lay = (sim.colony_food_storage_sugar >= needed_s and
                       sim.colony_food_storage_protein >= needed_p)

            if can_lay:
                # Consume resources
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p

                # Decide caste and find position
                caste = self._decide_caste()
                egg_pos = self._find_egg_position()

                # If a valid position was found, create and add the egg
                if egg_pos:
                    egg = BroodItem(BroodStage.EGG, caste, egg_pos, int(sim.ticks), sim)
                    sim.add_brood(egg)
            # else: # Optional: print("Queen cannot lay egg - insufficient food")

    def _decide_caste(self):
        """Decides whether to lay a worker or soldier egg based on colony ratio."""
        sim = self.simulation
        soldier_count = 0
        worker_count = 0

        # Count existing ants
        for a in sim.ants:
            if a.caste == AntCaste.SOLDIER:
                soldier_count += 1
            else:
                worker_count += 1
        # Count developing brood (larva/pupa)
        for b in sim.brood:
            if b.stage in [BroodStage.LARVA, BroodStage.PUPA]:
                if b.caste == AntCaste.SOLDIER:
                    soldier_count += 1
                else:
                    worker_count += 1

        total_population = soldier_count + worker_count
        current_ratio = 0.0
        if total_population > 0:
            current_ratio = soldier_count / total_population

        target_ratio = QUEEN_SOLDIER_RATIO_TARGET

        # Adjust probability based on current ratio vs target
        # More likely to lay soldier if below target ratio
        if current_ratio < target_ratio:
            # Higher chance for soldier if below target
            return AntCaste.SOLDIER if random.random() < 0.65 else AntCaste.WORKER
        elif random.random() < 0.04:  # Small base chance for soldier even if above ratio
            return AntCaste.SOLDIER
        else:  # Default to worker
            return AntCaste.WORKER

    def _find_egg_position(self):
        """Finds a suitable nearby valid cell to place a new egg."""
        sim = self.simulation
        # Get neighbors around the queen
        possible_spots = get_neighbors(self.pos, sim.grid_width, sim.grid_height)
        # Filter out obstacles
        valid_spots = [p for p in possible_spots if not sim.grid.is_obstacle(p)]

        if not valid_spots: return None  # No valid spots nearby

        # Get positions currently occupied by other brood items
        brood_positions = sim.get_brood_positions()
        # Prefer spots that are valid and not occupied by other brood
        free_valid_spots = [p for p in valid_spots if p not in brood_positions]

        if free_valid_spots:
            # Choose randomly from free spots
            return random.choice(free_valid_spots)
        else:
            # If all valid spots have brood, choose randomly from any valid spot
            # (allows stacking, might need refinement later)
            return random.choice(valid_spots)

    def take_damage(self, amount, attacker):
        """Handles queen taking damage, signals high alarm."""
        if self.hp <= 0: return  # Already dead
        self.hp -= amount
        if self.hp > 0:
            # Queen taking damage is a major threat
            grid = self.simulation.grid
            pos_int = self.pos
            # Drop very strong alarm and recruitment signals
            grid.add_pheromone(pos_int, P_ALARM_FIGHT * 8, "alarm")  # Increased
            grid.add_pheromone(pos_int, P_RECRUIT_DAMAGE * 8, "recruitment")  # Increased
        else:
            self.hp = 0
            # Queen died, simulation end logic is handled in AntSimulation

# --- Enemy Class ---
class Enemy:
    """Represents an enemy entity that attacks ants and the queen."""
    def __init__(self, pos, sim):
        self.pos = tuple(map(int, pos))
        self.simulation = sim
        self.hp = float(ENEMY_HP)
        self.max_hp = float(ENEMY_HP)
        self.attack_power = ENEMY_ATTACK
        self.move_delay_base = ENEMY_MOVE_DELAY # Ticks between moves
        self.move_delay_timer = rnd_uniform(0, self.move_delay_base) # Ticks until next move
        self.color = ENEMY_COLOR

    def update(self, speed_multiplier):
        """Updates enemy state: attacks nearby ants/queen or moves."""
        sim = self.simulation
        if speed_multiplier == 0.0: return  # Paused

        # --- Attack Logic ---
        pos_int = self.pos
        # Check neighbors (including current cell) for targets
        neighbors_int = get_neighbors(pos_int, sim.grid_width, sim.grid_height, include_center=True)
        target_ant = None
        queen_target = None

        # Prioritize attacking the Queen if adjacent
        for p_int in neighbors_int:
            # Use simulation's get_ant_at which includes the queen
            ant_or_queen = sim.get_ant_at(p_int)
            if ant_or_queen and ant_or_queen.hp > 0:
                # Check if it's the queen
                if isinstance(ant_or_queen, Queen):
                    queen_target = ant_or_queen
                    break  # Found queen, attack immediately
                # If not queen, store the first ant found as potential target
                elif target_ant is None:
                    target_ant = ant_or_queen

        # Choose target: Queen > Ant
        chosen_target = queen_target if queen_target else target_ant

        # If a target was found, attack and skip movement
        if chosen_target:
            self.attack(chosen_target)
            # print(f"Enemy at {self.pos} attacks {type(chosen_target).__name__} at {chosen_target.pos}") # Debug
            return  # Attacked, end turn

        # --- Movement Logic ---
        # Update move timer
        self.move_delay_timer -= speed_multiplier
        if self.move_delay_timer > 0:
            return  # Not time to move yet
        # Reset timer
        self.move_delay_timer += self.move_delay_base

        # Find valid moves
        possible_moves_int = get_neighbors(pos_int, sim.grid_width, sim.grid_height)
        valid_moves_int = []
        for m_int in possible_moves_int:
            # Check obstacle, other enemies, ants, and prey
            if (not sim.grid.is_obstacle(m_int) and
                    not sim.is_enemy_at(m_int, exclude_self=self) and
                    not sim.is_ant_at(m_int) and
                    not sim.is_prey_at(m_int)):
                valid_moves_int.append(m_int)

        # If valid moves exist, choose one
        if valid_moves_int:
            chosen_move_int = None
            nest_pos_int = sim.nest_pos  # Use dynamic nest position

            # Small chance to move towards the nest
            if random.random() < ENEMY_NEST_ATTRACTION:
                best_nest_move = None
                # Start with current distance as max
                min_dist_sq_to_nest = distance_sq(pos_int, nest_pos_int)
                # Find the valid move that gets closest to the nest
                for move in valid_moves_int:
                    d_sq = distance_sq(move, nest_pos_int)
                    if d_sq < min_dist_sq_to_nest:
                        min_dist_sq_to_nest = d_sq
                        best_nest_move = move
                # Choose the best nest move if found, otherwise random
                chosen_move_int = best_nest_move if best_nest_move else random.choice(valid_moves_int)
            else:
                # Default: random move
                chosen_move_int = random.choice(valid_moves_int)

            # Execute the move if chosen and different from current
            if chosen_move_int and chosen_move_int != self.pos:
                old_pos = self.pos
                self.pos = chosen_move_int
                sim.update_entity_position(self, old_pos, self.pos)  # Notify simulation

    def attack(self, target):
        """Deals damage to a target ant or queen."""
        sim = self.simulation # Get simulation instance
        # Check type and method existence for safety
        if isinstance(target, (Ant, Queen)) and hasattr(target, 'take_damage'):
            target_pos = target.pos # Get target position
            target.take_damage(self.attack_power, self)
            # --- NEW: Record attack for visual indicator ---
            sim.add_attack_indicator(self.pos, target_pos, ATTACK_INDICATOR_COLOR_ENEMY)
            # --- END NEW ---

    def take_damage(self, amount, attacker):
        """Reduces HP when attacked."""
        if self.hp <= 0: return # Already dead
        self.hp -= amount
        if self.hp > 0:
             # Get grid *before* checking HP and using it
             grid = self.simulation.grid # Get the grid
             pos_int = self.pos
             # Drop alarm pheromone (less amount than during active fight)
             grid.add_pheromone(pos_int, P_ALARM_FIGHT * 1.5, "alarm")  # Increased
        else:
             self.hp = 0
             # print(f"Enemy died at {self.pos}") # Optional debug

    def draw(self, surface):
        """Draws the enemy (spider-like)."""
        sim = self.simulation
        if not is_valid_pos(self.pos, sim.grid_width, sim.grid_height):
            return

        cs = sim.cell_size
        pos_px = (int(self.pos[0] * cs + cs / 2),
                  int(self.pos[1] * cs + cs / 2))

        # Body (slightly elongated ellipse)
        body_width = max(2, int(cs * 0.45))
        body_height = max(3, int(cs * 0.6))
        body_rect = pygame.Rect(pos_px[0] - body_width // 2, pos_px[1] - body_height // 2, body_width, body_height)
        body_color = self.color
        pygame.draw.ellipse(surface, body_color, body_rect)

        # Legs (8 thin lines)
        num_legs = 8
        leg_length = cs * 0.4
        leg_color = tuple(max(0, c - 60) for c in body_color) # Darker color for legs
        leg_thickness = max(1, cs // 10) # Make legs slightly thicker with larger cells

        # Define angles - slightly irregular spacing
        base_angles = [
            math.pi * 0.20, math.pi * 0.45, math.pi * 0.55, math.pi * 0.80, # One side
            math.pi * 1.20, math.pi * 1.45, math.pi * 1.55, math.pi * 1.80  # Other side
        ]

        # Attach point slightly offset towards the center top/bottom for visual appeal
        attach_offset_y = body_height * 0.15

        for i, angle in enumerate(base_angles):
            # Alternate attach points slightly for visual separation
            attach_y = pos_px[1] - attach_offset_y if i < 4 else pos_px[1] + attach_offset_y
            attach_point = (pos_px[0], attach_y)

            # Add slight randomness to angle
            angle += rnd_uniform(-0.08, 0.08)

            # Calculate end point
            end_x = attach_point[0] + leg_length * math.cos(angle)
            end_y = attach_point[1] + leg_length * math.sin(angle)

            # Draw leg
            pygame.draw.line(surface, leg_color, attach_point, (int(end_x), int(end_y)), leg_thickness)

        # Outline the body
        pygame.draw.ellipse(surface, (0, 0, 0), body_rect, 1) # Black outline

# --- Main Simulation Class ---
class AntSimulation:
    """Manages the overall simulation state, entities, drawing, and UI."""

    def __init__(self):
        self.app_running = True  # Flag to control the main application loop
        self.simulation_running = False  # Flag to control the current simulation run

        # --- Pygame and Display Initialization ---
        try:
            pygame.init()
            if not pygame.display.get_init():
                raise RuntimeError("Display module failed")
            os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center window
            pygame.display.set_caption("Ant Simulation - Responsive")

            # Determine Screen/Window Size
            screen_info = pygame.display.Info()
            monitor_width, monitor_height = screen_info.current_w, screen_info.current_h
            print(f"Detected Monitor Size: {monitor_width}x{monitor_height}")

            if USE_FULLSCREEN:
                self.screen_width = monitor_width
                self.screen_height = monitor_height
                display_flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
                print("Using Fullscreen Mode")
            else:
                target_w, target_h = DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
                if USE_SCREEN_PERCENT is not None and 0.1 <= USE_SCREEN_PERCENT <= 1.0:
                    target_w = int(monitor_width * USE_SCREEN_PERCENT)
                    target_h = int(monitor_height * USE_SCREEN_PERCENT)
                    print(f"Using {USE_SCREEN_PERCENT * 100:.0f}% of screen: {target_w}x{target_h}")
                else:
                    print(f"Using Default Window Size: {target_w}x{target_h}")
                # Ensure window is not larger than monitor
                self.screen_width = min(target_w, monitor_width)
                self.screen_height = min(target_h, monitor_height)
                display_flags = pygame.DOUBLEBUF  # Use default flags for windowed

            # Set the display mode
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), display_flags)
            print(f"Set Display Mode: {self.screen.get_width()}x{self.screen.get_height()}")

        except Exception as e:
            print(f"FATAL: Pygame/Display initialization failed: {e}")
            traceback.print_exc()
            self.app_running = False
            return  # Cannot continue

        # --- Calculate Grid Dimensions based on Screen ---
        self.cell_size = CELL_SIZE
        # Ensure grid dimensions are at least 1
        self.grid_width = max(1, self.screen.get_width() // self.cell_size)
        self.grid_height = max(1, self.screen.get_height() // self.cell_size)
        # Recalculate actual pixel width/height based on grid
        self.width = self.grid_width * self.cell_size
        self.height = self.grid_height * self.cell_size
        # Adjust screen surface size if calculated grid is smaller than requested window
        if self.width < self.screen.get_width() or self.height < self.screen.get_height():
            print(f"Adjusting screen surface to grid dimensions: {self.width}x{self.height}")
            self.screen = pygame.display.set_mode((self.width, self.height), display_flags)

        # --- Calculate Nest Position (Center of Grid) ---
        self.nest_pos = (self.grid_width // 2, self.grid_height // 2)
        print(
            f"Calculated Grid: {self.grid_width}x{self.grid_height}, Cell Size: {self.cell_size}, Nest: {self.nest_pos}")

        # --- Initialize Fonts (Scaled) ---
        self.font = None
        self.debug_font = None
        self.legend_font = None
        self._init_fonts()  # Initialize after screen size is known
        if not self.app_running:
            return  # Font init might fail

        # --- Simulation State Variables ---
        self.clock = pygame.time.Clock()
        self.grid = WorldGrid(self.grid_width, self.grid_height)
        self.end_game_reason = ""
        self.colony_generation = 0
        self.ticks = 0.0  # Use float for accumulated time with speed multiplier
        self.soldier_patrol_radius_sq = (NEST_RADIUS * SOLDIER_PATROL_RADIUS_MULTIPLIER) ** 2

        # Entity Lists and Position Lookups
        self.ants = []
        self.enemies = []
        self.brood = []
        self.prey = []
        self.queen: Queen | None = None
        self.ant_positions = {}  # { (x, y): ant_instance }
        self.enemy_positions = {}
        self.prey_positions = {}
        self.brood_positions = {}  # { (x, y): [brood_item1, brood_item2,...] }
        self.recent_attacks = []  # --- For attack indicators ---

        # --- NEW: Ant Position Array ---
        self.max_ants = MAX_ANTS  # Store max ants for array size
        # Use float for position to avoid potential issues, or keep int if sure
        # self.ant_positions_array = np.full((self.max_ants, 2), -1.0, dtype=np.float32) # Example with float
        self.ant_positions_array = np.full((self.max_ants, 2), -1, dtype=np.int16)  # Sticking with int16
        self.ant_indices = {}  # {ant_instance: index in array}
        self.next_ant_index = 0  # Tracks next available index (for efficient adding)

        # Colony Resources
        self.colony_food_storage_sugar = 0.0
        self.colony_food_storage_protein = 0.0

        # Timers
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_interval_ticks = ENEMY_SPAWN_RATE
        self.prey_spawn_timer = 0.0
        self.prey_spawn_interval_ticks = PREY_SPAWN_RATE
        self.food_replenish_timer = 0.0
        self.food_replenish_interval_ticks = FOOD_REPLENISH_RATE

        # UI State
        self.show_debug_info = True
        self.show_legend = False
        self.simulation_speed_index = DEFAULT_SPEED_INDEX
        self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
        self.buttons = self._create_buttons()  # Create after screen size and font are known

        # Drawing Surfaces
        self.static_background_surface = pygame.Surface((self.width, self.height))
        self.latest_frame_surface = None  # For network streaming
        self.food_dot_rng = random.Random()  # --- NEU: Dedizierter RNG für Futterpunkte ---
        self._prepare_static_background()  # Draw initial obstacles here

        # --- Spatial Grid ---
        self.spatial_grid = SpatialGrid(self.width, self.height, self.cell_size)  # Use pixel width/height

        # --- Start Optional Network Stream ---
        self._start_streaming_server_if_enabled()

        # --- Initial Simulation Reset ---
        if self.app_running:  # Only reset if init was successful so far
            self._reset_simulation()
            # Note: simulation_running flag is set to True inside _reset_simulation

    def _init_fonts(self):
        """Initializes fonts, scaling them based on screen height."""
        if not pygame.font.get_init():
             print("FATAL: Font module not initialized.")
             self.app_running = False
             return

        # Calculate scaling factor based on current height vs reference height
        scale_factor = self.screen.get_height() / REFERENCE_HEIGHT_FOR_SCALING
        # Clamp scale factor to avoid excessively large/small fonts
        scale_factor = max(0.7, min(1.5, scale_factor)) # Adjust clamps as needed

        # Calculate scaled font sizes (ensure they are at least 1)
        font_size = max(8, int(BASE_FONT_SIZE * scale_factor))
        debug_font_size = max(8, int(BASE_DEBUG_FONT_SIZE * scale_factor))
        legend_font_size = max(8, int(BASE_LEGEND_FONT_SIZE * scale_factor))
        print(f"Font scaling factor: {scale_factor:.2f} -> Sizes: Main={font_size}, Debug={debug_font_size}, Legend={legend_font_size}")

        try:
            # Try loading system fonts first
            try:
                self.font = pygame.font.SysFont("sans", font_size)
                self.debug_font = pygame.font.SysFont("monospace", debug_font_size)
                self.legend_font = pygame.font.SysFont("sans", legend_font_size)
                print("Using scaled system 'sans' and 'monospace' fonts.")
            except Exception:
                print("System fonts not found or scaling failed. Trying default font.")
                # Fallback to default Pygame font with scaled size
                self.font = pygame.font.Font(None, font_size)
                self.debug_font = pygame.font.Font(None, debug_font_size)
                self.legend_font = pygame.font.Font(None, legend_font_size)
                print("Using scaled Pygame default font.")

            # Final check if any font failed to load
            if not self.font or not self.debug_font or not self.legend_font:
                 raise RuntimeError("Font loading failed even with fallback.")

        except Exception as e:
            print(f"FATAL: Font initialization failed: {e}. Cannot render text.")
            self.font = None; self.debug_font = None; self.legend_font = None
            self.app_running = False # Cannot run without fonts

    def _start_streaming_server_if_enabled(self):
        """Starts the Flask streaming server in a thread if enabled."""
        global streaming_app, streaming_thread, stop_streaming_event
        if not ENABLE_NETWORK_STREAM or not Flask:
            if ENABLE_NETWORK_STREAM and not Flask:
                print("WARNING: ENABLE_NETWORK_STREAM=True, but Flask not installed. Streaming disabled.")
            return # Do nothing if disabled or Flask not available

        # --- Flask App Setup (within the method to access self.width/height) ---
        streaming_app = Flask(__name__)
        stop_streaming_event.clear() # Ensure event is clear before starting

        # Capture self.width and self.height for the template context
        template_width = self.width
        template_height = self.height

        @streaming_app.route('/')
        def index():
            """Serves the HTML page to display the stream."""
            # Pass the captured dimensions to the template
            return render_template_string(HTML_TEMPLATE, width=template_width, height=template_height)

        @streaming_app.route('/video_feed')
        def video_feed():
            """Route for the MJPEG stream."""
            return Response(stream_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        # --- End Flask App Setup ---

        # Start Flask in a daemon thread
        streaming_thread = threading.Thread(
             target=run_server,
             args=(streaming_app, STREAMING_HOST, STREAMING_PORT),
             daemon=True # Ensures thread exits when main program exits
             )
        streaming_thread.start()

    def _stop_streaming_server(self):
        """Signals the streaming thread to stop."""
        global streaming_thread, stop_streaming_event
        if streaming_thread and streaming_thread.is_alive():
            print(" * Stopping Flask server thread...")
            stop_streaming_event.set()
            # Note: Stopping Flask cleanly from another thread is tricky.
            # Setting the event and daemon=True is often sufficient for shutdown.
            # For more robust shutdown, might need requests to a shutdown endpoint.
            streaming_thread.join(timeout=1) # Optional: Wait briefly for thread

    def _reset_simulation(self):
        """Resets the simulation state to start a new colony."""
        print(f"\nResetting simulation (Kolonie {self.colony_generation + 1})...")
        self.ticks = 0.0
        self.ants.clear()
        self.enemies.clear()
        self.brood.clear()
        self.prey.clear()
        self.ant_positions.clear()
        self.enemy_positions.clear()
        self.prey_positions.clear()
        self.brood_positions.clear()
        self.queen = None # Clear queen reference

        # Reset resources
        self.colony_food_storage_sugar = INITIAL_COLONY_FOOD_SUGAR
        self.colony_food_storage_protein = INITIAL_COLONY_FOOD_PROTEIN

        # Reset timers
        self.enemy_spawn_timer = 0.0
        self.prey_spawn_timer = 0.0
        self.food_replenish_timer = 0.0

        self.end_game_reason = ""
        self.colony_generation += 1

        # Reset grid (pass the dynamic nest position)
        self.grid.reset(self.nest_pos)
        self._prepare_static_background() # Redraw obstacles

        # Spawn initial entities
        if not self._spawn_initial_entities():
            print("CRITICAL ERROR during simulation reset (entity spawn). Cannot continue.")
            self.simulation_running = False
            self.app_running = False
            self.end_game_reason = "Initialisierungsfehler"
            return

        # Reset speed and start simulation
        self.simulation_speed_index = DEFAULT_SPEED_INDEX
        self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
        self.simulation_running = True
        print(f"Kolonie {self.colony_generation} gestartet at {SPEED_MULTIPLIERS[self.simulation_speed_index]:.1f}x speed.")

    def _prepare_static_background(self):
        """Draws static elements (background color, obstacles) onto a surface."""
        self.static_background_surface.fill(MAP_BG_COLOR)
        cs = self.cell_size
        # Find coordinates where obstacles are True
        obstacle_coords = np.argwhere(self.grid.obstacles)
        # Draw rectangles for each obstacle coordinate
        base_r, base_g, base_b = OBSTACLE_COLOR
        var = OBSTACLE_COLOR_VARIATION
        for x, y in obstacle_coords:
            # --- NEW: Add color variation ---
            r = max(0, min(255, base_r + rnd(-var, var)))
            g = max(0, min(255, base_g + rnd(-var, var)))
            b = max(0, min(255, base_b + rnd(-var, var)))
            color = (r, g, b)
            # --- END NEW ---
            pygame.draw.rect(self.static_background_surface, color, (x * cs, y * cs, cs, cs)) # Use varied color
        print(f"Prepared static background with {len(obstacle_coords)} organic obstacles.")

    def _create_buttons(self):
        """Creates UI buttons with relative positioning."""
        buttons = []
        if not self.font: return buttons # Cannot create buttons without font

        # Define button properties
        # Scale button size slightly based on font size? Optional. Keep fixed for now.
        button_h = max(20, int(self.font.get_height() * 1.5)) # Height based on font
        button_w = max(60, int(button_h * 3.5)) # Width relative to height
        margin = 5 # Pixel margin between buttons

        button_definitions = [
            {"text": "Stats", "action": "toggle_debug", "key": pygame.K_d},
            {"text": "Legend", "action": "toggle_legend", "key": pygame.K_l},
            {"text": "Speed (-)", "action": "speed_down", "key": pygame.K_MINUS},
            {"text": "Speed (+)", "action": "speed_up", "key": pygame.K_PLUS},
            {"text": "Restart", "action": "restart", "key": None}, # No default key
            {"text": "Quit", "action": "quit", "key": pygame.K_ESCAPE},
        ]

        # Calculate total width needed for all buttons and margins
        num_buttons = len(button_definitions)
        total_buttons_width = num_buttons * button_w + (num_buttons - 1) * margin

        # Calculate starting X position to center the buttons horizontally
        start_x = (self.width - total_buttons_width) // 2
        start_y = margin # Place buttons near the top margin

        # Create button dictionaries
        for i, button_def in enumerate(button_definitions):
            # Calculate position for this button
            button_x = start_x + i * (button_w + margin)
            rect = pygame.Rect(button_x, start_y, button_w, button_h)
            buttons.append({
                "rect": rect,
                "text": button_def["text"],
                "action": button_def["action"],
                "key": button_def["key"] # Store associated key if any
            })

        return buttons

    def _spawn_initial_entities(self):
        """Spawns the queen, initial ants, enemies, and prey."""
        # Queen placement (already determined nest_pos)
        queen_pos = self._find_valid_queen_pos()
        if queen_pos:
            self.queen = Queen(queen_pos, self)
            print(f"Queen placed at {self.queen.pos}")
        else:
            print("CRITICAL: Cannot place Queen near calculated nest position.")
            return False  # Cannot proceed without queen

        # Spawn initial ants around the queen
        spawned_ants = 0
        attempts = 0
        max_att = INITIAL_ANTS * 25  # Max attempts to spawn ants
        queen_pos_int = self.queen.pos

        while spawned_ants < INITIAL_ANTS and attempts < max_att:
            # Spawn in a radius around the queen
            radius_offset = NEST_RADIUS + 1  # Spawn slightly outside inner nest
            angle = rnd_uniform(0, 2 * math.pi)
            dist = rnd_uniform(0, radius_offset)  # Distance from center
            # Calculate offset position
            ox = int(dist * math.cos(angle))
            oy = int(dist * math.sin(angle))
            # Calculate final position relative to queen
            pos = (queen_pos_int[0] + ox, queen_pos_int[1] + oy)

            # Decide caste based on target ratio
            caste = AntCaste.SOLDIER if random.random() < QUEEN_SOLDIER_RATIO_TARGET else AntCaste.WORKER
            # Try adding the ant at the calculated position
            if self.add_ant(pos, caste):
                spawned_ants += 1
            attempts += 1

        if spawned_ants < INITIAL_ANTS:
            print(f"Warning: Spawned only {spawned_ants}/{INITIAL_ANTS} initial ants.")
        else:
            print(f"Spawned {spawned_ants} initial ants.")

        # Spawn initial enemies
        enemies_spawned = sum(1 for _ in range(INITIAL_ENEMIES) if self.spawn_enemy())
        if enemies_spawned < INITIAL_ENEMIES:
            print(f"Warning: Spawned only {enemies_spawned}/{INITIAL_ENEMIES} initial enemies.")
        else:
            print(f"Spawned {enemies_spawned} initial enemies.")

        # Spawn initial prey
        prey_spawned = sum(1 for _ in range(INITIAL_PREY) if self.spawn_prey())
        if prey_spawned < INITIAL_PREY:
            print(f"Warning: Spawned only {prey_spawned}/{INITIAL_PREY} initial prey.")
        else:
            print(f"Spawned {prey_spawned} initial prey.")

        return True  # Spawn successful (even if counts are low)

    def _find_valid_queen_pos(self):
        """Finds a valid spot for the queen, starting from the calculated center."""
        base_int = self.nest_pos # Start at the ideal center

        # Check if the center itself is valid
        if is_valid_pos(base_int, self.grid_width, self.grid_height) and not self.grid.is_obstacle(base_int):
            return base_int

        # If center is blocked, check immediate neighbors
        neighbors = get_neighbors(base_int, self.grid_width, self.grid_height)
        random.shuffle(neighbors)
        for p_int in neighbors:
            if not self.grid.is_obstacle(p_int):
                return p_int # Found a valid neighbor

        # If immediate neighbors are blocked, check slightly further out
        for r in range(2, 5): # Check rings with radius 2, 3, 4
            perimeter = []
            # Iterate over the bounding box of the ring
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # Check if the point is on the perimeter of the ring
                    if abs(dx) == r or abs(dy) == r:
                        p_int = (base_int[0] + dx, base_int[1] + dy)
                        # Check if the position is valid and not an obstacle
                        if is_valid_pos(p_int, self.grid_width, self.grid_height) and not self.grid.is_obstacle(p_int):
                            perimeter.append(p_int)
            # If valid spots found in this ring, choose one randomly
            if perimeter:
                return random.choice(perimeter)

        # Very unlikely, but if no spot is found nearby:
        print("CRITICAL: Could not find any valid spot for Queen near nest center.")
        return None

    # --- Entity Management (Add, Remove, Update Position) ---

    def add_entity(self, entity, entity_list, position_dict):
        """Generic helper to add an entity if position is valid and empty."""
        pos_int = entity.pos
        # Check bounds, obstacle, and if position is already occupied in *any* dict
        if (is_valid_pos(pos_int, self.grid_width, self.grid_height) and
                not self.grid.is_obstacle(pos_int) and
                pos_int not in self.ant_positions and  # Check ant dict
                pos_int not in self.enemy_positions and  # Check enemy dict
                pos_int not in self.prey_positions and  # Check prey dict
                # Special check for brood list at position
                len(self.brood_positions.get(pos_int, [])) == 0 and
                (self.queen is None or pos_int != self.queen.pos)):  # Check queen pos

            entity_list.append(entity)
            # For single-entity-per-cell dicts:
            if position_dict is not self.brood_positions:
                position_dict[pos_int] = entity
            # For brood (list per cell):
            elif position_dict is self.brood_positions:
                if pos_int not in position_dict:
                    position_dict[pos_int] = []
                position_dict[pos_int].append(entity)
            # --- NEW: Add to Spatial Grid ---
            self.spatial_grid.add_entity(entity)
            return True
        # print(f"Failed to add {type(entity).__name__} at {pos_int}") # Debug failed adds
        return False

    def add_ant(self, pos, caste: AntCaste):
        """Adds a new ant to the simulation if the position is valid and empty."""
        pos_int = tuple(map(int, pos))
        # Basic validity check
        if not is_valid_pos(pos_int, self.grid_width, self.grid_height):
            # print(f"Invalid pos for ant: {pos_int}") # Debug
            return False

        # Check obstacle, existing entities (including queen)
        if (not self.grid.is_obstacle(pos_int) and
                not self.is_ant_at(pos_int) and  # Checks regular ants and queen
                not self.is_enemy_at(pos_int) and
                not self.is_prey_at(pos_int)):
            ant = Ant(pos_int, self, caste)
            self.ants.append(ant)
            self.ant_positions[pos_int] = ant
            # --- NEW: Add to Spatial Grid ---
            self.spatial_grid.add_entity(ant)

            # --- NEW: Add to Position Array ---
            if self.next_ant_index < self.max_ants:
                ant.index = self.next_ant_index
                self.ant_positions_array[ant.index] = pos_int
                self.ant_indices[ant] = ant.index
                self.next_ant_index += 1
            else:
                print("Warning: Max ants reached, cannot add more.")
                return False

            return True
        # else: # Debug failed placement
        # reason = []
        # if self.grid.is_obstacle(pos_int): reason.append("obstacle")
        # if self.is_ant_at(pos_int): reason.append("ant/queen")
        # if self.is_enemy_at(pos_int): reason.append("enemy")
        # if self.is_prey_at(pos_int): reason.append("prey")
        # print(f"Cannot add ant at {pos_int}. Reason: {', '.join(reason)}")
        return False

    def add_brood(self, brood_item: BroodItem):
        """Adds a brood item to the simulation."""
        pos_int = brood_item.pos
        # Brood can be placed on non-obstacle cells, even with other brood
        if is_valid_pos(pos_int, self.grid_width, self.grid_height) and not self.grid.is_obstacle(pos_int):
            self.brood.append(brood_item)
            # Add to the list associated with this position
            if pos_int not in self.brood_positions:
                self.brood_positions[pos_int] = []
            self.brood_positions[pos_int].append(brood_item)
            return True
        return False

    def remove_entity(self, entity, entity_list, position_dict):
        """Generic helper to remove an entity from lists and position lookups."""
        try:
            # Remove from the main list
            if entity in entity_list:
                 entity_list.remove(entity)

            pos = entity.pos # Get position before potential errors

            # Remove from position dictionary
            # Handle single entity dicts (ant, enemy, prey)
            if position_dict is not self.brood_positions:
                 if pos in position_dict and position_dict[pos] == entity:
                      del position_dict[pos]
            # Handle brood dictionary (list per cell)
            elif position_dict is self.brood_positions:
                 if pos in self.brood_positions:
                      # Remove the specific item from the list at that position
                      if entity in self.brood_positions[pos]:
                           self.brood_positions[pos].remove(entity)
                           # If the list becomes empty, remove the position key
                           if not self.brood_positions[pos]:
                                del self.brood_positions[pos]

        except ValueError:
            # Occurs if entity was already removed somehow
            # print(f"Warning: Attempted to remove {type(entity).__name__} which was not in list.")
            pass
        except KeyError:
             # Occurs if pos was not in dict (e.g., moved and removed simultaneously)
             # print(f"Warning: Attempted to remove entity from position {pos} which was not in dict.")
             pass

    def update_entity_position(self, entity, old_pos, new_pos):
        """Updates the position dictionary and spatial grid when an entity moves."""
        pos_dict = None
        if isinstance(entity, Ant):
            pos_dict = self.ant_positions
            # --- NEW: Update Position Array ---
            if entity.index >= 0:
                self.ant_positions_array[entity.index] = new_pos
        elif isinstance(entity, Enemy):
            pos_dict = self.enemy_positions
        elif isinstance(entity, Prey):
            pos_dict = self.prey_positions
        # Brood doesn't move, so no update needed for brood_positions here

        if pos_dict is not None:
            # Remove from old position if it was correctly registered there
            if old_pos in pos_dict and pos_dict[old_pos] == entity:
                del pos_dict[old_pos]
            # Add to new position
            pos_dict[new_pos] = entity
        # --- NEW: Update Spatial Grid ---
        self.spatial_grid.update_entity_position(entity, old_pos)

    def spawn_enemy(self):
        """Spawns a new enemy at a random valid location far from the nest."""
        tries = 0
        nest_pos_int = self.nest_pos
        # Define spawn area constraints (far from nest)
        min_dist_sq_from_nest = (MIN_FOOD_DIST_FROM_NEST + 5) ** 2 # Spawn further out than food
        max_tries = 80 # Attempts to find a spawn location

        while tries < max_tries:
            # Choose random coordinates within the grid
            pos_i = (rnd(0, self.grid_width - 1), rnd(0, self.grid_height - 1))

            # Check conditions: far from nest, not obstacle, not occupied
            if (distance_sq(pos_i, nest_pos_int) > min_dist_sq_from_nest and
                not self.grid.is_obstacle(pos_i) and
                not self.is_enemy_at(pos_i) and
                not self.is_ant_at(pos_i) and # Include queen check
                not self.is_prey_at(pos_i)):

                # Create and add the enemy
                enemy = Enemy(pos_i, self)
                self.enemies.append(enemy)
                self.enemy_positions[pos_i] = enemy
                return True # Successfully spawned
            tries += 1

        # print("Warning: Failed to spawn enemy after max tries.") # Optional debug
        return False # Failed to find a spot

    def spawn_prey(self):
        """Spawns a new prey item at a random valid location."""
        tries = 0
        nest_pos_int = self.nest_pos
         # Prey can spawn closer to the nest than enemies
        min_dist_sq_from_nest = (MIN_FOOD_DIST_FROM_NEST - 10) ** 2
        # Ensure min distance isn't negative if MIN_FOOD_DIST.. is small
        min_dist_sq_from_nest = max(0, min_dist_sq_from_nest)
        max_tries = 70

        while tries < max_tries:
            pos_i = (rnd(0, self.grid_width - 1), rnd(0, self.grid_height - 1))

            # Check conditions: distance, not obstacle, not occupied
            if (distance_sq(pos_i, nest_pos_int) > min_dist_sq_from_nest and
                not self.grid.is_obstacle(pos_i) and
                not self.is_enemy_at(pos_i) and
                not self.is_ant_at(pos_i) and # Include queen check
                not self.is_prey_at(pos_i)):

                # Create and add prey
                prey_item = Prey(pos_i, self)
                self.prey.append(prey_item)
                self.prey_positions[pos_i] = prey_item
                return True # Success
            tries += 1

        # print("Warning: Failed to spawn prey after max tries.") # Optional debug
        return False # Failed

    # --- Kill Methods ---

    def kill_ant(self, ant_to_remove: Ant, reason="unknown"):
        """Removes a dead ant from the simulation."""
        # print(f"Ant died at {ant_to_remove.pos}, reason: {reason}") # Debug
        self.remove_entity(ant_to_remove, self.ants, self.ant_positions)

        # --- NEW: Remove from Position Array ---
        if ant_to_remove.index >= 0:
            # Reset position in array
            self.ant_positions_array[ant_to_remove.index] = [-1, -1]
            # Remove from index lookup
            del self.ant_indices[ant_to_remove]
            # Mark index as free (can be reused)
            self.next_ant_index = min(self.next_ant_index, ant_to_remove.index)
            ant_to_remove.index = -1  # Mark as invalid

        # --- NEW: Remove from Spatial Grid ---
        self.spatial_grid.remove_entity(ant_to_remove)

        # Optional: Drop negative pheromone or small amount of food?

    def kill_enemy(self, enemy_to_remove: Enemy):
        """Removes a dead enemy and adds food resources to the grid."""
        pos_int = enemy_to_remove.pos
        self.remove_entity(enemy_to_remove, self.enemies, self.enemy_positions)

        # --- NEW: Remove from Spatial Grid ---
        self.spatial_grid.remove_entity(enemy_to_remove)

        # Add food resources at the enemy's death location if valid
        if is_valid_pos(pos_int, self.grid_width, self.grid_height) and not self.grid.is_obstacle(pos_int):
            fx, fy = pos_int
            grid = self.grid
            s_idx = FoodType.SUGAR.value
            p_idx = FoodType.PROTEIN.value
            try:
                # Add sugar, capped at max per cell
                grid.food[fx, fy, s_idx] = min(MAX_FOOD_PER_CELL,
                                               grid.food[fx, fy, s_idx] + ENEMY_TO_FOOD_ON_DEATH_SUGAR)
                # Add protein, capped at max per cell
                grid.food[fx, fy, p_idx] = min(MAX_FOOD_PER_CELL,
                                               grid.food[fx, fy, p_idx] + ENEMY_TO_FOOD_ON_DEATH_PROTEIN)
            except IndexError:
                # Should not happen with valid pos, but safety check
                pass

    def kill_prey(self, prey_to_remove: Prey):
        """Removes dead prey and adds protein resource to the grid."""
        pos_int = prey_to_remove.pos
        self.remove_entity(prey_to_remove, self.prey, self.prey_positions)

        # --- NEW: Remove from Spatial Grid ---
        self.spatial_grid.remove_entity(prey_to_remove)

        # Add protein resource at the prey's death location
        if is_valid_pos(pos_int, self.grid_width, self.grid_height) and not self.grid.is_obstacle(pos_int):
            fx, fy = pos_int
            grid = self.grid
            p_idx = FoodType.PROTEIN.value
            try:
                # Add protein, capped at max per cell
                grid.food[fx, fy, p_idx] = min(MAX_FOOD_PER_CELL, grid.food[fx, fy, p_idx] + PROTEIN_ON_DEATH)
            except IndexError:
                pass

    def kill_queen(self, queen_to_remove: Queen):
        """Handles the queen's death, ending the simulation run."""
        if self.queen == queen_to_remove:
            print(f"\n--- QUEEN DIED (Tick {int(self.ticks)}, Kolonie {self.colony_generation}) ---")
            print(f"    Food S:{self.colony_food_storage_sugar:.1f} P:{self.colony_food_storage_protein:.1f}")
            print(f"    Ants:{len(self.ants)}, Brood:{len(self.brood)}")
            self.queen = None # Remove queen reference
            self.simulation_running = False # Stop the current simulation run
            self.end_game_reason = "Königin gestorben"


    # --- Position Query Methods ---

    def is_ant_at(self, pos_int, exclude_self=None):
        """Checks if an ant or queen is at the given position."""
        # Check if the queen is at the position (and not excluded)
        if self.queen and self.queen.pos == pos_int and exclude_self != self.queen:
            return True
        # Check if a regular ant is at the position (and not excluded)
        ant = self.ant_positions.get(pos_int)
        return ant is not None and ant is not exclude_self

    def get_ant_at(self, pos_int):
        """Returns the ant or queen instance at the position, or None."""
        # Check if the queen is at the position (and not excluded)
        if self.queen and self.queen.pos == pos_int:
            return self.queen
        # --- NEW: Check Spatial Grid ---
        nearby_ants = self.spatial_grid.get_nearby_entities(pos_int, Ant)
        for ant in nearby_ants:
            if ant.pos == pos_int:
                return ant
        return None

    def is_enemy_at(self, pos_int, exclude_self=None):
        """Checks if an enemy is at the given position."""
        enemy = self.enemy_positions.get(pos_int)
        return enemy is not None and enemy is not exclude_self

    def get_enemy_at(self, pos_int):
        """Checks if an enemy is at the given position."""
        # --- NEW: Check Spatial Grid ---
        nearby_enemies = self.spatial_grid.get_nearby_entities(pos_int, Enemy)
        for enemy in nearby_enemies:
            if enemy.pos == pos_int:
                return enemy
        return None

    def is_prey_at(self, pos_int, exclude_self=None):
        """Checks if prey is at the given position."""
        prey_item = self.prey_positions.get(pos_int)
        return prey_item is not None and prey_item is not exclude_self

    def get_prey_at(self, pos_int):
        """Checks if prey is at the given position."""
        # --- NEW: Check Spatial Grid ---
        nearby_prey = self.spatial_grid.get_nearby_entities(pos_int, Prey)
        for prey in nearby_prey:
            if prey.pos == pos_int:
                return prey
        return None

    def get_brood_positions(self):
        """Returns a set of all positions currently containing brood."""
        return set(self.brood_positions.keys())

    def find_nearby_prey(self, pos_int, radius_sq):
        """Finds all living prey within a squared radius of a position."""
        nearby = []
        # --- NEW: Use Spatial Grid for initial filtering ---
        nearby_entities = self.spatial_grid.get_nearby_entities(pos_int, Prey)
        for p in nearby_entities:
            # Check if prey exists, is alive, and is within radius
            if p in self.prey and p.hp > 0 and distance_sq(pos_int, p.pos) <= radius_sq:
                nearby.append(p)
        return nearby

    def update(self):
        """Main simulation update step."""
        global latest_frame_bytes  # For streaming

        # Get speed multiplier (0.0 if paused)
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]
        if current_multiplier == 0.0:
            # Still increment ticks slightly when paused to allow UI updates
            # and prevent timers from completely freezing if paused for long
            self.ticks += 0.001
            return  # Skip simulation logic if paused

        # Increment simulation time
        self.ticks += current_multiplier

        # --- Update Queen ---
        if self.queen:
            self.queen.update(current_multiplier)
        # Check if queen died during her update (unlikely but possible)
        if not self.simulation_running: return  # Queen death stops the simulation

        # --- Update Brood ---
        hatched_pupae = []
        # Iterate over a copy, as brood list can change during iteration
        brood_copy = list(self.brood)
        for item in brood_copy:
            # Check if item still exists (might have been removed if invalid?)
            if item in self.brood:
                hatch_signal = item.update(self.ticks)  # Pass current tick
                if hatch_signal:  # Returns self if hatched
                    hatched_pupae.append(hatch_signal)

        # --- Handle Hatched Pupae ---
        for pupa in hatched_pupae:
            # Double check pupa still exists before removing/spawning
            if pupa in self.brood:
                self.remove_entity(pupa, self.brood, self.brood_positions)
                # Try to spawn the new ant
                self._spawn_hatched_ant(pupa.caste, pupa.pos)

        # --- Update Mobile Entities (Ants, Enemies, Prey) ---
        # Update copies and shuffle for fairness in interaction order
        ants_copy = list(self.ants);
        random.shuffle(ants_copy)
        enemies_copy = list(self.enemies);
        random.shuffle(enemies_copy)
        prey_copy = list(self.prey);
        random.shuffle(prey_copy)

        for a in ants_copy:
            # Check if ant still exists and is alive before updating
            if a in self.ants and a.hp > 0:
                a.update(current_multiplier)
        for e in enemies_copy:
            if e in self.enemies and e.hp > 0:
                e.update(current_multiplier)
        for p in prey_copy:
            if p in self.prey and p.hp > 0:
                p.update(current_multiplier)

        # --- Entity Cleanup (Remove dead or invalid entities) ---
        # Use list comprehensions to find entities to remove
        ants_to_remove = [a for a in self.ants if a.hp <= 0 or self.grid.is_obstacle(a.pos)]
        enemies_to_remove = [e for e in self.enemies if e.hp <= 0 or self.grid.is_obstacle(e.pos)]
        prey_to_remove = [p for p in self.prey if p.hp <= 0 or self.grid.is_obstacle(p.pos)]

        # Remove them using the kill methods
        for a in ants_to_remove: self.kill_ant(a, "cleanup")
        for e in enemies_to_remove: self.kill_enemy(e)
        for p in prey_to_remove: self.kill_prey(p)

        # Final check for the Queen
        if self.queen and (self.queen.hp <= 0 or self.grid.is_obstacle(self.queen.pos)):
            self.kill_queen(self.queen)
        if not self.simulation_running: return  # Queen death stops sim

        # --- Update Grid Systems ---
        if int(self.ticks) % 3 == 0:  # Nur alle 3 Ticks
            self.grid.update_pheromones(current_multiplier)

        # --- Spawning Timers ---
        # Enemy Spawning
        self.enemy_spawn_timer += current_multiplier
        if self.enemy_spawn_timer >= self.enemy_spawn_interval_ticks:
            self.enemy_spawn_timer %= self.enemy_spawn_interval_ticks
            # Limit total enemies based on initial count
            if len(self.enemies) < INITIAL_ENEMIES * 6:
                self.spawn_enemy()

        # Prey Spawning
        self.prey_spawn_timer += current_multiplier
        if self.prey_spawn_timer >= self.prey_spawn_interval_ticks:
            self.prey_spawn_timer %= self.prey_spawn_interval_ticks
            max_prey = INITIAL_PREY * 3  # Limit total prey
            if len(self.prey) < max_prey:
                self.spawn_prey()

        # Food Replenishment
        self.food_replenish_timer += current_multiplier
        if self.food_replenish_timer >= self.food_replenish_interval_ticks:
            self.food_replenish_timer %= self.food_replenish_interval_ticks
            self.grid.replenish_food(self.nest_pos)  # Pass nest pos for placement logic

        # --- Frame Capture for Network Streaming ---
        if ENABLE_NETWORK_STREAM and Flask and streaming_thread and streaming_thread.is_alive():
            # Check if draw() method has produced a surface to stream
            if self.latest_frame_surface:
                try:
                    # Use an in-memory buffer
                    frame_buffer = io.BytesIO()
                    # Save the current screen surface (captured in draw()) to the buffer as JPEG
                    pygame.image.save(self.latest_frame_surface, frame_buffer, ".jpg")
                    frame_buffer.seek(0)  # Rewind buffer to the beginning
                    # Update the global frame bytes under lock for the streamer thread
                    with latest_frame_lock:
                        latest_frame_bytes = frame_buffer.read()
                except pygame.error as e:
                    print(f"Pygame error during frame capture: {e}")
                except Exception as e:
                    print(f"Error capturing frame for streaming: {e}")
                    # Ensure latest_frame_bytes is None or empty on error?
                    with latest_frame_lock:
                        latest_frame_bytes = None

    def _spawn_hatched_ant(self, caste: AntCaste, pupa_pos_int: tuple):
        """Tries to spawn a newly hatched ant at or near the pupa's position."""
        # Try spawning at the exact pupa position first
        if self.add_ant(pupa_pos_int, caste):
            return True

        # If exact spot is blocked, try immediate neighbors
        neighbors = get_neighbors(pupa_pos_int, self.grid_width, self.grid_height)
        random.shuffle(neighbors)
        for pos_int in neighbors:
            if self.add_ant(pos_int, caste):
                return True

        # If neighbors are also blocked, try random spots near the nest center
        if self.queen: # Use queen position if available, else nest_pos
            base_pos = self.queen.pos
        else:
             base_pos = self.nest_pos

        for _ in range(15): # Try a few random spots near nest
            # Random offset within nest radius
            ox = rnd(-(NEST_RADIUS - 1), NEST_RADIUS - 1)
            oy = rnd(-(NEST_RADIUS - 1), NEST_RADIUS - 1)
            pos_int = (base_pos[0] + ox, base_pos[1] + oy)
            if self.add_ant(pos_int, caste):
                return True

        # Failed to spawn the ant anywhere sensible
        # print(f"Warning: Failed to spawn hatched {caste.name} near {pupa_pos_int}.")
        return False


    # --- Drawing Methods ---

    def draw_debug_info(self):
        """Draws simulation statistics and mouse hover info."""
        if not self.debug_font: return # Cannot draw without font

        # --- Performance & Basic Stats ---
        ant_c = len(self.ants)
        enemy_c = len(self.enemies)
        brood_c = len(self.brood)
        prey_c = len(self.prey)
        food_s = self.colony_food_storage_sugar
        food_p = self.colony_food_storage_protein
        tick_display = int(self.ticks)
        fps = self.clock.get_fps()

        # --- Caste & Brood Stage Breakdown ---
        w_c = sum(1 for a in self.ants if a.caste == AntCaste.WORKER)
        s_c = ant_c - w_c # Soldier count
        e_c = sum(1 for b in self.brood if b.stage == BroodStage.EGG)
        l_c = sum(1 for b in self.brood if b.stage == BroodStage.LARVA)
        p_c = brood_c - e_c - l_c # Pupa count

        # --- Speed Info ---
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]
        speed_text = f"Speed: Paused" if current_multiplier == 0.0 else f"Speed: {current_multiplier:.1f}x".replace(".0x", "x")

        # --- Text Lines ---
        texts=[f"Kolonie: {self.colony_generation}",
               f"Tick: {tick_display} FPS: {fps:.0f}",
               speed_text,
               f"Ants: {ant_c} (W:{w_c} S:{s_c})",
               f"Brood: {brood_c} (E:{e_c} L:{l_c} P:{p_c})",
               f"Enemies: {enemy_c}",
               f"Prey: {prey_c}",
               f"Food S:{food_s:.1f} P:{food_p:.1f}"]

        # --- Render Top-Left Stats ---
        y_start = 5 + self.buttons[0]['rect'].bottom # Position below buttons
        line_height = self.debug_font.get_height() + 1
        text_color = (240, 240, 240) # White/Light Grey
        for i, txt in enumerate(texts):
            try:
                surf = self.debug_font.render(txt, True, text_color)
                self.screen.blit(surf, (5, y_start + i * line_height))
            except Exception as e:
                # Print error but continue if possible
                print(f"Debug Font render error (line '{txt}'): {e}")

        # --- Mouse Hover Info ---
        try:
            mx, my = pygame.mouse.get_pos()
            # Convert mouse pixel coordinates to grid coordinates
            gx, gy = mx // self.cell_size, my // self.cell_size
            pos_i = (gx, gy)

            # Check if mouse is within grid bounds
            if is_valid_pos(pos_i, self.grid_width, self.grid_height):
                hover_lines = []
                # Check for entities at the grid cell
                entity = self.get_ant_at(pos_i) or self.get_enemy_at(pos_i) or self.get_prey_at(pos_i)
                if entity:
                    entity_pos_int = entity.pos # Use entity's actual position
                    hp_str = f"HP:{entity.hp:.0f}/{entity.max_hp}" if hasattr(entity, 'hp') else ""
                    if isinstance(entity, Queen):
                         hover_lines.extend([f"QUEEN @{entity_pos_int}", f"{hp_str} Age:{entity.age:.0f}"])
                    elif isinstance(entity, Ant):
                         carry_str = f"C:{entity.carry_amount:.1f}({entity.carry_type.name if entity.carry_type else '-'})"
                         age_str = f"Age:{entity.age:.0f}/{entity.max_age_ticks}"
                         move_str = f"Mv:{entity.last_move_info[:28]}" # Limit length
                         hover_lines.extend([f"{entity.caste.name} @{entity_pos_int}", f"S:{entity.state.name} {hp_str}", carry_str, age_str, move_str])
                    elif isinstance(entity, Enemy):
                         hover_lines.extend([f"ENEMY @{entity_pos_int}", f"{hp_str}"])
                    elif isinstance(entity, Prey):
                         hover_lines.extend([f"PREY @{entity_pos_int}", f"{hp_str}"])

                # Check for brood at the grid cell
                brood_at_pos = self.brood_positions.get(pos_i, [])
                if brood_at_pos:
                    hover_lines.append(f"Brood:{len(brood_at_pos)} @{pos_i}")
                    # Show details for first few brood items
                    for b in brood_at_pos[:3]:
                         hover_lines.append(f"-{b.stage.name[:1]}({b.caste.name[:1]}) {int(b.progress_timer)}/{b.duration}")

                # Cell Information (Obstacle, Food, Pheromones)
                is_obs = self.grid.is_obstacle(pos_i)
                obs_txt = " OBSTACLE" if is_obs else ""
                hover_lines.append(f"Cell:{pos_i}{obs_txt}")

                if not is_obs:
                    try:
                        # Food levels
                        foods = self.grid.food[pos_i[0], pos_i[1]]
                        food_txt = f"Food S:{foods[0]:.1f} P:{foods[1]:.1f}"
                        # Pheromone levels
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
                        hover_lines.append("Error reading cell data")

                # Render Hover Text (bottom-left)
                hover_color = (255, 255, 0) # Yellow
                # Calculate starting Y position from the bottom
                hover_y_start = self.height - (len(hover_lines) * line_height) - 5
                for i, line in enumerate(hover_lines):
                    surf = self.debug_font.render(line, True, hover_color)
                    self.screen.blit(surf, (5, hover_y_start + i * line_height))
        except Exception as e:
            # Catch potential errors during hover info gathering/rendering
            # print(f"Error drawing hover info: {e}") # Optional debug
            pass

    def _draw_legend(self):
        """Draws the simulation legend with relative positioning."""
        if not self.legend_font: return # Need font

        # Legend items: (Text, Color)
        legend_items = [
            ("Entities:", None), # Title
            ("Queen", QUEEN_COLOR),
            ("Worker (Search)", ANT_ATTRIBUTES[AntCaste.WORKER]["color"]),
            ("Worker (Return)", ANT_ATTRIBUTES[AntCaste.WORKER]["return_color"]),
            ("Soldier (Patrol)", ANT_ATTRIBUTES[AntCaste.SOLDIER]["color"]),
            ("Soldier (Return)", ANT_ATTRIBUTES[AntCaste.SOLDIER]["return_color"]),
            ("Ant (Escape)", WORKER_ESCAPE_COLOR),
            ("Ant (Defend)", ANT_DEFEND_COLOR),
            ("Ant (Hunt)", ANT_HUNT_COLOR),
            ("Enemy", ENEMY_COLOR),
            ("Prey", PREY_COLOR),
            ("Brood:", None), # Title
            ("Egg", EGG_COLOR[:3]), # Ignore alpha for swatch
            ("Larva", LARVA_COLOR[:3]),
            ("Pupa", PUPA_COLOR[:3]),
            ("Resources:", None), # Title
            ("Food (Sugar)", FOOD_COLORS[FoodType.SUGAR]),
            ("Food (Protein)", FOOD_COLORS[FoodType.PROTEIN]),
            ("Pheromones:", None), # Title
            ("Home", PHEROMONE_HOME_COLOR[:3]),
            ("Food S", PHEROMONE_FOOD_SUGAR_COLOR[:3]),
            ("Food P", PHEROMONE_FOOD_PROTEIN_COLOR[:3]),
            ("Alarm", PHEROMONE_ALARM_COLOR[:3]),
            ("Negative", PHEROMONE_NEGATIVE_COLOR[:3]),
            ("Recruit", PHEROMONE_RECRUITMENT_COLOR[:3]),
        ]

        # --- Positioning and Sizing ---
        padding = 5 # Padding inside legend box
        line_height = self.legend_font.get_height() + padding
        swatch_size = self.legend_font.get_height() - 1 # Size of color square
        # Estimate width needed (can be calculated more precisely if needed)
        # Based on longest text + swatch + padding
        max_text_width = 0
        for text, _ in legend_items:
             try:
                  text_width = self.legend_font.size(text)[0]
                  max_text_width = max(max_text_width, text_width)
             except Exception: pass # Ignore font errors for width calculation
        legend_width = max(100, swatch_size + max_text_width + 3 * padding)
        legend_height = len(legend_items) * line_height + padding # Total height

        # Position relative to top-right corner
        start_x = self.width - legend_width - 10 # 10px from right edge
        start_y = self.buttons[0]['rect'].bottom + 10 if self.buttons else 10 # 10px below buttons


        # --- Draw Legend Surface ---
        try:
             legend_surf = pygame.Surface((legend_width, legend_height), pygame.SRCALPHA)
             legend_surf.fill(LEGEND_BG_COLOR) # Semi-transparent background

             # Draw items onto the legend surface
             current_y = padding
             for text, color in legend_items:
                 text_x = padding
                 # Draw color swatch if color is provided
                 if color:
                     swatch_rect = pygame.Rect(padding, current_y + 1, swatch_size, swatch_size)
                     # Ensure color has no alpha or is opaque for swatch
                     draw_color = color[:3] if len(color) == 4 else color
                     pygame.draw.rect(legend_surf, draw_color, swatch_rect)
                     text_x += swatch_size + padding # Indent text after swatch
                 else:
                     # Title or section header - maybe bold or different rendering?
                     # For now, just draw text normally
                     pass

                 # Render and blit text
                 try:
                     surf = self.legend_font.render(text, True, LEGEND_TEXT_COLOR)
                     legend_surf.blit(surf, (text_x, current_y))
                 except Exception as e:
                      # Log error but continue drawing other items
                      print(f"Legend Font render error ('{text}'): {e}")
                 current_y += line_height

             # Blit the complete legend surface onto the main screen
             self.screen.blit(legend_surf, (start_x, start_y))

        except pygame.error as e:
             print(f"Error creating or drawing legend surface: {e}")
        except Exception as e:
             print(f"Unexpected error drawing legend: {e}")

    def draw(self):
        """Draws all simulation elements onto the screen."""
        current_multiplier = SPEED_MULTIPLIERS[self.simulation_speed_index]  # Needed for indicator update
        # 1. Draw grid (static bg, pheromones, food, nest area)
        self._draw_grid()
        # 2. Draw brood
        self._draw_brood()
        # 3. Draw queen
        self._draw_queen()
        # 4. Draw ants
        self._draw_ants() # Separated ant drawing
        # 5. Draw enemies
        self._draw_enemies() # Separated enemy drawing
        # 6. Draw prey
        self._draw_prey()
        # --- NEW: Draw Attack Indicators ---
        # Pass multiplier to let it decrement timers correctly
        self._draw_attack_indicators(current_multiplier)
        # --- UI Overlays ---
        # 7. Draw debug overlay if enabled
        if self.show_debug_info:
             self.draw_debug_info()
        # 8. Draw legend if enabled
        if self.show_legend:
             self._draw_legend()
        # 9. Draw UI buttons
        self._draw_buttons()

        # --- Network Streaming Frame Capture ---
        # Store a *copy* of the screen surface *before* flipping the display
        # if streaming is enabled.
        if ENABLE_NETWORK_STREAM and Flask:
             try:
                  # Capture the current state of self.screen
                  self.latest_frame_surface = self.screen.copy()
             except pygame.error as e:
                  print(f"Error copying screen surface for streaming: {e}")
                  self.latest_frame_surface = None # Indicate error
        else:
            self.latest_frame_surface = None # Ensure it's None if not streaming

        # --- Final Display Update ---
        # 10. Update the actual display to show the drawn frame
        pygame.display.flip()

    def _draw_grid(self):
        """Draws the static background, pheromones, food, and nest area."""
        cs = self.cell_size  # Local alias for cell size

        # 1. Blit the pre-rendered static background (obstacles)
        try:
            self.screen.blit(self.static_background_surface, (0, 0))
        except pygame.error as e:
             print(f"ERROR: Failed to blit static background: {e}")
             # Attempting to continue might lead to more errors, but let's try
        except Exception as e:
             print(f"ERROR: Unexpected error blitting static background: {e}")

        # 2. Draw Pheromones (using transparent surfaces)
        ph_info = {
            "home": (PHEROMONE_HOME_COLOR, self.grid.pheromones_home, PHEROMONE_MAX),
            "food_sugar": (PHEROMONE_FOOD_SUGAR_COLOR, self.grid.pheromones_food_sugar, PHEROMONE_MAX),
            "food_protein": (PHEROMONE_FOOD_PROTEIN_COLOR, self.grid.pheromones_food_protein, PHEROMONE_MAX),
            "alarm": (PHEROMONE_ALARM_COLOR, self.grid.pheromones_alarm, PHEROMONE_MAX),
            "negative": (PHEROMONE_NEGATIVE_COLOR, self.grid.pheromones_negative, PHEROMONE_MAX),
            "recruitment": (PHEROMONE_RECRUITMENT_COLOR, self.grid.pheromones_recruitment, RECRUITMENT_PHEROMONE_MAX),
        }

        min_alpha_for_draw = 5  # Don't draw extremely faint pheromones

        for ph_type, (base_col, arr, current_max) in ph_info.items():
            try:
                # Create a surface for this pheromone layer with alpha channel
                # Check for valid dimensions before creating surface
                if self.width <= 0 or self.height <= 0:
                    # print(f"WARN: Invalid dimensions for pheromone surface ({self.width}x{self.height}). Skipping {ph_type}.") # Optional Debug
                    continue
                ph_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

                norm_divisor = max(current_max / 2.5, 1.0)
                nz_coords = np.argwhere(arr > MIN_PHEROMONE_DRAW_THRESHOLD)

                # Draw rectangles for each significant pheromone cell
                for x, y in nz_coords:
                    if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                        continue

                    try:
                        val = arr[x, y]
                        norm_val = normalize(val, norm_divisor)
                        # Ensure base_col has alpha, default to 255 if not
                        alpha_base = base_col[3] if len(base_col) > 3 else 255
                        alpha = min(max(int(norm_val * alpha_base), 0), 255)

                        if alpha >= min_alpha_for_draw:
                            color = (*base_col[:3], alpha)
                            rect_coords = (x * cs, y * cs, cs, cs)
                            pygame.draw.rect(ph_surf, color, rect_coords)
                    except IndexError:
                        # print(f"WARN: Pheromone draw IndexError at {(x,y)} for {ph_type}") # Optional Debug
                        continue
                    except (ValueError, TypeError) as e:
                        # print(f"WARN: Pheromone draw Value/Type Error at {(x,y)} for {ph_type}: {e}") # Optional Debug
                        continue

                # Blit this pheromone layer onto the main screen
                self.screen.blit(ph_surf, (0, 0))
            except pygame.error as e:
                 print(f"ERROR: Pygame error during pheromone drawing for {ph_type}: {e}")
            except ValueError as e: # Catch potential numpy errors during argwhere or access
                 print(f"ERROR: ValueError during pheromone processing for {ph_type}: {e}")
            except Exception as e: # Catch any other unexpected errors
                 print(f"ERROR: Unexpected error during pheromone drawing for {ph_type}: {e}")
                 # traceback.print_exc() # Uncomment for full traceback if needed


        # 3. Draw Food (mit stabilen Punkten)
        food_drawn_count = 0 # Debug counter
        try:
            food_totals = np.sum(self.grid.food, axis=2)
            min_food_for_dot_check = 0.01
            food_nz_coords = np.argwhere(food_totals > min_food_for_dot_check)

            s_idx = FoodType.SUGAR.value
            p_idx = FoodType.PROTEIN.value
            s_col = FOOD_COLORS[FoodType.SUGAR]
            p_col = FOOD_COLORS[FoodType.PROTEIN]
            dot_radius = max(1, FOOD_DOT_RADIUS) # Ensure radius is at least 1

            # --- Verwenden Sie den dedizierten RNG ---
            if not hasattr(self, 'food_dot_rng') or self.food_dot_rng is None:
                print("ERROR: food_dot_rng not initialized!")
                # If RNG isn't there, we can't draw food properly, so skip the rest of food drawing.
                # Alternatively, initialize it here as a fallback, but better to fix __init__
                # self.food_dot_rng = random.Random()
            else:
                food_rng = self.food_dot_rng

                for x, y in food_nz_coords:
                     if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                         continue

                     try:
                         foods = self.grid.food[x, y]
                         s = foods[s_idx]
                         p = foods[p_idx]
                         total = s + p

                         if total < min_food_for_dot_check: continue

                         num_dots = max(1, min(FOOD_MAX_DOTS_PER_CELL, int(total * FOOD_DOTS_PER_UNIT)))

                         sr = s / total if total > 0 else 0.5
                         pr = 1.0 - sr
                         color_mixed = (int(s_col[0] * sr + p_col[0] * pr),
                                        int(s_col[1] * sr + p_col[1] * pr),
                                        int(s_col[2] * sr + p_col[2] * pr))

                         color = tuple(max(0, min(255, c)) for c in color_mixed)

                         cell_x_start = x * cs
                         cell_y_start = y * cs

                         # --- WICHTIG: Seed den RNG für diese Zelle deterministisch ---
                         # Berechne den Seed (wird wahrscheinlich ein numpy int)
                         cell_seed_np = x * self.grid_height + y
                         # Konvertiere den Seed explizit in einen Python int! <<< FIX
                         cell_seed_int = int(cell_seed_np)
                         food_rng.seed(cell_seed_int)
                         # -----------------------------------------------------------

                         # Ensure cell size is large enough for the dot range
                         if cs - (2 * dot_radius) <= 0:
                              if num_dots > 0: # Only draw if there should be food
                                 dot_x = cell_x_start + cs / 2
                                 dot_y = cell_y_start + cs / 2
                                 # Check if coordinates are valid before drawing
                                 if 0 <= int(dot_x) < self.width and 0 <= int(dot_y) < self.height:
                                     pygame.draw.circle(self.screen, color, (int(dot_x), int(dot_y)), dot_radius)
                                     food_drawn_count += 1
                              continue # Skip the loop below if cell too small

                         for i in range(num_dots):
                             # --- Verwenden Sie den geseedeten RNG für die Position ---
                             dot_x = cell_x_start + food_rng.uniform(dot_radius, cs - dot_radius)
                             dot_y = cell_y_start + food_rng.uniform(dot_radius, cs - dot_radius)
                             # ------------------------------------------------------
                             # Check if coordinates are valid before drawing
                             if 0 <= int(dot_x) < self.width and 0 <= int(dot_y) < self.height:
                                 pygame.draw.circle(self.screen, color, (int(dot_x), int(dot_y)), dot_radius)
                                 food_drawn_count += 1

                     except IndexError:
                         # print(f"WARN: Food draw IndexError at {(x,y)}") # Optional Debug
                         continue
                     except (ValueError, TypeError) as e:
                         # Use f-string for better error message formatting
                         print(f"ERROR: Value/Type Error drawing food at {(int(x), int(y))}: Color={color}, Error={e}")
                         continue

        except Exception as e:
            print(f"ERROR: General exception during food grid processing: {e}")
            traceback.print_exc() # Print full traceback for general errors


        # 4. Draw Nest Area Highlight (subtle overlay)
        r = NEST_RADIUS
        nx, ny = self.nest_pos
        center_x = int(nx * cs + cs / 2)
        center_y = int(ny * cs + cs / 2)
        try:
            radius_px = r * cs
            if radius_px <= 0: raise ValueError("Nest radius non-positive")
            surf_size = int(radius_px * 2)
            if surf_size <= 0: raise ValueError("Nest surface size non-positive")

            nest_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            nest_surf.fill((0, 0, 0, 0))
            pygame.draw.circle(nest_surf, (100, 100, 100, 35), (surf_size // 2, surf_size // 2), radius_px)
            blit_pos = (center_x - surf_size // 2, center_y - surf_size // 2)
            self.screen.blit(nest_surf, blit_pos)
        except (ValueError, pygame.error, OverflowError) as e:
            # print(f"WARN: Could not draw nest highlight: {e}") # Optional Debug
            pass # Silently ignore if nest drawing fails

    def _draw_brood(self):
        """Draws all brood items (eggs, larvae, pupae)."""
        # Iterate over a copy in case list changes during drawing (unlikely)
        for item in list(self.brood):
             # Check if item still exists and position is valid before drawing
             if item in self.brood and is_valid_pos(item.pos, self.grid_width, self.grid_height):
                  item.draw(self.screen) # Call brood item's own draw method

    def _draw_queen(self):
        """Draws the queen ant."""
        if not self.queen or not is_valid_pos(self.queen.pos, self.grid_width, self.grid_height):
             return # No queen or invalid position

        cs = self.cell_size
        # Calculate center pixel position
        pos_px = (int(self.queen.pos[0] * cs + cs / 2),
                  int(self.queen.pos[1] * cs + cs / 2))
        # Radius relative to cell size (make queen slightly larger)
        radius = max(2, int(cs / 1.4))
        # Draw main circle
        pygame.draw.circle(self.screen, self.queen.color, pos_px, radius)
        # Draw outline
        pygame.draw.circle(self.screen, (255, 255, 255), pos_px, radius, 1) # White outline

    def _draw_ants(self):
        """Draws all worker and soldier ants."""
        # Iterate over a copy of the ants list
        for a in list(self.ants):
            # Check if ant still exists and position is valid
            if a not in self.ants or not is_valid_pos(a.pos, self.grid_width, self.grid_height):
                continue
            a.draw(self.screen)

    def _draw_enemies(self):
         """Draws all enemy entities using their own draw method."""
         # Iterate over a copy of the list for safety during potential modifications
         for e in list(self.enemies):
             # Check if enemy still exists in the main list and has a valid position
             if e in self.enemies and is_valid_pos(e.pos, self.grid_width, self.grid_height):
                  try:
                      # --- KORREKTUR: Rufe die draw-Methode des Feind-Objekts auf ---
                      e.draw(self.screen)
                      # ---------------------------------------------------------
                  except Exception as draw_error:
                      # Catch potential errors within the specific enemy's draw method
                      print(f"ERROR: Failed to draw enemy at {e.pos}: {draw_error}")
                      # Optionally remove the problematic enemy or just skip drawing it
                      # For now, just log the error and continue

    def _draw_prey(self):
        """Draws all prey entities."""
        # Iterate over a copy of the list
        for p in list(self.prey):
             # Check existence and validity before drawing
             if p in self.prey and is_valid_pos(p.pos, self.grid_width, self.grid_height):
                  p.draw(self.screen) # Call prey's own draw method

    def _draw_buttons(self):
        """Draws the UI buttons."""
        # Need font and buttons list
        if not self.font or not self.buttons:
            return

        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            rect = button["rect"]
            text = button["text"]
            # Change color on hover
            color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else BUTTON_COLOR
            # Draw button background
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            # Draw button text
            try:
                text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)
            except Exception as e:
                # Log error but don't crash
                print(f"Button font render error ('{text}'): {e}")

    def handle_events(self):
        """Processes Pygame events for user input (quit, keys, mouse clicks)."""
        for event in pygame.event.get():
            # --- Quit Event ---
            if event.type == pygame.QUIT:
                self.simulation_running = False
                self.app_running = False
                self.end_game_reason = "Fenster geschlossen"
                return "quit_app" # Signal app quit

            # --- Key Presses ---
            if event.type == pygame.KEYDOWN:
                # Check against button key bindings first
                key_action = None
                for button in self.buttons:
                    if button.get("key") == event.key:
                         key_action = button["action"]
                         break
                if key_action:
                     action_result = self._handle_button_click(key_action)
                     # Map button actions to event results if needed
                     if key_action == "quit": return "quit_app"
                     if key_action == "restart": return "sim_stop"
                     if "speed" in key_action: return "speed_change"
                     return "ui_action" # Generic UI action

                # Handle keys not bound to buttons
                if event.key == pygame.K_d: # Toggle Debug (alternative binding)
                    self._handle_button_click("toggle_debug")
                    return "ui_action"
                if event.key == pygame.K_l: # Toggle Legend (alternative binding)
                     self._handle_button_click("toggle_legend")
                     return "ui_action"
                # Allow +/- on numpad as well for speed
                if event.key == pygame.K_KP_MINUS:
                     self._handle_button_click("speed_down")
                     return "speed_change"
                if event.key == pygame.K_KP_PLUS:
                     self._handle_button_click("speed_up")
                     return "speed_change"

            # --- Mouse Click ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                # Check if simulation is running (buttons active)
                if self.simulation_running:
                    mouse_pos = event.pos
                    for button in self.buttons:
                        if button["rect"].collidepoint(mouse_pos):
                            action_result = self._handle_button_click(button["action"])
                            # Map actions to event results
                            if button["action"] == "quit": return "quit_app"
                            if button["action"] == "restart": return "sim_stop"
                            if "speed" in button["action"]: return "speed_change"
                            return "ui_action" # Generic button click

        return None # No significant event processed

    def _handle_button_click(self, action):
        """Handles actions triggered by button presses or key bindings."""
        if action == "toggle_debug":
            self.show_debug_info = not self.show_debug_info
        elif action == "toggle_legend":
            self.show_legend = not self.show_legend
        elif action == "speed_down":
            current_index = self.simulation_speed_index
            # Find the new index, ensuring it doesn't go below 0
            new_index = max(0, current_index - 1)
            if new_index != self.simulation_speed_index:
                self.simulation_speed_index = new_index
                # Update target FPS based on the new speed index
                self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
                print(f"Speed decreased to {SPEED_MULTIPLIERS[self.simulation_speed_index]:.1f}x")
        elif action == "speed_up":
            current_index = self.simulation_speed_index
            max_index = len(SPEED_MULTIPLIERS) - 1
            # Find the new index, ensuring it doesn't exceed max
            new_index = min(max_index, current_index + 1)
            if new_index != self.simulation_speed_index:
                self.simulation_speed_index = new_index
                self.current_target_fps = TARGET_FPS_LIST[self.simulation_speed_index]
                print(f"Speed increased to {SPEED_MULTIPLIERS[self.simulation_speed_index]:.1f}x")
        elif action == "restart":
            self.simulation_running = False # Stop current run
            self.end_game_reason = "Neustart Button/Taste"
        elif action == "quit":
            self.simulation_running = False # Stop current run
            self.app_running = False # Signal app to exit
            self.end_game_reason = "Beenden Button/Taste"
        else:
            print(f"Warning: Unknown button action '{action}'")

        # Return value indicates if speed changed, could be used elsewhere
        return "speed_change" if "speed" in action else action

    def _show_end_game_dialog(self):
        """Displays a dialog when a simulation run ends, offering restart or quit."""
        # Ensure font is available
        if not self.font:
            print("Error: Cannot display end game dialog - font not loaded.")
            self.app_running = False # Cannot proceed without font
            return "quit"

        # Dialog dimensions and position (centered)
        dialog_w, dialog_h = 350, 180 # Slightly larger dialog
        dialog_x = (self.width - dialog_w) // 2
        dialog_y = (self.height - dialog_h) // 2

        # Button dimensions and positions within the dialog
        btn_w, btn_h = 120, 40 # Larger buttons
        btn_margin = 30
        # Position buttons centered horizontally near the bottom of the dialog
        btn_y = dialog_y + dialog_h - btn_h - 25 # Y pos relative to dialog bottom
        total_btn_width = btn_w * 2 + btn_margin
        btn_restart_x = dialog_x + (dialog_w - total_btn_width) // 2
        btn_quit_x = btn_restart_x + btn_w + btn_margin
        # Create button rectangles
        restart_rect = pygame.Rect(btn_restart_x, btn_y, btn_w, btn_h)
        quit_rect = pygame.Rect(btn_quit_x, btn_y, btn_w, btn_h)

        # Text content and color
        text_color = (240, 240, 240)
        title_text = f"Kolonie {self.colony_generation} Ende"
        reason_text = f"Grund: {self.end_game_reason}"

        # Semi-transparent overlay for background dimming
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill(END_DIALOG_BG_COLOR) # Use defined transparent black

        waiting_for_choice = True
        while waiting_for_choice and self.app_running:
            mouse_pos = pygame.mouse.get_pos()

            # --- Event Handling for Dialog ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.app_running = False
                    waiting_for_choice = False
                    return "quit" # Exit app directly
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.app_running = False
                    waiting_for_choice = False
                    return "quit" # Exit app on ESC
                # Check mouse clicks on buttons
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if restart_rect.collidepoint(mouse_pos):
                        return "restart" # Signal to restart simulation
                    if quit_rect.collidepoint(mouse_pos):
                        self.app_running = False # Signal to quit app
                        return "quit"

            # --- Drawing the Dialog ---
            # 1. Draw the overlay background
            self.screen.blit(overlay, (0, 0))
            # 2. Draw the dialog box background
            pygame.draw.rect(self.screen, (40, 40, 80), (dialog_x, dialog_y, dialog_w, dialog_h), border_radius=6) # Dark blue bg

            # 3. Render and draw text
            try:
                 # Title text centered near top
                 title_surf = self.font.render(title_text, True, text_color)
                 title_rect = title_surf.get_rect(center=(dialog_x + dialog_w // 2, dialog_y + 40)) # Adjusted Y
                 self.screen.blit(title_surf, title_rect)
                 # Reason text centered below title
                 reason_surf = self.font.render(reason_text, True, text_color)
                 reason_rect = reason_surf.get_rect(center=(dialog_x + dialog_w // 2, dialog_y + 80)) # Adjusted Y
                 self.screen.blit(reason_surf, reason_rect)
            except Exception as e: print(f"Dialog text render error: {e}") # Log errors

            # 4. Draw Buttons (with hover effect)
            # Restart Button
            r_color = BUTTON_HOVER_COLOR if restart_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, r_color, restart_rect, border_radius=4)
            try:
                 r_text_surf = self.font.render("Neu starten", True, BUTTON_TEXT_COLOR)
                 r_text_rect = r_text_surf.get_rect(center=restart_rect.center)
                 self.screen.blit(r_text_surf, r_text_rect)
            except Exception as e: print(f"Restart button text render error: {e}")
            # Quit Button
            q_color = BUTTON_HOVER_COLOR if quit_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, q_color, quit_rect, border_radius=4)
            try:
                 q_text_surf = self.font.render("Beenden", True, BUTTON_TEXT_COLOR)
                 q_text_rect = q_text_surf.get_rect(center=quit_rect.center)
                 self.screen.blit(q_text_surf, q_text_rect)
            except Exception as e: print(f"Quit button text render error: {e}")

            # Update display and control frame rate
            pygame.display.flip()
            self.clock.tick(30) # Lower FPS for dialog is fine

        # If loop exits without choice (e.g., app_running becomes false)
        return "quit"

    def add_attack_indicator(self, attacker_pos_grid, target_pos_grid, color):
         """Stores information about a recent attack for drawing."""
         if not is_valid_pos(attacker_pos_grid, self.grid_width, self.grid_height) or \
            not is_valid_pos(target_pos_grid, self.grid_width, self.grid_height):
             return # Ignore attacks involving invalid positions

         indicator = {
             "attacker_pos": attacker_pos_grid,
             "target_pos": target_pos_grid,
             "color": color,
             "timer": ATTACK_INDICATOR_DURATION_TICKS # Start timer
         }
         self.recent_attacks.append(indicator)

    def _draw_attack_indicators(self, speed_multiplier):
         """Draws visual effects for recent attacks and updates timers."""
         if not self.recent_attacks:
             return

         cs = self.cell_size
         cs_half = cs / 2
         indices_to_remove = []

         # Iterate backwards for safe removal
         for i in range(len(self.recent_attacks) - 1, -1, -1):
             indicator = self.recent_attacks[i]
             indicator["timer"] -= speed_multiplier # Decrease timer based on game speed

             if indicator["timer"] <= 0:
                 indices_to_remove.append(i)
                 continue

             # Calculate pixel positions
             attacker_px = (int(indicator["attacker_pos"][0] * cs + cs_half),
                            int(indicator["attacker_pos"][1] * cs + cs_half))
             target_px = (int(indicator["target_pos"][0] * cs + cs_half),
                          int(indicator["target_pos"][1] * cs + cs_half))

             # Calculate alpha based on remaining time (fade out)
             base_alpha = indicator["color"][3]
             current_alpha = max(0, min(255, int(base_alpha * (indicator["timer"] / ATTACK_INDICATOR_DURATION_TICKS))))

             # Draw a line or flash
             if current_alpha > 10: # Only draw if somewhat visible
                 line_color = (*indicator["color"][:3], current_alpha) # Apply faded alpha
                 try:
                      pygame.draw.line(self.screen, line_color, attacker_px, target_px, 2) # Thickness 2 line
                      # Optional: Draw small circles at ends
                      # pygame.draw.circle(self.screen, line_color, attacker_px, 3)
                      # pygame.draw.circle(self.screen, line_color, target_px, 3)
                 except TypeError: # Catch potential color format issues
                      pass


         # Remove expired indicators
         for index in sorted(indices_to_remove, reverse=True): # Remove from end first
              del self.recent_attacks[index]

    def run(self):
        """Main application loop: handles simulation runs and end dialog."""
        print("Starting Ant Simulation...")
        print("Controls: D=Debug | L=Legend | ESC=Quit | +/- = Speed")
        if ENABLE_NETWORK_STREAM and Flask:
            # Determine accessible IP (this is a guess, might not be correct)
            hostname = STREAMING_HOST if STREAMING_HOST != "0.0.0.0" else "localhost" # Default to localhost if 0.0.0.0
            # Try to get local IP (requires network connection)
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80)) # Connect to external server (doesn't send data)
                local_ip = s.getsockname()[0]
                s.close()
                hostname = local_ip
            except Exception:
                 pass # Stick with localhost if IP fetch fails
            print(f"INFO: Network stream *may be* available at http://{hostname}:{STREAMING_PORT}")
            if STREAMING_HOST == "0.0.0.0":
                 print("      (Streaming host 0.0.0.0 means access may be possible from other devices on your network)")

        # --- Main Application Loop ---
        while self.app_running:

            # --- Inner Simulation Loop ---
            # Runs as long as simulation_running is true and app_running is true
            while self.simulation_running and self.app_running:
                # Process user input
                event_action = self.handle_events()

                # Handle actions that affect the loops
                if event_action == "quit_app":
                     self.app_running = False # Ensure outer loop terminates
                     break # Exit inner loop immediately
                if event_action == "sim_stop":
                     break # Exit inner loop to show end dialog

                # Update simulation state
                self.update()

                # Draw the current frame
                self.draw()

                # Control frame rate
                self.clock.tick(self.current_target_fps)
            # --- End Inner Simulation Loop ---

            # Check if the app should exit completely
            if not self.app_running:
                break # Exit outer loop

            # --- Show End Game Dialog ---
            # If simulation stopped but app is still running
            if not self.end_game_reason: # Set default reason if none provided
                 self.end_game_reason = "Simulation beendet"
            # Show dialog and get user choice ('restart' or 'quit')
            choice = self._show_end_game_dialog()

            if choice == "restart":
                # Reset the simulation state for a new run
                self._reset_simulation()
                # simulation_running flag is set within _reset_simulation
            elif choice == "quit":
                # User chose Quit in the dialog
                self.app_running = False # Signal outer loop to terminate

        # --- Application Exit ---
        print("Exiting application.")
        self._stop_streaming_server() # Attempt to stop stream thread if running
        try:
            pygame.quit()
            print("Pygame shut down gracefully.")
        except Exception as e:
            print(f"Error during Pygame quit: {e}")

# --- Start Simulation ---
if __name__ == "__main__":
    print("Initializing simulation environment...")
    # Basic checks for required libraries
    if 'pygame' not in sys.modules: print("FATAL: Pygame module not imported correctly."); exit()
    if 'numpy' not in sys.modules: print("FATAL: NumPy module not imported correctly."); exit()

    print(f"Pygame version: {pygame.version.ver}")
    print(f"NumPy version: {np.__version__}")
    if Flask: print(f"Flask found. Network streaming enabled: {ENABLE_NETWORK_STREAM}")
    else: print("Flask not found (Network Streaming disabled).")

    initialization_success = False
    try:
        # Pygame initialization is now handled within AntSimulation.__init__
        # We just need to ensure the class can be instantiated.
        print("Attempting to initialize AntSimulation...")
        simulation_instance = AntSimulation()
        # Check if initialization within the class failed
        if simulation_instance.app_running:
             initialization_success = True
             print("AntSimulation initialized successfully.")
        else:
             print("AntSimulation initialization failed (check logs above).")

    except Exception as e:
        print(f"\n--- FATAL ERROR DURING INITIALIZATION ---")
        traceback.print_exc()
        print("-----------------------------------------")
        initialization_success = False
        # Attempt cleanup even on init error
        try: pygame.quit()
        except Exception: pass
        input("Press Enter to Exit.")
        exit()


    if initialization_success:
        print("\nStarting simulation run...")
        try:
            # Run the main simulation loop
            simulation_instance.run()

        except Exception as e:
            # Catch unexpected errors during the simulation run
            print("\n--- CRITICAL UNHANDLED EXCEPTION DURING SIMULATION RUN ---")
            traceback.print_exc()
            print("----------------------------------------------------------")
            print("Attempting to exit gracefully...")
            try:
                # Try to stop simulation and clean up
                if simulation_instance:
                     simulation_instance.app_running = False # Ensure loops exit
                     simulation_instance._stop_streaming_server() # Try stopping thread
                pygame.quit()
            except Exception as cleanup_e:
                 print(f"Error during cleanup after exception: {cleanup_e}")
            input("Press Enter to Exit.")

    print("\nSimulation process finished.")
# --- END OF FILE antsim.py ---
