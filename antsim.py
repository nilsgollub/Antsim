# Standard Library Imports
import sys
import os
import random
import math
import time
import io
import traceback
import threading
from enum import Enum, auto

# Third-Party Imports
try:
    import pygame
except ImportError:
    print("FATAL: Pygame is required but not found. Install it: pip install pygame")
    sys.exit()
try:
    import numpy as np
except ImportError:
    print("FATAL: NumPy is required but not found. Install it: pip install numpy")
    sys.exit()
try:
    from flask import Flask, Response, render_template_string
except ImportError:
    # Flask is optional, only needed for network streaming.
    # Set names to None if import fails, so checks later in the code work.
    Flask = None
    Response = None
    render_template_string = None
    # Print a generic info message. The decision to USE Flask happens later.
    print(
        "INFO: Flask library not found. Install Flask for optional network streaming feature: pip install Flask"
    )
try:
    import scipy.ndimage
except ImportError:
    print("FATAL: SciPy is required but not found (needed for pheromone diffusion). Install it: pip install scipy")
    sys.exit()

# --- Print Versions (Optional but helpful for debugging) ---
# (Restlicher Code zum Drucken der Versionen bleibt gleich)
print(f"Python Version: {sys.version.split()[0]}")
print(f"Pygame Version: {pygame.version.ver}")
print(f"NumPy Version: {np.__version__}")
try:
    import scipy
    print(f"SciPy Version: {scipy.__version__}")
except NameError: pass # Already handled import error above
if Flask:
     try:
          import flask
          print(f"Flask Version: {flask.__version__}")
     except Exception:
          print("Flask imported but version unknown.")
else:
     print("Flask not imported.") # Explicitly state if Flask is None


# --- Configuration Constants ---
# --- Screen/Grid Handling ---
# Option 1: Use Fullscreen
USE_FULLSCREEN = False
# Option 2: Use Specific Window Size (if not fullscreen)
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
# Option 3: Use a percentage of the detected screen size (if not fullscreen)
# Example: USE_SCREEN_PERCENT = 0.8 # Use 80% of the screen width/height
# Set to a float between 0.1 and 1.0 or None to use DEFAULT_WINDOW_WIDTH/HEIGHT
USE_SCREEN_PERCENT = None

# Base size for grid cells. Affects visual detail and simulation scale.
CELL_SIZE = 16

# Simulation speed control: Target Frames Per Second
# The simulation logic advances one step (tick) per frame update.
# Higher FPS means faster simulation speed.
TARGET_FPS = 20  # Target updates (and ticks) per second << NEW

# --- THESE ARE CALCULATED in AntSimulation.__init__ based on screen/window size ---
# GRID_WIDTH, GRID_HEIGHT, WIDTH, HEIGHT, NEST_POS

# --- Logging Configuration ---
# Enable logging of simulation statistics to a CSV file
ENABLE_SIMULATION_LOGGING = False  # Set to False to disable logging
# Name of the log file
SIMULATION_LOG_FILE = "antsim_log.csv"
# Interval (in simulation ticks) for writing log entries
LOGGING_INTERVAL_TICKS = 100
# Header row for the CSV log file
LOG_HEADER = "Tick,Generation,Sugar,Protein,Ants,Workers,Soldiers,Eggs,Larvae,Pupae,Enemies,Prey\n"

# --- World Colors & Properties ---
# Background color of the simulation map
MAP_BG_COLOR = (20, 20, 10)  # Dark brownish-green

# --- Nest Parameters ---
# Radius of the nest area (in grid cells) from the center.
# Determines queen's egg-laying area and initial ant spawn zone.
NEST_RADIUS = 3

# --- Food Parameters ---
# Number of distinct food types (used for array dimensions)
NUM_FOOD_TYPES = 2  # Corresponds to FoodType Enum (SUGAR, PROTEIN)
# Number of initial food clusters placed at the start of a simulation run
INITIAL_FOOD_CLUSTERS = 2
# Approximate total amount of food units per initial cluster
FOOD_PER_CLUSTER = 250
# Radius (in grid cells) around which food items in a cluster are spread
FOOD_CLUSTER_RADIUS = 5
# Minimum distance (in grid cells) initial food clusters must be from the nest center
MIN_FOOD_DIST_FROM_NEST = 30
# Maximum amount of food units allowed in a single grid cell
MAX_FOOD_PER_CELL = 100.0
# Initial amount of sugar stored in the colony's reserves
INITIAL_COLONY_FOOD_SUGAR = 200.0
# Initial amount of protein stored in the colony's reserves
INITIAL_COLONY_FOOD_PROTEIN = 200.0
# Food amount threshold in a cell considered "rich" (triggers recruitment pheromone)
RICH_FOOD_THRESHOLD = 50.0
# Colony storage threshold below which a food type is considered critically low,
# affecting ant search behavior and queen egg-laying.
CRITICAL_FOOD_THRESHOLD = 200.0
# Interval (in simulation ticks) at which new food clusters are potentially added
# (adjusted dynamically based on map size in __init__)
FOOD_REPLENISH_RATE = 20000
# --- Food Drawing Parameters ---
# Controls the density of dots used to visualize food. Lower value = more dots per unit.
FOOD_DOTS_PER_UNIT = 0.15
# Maximum number of dots drawn per cell, regardless of food amount (performance/visual limit)
FOOD_MAX_DOTS_PER_CELL = 25
# Pixel radius of individual food dots
FOOD_DOT_RADIUS = 1

# --- Obstacle Parameters ---
# Number of obstacle 'areas' to generate on the map
NUM_OBSTACLES = 5
# Color of obstacle cells
OBSTACLE_COLOR = (100, 100, 100)  # Grey
# Minimum number of small circular 'clumps' that form a single obstacle area
MIN_OBSTACLE_CLUMPS = 1
# Maximum number of small circular 'clumps' that form a single obstacle area
MAX_OBSTACLE_CLUMPS = 5
# Minimum radius (in grid cells) of a single obstacle clump
MIN_OBSTACLE_CLUMP_RADIUS = 1
# Maximum radius (in grid cells) of a single obstacle clump
MAX_OBSTACLE_CLUMP_RADIUS = 4
# Radius (in grid cells) controlling how far clumps spread from the obstacle area center
OBSTACLE_CLUSTER_SPREAD_RADIUS = 5
# +/- range for random color variation applied to individual obstacle cells for texture
OBSTACLE_COLOR_VARIATION = 15

# --- Pheromone Parameters ---
# Maximum strength value for most pheromone types
PHEROMONE_MAX = 1000.0
# Decay factor applied each tick to most pheromones (value < 1.0)
# Example: 0.999 means pheromone strength becomes 99.9% of its previous value each tick.
PHEROMONE_DECAY = 0.9995
# Diffusion rate controls how much pheromone spreads to neighbors each tick (using Gaussian filter sigma)
# Higher values cause faster spreading and dilution. Needs careful tuning with decay.
PHEROMONE_DIFFUSION_SIGMA = 0.2#0.32  # Sigma for Gaussian filter diffusion << RENAMED/CLARIFIED
# Decay factor for negative pheromones (can decay faster/slower than others)
NEGATIVE_PHEROMONE_DECAY = 0.995
# Diffusion rate (sigma) for negative pheromones
NEGATIVE_PHEROMONE_DIFFUSION_SIGMA = 0.32  # << RENAMED/CLARIFIED
# Decay factor for recruitment pheromones
RECRUITMENT_PHEROMONE_DECAY = 0.98
# Diffusion rate (sigma) for recruitment pheromones
RECRUITMENT_PHEROMONE_DIFFUSION_SIGMA = 0.32  # << RENAMED/CLARIFIED
# Maximum strength value specifically for recruitment pheromones
RECRUITMENT_PHEROMONE_MAX = 500.0
# Minimum pheromone strength required to be drawn (performance optimization)
MIN_PHEROMONE_DRAW_THRESHOLD = 0.5
# --- Pheromone Drop Amounts (Strength added when dropped) ---
P_HOME_RETURNING = 20.0           # Dropped by ants returning to nest (trail home)
P_FOOD_RETURNING_TRAIL = 50.0    # Food pheromone dropped by ants returning with food
P_FOOD_AT_SOURCE = 500.0          # Food pheromone dropped directly at food source upon pickup
P_ALARM_FIGHT = 200.0             # Alarm pheromone dropped during combat (by ants or enemies being hit) << INCREASED
P_NEGATIVE_SEARCH = 10.0          # Negative pheromone dropped when searching empty areas
P_RECRUIT_FOOD = 400.0            # Recruitment pheromone dropped at rich food sources
P_RECRUIT_DAMAGE = 250.0          # Recruitment pheromone dropped when an ant/queen is damaged << INCREASED
P_RECRUIT_DAMAGE_SOLDIER = 350.0  # Stronger recruitment signal if a soldier is damaged << INCREASED
P_RECRUIT_PREY = 150.0            # Recruitment pheromone dropped when prey is killed << INCREASED
# --- Pheromone Influence Weights (How strongly ants react to pheromones) ---
# These weights are multiplied by the pheromone strength found in neighboring cells
# during the ant's decision-making process (_score_moves_* methods).
# --- Returning to Nest State ---
W_HOME_PHEROMONE_RETURN = 45.0        # Attraction to home pheromone when returning
W_NEST_DIRECTION_RETURN = 250.0       # Strong bias towards moving closer to the nest coordinates when returning
# --- Searching State ---
# Base attraction to food pheromones when searching (colony needs are moderate)
W_FOOD_PHEROMONE_SEARCH_BASE = 40.0
# Attraction to food pheromones when the specific food type is NOT critically needed
W_FOOD_PHEROMONE_SEARCH_LOW_NEED = 5.0
# Repulsion from food pheromones of the OPPOSITE type when the colony critically needs the OTHER type
W_FOOD_PHEROMONE_SEARCH_AVOID = -10.0
# Attraction to home pheromones when searching (typically low or zero, ants explore outwards)
W_HOME_PHEROMONE_SEARCH = 0.0
# VERY strong attraction to a food pheromone type if that type is CRITICALLY low in the colony
W_FOOD_PHEROMONE_SEARCH_CRITICAL_NEED = 100.0
# Repulsion/avoidance factor for a food pheromone type if the OTHER type is critically needed
# (Set to 0.0 to just ignore the non-needed type, negative to actively avoid)
W_FOOD_PHEROMONE_SEARCH_CRITICAL_AVOID = 0.0
# Repulsion from alarm pheromones (negative value = avoidance)
W_ALARM_PHEROMONE = -10.0
# Repulsion from negative pheromones (avoid explored/empty areas)
W_NEGATIVE_PHEROMONE = -10.0
# Strong attraction to recruitment pheromones (signals important locations like rich food or danger)
W_RECRUITMENT_PHEROMONE = 250.0
# Penalty for searching ants moving towards/staying near the nest center (encourages exploration)
W_AVOID_NEST_SEARCHING = -400.0
# --- Patrolling State (Soldiers) ---
# Repulsion from moving away from the nest center while patrolling within radius
W_NEST_DIRECTION_PATROL = -10.0
# --- Defending State ---
# Strong attraction to the estimated source of alarm/recruitment signals
W_ALARM_SOURCE_DEFEND = 500.0
# --- Hunting State ---
# Strong attraction towards the targeted prey's location
W_HUNTING_TARGET = 350.0
# --- General Movement Modifiers ---
# Bonus for continuing in the same direction as the last move (inertia)
W_PERSISTENCE = 0.5
# Random noise added to move scores to break ties and add variability
W_RANDOM_NOISE = 0.2
# Penalty for moving to a cell recently visited (part of path history)
W_AVOID_HISTORY = -200.0
# Penalty applied to moves towards cells occupied by other ants (simple repulsion)
# Note: Direct blocking is handled separately; this is a scoring penalty.
W_REPULSION = 20.0  # Seems low, maybe increase? But blocking exists.

# --- Probabilistic Choice Parameters ---
# Temperature parameter for probabilistic move selection (SEARCHING, PATROLLING states).
# Higher temp -> more randomness/exploration. Lower temp -> more greedy choice (highest score).
PROBABILISTIC_CHOICE_TEMP = 0.5
# Minimum score required for a move to be considered in probabilistic selection
# (prevents tiny negative scores dominating if all scores are negative).
MIN_SCORE_FOR_PROB_CHOICE = 0.01

# --- Ant Parameters ---
# Initial number of ants spawned at the start
INITIAL_ANTS = 10
# Maximum number of ants allowed in the simulation (includes queen if counted, currently not)
MAX_ANTS = 50
# Queen's starting Hit Points
QUEEN_HP = 1000
# Average lifespan of a worker ant in simulation ticks
WORKER_MAX_AGE_MEAN = 12000
# Standard deviation for worker lifespan, introducing variability
WORKER_MAX_AGE_STDDEV = 2000
# Length of ant's path history (in ticks) - used for avoidance
WORKER_PATH_HISTORY_LENGTH = 8
# Number of consecutive ticks an ant must be blocked before triggering ESCAPING state
WORKER_STUCK_THRESHOLD = 30
# Duration (in ticks) an ant stays in ESCAPING state
WORKER_ESCAPE_DURATION = 30
# Interval (in ticks) at which ants consume food from colony storage
WORKER_FOOD_CONSUMPTION_INTERVAL = 100
# Radius around the nest (as multiplier of NEST_RADIUS) soldiers try to stay within when patrolling
SOLDIER_PATROL_RADIUS_MULTIPLIER = 0.2
# Combined alarm/recruitment pheromone threshold required for a soldier to switch to DEFENDING state
SOLDIER_DEFEND_ALARM_THRESHOLD = 100.0
# Radius (in grid cells) used by DEFENDING ants to scan for the strongest alarm/recruitment signal source
ALARM_SEARCH_RADIUS_SIGNAL = 10
# Radius (in grid cells) used by DEFENDING ants for random searching if no strong signal is found nearby
ALARM_SEARCH_RADIUS_RANDOM = 20
# Visual range of ants (in grid cells) for detecting enemies directly
ANT_VISUAL_RANGE = 6

# --- Brood Cycle Parameters ---
# Interval (in ticks) the queen attempts to lay an egg (can be modified by food availability)
QUEEN_EGG_LAY_RATE = 60
# Amount of sugar consumed from colony storage per egg laid
QUEEN_FOOD_PER_EGG_SUGAR = 1.0
# Amount of protein consumed from colony storage per egg laid
QUEEN_FOOD_PER_EGG_PROTEIN = 1.5
# Target ratio of soldiers in the total ant population (influences caste of new eggs)
QUEEN_SOLDIER_RATIO_TARGET = 0.15
# Duration (in ticks) for an egg to hatch into a larva
EGG_DURATION = 500
# Duration (in ticks) for a larva to develop into a pupa
LARVA_DURATION = 800
# Duration (in ticks) for a pupa to hatch into an adult ant
PUPA_DURATION = 600
# Amount of protein consumed by a larva from colony storage at each feeding interval
LARVA_FOOD_CONSUMPTION_PROTEIN = 0.05
# Amount of sugar consumed by a larva from colony storage at each feeding interval
LARVA_FOOD_CONSUMPTION_SUGAR = 0.01
# Interval (in ticks) at which larvae need to be fed
LARVA_FEED_INTERVAL = 50

# --- Enemy Parameters ---
# Initial number of enemies spawned at the start
INITIAL_ENEMIES = 1
# Enemy starting Hit Points
ENEMY_HP = 60
# Damage dealt by an enemy per attack
ENEMY_ATTACK = 10
# Delay (in ticks) between enemy moves
ENEMY_MOVE_DELAY = 4
# Interval (in ticks) at which new enemies are potentially spawned
ENEMY_SPAWN_RATE = 1000
# Amount of sugar added to the grid cell when an enemy dies
ENEMY_TO_FOOD_ON_DEATH_SUGAR = 10.0
# Amount of protein added to the grid cell when an enemy dies
ENEMY_TO_FOOD_ON_DEATH_PROTEIN = 20.0
# Probability (per move attempt) that an enemy will choose to move towards the nest center
ENEMY_NEST_ATTRACTION = 0.05

# --- Prey Parameters ---
# Initial number of prey spawned at the start
INITIAL_PREY = 5
# Prey starting Hit Points
PREY_HP = 25
# Delay (in ticks) between prey moves
PREY_MOVE_DELAY = 2
# Interval (in ticks) at which new prey are potentially spawned
PREY_SPAWN_RATE = 1000
# Amount of protein added to the grid cell when prey dies
PROTEIN_ON_DEATH = 20.0
# Squared radius (in grid cells) within which prey will detect and flee from ants
PREY_FLEE_RADIUS_SQ = 5 * 5

# --- Network Streaming ---
# Enable Flask web server for streaming simulation view (requires Flask installation)
ENABLE_NETWORK_STREAM = False
# Host IP address for the streaming server ('0.0.0.0' allows access from other machines)
STREAMING_HOST = "0.0.0.0"
# Port number for the streaming server
STREAMING_PORT = 5000
# JPEG quality for streamed frames (0-100, higher is better quality but more data)
STREAM_FRAME_QUALITY = 75
# Maximum frames per second for the network stream (limits bandwidth usage)
STREAM_FPS_LIMIT = 15

# --- UI Colors ---
BUTTON_COLOR = (80, 80, 150)          # Normal button color
BUTTON_HOVER_COLOR = (100, 100, 180)   # Button color when mouse hovers over it
BUTTON_TEXT_COLOR = (240, 240, 240)    # Text color on buttons
END_DIALOG_BG_COLOR = (0, 0, 0, 180)   # Background color for end game dialog (semi-transparent black)
LEGEND_BG_COLOR = (10, 10, 30, 180)    # Background color for legend box (semi-transparent dark blue)
LEGEND_TEXT_COLOR = (230, 230, 230)    # Text color in the legend

# --- Auto Restart ---
# Delay (in seconds) before automatically restarting the simulation after the Queen dies
AUTO_RESTART_DELAY_SECONDS = 10

# --- Font Size Scaling ---
# Base font sizes, scaled dynamically based on screen height in _init_fonts()
BASE_FONT_SIZE = 20          # Base size for UI text (buttons, dialogs)
BASE_DEBUG_FONT_SIZE = 16    # Base size for debug overlay text
BASE_LEGEND_FONT_SIZE = 15   # Base size for legend text
# The window height used as the reference for font scaling calculations
REFERENCE_HEIGHT_FOR_SCALING = DEFAULT_WINDOW_HEIGHT

# --- Attack Indicators ---
# Color for ant attack indicators (e.g., lines drawn during attacks)
ATTACK_INDICATOR_COLOR_ANT = (255, 255, 100, 255)  # Yellowish flash
# Color for enemy attack indicators
ATTACK_INDICATOR_COLOR_ENEMY = (255, 100, 100, 255) # Reddish flash
# Duration (in ticks) the attack indicator visual effect lasts
ATTACK_INDICATOR_DURATION_TICKS = 6

# --- Enums ---
# Define enumerations for distinct simulation states, types, etc.
# Using Enums improves readability and helps prevent errors from typos.

class AntState(Enum):
    """Defines the possible behavioral states of an ant."""
    SEARCHING = auto()          # Looking for food or other tasks
    RETURNING_TO_NEST = auto()  # Carrying food back to the nest
    ESCAPING = auto()           # Trying to get unstuck from a blocking situation
    PATROLLING = auto()         # Soldier specific: Guarding the nest area
    DEFENDING = auto()          # Engaging threats (enemies) or reacting to alarms
    HUNTING = auto()            # Actively pursuing a specific prey item
    # TENDING_BROOD = auto()    # Placeholder for potential future brood care behavior

class BroodStage(Enum):
    """Defines the developmental stages of ant brood."""
    EGG = auto()
    LARVA = auto()
    PUPA = auto()

class AntCaste(Enum):
    """Defines the different functional types (castes) of ants."""
    WORKER = auto()
    SOLDIER = auto()

class FoodType(Enum):
    """Defines the types of food resources available."""
    # Using integer values allows direct use as indices in NumPy arrays (e.g., self.grid.food)
    SUGAR = 0
    PROTEIN = 1

# --- Ant Caste Attributes ---
# Defines base parameters for each ant caste.
# These values are used when creating new ants.
ANT_ATTRIBUTES = {
    AntCaste.WORKER: {
        "hp": 50,
        "attack": 3,
        "capacity": 2.5,
        "move_cooldown_base": 0, # <<< RENAMED from speed_delay
        "color": (200, 200, 200),
        "return_color": (200, 255, 200),
        "food_consumption_sugar": 0.02,
        "food_consumption_protein": 0.005,
        "description": "Worker",
        "size_factor": 2.5,
        "head_size_factor": 0.4,
    },
    AntCaste.SOLDIER: {
        "hp": 90,
        "attack": 10,
        "capacity": 0.2,
        "move_cooldown_base": 1, # <<< RENAMED from speed_delay
        "color": (230, 200, 200),
        "return_color": (255, 230, 200),
        "food_consumption_sugar": 0.025,
        "food_consumption_protein": 0.01,
        "description": "Soldier",
        "size_factor": 1.8,
        "head_size_factor": 0.6,
    },
}
# --- Other Colors ---

# Ant Colors based on State/Action
ANT_BASE_COLOR = (200, 200, 200)         # Medium gray (currently less used, caste colors preferred)
QUEEN_COLOR = (200, 200, 255)               # Blue
WORKER_ESCAPE_COLOR = (255, 255, 0)     # Yellow (when in ESCAPING state)
ANT_DEFEND_COLOR = (255, 0, 0)          # Red (when in DEFENDING state)
ANT_HUNT_COLOR = (0, 255, 255)          # Cyan (when in HUNTING state)

# Other Entity Colors
ENEMY_COLOR = (200, 0, 0)               # Red
PREY_COLOR = (0, 100, 0)                # Dark green

# Food Colors
FOOD_COLORS = {
    FoodType.SUGAR: (200, 200, 255),    # Light Blue/Purple
    FoodType.PROTEIN: (255, 180, 180),  # Light Red/Pink
}
FOOD_COLOR_MIX = (230, 200, 230)        # Default mix color for UI/Legend where specific type isn't shown

# Pheromone Colors (RGBA format, Alpha controls intensity/visibility)
PHEROMONE_HOME_COLOR = (0, 0, 255, 50)              # Blue trail to nest
PHEROMONE_FOOD_SUGAR_COLOR = (180, 180, 255, 50)   # Light Blue/Purple trail to sugar
PHEROMONE_FOOD_PROTEIN_COLOR = (255, 160, 160, 50)  # Light Red/Pink trail to protein
PHEROMONE_ALARM_COLOR = (255, 0, 0, 80)             # Red signal for danger
PHEROMONE_NEGATIVE_COLOR = (150, 150, 150, 10)      # Grey signal for explored/empty areas
PHEROMONE_RECRUITMENT_COLOR = (255, 0, 255, 80)     # Magenta/Pink signal for important events (rich food, damage)

# Brood Colors (RGBA format, Alpha adds density effect)
EGG_COLOR = (255, 255, 255, 200)      # White
LARVA_COLOR = (255, 255, 200, 220)    # Pale Yellow
PUPA_COLOR = (200, 180, 150, 220)     # Beige/Brown

# Define shorter aliases for frequently used random functions
# Enhances readability in the code where random numbers are generated.
rnd = random.randint          # Generates a random integer within a specified range (inclusive).
rnd_gauss = random.gauss        # Generates a random float from a Gaussian distribution (bell curve).
rnd_uniform = random.uniform    # Generates a random float uniformly distributed within a specified range.

# --- Network Streaming Setup ---
# These variables manage the optional Flask web server for streaming the simulation view.

# The Flask application instance. Initialized if streaming is enabled.
streaming_app = None
# The thread running the Flask server.
streaming_thread = None
# A lock to ensure thread-safe access to the latest captured frame data.
latest_frame_lock = threading.Lock()
# Holds the bytes of the most recently captured JPEG frame for streaming.
latest_frame_bytes = None
# An event flag used to signal the streaming thread to stop gracefully.
stop_streaming_event = threading.Event()

# Basic HTML page template for viewing the stream in a web browser.
# Uses Flask's template rendering to insert the correct image dimensions.
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
    <img src="{{{{ url_for('video_feed') }}}}" width="{{{{ width }}}}" height="{{{{ height }}}}">
  </body>
</html>
"""

def stream_frames():
    """
    Generator function that yields JPEG image frames for the MJPEG stream.

    Continuously checks for the latest captured frame (`latest_frame_bytes`)
    and yields it formatted as part of a multipart HTTP response.
    Includes throttling based on `STREAM_FPS_LIMIT` to control bandwidth.
    """
    global latest_frame_bytes  # Access the globally shared frame data
    last_yield_time = time.time()
    # Minimum time interval between yielding frames based on the desired stream FPS
    min_interval = 1.0 / STREAM_FPS_LIMIT if STREAM_FPS_LIMIT > 0 else 0

    print("Streamer thread: Starting frame generation loop.")
    while not stop_streaming_event.is_set():
        current_time = time.time()
        # Throttle: Wait if the minimum interval hasn't passed since the last yield
        if min_interval > 0 and current_time - last_yield_time < min_interval:
            # Sleep briefly to avoid busy-waiting
            time.sleep(min_interval / 5)
            continue

        frame_data = None
        # Safely access the shared frame data using the lock
        with latest_frame_lock:
            if latest_frame_bytes:
                frame_data = latest_frame_bytes
                # Optional: Clear latest_frame_bytes after reading to ensure
                # only *new* frames are sent. This might cause the stream
                # to pause if the simulation frame rate is lower than STREAM_FPS_LIMIT.
                # latest_frame_bytes = None

        if frame_data:
            try:
                # Yield the frame data in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                last_yield_time = current_time
            except Exception as e:
                # Log errors during streaming (e.g., client disconnected)
                print(f"Streamer thread: Error yielding frame: {e}")
                # Avoid busy-looping on persistent errors
                time.sleep(0.5)
        else:
            # No new frame available, wait briefly before checking again
            sleep_duration = min_interval / 2 if min_interval > 0 else 0.05
            time.sleep(sleep_duration)

    print("Streamer thread: Stop event received, exiting frame generation loop.")

def run_server(app: Flask, host: str, port: int):
    """
    Runs the Flask web server in the current thread.

    Designed to be executed in a separate thread to avoid blocking the main
    simulation loop. It starts the Flask development server to handle HTTP
    requests for the stream.

    Args:
        app: The Flask application instance to run.
        host: The hostname or IP address to bind the server to.
        port: The port number to bind the server to.
    """
    print(f" * Starting Flask server thread on http://{host}:{port}")
    try:
        # Run the Flask app.
        # `threaded=True` allows handling multiple client requests concurrently.
        # `debug=False` is recommended for stability when run threaded.
        # `use_reloader=False` prevents Flask from restarting the server process,
        # which is problematic when run inside another application's thread.
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
    except OSError as e:
        # Catch specific OS errors like "Address already in use"
        print(f"FATAL: Could not start Flask server: {e}")
        print(f"       Is another process using port {port}?")
        # Optionally, signal the main thread to stop if the server is critical.
        # stop_streaming_event.set() # Example: Signal main thread problems
    except Exception as e:
        # Catch any other unexpected errors during server startup or runtime.
        print(f"FATAL: Flask server encountered an error: {e}")
        traceback.print_exc() # Print detailed traceback
    finally:
        # This block executes when the server stops (either normally or due to error).
        print(" * Flask server thread stopped.")
# --- Helper Functions ---

def is_valid_pos(pos: tuple, grid_width: int, grid_height: int) -> bool:
    """
    Checks if a grid position (column, row) is within the valid boundaries.

    Args:
        pos: A tuple representing the position (x, y) or (column, row).
        grid_width: The total number of columns in the grid.
        grid_height: The total number of rows in the grid.

    Returns:
        True if the position is valid (within 0 <= x < grid_width and
        0 <= y < grid_height), False otherwise. Returns False for invalid
        input types (non-tuple, wrong length).
    """
    # Basic type and length check for the position tuple
    if not isinstance(pos, tuple) or len(pos) != 2:
        return False
    x, y = pos
    # Check if both x and y coordinates are within the grid dimensions
    # Note: Grid coordinates are 0-indexed.
    return 0 <= x < grid_width and 0 <= y < grid_height

def get_neighbors(pos: tuple, grid_width: int, grid_height: int, include_center: bool = False) -> list[tuple]:
    """
    Gets valid integer neighbor coordinates for a given grid position.

    Includes the 8 adjacent cells (Moore neighborhood). Can optionally
    include the center cell itself. Returns only coordinates that are
    within the grid boundaries.

    Args:
        pos: The center position (x, y) as a tuple. Floats will be truncated.
        grid_width: The width of the grid.
        grid_height: The height of the grid.
        include_center: If True, the original `pos` will be included in the
                        list if it's valid. Defaults to False.

    Returns:
        A list of valid neighbor position tuples (int, int). Returns an
        empty list if the input `pos` itself is invalid.
    """
    try:
        # Ensure the center position is treated as integers
        x_int, y_int = int(pos[0]), int(pos[1])
    except (TypeError, ValueError, IndexError):
        return [] # Invalid input position

    # Check if the center position is even valid before proceeding
    if not (0 <= x_int < grid_width and 0 <= y_int < grid_height):
        return []

    neighbors = []
    # Iterate through the 3x3 grid centered around the position
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # Skip the center cell if not requested
            if dx == 0 and dy == 0 and not include_center:
                continue

            # Calculate neighbor coordinates
            n_pos = (x_int + dx, y_int + dy)

            # Check if the neighbor is within the grid boundaries
            if 0 <= n_pos[0] < grid_width and 0 <= n_pos[1] < grid_height:
                neighbors.append(n_pos)

    return neighbors

def distance_sq(pos1: tuple, pos2: tuple) -> float:
    """
    Calculates the squared Euclidean distance between two points.

    Using squared distance avoids the computationally more expensive
    square root operation and is sufficient for comparing distances.
    Handles potential errors by returning infinity if inputs are invalid.

    Args:
        pos1: The first position tuple (x1, y1). Coordinates can be int or float.
        pos2: The second position tuple (x2, y2). Coordinates can be int or float.

    Returns:
        The squared Euclidean distance as a float. Returns float('inf')
        if either input is invalid (e.g., not a tuple, wrong length,
        or contains non-numeric types).
    """
    try:
        # Unpack coordinates from the input tuples
        x1, y1 = pos1
        x2, y2 = pos2
        # Calculate the squared difference for x and y coordinates
        dx_sq = (x1 - x2) ** 2
        dy_sq = (y1 - y2) ** 2
        # Return the sum as a float
        return float(dx_sq + dy_sq)
    except (TypeError, ValueError, IndexError):
        # Catch errors if inputs are not tuples, have wrong length, or contain non-numeric data
        # print(f"Warning: Invalid input for distance_sq: {pos1}, {pos2}") # Optional debug log
        # Return infinity to signify an invalid distance calculation
        return float("inf")

def normalize(value: float | int, max_val: float | int) -> float:
    """
    Normalizes a value to the range [0.0, 1.0] based on a maximum value.

    The normalization is clamped, meaning values below 0 will result in 0.0,
    and values above `max_val` will result in 1.0. If `max_val` is zero or
    negative, the function returns 0.0 to avoid division by zero.

    Args:
        value: The numerical value to normalize.
        max_val: The maximum possible value for the normalization scale.

    Returns:
        The normalized value as a float between 0.0 and 1.0 (inclusive).
    """
    # Handle division by zero or invalid maximum value case
    if max_val <= 0:
        return 0.0

    # Calculate the normalized value
    # Convert to float for potential division
    norm_val = float(value) / float(max_val)

    # Clamp the result to the range [0.0, 1.0]
    return min(1.0, max(0.0, norm_val))

# --- Brood Class ---
class BroodItem:
    """
    Represents an item of brood (egg, larva, pupa) within the nest.
    Manages its own development progress and state transitions.
    """
    def __init__(
        self, stage: BroodStage, caste: AntCaste, position: tuple, current_tick: int,
        simulation  # Pass simulation reference for accessing grid dimensions, etc.
    ):
        """
        Initializes a brood item.

        Args:
            stage: The initial developmental stage (EGG, LARVA, PUPA).
            caste: The caste the ant will become (WORKER, SOLDIER).
            position: The (x, y) grid coordinates where the brood is placed.
            current_tick: The simulation tick at which the brood item was created.
            simulation: Reference to the main AntSimulation object.
        """
        self.stage = stage
        self.caste = caste
        self.pos = tuple(map(int, position))  # Ensure integer coordinates
        self.creation_tick = current_tick
        self.progress_timer = 0.0  # Ticks accumulated towards next stage duration
        self.last_feed_check = current_tick  # Tick when larva last attempted to feed
        self.simulation = simulation  # Store simulation reference

        # Set duration, color, and drawing radius based on the initial stage
        cell_size = self.simulation.cell_size
        if self.stage == BroodStage.EGG:
            self.duration = EGG_DURATION
            self.color = EGG_COLOR
            self.radius = max(1, cell_size // 8)  # Min radius 1 pixel
        elif self.stage == BroodStage.LARVA:
            self.duration = LARVA_DURATION
            self.color = LARVA_COLOR
            self.radius = max(1, cell_size // 4)
        elif self.stage == BroodStage.PUPA:
            self.duration = PUPA_DURATION
            self.color = PUPA_COLOR
            self.radius = max(1, int(cell_size / 3.5))
        else:
            # Should not happen with valid stages, but default values
            self.duration = 0
            self.color = (0, 0, 0, 0)  # Invisible
            self.radius = 0

    def update(self, current_tick: int):
        """
        Updates the brood item's development progress for one simulation tick.

        Handles stage transitions (Egg -> Larva -> Pupa) and food consumption
        for larvae. Returns the brood item itself if it hatches (Pupa -> Ant),
        otherwise returns None.

        Args:
            current_tick: The current simulation tick.

        Returns:
            The BroodItem instance if it hatches this tick, otherwise None.
        """
        sim = self.simulation
        growth_factor = 1.0  # Standard growth rate is 1 tick per update call

        # Larvae consume food and may grow slower if resources are scarce
        if self.stage == BroodStage.LARVA:
            # Check food consumption only if the interval has passed
            if current_tick - self.last_feed_check >= LARVA_FEED_INTERVAL:
                self.last_feed_check = current_tick
                needed_p = LARVA_FOOD_CONSUMPTION_PROTEIN
                needed_s = LARVA_FOOD_CONSUMPTION_SUGAR
                consumed_p = False
                consumed_s = False

                # Consume protein if available
                if sim.colony_food_storage_protein >= needed_p:
                    sim.colony_food_storage_protein -= needed_p
                    consumed_p = True

                # Consume sugar if available
                if sim.colony_food_storage_sugar >= needed_s:
                    sim.colony_food_storage_sugar -= needed_s
                    consumed_s = True

                # Reduce growth slightly if either food type was missing
                if not consumed_p or not consumed_s:
                    growth_factor *= 0.75  # Example: Reduce growth to 75%

        # Update development progress by the calculated growth factor
        self.progress_timer += growth_factor

        # Check for stage progression if duration is reached
        if self.progress_timer >= self.duration:
            if self.stage == BroodStage.EGG:
                # Transition to Larva
                self.stage = BroodStage.LARVA
                self.progress_timer = 0.0  # Reset progress timer
                self.duration = LARVA_DURATION
                self.color = LARVA_COLOR
                self.radius = max(1, sim.cell_size // 4)
                self.last_feed_check = current_tick # Reset feed check timer for larva
                return None  # Still brood

            elif self.stage == BroodStage.LARVA:
                # Optional check: Could add a condition here to prevent pupation
                # if the larva missed its last feeding (e.g., `if not consumed_p or not consumed_s: return None`)
                # Transition to Pupa
                self.stage = BroodStage.PUPA
                self.progress_timer = 0.0
                self.duration = PUPA_DURATION
                self.color = PUPA_COLOR
                self.radius = max(1, int(sim.cell_size / 3.5))
                return None  # Still brood

            elif self.stage == BroodStage.PUPA:
                # Pupa hatches! Return self to signal hatching.
                return self

        return None  # Still developing

    def draw(self, surface: pygame.Surface):
        """
        Draws the brood item onto the specified Pygame surface.

        Args:
            surface: The Pygame surface to draw on.
        """
        sim = self.simulation
        # Basic validation before drawing
        if not is_valid_pos(self.pos, sim.grid_width, sim.grid_height) or self.radius <= 0:
            return

        # Calculate center pixel coordinates for drawing
        center_x = int(self.pos[0] * sim.cell_size + sim.cell_size // 2)
        center_y = int(self.pos[1] * sim.cell_size + sim.cell_size // 2)
        draw_pos = (center_x, center_y)

        # Draw the main circle representing the brood item
        pygame.draw.circle(surface, self.color, draw_pos, self.radius)

        # Add an outline to pupae to indicate the future ant's caste
        if self.stage == BroodStage.PUPA:
            outline_col = (50, 50, 50) if self.caste == AntCaste.WORKER else (100, 0, 0)
            pygame.draw.circle(surface, outline_col, draw_pos, self.radius, 1) # 1px outline

# --- Grid Class ---
class WorldGrid:
    """
    Manages the simulation grid, holding information about food, obstacles,
    and pheromone levels at each cell.
    """
    def __init__(self, grid_width: int, grid_height: int):
        """
        Initializes the world grid with the specified dimensions.

        Creates NumPy arrays to store grid data efficiently.

        Args:
            grid_width: The number of columns in the grid.
            grid_height: The number of rows in the grid.

        Raises:
            ValueError: If grid_width or grid_height are not positive integers.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        print(f"Initializing WorldGrid with dimensions: {grid_width}x{grid_height}")

        # Ensure dimensions are valid before creating large NumPy arrays
        if not (grid_width > 0 and grid_height > 0):
            raise ValueError(f"Invalid grid dimensions for WorldGrid: {grid_width}x{grid_height}")

        # --- Initialize Grid Data Arrays ---
        # Food: Stores amount of each food type per cell.
        # Shape: (width, height, num_food_types)
        self.food = np.zeros((grid_width, grid_height, NUM_FOOD_TYPES), dtype=np.float32)

        # Obstacles: Boolean flag indicating if a cell is blocked.
        # Shape: (width, height)
        self.obstacles = np.zeros((grid_width, grid_height), dtype=bool)

        # Pheromones: Separate arrays for each type.
        # Shape: (width, height) for each type.
        # Using float32 for performance and memory efficiency.
        self.pheromones_home = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_alarm = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_negative = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_recruitment = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_food_sugar = np.zeros((grid_width, grid_height), dtype=np.float32)
        self.pheromones_food_protein = np.zeros((grid_width, grid_height), dtype=np.float32)

    def reset(self, nest_pos: tuple):
        """
        Resets the grid state for a new simulation run.

        Clears all food and pheromones, then places new obstacles and initial
        food clusters based on the provided nest position.

        Args:
            nest_pos: The (x, y) coordinates of the colony's nest center.
                      Used to ensure obstacles and initial food are placed
                      appropriately relative to the nest.
        """
        print("Resetting WorldGrid state...")
        # Clear all dynamic grid data by filling arrays with zeros
        self.food.fill(0)
        self.obstacles.fill(0)  # Obstacles are recalculated below
        self.pheromones_home.fill(0)
        self.pheromones_alarm.fill(0)
        self.pheromones_negative.fill(0)
        self.pheromones_recruitment.fill(0)
        self.pheromones_food_sugar.fill(0)
        self.pheromones_food_protein.fill(0)

        # Generate and place new obstacles, avoiding the nest area
        self.place_obstacles(nest_pos)

        # Place initial food clusters away from the nest
        self.place_food_clusters(nest_pos)
        print("WorldGrid reset complete.")

    def place_food_clusters(self, nest_pos: tuple):
        """
        Places initial food clusters on the grid at the start of a simulation.

        Clusters are placed randomly, attempting to position them a minimum
        distance away from the nest and avoiding obstacle cells.

        Args:
            nest_pos: The (x, y) integer coordinates of the nest center.
        """
        print(f"Placing {INITIAL_FOOD_CLUSTERS} initial food clusters...")
        nest_pos_int = tuple(map(int, nest_pos))
        min_dist_sq = MIN_FOOD_DIST_FROM_NEST ** 2  # Use squared distance for efficiency

        for i in range(INITIAL_FOOD_CLUSTERS):
            # Alternate food types for initial clusters if multiple types exist
            food_type_index = i % NUM_FOOD_TYPES
            food_type_enum = FoodType(food_type_index) # Get the enum member
            print(f"  Placing cluster {i+1} ({food_type_enum.name})...")

            attempts = 0
            max_placement_attempts_center = 150 # Max tries to find a good center spot
            cx, cy = 0, 0 # Cluster center coordinates
            found_spot = False

            # --- Try finding a suitable cluster center ---
            # Attempt 1: Find a spot far from the nest and not on an obstacle
            while attempts < max_placement_attempts_center and not found_spot:
                cx = rnd(0, self.grid_width - 1)
                cy = rnd(0, self.grid_height - 1)
                pos_check = (cx, cy)
                if (is_valid_pos(pos_check, self.grid_width, self.grid_height) and
                        not self.obstacles[cx, cy] and
                        distance_sq(pos_check, nest_pos_int) > min_dist_sq):
                    found_spot = True
                attempts += 1

            # Attempt 2 (Fallback): Find any non-obstacle spot if the first attempt failed
            if not found_spot:
                print(f"    Warning: Could not find spot far from nest for cluster {i+1}. Trying any non-obstacle spot.")
                attempts = 0
                max_placement_attempts_fallback = 200
                while attempts < max_placement_attempts_fallback:
                    cx = rnd(0, self.grid_width - 1)
                    cy = rnd(0, self.grid_height - 1)
                    pos_check = (cx, cy)
                    if is_valid_pos(pos_check, self.grid_width, self.grid_height) and not self.obstacles[cx, cy]:
                        found_spot = True
                        break
                    attempts += 1

            # Attempt 3 (Last Resort): Place anywhere if still no spot found (should be rare)
            if not found_spot:
                print(f"    Warning: Could not find any non-obstacle spot for cluster {i+1}. Placing randomly.")
                cx = rnd(0, self.grid_width - 1)
                cy = rnd(0, self.grid_height - 1)

            print(f"    Cluster {i+1} center set to: ({cx}, {cy})")

            # --- Distribute food items around the chosen center ---
            added_amount = 0.0
            target_food_amount = FOOD_PER_CLUSTER
            # Allow more attempts than target amount to account for randomness and obstacles
            max_food_placement_attempts = int(target_food_amount * 2.5)

            for _ in range(max_food_placement_attempts):
                # Stop if the target amount for the cluster has been reached
                if added_amount >= target_food_amount:
                    break

                # Place food with a Gaussian distribution around the cluster center (cx, cy)
                # Use half the cluster radius as the standard deviation for a tighter spread
                try:
                    fx = cx + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                    fy = cy + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                except OverflowError:
                     # Fallback if gaussian calculation fails (e.g., huge radius)
                     fx = cx + rnd(-FOOD_CLUSTER_RADIUS, FOOD_CLUSTER_RADIUS)
                     fy = cy + rnd(-FOOD_CLUSTER_RADIUS, FOOD_CLUSTER_RADIUS)


                # Check if the calculated food position is valid and not an obstacle
                if (0 <= fx < self.grid_width and 0 <= fy < self.grid_height and
                        not self.obstacles[fx, fy]):

                    # Determine amount to add (random small amount per placement)
                    amount_to_add = rnd_uniform(0.5, 1.0) * (MAX_FOOD_PER_CELL / 8)
                    current_amount = self.food[fx, fy, food_type_index]

                    # Ensure the cell doesn't exceed the maximum food capacity
                    new_amount = min(MAX_FOOD_PER_CELL, current_amount + amount_to_add)
                    actual_added = new_amount - current_amount

                    # Add the food if the amount is positive
                    if actual_added > 0:
                        self.food[fx, fy, food_type_index] = new_amount
                        added_amount += actual_added

            print(f"    Cluster {i+1} placed ~{added_amount:.1f} units of {food_type_enum.name}.")

        print(f"Finished placing initial food clusters.")

    def place_obstacles(self, nest_pos: tuple):
        """
        Places obstacles on the grid using a clumping algorithm for more organic shapes.

        Generates several obstacle 'areas'. Each area consists of multiple overlapping
        circular 'clumps'. Ensures obstacles do not overlap the defined nest area.

        Args:
            nest_pos: The (x, y) integer coordinates of the nest center.
        """
        print(f"Placing {NUM_OBSTACLES} organic obstacle areas...")
        nest_center_int = tuple(map(int, nest_pos))
        # Define a buffer zone around the nest where obstacles cannot be placed.
        # Use squared distance for efficiency. Increased buffer for rounded shapes.
        nest_clearance_radius = NEST_RADIUS + 4
        nest_clearance_radius_sq = nest_clearance_radius ** 2

        # --- Pre-calculate the nest area set for faster checking ---
        nest_area = set()
        # Determine bounding box around the nest clearance area
        min_x_nest = max(0, nest_center_int[0] - nest_clearance_radius)
        max_x_nest = min(self.grid_width - 1, nest_center_int[0] + nest_clearance_radius)
        min_y_nest = max(0, nest_center_int[1] - nest_clearance_radius)
        max_y_nest = min(self.grid_height - 1, nest_center_int[1] + nest_clearance_radius)

        # Iterate within the bounding box and add cells within the circular radius to the set
        for x in range(min_x_nest, max_x_nest + 1):
            for y in range(min_y_nest, max_y_nest + 1):
                if distance_sq((x, y), nest_center_int) <= nest_clearance_radius_sq:
                    nest_area.add((x, y))
        # --- End Nest Area Calculation ---

        placed_count = 0
        # Allow significantly more attempts than the target number of obstacles
        # because placement attempts can fail (e.g., overlap nest, overlap existing obstacle).
        max_obstacle_attempts = NUM_OBSTACLES * 25

        # --- Try to place each obstacle area ---
        for attempt_num in range(max_obstacle_attempts):
            # Stop if the target number of obstacles has been placed
            if placed_count >= NUM_OBSTACLES:
                break

            attempts_per_obstacle = 0
            # Max tries to find a valid spot for *one* obstacle area
            max_attempts_per_obstacle = 35
            placed_this_obstacle = False

            # --- Try finding a valid center and generating clumps for one obstacle area ---
            while attempts_per_obstacle < max_attempts_per_obstacle and not placed_this_obstacle:
                attempts_per_obstacle += 1

                # 1. Choose a potential center for the obstacle cluster (randomly on the grid)
                cluster_cx = rnd(0, self.grid_width - 1)
                cluster_cy = rnd(0, self.grid_height - 1)

                # 2. Quick check: Is the chosen center itself inside the nest clearance zone?
                if (cluster_cx, cluster_cy) in nest_area:
                    continue # Try a different center

                # 3. Generate clumps for this potential obstacle area
                num_clumps = rnd(MIN_OBSTACLE_CLUMPS, MAX_OBSTACLE_CLUMPS)
                 # Store potential obstacle cells for this attempt
                obstacle_cells_this_attempt = set()
                # Flag to track if this obstacle attempt is valid
                can_place_this = True

                for _ in range(num_clumps):
                    # Calculate clump center relative to the main cluster center (Gaussian spread)
                    try:
                        clump_offset_x = int(rnd_gauss(0, OBSTACLE_CLUSTER_SPREAD_RADIUS * 0.5))
                        clump_offset_y = int(rnd_gauss(0, OBSTACLE_CLUSTER_SPREAD_RADIUS * 0.5))
                    except OverflowError: # Fallback if gaussian fails
                        clump_offset_x = rnd(-OBSTACLE_CLUSTER_SPREAD_RADIUS, OBSTACLE_CLUSTER_SPREAD_RADIUS)
                        clump_offset_y = rnd(-OBSTACLE_CLUSTER_SPREAD_RADIUS, OBSTACLE_CLUSTER_SPREAD_RADIUS)

                    clump_cx = cluster_cx + clump_offset_x
                    clump_cy = cluster_cy + clump_offset_y
                    clump_radius = rnd(MIN_OBSTACLE_CLUMP_RADIUS, MAX_OBSTACLE_CLUMP_RADIUS)
                    clump_radius_sq = clump_radius ** 2

                    # Iterate through the bounding box of the current clump
                    min_x_clump = max(0, clump_cx - clump_radius)
                    max_x_clump = min(self.grid_width - 1, clump_cx + clump_radius)
                    min_y_clump = max(0, clump_cy - clump_radius)
                    max_y_clump = min(self.grid_height - 1, clump_cy + clump_radius)

                    for x in range(min_x_clump, max_x_clump + 1):
                        for y in range(min_y_clump, max_y_clump + 1):
                            pos_check = (x, y)
                            # Check distance from clump center and if it overlaps the nest area
                            if distance_sq(pos_check, (clump_cx, clump_cy)) <= clump_radius_sq:
                                if pos_check in nest_area:
                                     # Clump hit the nest, abandon this obstacle attempt
                                    can_place_this = False
                                    break # Exit inner y loop
                                # Add valid cell to the temporary set for this obstacle
                                if is_valid_pos(pos_check, self.grid_width, self.grid_height):
                                    obstacle_cells_this_attempt.add(pos_check)
                        if not can_place_this:
                            break # Exit outer x loop if inner loop failed
                    if not can_place_this:
                        break # Exit clump generation loop if this clump failed

                # 4. If all clumps were valid (didn't hit nest) and generated cells:
                if can_place_this and obstacle_cells_this_attempt:
                    # Check if any proposed cell overlaps with *already placed permanent* obstacles
                    is_clear_of_existing = True
                    for cell in obstacle_cells_this_attempt:
                        # Need to check bounds again before accessing self.obstacles array
                        # Check both out-of-bounds and existing obstacle flag
                        if not (0 <= cell[0] < self.grid_width and 0 <= cell[1] < self.grid_height) or \
                           self.obstacles[cell[0], cell[1]]:
                            is_clear_of_existing = False
                            break

                    # 5. Place the obstacle if the area is clear of existing ones
                    if is_clear_of_existing:
                        for x, y in obstacle_cells_this_attempt:
                            # Bounds check should be redundant here, but for safety:
                            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                                self.obstacles[x, y] = True
                        placed_this_obstacle = True
                        placed_count += 1
                        # Optional detailed log:
                        # print(f"    Placed obstacle area {placed_count} centered near ({cluster_cx},{cluster_cy})")

            # End of attempts for one obstacle

        # --- Final Report ---
        if placed_count < NUM_OBSTACLES:
            print(f"Warning: Placed only {placed_count}/{NUM_OBSTACLES} obstacle areas after {max_obstacle_attempts} attempts.")
        else:
            print(f"Placed {placed_count} obstacle areas successfully.")

    def is_obstacle(self, pos: tuple) -> bool:
        """
        Checks if a given grid cell contains an obstacle.

        Args:
            pos: The (x, y) integer coordinates of the cell to check.

        Returns:
            True if the cell contains an obstacle or is outside the grid
            boundaries, False otherwise. Handles potential errors by treating
            invalid input as an obstacle.
        """
        try:
            x, y = map(int, pos) # Ensure integer coordinates
            # Check bounds first to prevent IndexError
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                # Access the pre-computed boolean obstacle map
                return self.obstacles[x, y]
            else:
                # Treat out-of-bounds positions as obstacles
                return True
        except (IndexError, TypeError, ValueError):
            # Treat errors during coordinate access or conversion as obstacles
            # print(f"Warning: Error checking obstacle at {pos}") # Optional debug
            return True

    def get_pheromone(self, pos: tuple, ph_type: str = "home", food_type: FoodType | None = None) -> float:
        """
        Gets the pheromone strength at a specific grid cell for a given type.

        Args:
            pos: The (x, y) integer coordinates of the cell.
            ph_type: The type of pheromone to retrieve (e.g., "home", "alarm",
                     "food", "negative", "recruitment"). Defaults to "home".
            food_type: Required only if `ph_type` is "food". Specifies which
                       food pheromone (SUGAR or PROTEIN) to get. Defaults to None.

        Returns:
            The pheromone strength as a float. Returns 0.0 if the position is
            invalid, out of bounds, or the pheromone type/food type combination
            is invalid.
        """
        try:
            x, y = map(int, pos) # Ensure integer coordinates
            # Check bounds first to prevent IndexError
            if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                return 0.0  # Out of bounds
        except (ValueError, TypeError, IndexError):
            return 0.0  # Invalid input position

        # Access the corresponding pheromone array based on type
        # Use try-except for the array access itself as a final safety net,
        # although the bounds check above should prevent IndexErrors here.
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
                # Check food_type when ph_type is "food"
                if food_type == FoodType.SUGAR:
                    return self.pheromones_food_sugar[x, y]
                elif food_type == FoodType.PROTEIN:
                    return self.pheromones_food_protein[x, y]
                else:
                    # print(f"Warning: food_type required for ph_type='food' at {pos}") # Optional debug
                    return 0.0 # Invalid food type specified
            else:
                # print(f"Warning: Unknown pheromone type '{ph_type}' requested at {pos}") # Optional debug
                return 0.0  # Unknown pheromone type requested
        except IndexError:
            # This should theoretically not happen if bounds check passed, but safety first.
            # print(f"Warning: Unexpected IndexError accessing pheromone '{ph_type}' at {pos}") # Optional debug
            return 0.0

    def add_pheromone(self, pos: tuple, amount: float, ph_type: str = "home", food_type: FoodType | None = None):
        """
        Adds a specified amount of pheromone to a grid cell, respecting maximum values.

        Pheromones are not added to obstacle cells or invalid positions.
        The amount added is clamped to the maximum allowed for that pheromone type.

        Args:
            pos: The (x, y) integer coordinates of the cell.
            amount: The amount of pheromone strength to add (should be positive).
            ph_type: The type of pheromone to add (e.g., "home", "alarm",
                     "food", "negative", "recruitment"). Defaults to "home".
            food_type: Required only if `ph_type` is "food". Specifies which
                       food pheromone (SUGAR or PROTEIN) to add. Defaults to None.
        """
        # Ignore if the amount to add is non-positive
        if amount <= 0:
            return

        try:
            x, y = map(int, pos) # Ensure integer coordinates
            # --- Critical Checks ---
            # 1. Check bounds
            if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                # print(f"Debug: Add pheromone out of bounds: {pos}") # Optional debug
                return # Out of bounds
            # 2. Check if it's an obstacle
            if self.obstacles[x, y]:
                 # print(f"Debug: Add pheromone on obstacle: {pos}") # Optional debug
                 return # Cannot add pheromone to obstacles
            # --- End Critical Checks ---

        except (ValueError, TypeError, IndexError):
            # print(f"Warning: Invalid position {pos} for add_pheromone.") # Optional debug
            return # Invalid input position or error checking obstacle

        # Determine the target pheromone array and maximum value
        target_array = None
        max_value = PHEROMONE_MAX  # Default max for most types

        if ph_type == "home":
            target_array = self.pheromones_home
        elif ph_type == "alarm":
            target_array = self.pheromones_alarm
        elif ph_type == "negative":
            target_array = self.pheromones_negative
        elif ph_type == "recruitment":
            target_array = self.pheromones_recruitment
            max_value = RECRUITMENT_PHEROMONE_MAX # Use specific max for recruitment
        elif ph_type == "food":
            if food_type == FoodType.SUGAR:
                target_array = self.pheromones_food_sugar
            elif food_type == FoodType.PROTEIN:
                target_array = self.pheromones_food_protein
            else:
                # print(f"Warning: Invalid food_type for add_pheromone ph_type='food' at {pos}") # Optional debug
                return # Invalid food type specified
        else:
            # print(f"Warning: Unknown ph_type '{ph_type}' for add_pheromone at {pos}") # Optional debug
            return # Invalid pheromone type

        # Add the pheromone if a valid target array was identified
        if target_array is not None:
            try:
                # Add amount and clamp to the maximum value for this pheromone type
                current_value = target_array[x, y]
                target_array[x, y] = min(current_value + amount, max_value)
            except IndexError:
                 # Safety check, should have been caught by bounds check above
                 # print(f"Warning: Unexpected IndexError adding pheromone '{ph_type}' at {pos}") # Optional debug
                 pass

    def update_pheromones(self):
        """
        Applies decay and diffusion to all pheromone maps for one simulation tick.

        Handles decay based on predefined constants and diffusion using a
        Gaussian filter, respecting obstacles. Also clamps maximum values
        and removes negligible amounts below a threshold.
        """
        # --- 1. Apply Decay ---
        # Multiply each pheromone map by its corresponding decay factor.
        self.pheromones_home *= PHEROMONE_DECAY
        self.pheromones_alarm *= PHEROMONE_DECAY  # Using common decay for alarm
        self.pheromones_food_sugar *= PHEROMONE_DECAY
        self.pheromones_food_protein *= PHEROMONE_DECAY
        self.pheromones_negative *= NEGATIVE_PHEROMONE_DECAY
        self.pheromones_recruitment *= RECRUITMENT_PHEROMONE_DECAY

        # --- 2. Apply Diffusion (Optimized using SciPy Gaussian Filter) ---
        # Create a mask of where diffusion *can* occur (not on obstacles)
        # `~` inverts the boolean array: True where there is NO obstacle.
        obstacle_mask = ~self.obstacles

        # List of pheromone arrays and their corresponding diffusion sigma values
        # Sigma controls the spread radius of the Gaussian filter.
        arrays_sigmas = [
            (self.pheromones_home, PHEROMONE_DIFFUSION_SIGMA),
            (self.pheromones_food_sugar, PHEROMONE_DIFFUSION_SIGMA),
            (self.pheromones_food_protein, PHEROMONE_DIFFUSION_SIGMA),
            (self.pheromones_alarm, PHEROMONE_DIFFUSION_SIGMA), # Using common sigma
            (self.pheromones_negative, NEGATIVE_PHEROMONE_DIFFUSION_SIGMA),
            (self.pheromones_recruitment, RECRUITMENT_PHEROMONE_DIFFUSION_SIGMA)
        ]

        for arr, sigma in arrays_sigmas:
            if sigma > 0: # Only apply filter if diffusion is enabled (sigma > 0)
                try:
                    # Apply obstacle mask *before* diffusion: Pheromones on obstacles
                    # should neither diffuse nor receive diffusion.
                    arr_masked = arr * obstacle_mask

                    # Apply Gaussian filter for diffusion.
                    # 'constant' mode with cval=0 treats edges as having zero pheromone.
                    diffused = scipy.ndimage.gaussian_filter(
                        arr_masked, sigma=sigma, mode='constant', cval=0.0
                    )

                    # Update the original array with the diffused values.
                    # The mask ensures obstacles remain at 0 (or their decayed value if filter skipped).
                    # arr[:] = diffused * obstacle_mask # Apply mask again ensures obstacles zeroed
                    # Simpler: Directly assign the diffused result, as obstacles were zeroed before filtering.
                    arr[:] = diffused

                except Exception as e:
                    print(f"Error during '{sigma}' sigma diffusion: {e}")
                    # Continue with other pheromones if one fails

        # --- 3. Clamp maximum values and remove negligible amounts ---
        min_pheromone_threshold = 0.01  # Values below this are set to 0 after decay/diffusion

        # List of pheromone arrays and their maximum allowed values
        pheromone_arrays_max = [
            (self.pheromones_home, PHEROMONE_MAX),
            (self.pheromones_food_sugar, PHEROMONE_MAX),
            (self.pheromones_food_protein, PHEROMONE_MAX),
            (self.pheromones_alarm, PHEROMONE_MAX),
            (self.pheromones_negative, PHEROMONE_MAX), # Uses standard max
            (self.pheromones_recruitment, RECRUITMENT_PHEROMONE_MAX) # Special max
        ]

        for arr, max_val in pheromone_arrays_max:
            # Clamp values between 0 and the maximum allowed for this type
            np.clip(arr, 0, max_val, out=arr)
            # Zero out tiny amounts below the threshold for performance/visual cleanliness
            arr[arr < min_pheromone_threshold] = 0

    def replenish_food(self, nest_pos: tuple):
        """
        Periodically adds new, smaller food clusters to the world.

        Called at regular intervals (FOOD_REPLENISH_RATE) during the simulation.
        Places fewer clusters with less food than the initial placement,
        avoiding the immediate nest area and obstacles.

        Args:
            nest_pos: The (x, y) integer coordinates of the nest center.
        """
        nest_pos_int = tuple(map(int, nest_pos))
        min_dist_sq = MIN_FOOD_DIST_FROM_NEST ** 2 # Use squared distance

        # Replenish fewer clusters than initially placed (e.g., one third)
        num_clusters_to_add = max(1, INITIAL_FOOD_CLUSTERS // 3)
        print(f"Replenishing food: Adding {num_clusters_to_add} new clusters...")

        for i in range(num_clusters_to_add):
            # Choose a random food type for replenishment clusters
            food_type_index = rnd(0, NUM_FOOD_TYPES - 1)
            food_type_enum = FoodType(food_type_index)

            attempts = 0
            max_placement_attempts_center = 150
            cx, cy = 0, 0
            found_spot = False

            # --- Try finding a suitable cluster center ---
            # Attempt 1: Far from nest, not on obstacle
            while attempts < max_placement_attempts_center and not found_spot:
                cx = rnd(0, self.grid_width - 1)
                cy = rnd(0, self.grid_height - 1)
                pos_check = (cx, cy)
                if (is_valid_pos(pos_check, self.grid_width, self.grid_height) and
                        not self.obstacles[cx, cy] and
                        distance_sq(pos_check, nest_pos_int) > min_dist_sq):
                    found_spot = True
                attempts += 1

            # Attempt 2 (Fallback): Any non-obstacle spot
            if not found_spot:
                attempts = 0
                max_placement_attempts_fallback = 200
                while attempts < max_placement_attempts_fallback:
                    cx = rnd(0, self.grid_width - 1)
                    cy = rnd(0, self.grid_height - 1)
                    pos_check = (cx, cy)
                    if is_valid_pos(pos_check, self.grid_width, self.grid_height) and not self.obstacles[cx, cy]:
                        found_spot = True
                        break
                    attempts += 1

            # Attempt 3 (Last Resort): Place anywhere
            if not found_spot:
                cx = rnd(0, self.grid_width - 1)
                cy = rnd(0, self.grid_height - 1)

            # --- Distribute food around the chosen center ---
            added_amount = 0.0
            # Replenish roughly half the amount of an initial cluster
            target_food_amount = FOOD_PER_CLUSTER // 2
            max_food_placement_attempts = int(target_food_amount * 2.5)

            for _ in range(max_food_placement_attempts):
                if added_amount >= target_food_amount:
                    break
                # Gaussian spread around center
                try:
                    fx = cx + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                    fy = cy + int(rnd_gauss(0, FOOD_CLUSTER_RADIUS / 2))
                except OverflowError:
                     fx = cx + rnd(-FOOD_CLUSTER_RADIUS, FOOD_CLUSTER_RADIUS)
                     fy = cy + rnd(-FOOD_CLUSTER_RADIUS, FOOD_CLUSTER_RADIUS)

                # Check validity and add smaller amounts during replenishment
                if (0 <= fx < self.grid_width and 0 <= fy < self.grid_height and
                        not self.obstacles[fx, fy]):
                    # Add smaller chunks compared to initial placement
                    amount_to_add = rnd_uniform(0.3, 0.8) * (MAX_FOOD_PER_CELL / 10)
                    current_amount = self.food[fx, fy, food_type_index]
                    new_amount = min(MAX_FOOD_PER_CELL, current_amount + amount_to_add)
                    actual_added = new_amount - current_amount
                    if actual_added > 0:
                        self.food[fx, fy, food_type_index] = new_amount
                        added_amount += actual_added

            print(f"  Replenished cluster {i+1} near ({cx},{cy}) with ~{added_amount:.1f} {food_type_enum.name}.")

# --- Spatial Grid Class ---
class SpatialGrid:
    """
    Divides the game world into a coarser grid for efficient spatial queries.

    Used to quickly find entities (like ants, enemies, prey) near a specific
    location without checking every entity in the simulation. Entities are
    stored in lists associated with the grid cell they occupy. Queries check
    the target cell and its immediate neighbors.

    Note: This grid uses pixel coordinates for its cell calculation, based on
    the main simulation's `CELL_SIZE`.
    """

    def __init__(self, world_width_px: int, world_height_px: int, cell_size_px: int):
        """
        Initializes the spatial grid.

        Args:
            world_width_px: The total width of the simulation world in pixels.
            world_height_px: The total height of the simulation world in pixels.
            cell_size_px: The size (width and height) of each cell in the
                          spatial grid, in pixels. Should typically match the
                          simulation's `CELL_SIZE`.
        """
        # Dimensions of the coarse grid (number of cells)
        # Use max(1, ...) to prevent zero-size cells if world size is smaller than cell size
        self.grid_cols = max(1, world_width_px // cell_size_px)
        self.grid_rows = max(1, world_height_px // cell_size_px)
        self.cell_size = cell_size_px
        # The main data structure: a dictionary where keys are cell coordinates (col, row)
        # and values are lists of entities currently within that cell.
        self.grid: dict[tuple[int, int], list] = {}
        # print(f"SpatialGrid initialized: {self.grid_cols}x{self.grid_rows} cells, size {self.cell_size}px")

    def _get_cell_coords(self, pos_px: tuple) -> tuple[int, int]:
        """
        Calculates the spatial grid cell coordinates (column, row) for a given
        pixel position.

        Args:
            pos_px: The (x, y) pixel coordinates tuple.

        Returns:
            A tuple (column, row) representing the cell coordinates in the
            spatial grid. Returns (-1, -1) if input is invalid.
        """
        try:
            x, y = pos_px
            # Use integer division to find the cell column and row
            # Clamp coordinates to be within the grid bounds, just in case
            # pixel coordinates somehow go slightly out of world bounds.
            cell_col = max(0, min(self.grid_cols - 1, int(x // self.cell_size)))
            cell_row = max(0, min(self.grid_rows - 1, int(y // self.cell_size)))
            return cell_col, cell_row
        except (TypeError, ValueError, IndexError):
            # print(f"Warning: Invalid input for _get_cell_coords: {pos_px}") # Optional debug
            return -1, -1  # Return invalid coordinates on error

    def add_entity(self, entity):
        """
        Adds an entity to the spatial grid based on its current position.

        Args:
            entity: The simulation entity object (must have a `pos` attribute
                    containing pixel coordinates).
        """
        # Get the cell coordinates for the entity's position
        cell_coords = self._get_cell_coords(entity.pos)
        if cell_coords == (-1, -1): return  # Skip if coords are invalid

        # If this cell coordinate is not yet in the grid dictionary, create a new list for it
        if cell_coords not in self.grid:
            self.grid[cell_coords] = []
        # Append the entity to the list associated with its cell
        # Avoid adding duplicates if logic error elsewhere causes double add
        if entity not in self.grid[cell_coords]:
            self.grid[cell_coords].append(entity)

    def remove_entity(self, entity):
        """
        Removes an entity from the spatial grid.

        Args:
            entity: The simulation entity object to remove.
        """
        # Get the cell coordinates for the entity's *current* position
        cell_coords = self._get_cell_coords(entity.pos)
        if cell_coords == (-1, -1): return  # Skip if coords are invalid

        # Check if the cell exists in the grid and the entity is in that cell's list
        if cell_coords in self.grid and entity in self.grid[cell_coords]:
            try:
                self.grid[cell_coords].remove(entity)
                # Optional: If the cell's list becomes empty, remove the key from the grid dictionary
                # This might save some memory but adds overhead. Keep it simple for now.
                # if not self.grid[cell_coords]:
                #     del self.grid[cell_coords]
            except ValueError:
                # Should not happen if `entity in list` check passed, but safety first
                # print(f"Warning: Entity {entity} not found in cell {cell_coords} list during removal.") # Optional debug
                pass

    def update_entity_position(self, entity, old_pos_px: tuple):
        """
        Updates an entity's position within the spatial grid.

        Moves the entity from its old cell to its new cell if its position
        change resulted in crossing a cell boundary.

        Args:
            entity: The entity that moved (must have updated `pos` attribute).
            old_pos_px: The entity's previous pixel coordinates tuple (x_old, y_old).
        """
        # Calculate the cell coordinates for both the old and new positions
        old_cell_coords = self._get_cell_coords(old_pos_px)
        new_cell_coords = self._get_cell_coords(entity.pos)

        # Only perform updates if the entity actually moved to a different cell
        if old_cell_coords != new_cell_coords:
            # Remove from the old cell's list (if it was correctly registered there)
            if old_cell_coords != (-1, -1) and old_cell_coords in self.grid and entity in self.grid[
                old_cell_coords]:
                try:
                    self.grid[old_cell_coords].remove(entity)
                except ValueError:
                    pass  # Ignore if already removed somehow

            # Add to the new cell's list
            if new_cell_coords != (-1, -1):
                if new_cell_coords not in self.grid:
                    self.grid[new_cell_coords] = []
                # Avoid adding duplicates
                if entity not in self.grid[new_cell_coords]:
                    self.grid[new_cell_coords].append(entity)

    def get_nearby_entities(self, pos_px: tuple, radius_px: int, entity_type=None) -> list:
        """
        Gets entities near a given pixel position within a specified radius.

        Optimized to check only the necessary spatial grid cells that overlap
        with the circular search area.

        Args:
            pos_px: The center pixel position (x, y) of the search area.
            radius_px: The radius of the search area in pixels.
            entity_type: Optional. If specified, only returns entities of this
                         type (e.g., Ant, Enemy). Defaults to None (returns all types).

        Returns:
            A list of entity objects found within the specified radius.
            The list might contain duplicates if an entity spans multiple cells (unlikely with current setup).
        """
        nearby_entities = []
        center_x, center_y = pos_px

        # Calculate the range of spatial grid cells that the search radius could possibly overlap
        min_col = max(0, int((center_x - radius_px) // self.cell_size))
        max_col = min(self.grid_cols - 1, int((center_x + radius_px) // self.cell_size))
        min_row = max(0, int((center_y - radius_px) // self.cell_size))
        max_row = min(self.grid_rows - 1, int((center_y + radius_px) // self.cell_size))

        # Iterate through the potentially relevant cells
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                check_cell = (col, row)
                # Check if this cell exists in the grid and has entities
                if check_cell in self.grid:
                    # Iterate through entities within this potentially relevant cell
                    # Use list() to avoid issues if the list is modified elsewhere during iteration (safer)
                    for entity in list(self.grid[check_cell]):
                        # Optional type filtering
                        if entity_type is None or isinstance(entity, entity_type):
                            # Final precise check: Calculate actual distance from the entity
                            # to the search center and compare with the radius.
                            # We assume entity.pos holds pixel coordinates.
                            entity_pos_px = entity.pos
                            if distance_sq(pos_px, entity_pos_px) <= radius_px ** 2:
                                nearby_entities.append(entity)

        return nearby_entities

# --- Prey Class ---
class Prey:
    """
    Represents a small, passive creature that ants can hunt for protein.
    Moves randomly but attempts to flee when ants get too close.
    """

    def __init__(self, pos_grid: tuple, sim):
        """
        Initializes a new prey item.

        Args:
            pos_grid: The initial (x, y) grid coordinates.
            sim: Reference to the main AntSimulation object.
        """
        # --- Core Attributes ---
        self.pos = tuple(map(int, pos_grid)) # Grid coordinates (x, y)
        self.simulation = sim
        self.hp = float(PREY_HP)
        self.max_hp = float(PREY_HP)
        self.color = PREY_COLOR

        # --- Movement ---
        # Timer determining when the next move attempt occurs.
        # Initialized randomly to stagger prey movement.
        self.move_delay_timer = rnd_uniform(0, PREY_MOVE_DELAY)

        # --- State/Interaction ---
        self.is_dying = False # Flag to prevent multiple death processing

    def update(self):
        """
        Updates the prey's state for one simulation tick.

        Handles fleeing behavior from nearby ants and movement.
        """
        sim = self.simulation

        # If already marked for death, do nothing further
        if self.is_dying:
            return

        # --- Fleeing Behavior ---
        pos_int = self.pos
        flee_target_pos = None # Position of the nearest ant causing fleeing
        min_dist_sq_found = PREY_FLEE_RADIUS_SQ # Use configured flee radius

        # Define the search area around the prey based on flee radius
        # Convert grid radius to pixel radius for spatial grid query
        search_radius_px = (PREY_FLEE_RADIUS_SQ ** 0.5) * sim.cell_size
        center_px = (pos_int[0] * sim.cell_size + sim.cell_size // 2,
                     pos_int[1] * sim.cell_size + sim.cell_size // 2)

        # Use spatial grid to find nearby ants efficiently
        nearby_ants = sim.spatial_grid.get_nearby_entities(center_px, search_radius_px, Ant)

        for ant in nearby_ants:
            # Check if ant is alive and calculate distance
            if ant.hp > 0:
                d_sq = distance_sq(pos_int, ant.pos) # Distance in grid coordinates
                if d_sq < min_dist_sq_found:
                    min_dist_sq_found = d_sq
                    flee_target_pos = ant.pos # Store grid position of the ant

        # --- Movement Logic ---
        self.move_delay_timer -= 1 # Decrement timer by 1 each tick
        if self.move_delay_timer > 0:
            return # Not time to move yet

        # Reset timer for the next move attempt
        self.move_delay_timer += PREY_MOVE_DELAY

        # Get potential neighboring grid cells
        possible_moves = get_neighbors(pos_int, sim.grid_width, sim.grid_height)

        # Filter valid moves: not obstacle, not occupied by enemy/ant/other prey
        valid_moves = [
            m for m in possible_moves if
            not sim.grid.is_obstacle(m) and
            not sim.is_enemy_at(m) and
            not sim.is_ant_at(m) and
            not sim.is_prey_at(m, exclude_self=self) # Check for other prey
        ]

        if not valid_moves:
            return # No valid place to move

        chosen_move = None
        # --- Choose Move: Flee or Random ---
        if flee_target_pos:
            # Currently fleeing from an ant
            flee_dx = pos_int[0] - flee_target_pos[0]
            flee_dy = pos_int[1] - flee_target_pos[1]
            best_flee_move = None
            max_flee_score = -float("inf")

            # Score potential moves based on direction away from the flee target
            for move in valid_moves:
                move_dx = move[0] - pos_int[0]
                move_dy = move[1] - pos_int[1]

                # Normalize flee direction vector (approximation)
                dist_approx = max(1, abs(flee_dx) + abs(flee_dy))
                norm_flee_dx = flee_dx / dist_approx
                norm_flee_dy = flee_dy / dist_approx

                # Score = Alignment with flee direction + small bonus for increasing distance
                alignment_score = move_dx * norm_flee_dx + move_dy * norm_flee_dy
                # Add slight preference for moves that increase distance
                distance_score = distance_sq(move, flee_target_pos) * 0.05

                score = alignment_score + distance_score

                if score > max_flee_score:
                    max_flee_score = score
                    best_flee_move = move

            # Choose the best flee move, or a random valid one if no clear best direction
            chosen_move = best_flee_move if best_flee_move else random.choice(valid_moves)
        else:
            # Not fleeing, choose a random valid move
            chosen_move = random.choice(valid_moves)

        # --- Execute the move ---
        if chosen_move and chosen_move != self.pos:
            old_pos = self.pos
            self.pos = chosen_move
            # IMPORTANT: Update the simulation's position tracking and spatial grid
            sim.update_entity_position(self, old_pos, self.pos)

    def take_damage(self, amount: float, attacker):
        """
        Reduces the prey's HP when attacked. Marks prey for removal if HP drops to zero.

        Args:
            amount: The amount of damage to inflict.
            attacker: The entity that dealt the damage (used for potential future logic).
        """
        # Ignore damage if already dying or dead
        if self.is_dying or self.hp <= 0:
            return

        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            self.is_dying = True # Mark for removal in the main simulation loop
            # print(f"Prey marked as dying at {self.pos}") # Optional debug

    def draw(self, surface: pygame.Surface):
        """
        Draws the prey onto the specified Pygame surface as a beetle-like shape.

        Args:
            surface: The Pygame surface to draw on.
        """
        sim = self.simulation
        # Basic validation
        if not is_valid_pos(self.pos, sim.grid_width, sim.grid_height):
            return

        cs = sim.cell_size
        # Calculate center pixel position for drawing
        pos_px = (int(self.pos[0] * cs + cs / 2),
                  int(self.pos[1] * cs + cs / 2))

        # --- Body (Oval/Ellipse) ---
        body_width = max(3, int(cs / 1.8))
        body_height = max(2, int(cs / 2.4))
        # Create rect centered at pos_px
        body_rect = pygame.Rect(pos_px[0] - body_width // 2,
                                pos_px[1] - body_height // 2,
                                body_width, body_height)
        pygame.draw.ellipse(surface, self.color, body_rect)

        # --- Shell Line (Elytra Division) ---
        line_color = tuple(max(0, c - 40) for c in self.color) # Darker shade
        line_start = (pos_px[0], body_rect.top)
        line_end = (pos_px[0], body_rect.bottom)
        pygame.draw.line(surface, line_color, line_start, line_end, 1) # 1px line

        # --- Optional: Tiny Antennae ---
        antenna_length = body_width * 0.3
        antenna_angle = 0.3 # Radians offset from front-center
        # Antennae originate slightly back from the front of the body ellipse
        antenna_origin_x = pos_px[0]
        antenna_origin_y = body_rect.top + body_height * 0.2

        # Left Antenna (pointing upwards-left relative to body orientation)
        angle_l = math.pi * 1.5 - antenna_angle # Angle assumes body points 'up'
        end_lx = antenna_origin_x + antenna_length * math.cos(angle_l)
        end_ly = antenna_origin_y + antenna_length * math.sin(angle_l)
        pygame.draw.line(surface, line_color, (antenna_origin_x, antenna_origin_y), (int(end_lx), int(end_ly)), 1)

        # Right Antenna (pointing upwards-right)
        angle_r = math.pi * 1.5 + antenna_angle
        end_rx = antenna_origin_x + antenna_length * math.cos(angle_r)
        end_ry = antenna_origin_y + antenna_length * math.sin(angle_r)
        pygame.draw.line(surface, line_color, (antenna_origin_x, antenna_origin_y), (int(end_rx), int(end_ry)), 1)

        # --- Outline the Body ---
        pygame.draw.ellipse(surface, (0, 0, 0), body_rect, 1) # Black 1px outline

# --- Ant Class ---
class Ant:
    """
    Represents an individual ant in the simulation.
    Manages its own state, movement, interactions, pheromone dropping, and basic needs.
    """

    def __init__(self, pos_grid: tuple, simulation, caste: AntCaste):
        """
        Initializes a new ant.

        Args:
            pos_grid: The initial (x, y) grid coordinates.
            simulation: Reference to the main AntSimulation object.
            caste: The caste of the ant (WORKER or SOLDIER).
        """
        self.pos = tuple(map(int, pos_grid))
        self.simulation = simulation
        self.caste = caste
        attrs = ANT_ATTRIBUTES[caste] # Get attributes for this caste

        # --- Core Attributes from Caste ---
        self.hp = float(attrs["hp"])
        self.max_hp = float(attrs["hp"])
        self.attack_power = attrs["attack"]
        self.max_capacity = attrs["capacity"]
        # <<< KORRIGIERTE ZEILE: Verwende den neuen Schlüsselnamen >>>
        self.move_cooldown_base = attrs["move_cooldown_base"]
        # <<< ENDE KORRIGIERTE ZEILE >>>
        self.search_color = attrs["color"]
        self.return_color = attrs["return_color"]
        self.food_consumption_sugar = attrs["food_consumption_sugar"]
        self.food_consumption_protein = attrs["food_consumption_protein"]
        self.size_factor = attrs["size_factor"]
        self.head_size_factor = attrs["head_size_factor"]

        # --- State Variables ---
        self.state = AntState.SEARCHING
        self.carry_amount = 0.0
        self.carry_type: FoodType | None = None
        self.age = 0
        self.max_age_ticks = int(rnd_gauss(WORKER_MAX_AGE_MEAN, WORKER_MAX_AGE_STDDEV))
        self.is_dying = False

        # --- Movement and Pathfinding ---
        self.path_history: list[tuple[int, int]] = []
        self.history_timestamps: list[int] = []
        self.move_cooldown_timer = 0 # Start ready to move
        self.last_move_direction = (0, 1) # Default facing 'down'
        self.stuck_timer = 0
        self.escape_timer = 0

        # --- Status/Interaction ---
        self.last_move_info = "Born"
        self.just_picked_food = False
        self.visible_enemies: list[Enemy] = []

        # --- Timers ---
        self.food_consumption_timer = rnd(0, WORKER_FOOD_CONSUMPTION_INTERVAL)

        # --- Targeting/State Specific ---
        self.last_known_alarm_pos: tuple | None = None
        self.target_prey: Prey | None = None

        # --- Simulation Integration ---
        self.index = -1 # Assigned by simulation.add_ant

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
        head_size = int(ant_size * self.head_size_factor)  # Changed: Use head_size_factor
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
        """
        Checks environmental conditions and internal status to potentially
        change the ant's current behavioral state (AntState).

        This method handles transitions like:
        - Workers switching to HUNTING if needed and prey is nearby.
        - Soldiers switching between PATROLLING, DEFENDING, HUNTING, and SEARCHING
          based on threats (pheromones, visible enemies), proximity to the nest,
          and target availability.
        - Ants finishing ESCAPING state.
        """
        sim = self.simulation
        pos_int = self.pos
        nest_pos_int = sim.nest_pos  # Use dynamic nest position from simulation

        # --- Handle ESCAPING state timeout ---
        if self.state == AntState.ESCAPING:
            self.escape_timer -= 1 # Decrement timer by 1 tick
            if self.escape_timer <= 0:
                # Escape finished, revert to default state based on caste
                next_state = AntState.PATROLLING if self.caste == AntCaste.SOLDIER else AntState.SEARCHING
                self._switch_state(next_state, "EscapeEnd")
                return # State changed, exit early

        # --- Worker: Opportunity Hunting ---
        # If searching, needs protein, has no food/target, check for nearby prey.
        if (self.caste == AntCaste.WORKER and
                self.state == AntState.SEARCHING and
                self.carry_amount == 0 and not self.target_prey and
                # Check if protein is relatively low (e.g., below 2.5x critical threshold)
                sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * 2.5):

            # Check for nearby prey using spatial grid
            # Search radius slightly larger than prey's flee radius
            search_radius_px = (PREY_FLEE_RADIUS_SQ**0.5 + 2) * sim.cell_size
            center_px = (pos_int[0] * sim.cell_size + sim.cell_size / 2,
                         pos_int[1] * sim.cell_size + sim.cell_size / 2)
            nearby_prey = sim.spatial_grid.get_nearby_entities(center_px, search_radius_px, Prey)

            # Filter for living prey only
            living_nearby_prey = [p for p in nearby_prey if p.hp > 0]

            if living_nearby_prey:
                # Sort by distance (grid coordinates) to target the closest one
                living_nearby_prey.sort(key=lambda p: distance_sq(pos_int, p.pos))
                self.target_prey = living_nearby_prey[0]
                self._switch_state(AntState.HUNTING, f"HuntOpp@{self.target_prey.pos}")
                return # State changed

        # --- Soldier: State Management (Patrol/Defend/Hunt/Search) ---
        if self.caste == AntCaste.SOLDIER:
            # Don't override critical states like returning or escaping
            if self.state in [AntState.ESCAPING, AntState.RETURNING_TO_NEST]:
                return # Skip soldier logic if in these states

            grid = sim.grid
            # Check local threat level (pheromones and visible enemies)
            # 1. Pheromone check in a radius
            max_alarm = 0.0
            max_recruit = 0.0
            pheromone_check_radius_sq = 10 * 10 # Check 10 cells radius
            x0, y0 = pos_int
            min_scan_x = max(0, x0 - 10)
            max_scan_x = min(sim.grid_width - 1, x0 + 10)
            min_scan_y = max(0, y0 - 10)
            max_scan_y = min(sim.grid_height - 1, y0 + 10)

            for i in range(min_scan_x, max_scan_x + 1):
                for j in range(min_scan_y, max_scan_y + 1):
                    p_int = (i, j)
                    if distance_sq(pos_int, p_int) <= pheromone_check_radius_sq:
                        max_alarm = max(max_alarm, grid.get_pheromone(p_int, "alarm"))
                        max_recruit = max(max_recruit, grid.get_pheromone(p_int, "recruitment"))

            # Combine pheromone signals (weighted)
            pheromone_threat_signal = max_alarm + max_recruit * 0.6

            # 2. Check for directly visible enemies (updated in main ant update)
            has_visible_enemy = bool(self.visible_enemies)

            # --- Soldier State Decision Logic ---
            is_near_nest = distance_sq(pos_int, nest_pos_int) <= sim.soldier_patrol_radius_sq
            # Threshold for switching to DEFEND based on pheromones
            pheromone_defend_threshold = SOLDIER_DEFEND_ALARM_THRESHOLD * 0.7 # Slightly lower threshold

            # Priority 1: Switch to DEFENDING if high pheromone threat or visible enemy
            if has_visible_enemy or pheromone_threat_signal > pheromone_defend_threshold:
                if self.state != AntState.DEFENDING:
                    reason = "VisEnemy!" if has_visible_enemy else f"PherThreat({pheromone_threat_signal:.0f})!"
                    self._switch_state(AntState.DEFENDING, reason)
                    return # State changed

            # Priority 2: Switch to HUNTING if prey is nearby (and not defending/hunting)
            elif self.state not in [AntState.DEFENDING, AntState.HUNTING]:
                # Reuse worker search logic, but soldiers hunt more readily
                search_radius_px = (PREY_FLEE_RADIUS_SQ**0.5 + 2) * sim.cell_size
                center_px = (pos_int[0] * sim.cell_size + sim.cell_size / 2,
                             pos_int[1] * sim.cell_size + sim.cell_size / 2)
                nearby_prey = sim.spatial_grid.get_nearby_entities(center_px, search_radius_px, Prey)
                living_nearby_prey = [p for p in nearby_prey if p.hp > 0]

                if living_nearby_prey:
                    living_nearby_prey.sort(key=lambda p: distance_sq(pos_int, p.pos))
                    self.target_prey = living_nearby_prey[0]
                    self._switch_state(AntState.HUNTING, f"SHuntOpp@{self.target_prey.pos}")
                    return # State changed

            # Priority 3: Revert from DEFENDING if threat is low
            elif self.state == AntState.DEFENDING:
                # Condition to stop defending: no visible enemy AND pheromone signal below threshold
                if not has_visible_enemy and pheromone_threat_signal < pheromone_defend_threshold * 0.8: # Hysteresis
                    next_state = AntState.PATROLLING if is_near_nest else AntState.SEARCHING
                    self._switch_state(next_state, f"DefEnd({pheromone_threat_signal:.0f})")
                    return # State changed

            # Priority 4: Maintain PATROLLING/SEARCHING based on location
            elif is_near_nest and self.state != AntState.PATROLLING:
                # If near nest but not patrolling (and not defending/hunting), start patrolling
                self._switch_state(AntState.PATROLLING, "NearNest->Patrol")
            elif not is_near_nest and self.state == AntState.PATROLLING:
                # If patrolled too far from nest, switch to general searching
                self._switch_state(AntState.SEARCHING, "PatrolFar->Search")
            # Implicitly: If near nest and already patrolling, state remains PATROLLING.
            # Implicitly: If far from nest and already searching, state remains SEARCHING.

        # --- Add other state transition logic as needed ---
        # (E.g., Tending Brood - currently not implemented)

    def _update_path_history(self, new_pos_int: tuple):
        """
        Adds the ant's new position to its path history and removes old entries.

        History is pruned based on a time duration (`WORKER_PATH_HISTORY_LENGTH`)
        rather than just a fixed number of steps.

        Args:
            new_pos_int: The new (x, y) grid coordinate tuple to add.
        """
        current_sim_ticks = int(self.simulation.ticks)

        # Only add if it's a different position from the last recorded one
        if not self.path_history or self.path_history[-1] != new_pos_int:
            self.path_history.append(new_pos_int)
            self.history_timestamps.append(current_sim_ticks)

            # Prune old history based on time duration
            # WORKER_PATH_HISTORY_LENGTH defines how many ticks back to remember
            cutoff_time = current_sim_ticks - WORKER_PATH_HISTORY_LENGTH
            cutoff_index = 0
            # Find the index of the first entry that is *not* older than the cutoff time
            while (cutoff_index < len(self.history_timestamps) and
                   self.history_timestamps[cutoff_index] < cutoff_time):
                cutoff_index += 1

            # Keep only the recent part of the history (from cutoff_index onwards)
            # Slicing creates new lists, which is generally fine here
            if cutoff_index > 0:
                self.path_history = self.path_history[cutoff_index:]
                self.history_timestamps = self.history_timestamps[cutoff_index:]

    def _is_in_history(self, pos_int: tuple) -> bool:
        """
        Checks if a given position is present in the ant's recent path history.

        Used primarily for move scoring to discourage revisiting recently
        explored locations.

        Args:
            pos_int: The (x, y) grid coordinate tuple to check.

        Returns:
            True if the position exists in `self.path_history`, False otherwise.
        """
        # Simple check for membership in the list of recent positions
        return pos_int in self.path_history

    def _clear_path_history(self):
        """
        Clears the ant's path history and associated timestamps.

        Typically called when the ant undergoes a major state change where
        remembering the immediate past path is no longer relevant (e.g.,
        switching from searching to returning).
        """
        self.path_history.clear()
        self.history_timestamps.clear()

    def _filter_valid_moves(self, potential_neighbors_int: list[tuple],
                             ignore_history_near_nest: bool = False) -> list[tuple]:
        """
        Filters a list of potential neighboring moves based on validity criteria.

        Checks for:
        - Grid obstacles.
        - Presence of the Queen or other ants (dynamic obstacles).
        - Optionally ignores path history avoidance when near the nest (controlled by flag).

        Args:
            potential_neighbors_int: A list of potential (x, y) grid coordinate tuples to check.
            ignore_history_near_nest: If True, path history avoidance is disabled
                                      when the ant is currently close to the nest.
                                      Useful for preventing ants getting stuck when returning.
                                      Defaults to False.

        Returns:
            A list containing only the valid (x, y) grid coordinate tuples from the input list.
        """
        sim = self.simulation
        valid_moves_int = [] # List to store the valid moves found
        q_pos_int = sim.queen.pos if sim.queen else None # Get queen's position if she exists
        ant_current_pos_int = self.pos
        nest_pos_int = sim.nest_pos # Get nest center position

        # Determine if the ant is currently near the nest center
        # Use a slightly larger radius than the nest itself for the check
        is_near_nest_now = distance_sq(ant_current_pos_int, nest_pos_int) <= (NEST_RADIUS + 2)**2

        # Decide whether to actively avoid path history for this filtering step
        # Avoid history unless the flag is set AND the ant is currently near the nest.
        check_history_flag = not (ignore_history_near_nest and is_near_nest_now)

        # Iterate through each potential neighboring position
        for n_pos_int in potential_neighbors_int:
            # --- Check Blocking Conditions ---

            # 1. Path History Avoidance (if enabled)
            history_block = check_history_flag and self._is_in_history(n_pos_int)
            if history_block:
                continue # Skip this neighbor if blocked by history

            # 2. Queen's Position
            is_queen_pos = (n_pos_int == q_pos_int)
            if is_queen_pos:
                continue # Skip this neighbor if it's the queen's spot

            # 3. Grid Obstacle
            # Use the grid's method which also handles out-of-bounds checks
            is_obstacle_pos = sim.grid.is_obstacle(n_pos_int)
            if is_obstacle_pos:
                continue # Skip this neighbor if it's a permanent obstacle

            # 4. Other Ants
            # Check if another ant (but not itself) occupies the position
            is_ant_pos = sim.is_ant_at(n_pos_int, exclude_self=self)
            if is_ant_pos:
                continue # Skip this neighbor if blocked by another ant

            # --- If no blocking conditions met, the move is valid ---
            valid_moves_int.append(n_pos_int)

        return valid_moves_int

    def _choose_move(self) -> tuple | None:
        """
        Determines the best neighboring cell for the ant to move into.

        This involves:
        1. Getting potential neighboring cells.
        2. Filtering out invalid moves (obstacles, other ants, sometimes history).
        3. Handling cases where the ant is blocked.
        4. Scoring the remaining valid moves based on the ant's current state and goals.
        5. Selecting the final move, either deterministically (best score) or
           probabilistically (weighted random choice based on score).

        Returns:
            The chosen (x, y) grid coordinate tuple for the next move,
            or None if no valid move is possible.
        """
        sim = self.simulation
        current_pos_int = self.pos

        # 1. Get potential neighbors (8 adjacent cells)
        potential_neighbors_int = get_neighbors(current_pos_int, sim.grid_width, sim.grid_height)
        if not potential_neighbors_int:
            self.last_move_info = "No neighbors" # Should only happen if grid is 1x1
            return None # Cannot move

        # 2. Filter valid primary moves (avoid obstacles, ants, queen, and usually history)
        # Allow moving into history when returning near nest to avoid getting stuck.
        ignore_hist_near_nest = (self.state == AntState.RETURNING_TO_NEST)
        valid_neighbors_int = self._filter_valid_moves(potential_neighbors_int, ignore_hist_near_nest)

        # 3. Handle blocked situations
        chosen_move_int = None
        if not valid_neighbors_int:
            # If completely blocked by dynamic obstacles (ants) or history:
            # Try a fallback: Allow moving into history, but still avoid obstacles/queen/other ants.
            # Essentially, re-filter potential neighbors ignoring the history check.
            self.last_move_info = "Blocked; Fallback"
            fallback_neighbors_int = self._filter_valid_moves(potential_neighbors_int, ignore_history_near_nest=True) # Force ignore history

            # If fallback options exist, choose one (e.g., randomly or based on oldest history)
            if fallback_neighbors_int:
                 # Optional: Prioritize fallback moves that are *older* in history.
                 # fallback_neighbors_int.sort(key=lambda p: self.path_history.index(p) if p in self.path_history else -1)
                 # return fallback_neighbors_int[0] # Choose oldest visited

                 # Simpler fallback: Random choice among allowed fallback moves
                 chosen_move_int = random.choice(fallback_neighbors_int)
                 self.last_move_info += "->RandFallback"
                 return chosen_move_int # Return the fallback move directly
            else:
                 # Truly blocked by permanent obstacles or queen/ants even in fallback
                 self.last_move_info = "Truly Blocked"
                 return None # Cannot move

        # --- If primary valid moves exist, proceed with scoring ---

        # 4. Score valid moves based on current state
        # ESCAPING state has special logic (prioritize unvisited cells)
        if self.state == AntState.ESCAPING:
            # Filter valid moves to only those NOT in history
            escape_moves_int = [p for p in valid_neighbors_int if not self._is_in_history(p)]
            if escape_moves_int:
                self.last_move_info = "Esc->Unhist"
                chosen_move_int = random.choice(escape_moves_int)
            else:
                # If all valid escape moves are in history, pick a random valid one
                self.last_move_info = "Esc->Hist"
                chosen_move_int = random.choice(valid_neighbors_int)
            return chosen_move_int # Return escape move directly

        # For other states, use state-specific scoring functions
        scoring_functions = {
            AntState.RETURNING_TO_NEST: self._score_moves_returning,
            AntState.SEARCHING: self._score_moves_searching,
            AntState.PATROLLING: self._score_moves_patrolling,
            AntState.DEFENDING: self._score_moves_defending,
            AntState.HUNTING: self._score_moves_hunting,
            # Add other states like TENDING_BROOD if implemented
        }
        # Default to searching score if state is unknown (shouldn't happen)
        score_func = scoring_functions.get(self.state, self._score_moves_searching)

        # Calculate scores for all valid moves
        # <<< KORRIGIERTE ZEILE: Argument self.just_picked_food entfernt >>>
        move_scores = score_func(valid_neighbors_int)
        # <<< ENDE KORRIGIERTE ZEILE >>>


        # If scoring resulted in no scores (e.g., error or no moves scorable):
        if not move_scores:
            self.last_move_info = f"No scores({self.state.name})"
            # Fallback: choose randomly from the valid (but unscored) neighbours
            return random.choice(valid_neighbors_int)

        # 5. Select the best move based on scores and state logic
        if self.state == AntState.RETURNING_TO_NEST:
            # Returning uses a specific selection logic (prioritizes getting closer)
            chosen_move_int = self._select_best_move_returning(move_scores, valid_neighbors_int)
        elif self.state in [AntState.DEFENDING, AntState.HUNTING]:
            # Urgent states: Pick the move with the absolute highest score (deterministic)
            chosen_move_int = self._select_best_move(move_scores, valid_neighbors_int)
        else:  # SEARCHING, PATROLLING (and any others using probabilistic)
            # Use probabilistic choice to allow exploration and avoid local optima
            chosen_move_int = self._select_probabilistic_move(move_scores, valid_neighbors_int)

        # Final fallback if selection somehow failed (should be rare)
        if chosen_move_int is None:
            self.last_move_info += "(SelectFailed!)"
            chosen_move_int = random.choice(valid_neighbors_int)

        return chosen_move_int

    # --- Scoring Methods for Different States ---

    def _score_moves_searching(self, valid_neighbors_int: list[tuple]) -> dict[tuple, float]:
        """
        Scores potential moves for an ant in the SEARCHING state.

        Prioritizes exploration and finding food based on colony needs.
        Adapts weights for food pheromones depending on whether sugar or
        protein is critically low. Also considers recruitment signals and
        avoids the nest area, negative pheromones, and alarm signals.

        Args:
            valid_neighbors_int: A list of valid neighboring (x, y) grid tuples.

        Returns:
            A dictionary where keys are the valid neighbor tuples and values
            are their calculated scores (float).
        """
        scores = {}
        sim = self.simulation
        grid = sim.grid
        nest_pos_int = sim.nest_pos

        # --- Determine Dynamic Food Pheromone Weights based on Colony Needs ---
        sugar_critically_needed = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD
        protein_critically_needed = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD

        # Default weights (low interest if nothing is critical)
        w_sugar = W_FOOD_PHEROMONE_SEARCH_LOW_NEED
        w_protein = W_FOOD_PHEROMONE_SEARCH_LOW_NEED
        current_search_focus = "LowNeed" # Debug info

        if sugar_critically_needed and not protein_critically_needed:
            # Only Sugar critical: Strongly attract to sugar, potentially avoid protein
            w_sugar = W_FOOD_PHEROMONE_SEARCH_CRITICAL_NEED
            w_protein = W_FOOD_PHEROMONE_SEARCH_CRITICAL_AVOID # Avoid/ignore protein trails
            current_search_focus = "Crit(S)"
        elif protein_critically_needed and not sugar_critically_needed:
            # Only Protein critical: Strongly attract to protein, potentially avoid sugar
            w_protein = W_FOOD_PHEROMONE_SEARCH_CRITICAL_NEED
            w_sugar = W_FOOD_PHEROMONE_SEARCH_CRITICAL_AVOID # Avoid/ignore sugar trails
            current_search_focus = "Crit(P)"
        elif sugar_critically_needed and protein_critically_needed:
            # Both critical: Use base weights, possibly slightly favoring the relatively scarcer one
            w_sugar = W_FOOD_PHEROMONE_SEARCH_BASE
            w_protein = W_FOOD_PHEROMONE_SEARCH_BASE
            # Optional slight bias towards the more critical resource
            if sim.colony_food_storage_sugar <= sim.colony_food_storage_protein:
                w_sugar *= 1.1
                w_protein *= 0.9
            else:
                w_protein *= 1.1
                w_sugar *= 0.9
            current_search_focus = "Crit(Both)"
        # else: # Neither critical, use default LowNeed weights assigned above

        # Reduce food pheromone interest for soldiers
        if self.caste == AntCaste.SOLDIER:
            w_sugar *= 0.1   # Soldiers are much less interested in food trails
            w_protein *= 0.1

        # --- Nest Avoidance Radius ---
        # Define a squared radius around the nest to discourage searching within
        nest_search_avoid_radius_sq = (NEST_RADIUS * 2.5)**2

        # --- Score Each Valid Neighbor ---
        for n_pos_int in valid_neighbors_int:
            # Start with the base score (persistence, noise)
            score = self._score_moves_base(n_pos_int)

            # --- Pheromone Influence ---
            home_ph = grid.get_pheromone(n_pos_int, "home")
            sugar_ph = grid.get_pheromone(n_pos_int, "food", FoodType.SUGAR)
            protein_ph = grid.get_pheromone(n_pos_int, "food", FoodType.PROTEIN)
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            # Apply dynamically calculated weights for food pheromones
            score += sugar_ph * w_sugar
            score += protein_ph * w_protein

            # Recruitment pheromone (strong signal for important events)
            # Soldiers might react slightly more strongly
            recruit_w = W_RECRUITMENT_PHEROMONE * (1.2 if self.caste == AntCaste.SOLDIER else 1.0)
            score += recr_ph * recruit_w

            # Avoidance pheromones
            score += neg_ph * W_NEGATIVE_PHEROMONE # Avoid explored/empty areas
            score += alarm_ph * W_ALARM_PHEROMONE   # Avoid danger zones

            # Home pheromone influence (usually low/zero when searching outwards)
            score += home_ph * W_HOME_PHEROMONE_SEARCH

            # --- Nest Avoidance Penalty ---
            # Apply penalty for moving towards/staying near the nest while searching
            if distance_sq(n_pos_int, nest_pos_int) <= nest_search_avoid_radius_sq:
                score += W_AVOID_NEST_SEARCHING # Negative weight acts as penalty

            # Store the final score for this potential move
            scores[n_pos_int] = score

        # Update last move info for debugging purposes (reflects the search focus)
        # This might overwrite reasons from _switch_state, consider appending or conditional setting.
        # For now, let's keep it simple:
        # self.last_move_info = f"Search({current_search_focus})"

        return scores

    def _score_moves_base(self, neighbor_pos_int: tuple) -> float:
        """
        Calculates base score components common to most movement decisions.

        Includes:
        - Persistence bonus: Encourages continuing in the same direction.
        - Random noise: Adds variability to break ties and aid exploration.
        - (Optionally) History penalty: Although primarily handled by filtering,
          a small scoring penalty could also be applied here if needed.

        Args:
            neighbor_pos_int: The potential next move (x, y) tuple being scored.

        Returns:
            The base score component for the move as a float.
        """
        score = 0.0
        current_pos_int = self.pos

        # 1. Persistence Bonus: Add score if moving in the same direction as the last move.
        move_dx = neighbor_pos_int[0] - current_pos_int[0]
        move_dy = neighbor_pos_int[1] - current_pos_int[1]
        move_dir = (move_dx, move_dy)
        # Check if the calculated move direction matches the last recorded one
        # and ensure the ant actually moved last time (direction is not (0,0))
        if move_dir == self.last_move_direction and move_dir != (0, 0):
            score += W_PERSISTENCE

        # 2. Random Noise: Add a small random value to the score.
        # This helps break ties between equally good moves and prevents ants
        # from getting stuck in repetitive patterns in symmetric environments.
        score += rnd_uniform(-W_RANDOM_NOISE, W_RANDOM_NOISE)

        # 3. History Penalty (Optional Scoring Component):
        # Path history is primarily handled by `_filter_valid_moves`. However,
        # if we wanted a "softer" avoidance (allow moving into history but
        # prefer not to), a negative score could be added here.
        # Example:
        if self._is_in_history(neighbor_pos_int):
            score += W_AVOID_HISTORY # W_AVOID_HISTORY should be negative

        # 4. Repulsion from other Ants (Optional Scoring Component):
        # Check if the neighbor cell is occupied by another ant. Add penalty.
        # This complements the hard blocking in `_filter_valid_moves`.
        if self.simulation.is_ant_at(neighbor_pos_int, exclude_self=self):
            score -= W_REPULSION # W_REPULSION is positive, so subtract

        return score

    def _score_moves_returning(self, valid_neighbors_int: list[tuple]) -> dict[tuple, float]:
        """
        Scores potential moves for an ant in the RETURNING_TO_NEST state.

        Prioritizes:
        1. Moving closer to the nest.
        2. Following the 'home' pheromone trail.
        3. Avoiding negative and alarm pheromones (less strongly than when searching).

        Args:
            valid_neighbors_int: A list of valid neighboring (x, y) grid tuples.

        Returns:
            A dictionary where keys are the valid neighbor tuples and values
            are their calculated scores (float).
        """
        scores = {}
        sim = self.simulation
        grid = sim.grid
        current_pos_int = self.pos
        nest_pos_int = sim.nest_pos # Target destination

        # Calculate the ant's current distance to the nest (squared)
        dist_sq_now = distance_sq(current_pos_int, nest_pos_int)

        # Optionally prepare log data if detailed decision logging is needed
        # log_entry = { ... } # Example placeholder

        for n_pos_int in valid_neighbors_int:
            # Start with the base score (persistence, noise)
            score = self._score_moves_base(n_pos_int)

            # --- Pheromone Influence ---
            home_ph = grid.get_pheromone(n_pos_int, "home")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            neg_ph = grid.get_pheromone(n_pos_int, "negative")

            # Strong attraction to home pheromone trail
            score += home_ph * W_HOME_PHEROMONE_RETURN

            # --- Directional Bias ---
            # Calculate distance from the potential next step to the nest
            dist_sq_next = distance_sq(n_pos_int, nest_pos_int)
            # Add a significant bonus if this move gets the ant closer to the nest
            if dist_sq_next < dist_sq_now:
                score += W_NEST_DIRECTION_RETURN

            # --- Avoidance (less critical than when searching) ---
            # Slightly avoid alarm signals
            score += alarm_ph * W_ALARM_PHEROMONE * 0.3
            # Slightly avoid negative pheromones (previously explored areas)
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.4

            # Store the final score for this potential move
            scores[n_pos_int] = score

            # Optionally add detailed scoring factors to log entry
            # if log_entry: log_entry["moves_considered"].append({ ... })

        # Optionally log the complete decision process
        # if log_entry: sim.log_ant_decision(log_entry)

        return scores

    def _score_moves_patrolling(self, valid_neighbors_int: list[tuple]) -> dict[tuple, float]:
        """
        Scores potential moves for a soldier ant in the PATROLLING state.

        Prioritizes:
        1. Staying within the designated patrol radius around the nest.
        2. Investigating alarm and recruitment pheromones.
        3. Moving towards directly visible enemies.
        4. Slightly avoiding negative pheromones.

        Args:
            valid_neighbors_int: A list of valid neighboring (x, y) grid tuples.

        Returns:
            A dictionary where keys are the valid neighbor tuples and values
            are their calculated scores (float).
        """
        scores = {}
        sim = self.simulation
        grid = sim.grid
        current_pos_int = self.pos
        nest_pos_int = sim.nest_pos

        # Calculate current distance from nest (squared)
        dist_sq_current = distance_sq(current_pos_int, nest_pos_int)
        # Use the configured patrol radius squared from simulation
        patrol_radius_sq = sim.soldier_patrol_radius_sq
        # Define an outer boundary slightly beyond the patrol radius to strongly discourage leaving
        outer_boundary_sq = patrol_radius_sq * 1.4

        # Base weight for attracting soldiers to alarm signals
        # (May be adjusted based on colony situation, e.g., low queen health)
        alarm_attraction_weight = W_ALARM_PHEROMONE * -0.5 # Invert base avoidance for attraction
        # Example adjustment (currently disabled): Increase attraction if queen is weak
        # if sim.queen and sim.queen.hp < sim.queen.max_hp * 0.25:
        #     alarm_attraction_weight *= 1.75

        # Strong incentive to move towards directly visible enemies
        visible_enemy_weight = 500.0

        # --- Score Each Valid Neighbor ---
        for n_pos_int in valid_neighbors_int:
            # Start with the base score (persistence, noise)
            score = self._score_moves_base(n_pos_int)

            # --- Pheromone Influence ---
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            recr_ph = grid.get_pheromone(n_pos_int, "recruitment")

            # Moderate attraction to recruitment signals
            score += recr_ph * W_RECRUITMENT_PHEROMONE * 0.7
            # Attraction to alarm signals (investigation)
            score += alarm_ph * alarm_attraction_weight
            # Slight avoidance of negative pheromones
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.5

            # --- Visible Enemy Influence ---
            # Check if moving to `n_pos_int` brings the ant closer to any visible enemy.
            if self.visible_enemies:
                min_dist_sq_to_visible_enemy_next = float('inf')
                # Find the minimum squared distance from the potential *next* position
                # to any currently visible enemy.
                for enemy in self.visible_enemies:
                    # Ensure enemy still exists and is alive before calculating distance
                    if enemy in sim.enemies and enemy.hp > 0:
                         dist_sq_enemy = distance_sq(n_pos_int, enemy.pos)
                         min_dist_sq_to_visible_enemy_next = min(min_dist_sq_to_visible_enemy_next, dist_sq_enemy)

                # Add score bonus inversely proportional to the distance to the nearest visible enemy.
                # The closer the move gets, the higher the bonus. Added 1 to denominator to avoid division by zero.
                if min_dist_sq_to_visible_enemy_next < float('inf'):
                    score += visible_enemy_weight / (min_dist_sq_to_visible_enemy_next + 1)

            # --- Directional Control for Patrolling ---
            dist_sq_next = distance_sq(n_pos_int, nest_pos_int)

            # Penalize moving further away if *already inside* the patrol radius
            if dist_sq_current <= patrol_radius_sq and dist_sq_next > dist_sq_current:
                score += W_NEST_DIRECTION_PATROL # Negative weight discourages moving away

            # Strong penalty for moving beyond the outer boundary
            if dist_sq_next > outer_boundary_sq:
                score -= 8000 # Strong discouragement

            # Store the final score for this potential move
            scores[n_pos_int] = score

        self.last_move_info = "Patrolling" # Update debug info
        return scores

    def _score_moves_defending(self, valid_neighbors_int: list[tuple]) -> dict[tuple, float]:
        """
        Scores potential moves for an ant in the DEFENDING state.

        Prioritizes moving towards the source of danger, typically identified by
        the strongest nearby alarm or recruitment pheromone signals, or towards
        directly visible enemies. If no clear target is identified, may perform
        a random search around the last known position or current position.

        Args:
            valid_neighbors_int: A list of valid neighboring (x, y) grid tuples.

        Returns:
            A dictionary where keys are the valid neighbor tuples and values
            are their calculated scores (float).
        """
        scores = {}
        sim = self.simulation
        grid = sim.grid
        current_pos_int = self.pos

        # --- Update Target Location (last_known_alarm_pos) ---
        # Periodically re-scan the vicinity for the strongest threat signal,
        # or if the current target is lost (set to None).
        # The probability (e.g., 0.2) introduces stochasticity in target re-evaluation.
        if self.last_known_alarm_pos is None or random.random() < 0.2:
            best_signal_pos = None
            max_signal_strength = -1.0
            # Scan a radius around the ant for signals
            search_radius_sq = ALARM_SEARCH_RADIUS_SIGNAL ** 2
            x0, y0 = current_pos_int
            min_scan_x = max(0, x0 - ALARM_SEARCH_RADIUS_SIGNAL)
            max_scan_x = min(sim.grid_width - 1, x0 + ALARM_SEARCH_RADIUS_SIGNAL)
            min_scan_y = max(0, y0 - ALARM_SEARCH_RADIUS_SIGNAL)
            max_scan_y = min(sim.grid_height - 1, y0 + ALARM_SEARCH_RADIUS_SIGNAL)

            visible_enemy_positions = {enemy.pos for enemy in self.visible_enemies if enemy.hp > 0}

            for i in range(min_scan_x, max_scan_x + 1):
                for j in range(min_scan_y, max_scan_y + 1):
                    p_int = (i, j)
                    if distance_sq(current_pos_int, p_int) <= search_radius_sq:
                        # Combine alarm and recruitment signals (weighted)
                        signal = (grid.get_pheromone(p_int, "alarm") * 1.2 +
                                  grid.get_pheromone(p_int, "recruitment") * 0.8)

                        # Add significant bonus if an enemy is directly visible at this location
                        # This strongly biases the ant towards engaging visible threats.
                        if p_int in visible_enemy_positions:
                            signal += 600 # Strong incentive

                        # Update the best signal found so far
                        if signal > max_signal_strength:
                            max_signal_strength = signal
                            best_signal_pos = p_int

            # Update the ant's target if a sufficiently strong signal was found
            # A threshold prevents chasing faint, residual signals.
            signal_threshold = 80.0
            if max_signal_strength > signal_threshold:
                self.last_known_alarm_pos = best_signal_pos
            else:
                # Signal faded or too weak, lose the specific target
                # This might trigger random searching or state change later.
                self.last_known_alarm_pos = None

        # --- Score Moves Towards Target (or Random Search) ---
        target_pos = self.last_known_alarm_pos # Use the potentially updated target
        # Calculate distance to target if one exists
        dist_now_sq = distance_sq(current_pos_int, target_pos) if target_pos else float('inf')

        for n_pos_int in valid_neighbors_int:
            # Start with the base score (persistence, noise)
            score = self._score_moves_base(n_pos_int)

            # --- Immediate Engagement Bonus ---
            # Very high bonus for moving onto a cell with an enemy (visible or not)
            # This encourages direct combat when adjacent.
            enemy_at_n_pos = sim.get_enemy_at(n_pos_int)
            if enemy_at_n_pos and enemy_at_n_pos.hp > 0:
                score += 15000 # Prioritize engaging adjacent enemies

            # --- Movement based on Target ---
            if target_pos:
                # If a target position is known (strong signal source)
                dist_next_sq = distance_sq(n_pos_int, target_pos)
                # Bonus for getting closer to the target position
                if dist_next_sq < dist_now_sq:
                    score += W_ALARM_SOURCE_DEFEND # Attracted towards the signal source

                # Include pheromone values at the next step as well
                alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
                recr_ph = grid.get_pheromone(n_pos_int, "recruitment")
                score += (alarm_ph + recr_ph) * 0.5 # Additive bonus from pheromones at target step

            else:
                # --- Random Search if No Target ---
                # If no strong signal target is identified, perform a random walk
                # within a certain radius to search for the threat.
                # This prevents ants from freezing when signals dissipate.
                # Simple approach: Add score based on proximity to a random point
                # within the search radius. (More sophisticated random walks exist).
                random_offset_x = rnd(-ALARM_SEARCH_RADIUS_RANDOM, ALARM_SEARCH_RADIUS_RANDOM)
                random_offset_y = rnd(-ALARM_SEARCH_RADIUS_RANDOM, ALARM_SEARCH_RADIUS_RANDOM)
                # Calculate a temporary random destination near the ant's current position
                random_search_dest = (current_pos_int[0] + random_offset_x,
                                      current_pos_int[1] + random_offset_y)
                # Score moves based on getting closer to this temporary random destination
                dist_rand_sq = distance_sq(n_pos_int, random_search_dest)
                # Inverse relationship (smaller distance = higher score), simple linear scaling
                score += (ALARM_SEARCH_RADIUS_RANDOM * 2 - dist_rand_sq**0.5) * 0.5 # Adjust scaling factor as needed

            # Store the final score for this potential move
            scores[n_pos_int] = score

        self.last_move_info = f"Defend({target_pos})" if target_pos else "Defend(Search)"
        return scores

    def _score_moves_hunting(self, valid_neighbors_int: list[tuple]) -> dict[tuple, float]:
        """
        Scores potential moves for an ant in the HUNTING state.

        Prioritizes moving directly towards the ant's assigned `target_prey`.
        Includes minor avoidance of negative and alarm pheromones.

        Args:
            valid_neighbors_int: A list of valid neighboring (x, y) grid tuples.

        Returns:
            A dictionary where keys are the valid neighbor tuples and values
            are their calculated scores (float). Returns base scores if no
            valid target prey is assigned.
        """
        scores = {}
        sim = self.simulation
        grid = sim.grid
        current_pos_int = self.pos

        # Get the target prey's current position
        # Check if target exists, is still in the simulation's prey list, and is alive
        target_pos = None
        if (self.target_prey and
                self.target_prey in sim.prey and
                self.target_prey.hp > 0):
            target_pos = self.target_prey.pos
        else:
            # If target is invalid (None, removed, or dead), revert to searching behavior temporarily
            # The state should ideally be updated soon by _update_state
            self.last_move_info = "Hunt(NoTgt!)"
            # Return scores based only on base function (persistence, noise)
            return {n_pos_int: self._score_moves_base(n_pos_int) for n_pos_int in valid_neighbors_int}

        # Calculate current distance to the target prey
        dist_sq_now = distance_sq(current_pos_int, target_pos)

        # --- Score Each Valid Neighbor ---
        for n_pos_int in valid_neighbors_int:
            # Start with the base score
            score = self._score_moves_base(n_pos_int)

            # --- Directional Bias towards Prey ---
            dist_sq_next = distance_sq(n_pos_int, target_pos)
            # Add strong bonus for getting closer to the prey
            if dist_sq_next < dist_sq_now:
                score += W_HUNTING_TARGET

            # --- Minor Pheromone Avoidance ---
            # Slightly avoid strong negative or alarm signals while hunting
            alarm_ph = grid.get_pheromone(n_pos_int, "alarm")
            neg_ph = grid.get_pheromone(n_pos_int, "negative")
            score += alarm_ph * W_ALARM_PHEROMONE * 0.1   # Weak avoidance
            score += neg_ph * W_NEGATIVE_PHEROMONE * 0.2 # Slight avoidance

            # Store the final score
            scores[n_pos_int] = score

        self.last_move_info = f"Hunt({target_pos})" # Update debug info
        return scores

    def _select_best_move(self, move_scores: dict[tuple, float], valid_neighbors_int: list[tuple]) -> tuple | None:
        """
        Selects the move with the absolute highest score (deterministic choice).

        Used for states where exploration is not desired and the ant should
        take the objectively best action based on the calculated scores (e.g.,
        DEFENDING, HUNTING). Handles ties by choosing randomly among the
        equally highest-scoring moves.

        Args:
            move_scores: A dictionary mapping valid neighbor tuples (x, y) to
                         their calculated scores.
            valid_neighbors_int: The list of valid neighbor tuples considered.
                                 Used as a fallback if `move_scores` is empty
                                 or contains no valid scores.

        Returns:
            The chosen (x, y) grid coordinate tuple, or None if no valid choice
            could be made (e.g., no valid neighbors or scores).
        """
        # Handle edge case: No valid neighbors provided (should be caught earlier, but safety)
        if not valid_neighbors_int:
            self.last_move_info += "(Best:NoValidNeigh!)"
            return None
        # Handle edge case: No scores calculated (should be caught earlier, but safety)
        if not move_scores:
            self.last_move_info += "(Best:NoScores!)"
            # Fallback to random choice among the valid neighbors if scores are missing
            return random.choice(valid_neighbors_int)

        best_score = -float("inf")  # Initialize with lowest possible score
        best_moves_int = []       # List to hold moves tied for the best score

        # Find the highest score among the calculated move scores
        for pos_int, score in move_scores.items():
            # Ensure score is valid (not NaN, etc.) before comparing
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                # print(f"Warning: Invalid score {score} for move {pos_int} in _select_best_move") # Optional debug
                continue  # Skip invalid scores

            if score > best_score:
                # Found a new best score
                best_score = score
                best_moves_int = [pos_int]  # Reset list with the new best move
            elif score == best_score:
                # Found a move with the same best score (tie)
                best_moves_int.append(pos_int)  # Add to the list of tied best moves

        # If no best moves were found (e.g., all scores were invalid):
        if not best_moves_int:
            self.last_move_info += "(Best:NoBestFound!)"
            # Fallback to a random choice among all originally valid neighbors
            return random.choice(valid_neighbors_int)

        # If there are ties for the best score, choose randomly among them
        chosen_int = random.choice(best_moves_int)

        # Update debug info with the chosen move and its score
        score_val = move_scores.get(chosen_int, -999)  # Get score for logging
        state_prefix = self.state.name[:3]             # Abbreviated state name
        self.last_move_info = f"{state_prefix} Best->{chosen_int} (S:{score_val:.1f})"

        return chosen_int

    def _select_best_move_returning(self, move_scores: dict[tuple, float],
                                     valid_neighbors_int: list[tuple]) -> tuple | None:
        """
        Selects the best move for an ant returning to the nest.

        Prioritizes moves that reduce the distance to the nest. Among moves
        that get closer, or among other moves if none get closer, it selects
        the one with the highest score. Ties in score are now broken RANDOMLY.

        Args:
            move_scores: Dictionary mapping valid neighbor tuples (x, y) to scores.
            valid_neighbors_int: List of valid neighbor tuples considered.

        Returns:
            The chosen (x, y) grid coordinate tuple, or None if no valid choice.
        """
        # (Input validation remains the same)
        if not valid_neighbors_int:
            self.last_move_info += "(R: NoValidNeigh!)"
            return None
        if not move_scores:
            self.last_move_info += "(R: NoScores!)"
            return random.choice(valid_neighbors_int) # Fallback if scores missing

        sim = self.simulation
        current_pos_int = self.pos
        nest_pos_int = sim.nest_pos
        dist_sq_now = distance_sq(current_pos_int, nest_pos_int)

        # (Separation into closer_moves_scores and other_moves_scores remains the same)
        closer_moves_scores = {}
        other_moves_scores = {}
        for pos_int_cand, score in move_scores.items():
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                continue # Skip invalid scores
            if distance_sq(pos_int_cand, nest_pos_int) < dist_sq_now:
                closer_moves_scores[pos_int_cand] = score
            else:
                other_moves_scores[pos_int_cand] = score

        # (Determining target_pool_scores remains the same)
        target_pool_scores = {}
        selection_type = ""
        if closer_moves_scores:
            target_pool_scores = closer_moves_scores
            selection_type = "Closer"
        elif other_moves_scores:
            target_pool_scores = other_moves_scores
            selection_type = "Other"
        else:
            target_pool_scores = {k: v for k, v in move_scores.items() if math.isfinite(v)}
            selection_type = "All(Fallback)"
            if not target_pool_scores:
                 self.last_move_info += "(R: No Scorable Moves!)"
                 return random.choice(valid_neighbors_int)

        # (Finding best_score remains the same)
        best_score = -float("inf")
        best_scoring_moves = []
        for pos_int_cand, score in target_pool_scores.items():
            if score > best_score:
                best_score = score
                best_scoring_moves = [pos_int_cand]
            elif score == best_score:
                best_scoring_moves.append(pos_int_cand)

        if not best_scoring_moves:
            self.last_move_info += f"(R: NoBestInPool {selection_type})"
            return random.choice(valid_neighbors_int)

        # --- KORREKTUR: Tie-breaking - Wähle zufällig bei Gleichstand ---
        chosen_int = random.choice(best_scoring_moves)

        if len(best_scoring_moves) == 1:
            self.last_move_info = f"R({selection_type})Best->{chosen_int} (S:{best_score:.1f})"
        else:
            # Wenn es mehrere beste Züge gab, logge, dass ein zufälliger gewählt wurde.
            self.last_move_info = f"R({selection_type})RandTie->{chosen_int} (S:{best_score:.1f})"
        # --- ENDE KORREKTUR ---

        return chosen_int

    def _select_probabilistic_move(self, move_scores: dict[tuple, float],
                                    valid_neighbors_int: list[tuple]) -> tuple | None:
        """
        Selects a move probabilistically based on scores using a temperature parameter.

        Higher scores have a higher probability of being chosen, but lower scores
        still have a chance, allowing for exploration. Used in states like
        SEARCHING and PATROLLING.

        Args:
            move_scores: Dictionary mapping valid neighbor tuples (x, y) to scores.
            valid_neighbors_int: List of valid neighbor tuples considered.

        Returns:
            The chosen (x, y) grid coordinate tuple, or None if no valid choice.
        """
        # --- Input Validation ---
        if not valid_neighbors_int:
             self.last_move_info += "(Prob:NoValidNeigh!)"
             return None
        if not move_scores:
             self.last_move_info += "(Prob:NoScores!)"
             # Fallback: If scores are missing, choose randomly from valid neighbors
             return random.choice(valid_neighbors_int)

        # Extract positions and scores, ensuring scores are valid floats
        positions_int = []
        scores_raw = []
        for pos, score in move_scores.items():
            # Ensure the position is actually in the valid list and score is finite
            if pos in valid_neighbors_int and isinstance(score, (int, float)) and math.isfinite(score):
                positions_int.append(pos)
                scores_raw.append(float(score))
            # else: # Optional debug for skipped scores/positions
                # print(f"Debug Prob: Skipping pos {pos} or score {score}")

        # Handle cases after filtering invalid scores
        if not positions_int:
            self.last_move_info += "(Prob:NoValidScores!)"
            # Fallback: Choose randomly from the original valid list if no finite scores found
            return random.choice(valid_neighbors_int)
        if len(positions_int) == 1:
            # Only one valid option remains, choose it directly
            chosen_int = positions_int[0]
            score_val = scores_raw[0]
            state_prefix = self.state.name[:3]
            self.last_move_info = f"{state_prefix} Prob->{chosen_int} (Only S:{score_val:.1f})"
            return chosen_int

        # --- Score Normalization and Weight Calculation ---
        scores_np = np.array(scores_raw, dtype=np.float64) # Use float64 for precision

        # Normalize scores to be non-negative for weighting
        # Shift scores so the minimum score becomes a small positive value (avoids issues with zero/negative scores)
        min_score = np.min(scores_np)
        shifted_scores = scores_np - min_score + 0.01 # Add epsilon

        # Apply temperature parameter for randomness tuning
        # Clamp temperature to prevent extreme values leading to overflow/underflow
        temp = max(0.1, min(PROBABILISTIC_CHOICE_TEMP, 5.0)) # Ensure temp is within a reasonable range

        # Calculate weights = score ^ temperature
        # Use clipping before power calculation to prevent potential overflow with very large scores/temps
        clipped_scores = np.clip(shifted_scores, 0, 1e6) # Limit base to prevent huge numbers
        try:
             # Weights emphasize higher scores more strongly as temp increases (less random)
             # Weights approach uniformity as temp approaches 0 (more random - though clamped > 0.1)
             # If temp == 1, weights are proportional to shifted scores.
             weights = np.power(clipped_scores, 1.0 / temp) # Inverse relationship for desired effect
             # Ensure weights have a minimum value to avoid zero probabilities for low scores
             weights = np.maximum(MIN_SCORE_FOR_PROB_CHOICE, weights)
        except (OverflowError, ValueError) as e:
             # Fallback if power calculation fails (e.g., huge scores even after clipping)
             print(f"Warning: Overflow/ValueError in probabilistic weight calc. Scores: {clipped_scores}, Temp: {temp}, Error: {e}")
             # Use linear scaling as a fallback (equivalent to temp=1)
             weights = np.maximum(MIN_SCORE_FOR_PROB_CHOICE, clipped_scores)

        # --- Probability Calculation and Selection ---
        total_weight = np.sum(weights)

        # --- Sanity Check Weights and Probabilities ---
        if total_weight <= 1e-9 or not np.isfinite(total_weight) or np.any(~np.isfinite(weights)):
            # Handle invalid weights (e.g., all zero, NaN, Inf)
            self.last_move_info += f"({self.state.name[:3]}:InvW)"
            # Fallback: Choose the move with the original highest score deterministically
            best_s_idx = np.argmax(scores_np) # Index of highest original score
            chosen_int = positions_int[best_s_idx]
            score_val = scores_np[best_s_idx]
            state_prefix = self.state.name[:3]
            self.last_move_info = f"{state_prefix} ProbFW->{chosen_int} (Best S:{score_val:.1f})"
            return chosen_int

        # Calculate probabilities
        probabilities = weights / total_weight

        # --- Final Selection using Probabilities ---
        try:
            # Use numpy's random.choice with the calculated probabilities
            chosen_index = np.random.choice(len(positions_int), p=probabilities)
            chosen_int = positions_int[chosen_index]

            # Update debug info
            score_val = scores_raw[chosen_index] # Get original score for chosen move
            state_prefix = self.state.name[:3]
            self.last_move_info = f"{state_prefix} Prob->{chosen_int} (S:{score_val:.1f})"
            return chosen_int
        except ValueError as e:
            # Catch errors like "probabilities do not sum to 1" (should be rare after normalization)
            print(f"WARN: Probabilistic choice ValueError ({self.state.name}): {e}. Sum={np.sum(probabilities)}, Probs={probabilities}")
            self.last_move_info += "(ProbValErr!)"
            # Fallback to deterministic best choice if probability selection fails
            best_s_idx = np.argmax(scores_np)
            chosen_int = positions_int[best_s_idx]
            score_val = scores_np[best_s_idx]
            state_prefix = self.state.name[:3]
            self.last_move_info = f"{state_prefix} ProbFV->{chosen_int} (Best S:{score_val:.1f})"
            return chosen_int

    def _switch_state(self, new_state: AntState, reason: str):
        """
        Changes the ant's current behavioral state (AntState).

        Logs the reason for the state change and performs necessary cleanup,
        such as clearing the path history and resetting state-specific targets.

        Args:
            new_state: The new AntState enum member to switch to.
            reason: A short string describing the reason for the state change
                    (used for logging/debugging via `last_move_info`).
        """
        # Only perform actions if the state is actually changing
        if self.state != new_state:
            # print(f"Ant {id(self)}: {self.state.name} -> {new_state.name} ({reason}) at {self.pos}") # Optional detailed debug log
            old_state = self.state # Store old state if needed for specific logic
            self.state = new_state
            self.last_move_info = reason # Update debug info with the reason

            # --- Reset State-Specific Variables ---
            # Clear path history as the ant's goal has fundamentally changed
            self._clear_path_history()

            # Reset targets relevant to the *old* state
            if old_state == AntState.HUNTING:
                self.target_prey = None
            if old_state == AntState.DEFENDING:
                self.last_known_alarm_pos = None

            # Reset stuck timer whenever the state changes, giving the ant
            # a fresh chance in the new state.
            self.stuck_timer = 0

            # Reset escape timer if leaving the ESCAPING state
            if old_state == AntState.ESCAPING:
                self.escape_timer = 0

    def update(self):
        """
        Main update logic for the ant, executed once per simulation tick.

        Handles aging, eating, visual updates, state transitions, interactions
        (attacking enemies/prey), movement decisions, pheromone dropping,
        and food handling.
        """
        # If already marked for death, do nothing further
        if self.is_dying:
            return

        sim = self.simulation
        grid = sim.grid
        current_pos_int = self.pos # Store current position for comparisons

        # --- 1. Basic Needs: Aging and Starvation ---
        self.age += 1 # Increment age by one tick
        if self.age >= self.max_age_ticks:
            self.hp = 0 # Die of old age
            self.last_move_info = "Died of old age"
            self.is_dying = True # Mark for removal
            return # No further actions

        # Food consumption check based on interval timer
        self.food_consumption_timer += 1
        if self.food_consumption_timer >= WORKER_FOOD_CONSUMPTION_INTERVAL:
            self.food_consumption_timer = 0 # Reset timer
            needed_s = self.food_consumption_sugar
            needed_p = self.food_consumption_protein

            # Ants eat directly from shared colony storage
            can_eat = (sim.colony_food_storage_sugar >= needed_s and
                       sim.colony_food_storage_protein >= needed_p)

            if can_eat:
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p
            else:
                self.hp = 0 # Starved due to lack of colony resources
                self.last_move_info = "Starved"
                self.is_dying = True # Mark for removal
                return # No further actions

        # --- 2. Visual Perception ---
        # Update the list of enemies currently visible to the ant
        self._update_visible_enemies()

        # --- 3. State Management ---
        # Check environment and internal status for potential state changes
        # This might switch the ant's state (e.g., from SEARCHING to HUNTING)
        self._update_state()

        # Check if died during state update logic (e.g., if starvation check moved there - currently not)
        if self.hp <= 0 or self.is_dying: return

        # --- 4. Interaction Checks (Attack) ---
        # Check immediate surroundings (neighbors + current cell) for targets
        neighbors_int = get_neighbors(current_pos_int, sim.grid_width, sim.grid_height, include_center=True)
        adjacent_enemy = None
        adjacent_prey_to_attack = None

        # Priority: Attack adjacent enemies
        for p_int in neighbors_int:
            enemy = sim.get_enemy_at(p_int)
            if enemy and enemy.hp > 0:
                adjacent_enemy = enemy
                break # Target the first adjacent enemy found

        if adjacent_enemy:
            # --- Attack Enemy ---
            self.attack(adjacent_enemy)
            # Drop alarm pheromone at the ant's current location during combat
            grid.add_pheromone(current_pos_int, P_ALARM_FIGHT, "alarm")
            self.stuck_timer = 0 # Reset stuck timer during fight
            self.target_prey = None # Stop hunting if fighting an enemy
            self.last_move_info = f"FightEnemy@{adjacent_enemy.pos}"
            # Ensure state is DEFENDING when fighting
            if self.state != AntState.DEFENDING:
                self._switch_state(AntState.DEFENDING, "EnemyContact!")
            # --- Skip movement this tick after attacking ---
            return

        # Priority: Attack adjacent prey (if not fighting enemy)
        # Check if hunting target is adjacent, or if opportunistic attack is possible
        prey_in_range = [] # Collect all living prey in neighboring cells
        for p_int in neighbors_int:
            prey = sim.get_prey_at(p_int)
            if prey and prey.hp > 0:
                prey_in_range.append(prey)

        if prey_in_range:
            should_attack_prey = False
            # If HUNTING and the specific target is adjacent:
            if (self.state == AntState.HUNTING and self.target_prey and
                    self.target_prey in prey_in_range):
                 # Check position match explicitly for safety
                 if self.target_prey.pos in neighbors_int:
                      adjacent_prey_to_attack = self.target_prey
                      should_attack_prey = True
            # If not returning/defending, consider opportunistic attack:
            elif self.state not in [AntState.RETURNING_TO_NEST, AntState.DEFENDING]:
                # Check if colony needs protein (worker) or if soldier (always hunts opportunity)
                colony_needs_protein = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * 2
                can_hunt_opportunistically = (self.caste == AntCaste.SOLDIER or
                                              (self.caste == AntCaste.WORKER and colony_needs_protein))
                if can_hunt_opportunistically:
                    # Prefer prey on adjacent cells over prey on the same cell
                    adjacent_prey = [p for p in prey_in_range if p.pos != current_pos_int]
                    if adjacent_prey:
                        adjacent_prey_to_attack = random.choice(adjacent_prey)
                        should_attack_prey = True
                    else: # Only prey on current cell is available
                        adjacent_prey_to_attack = random.choice(prey_in_range) # prey_on_cell
                        should_attack_prey = True

            # Perform prey attack if decided
            if should_attack_prey and adjacent_prey_to_attack:
                # --- Attack Prey ---
                self.attack(adjacent_prey_to_attack)
                self.stuck_timer = 0 # Reset stuck timer
                self.last_move_info = f"AtkPrey@{adjacent_prey_to_attack.pos}"

                # Check if prey was killed by the attack
                if adjacent_prey_to_attack.hp <= 0:
                    killed_prey_pos = adjacent_prey_to_attack.pos
                    sim.kill_prey(adjacent_prey_to_attack) # Notify simulation to remove prey and add food
                    # Drop pheromones at the kill site
                    grid.add_pheromone(killed_prey_pos, P_FOOD_AT_SOURCE, "food", FoodType.PROTEIN)
                    grid.add_pheromone(killed_prey_pos, P_RECRUIT_PREY, "recruitment")
                    # If this was the hunted target, clear the target
                    if self.target_prey == adjacent_prey_to_attack:
                        self.target_prey = None
                    # Switch state after kill (e.g., back to searching/patrolling)
                    next_s = AntState.SEARCHING if self.caste == AntCaste.WORKER else AntState.PATROLLING
                    self._switch_state(next_s, "PreyKilled")
                # --- Skip movement this tick after attacking ---
                return

        # --- 5. Movement Cooldown ---
        if self.move_cooldown_timer > 0:
            self.move_cooldown_timer -= 1
            return # Cannot move yet, wait for cooldown

        # Reset cooldown timer for the next cycle (applies *after* moving this tick)
        self.move_cooldown_timer = self.move_cooldown_base

        # --- 6. Choose and Execute Move ---
        old_pos_int = current_pos_int # Remember position before moving
        # Store and reset flag for pheromone logic after food pickup
        local_just_picked = self.just_picked_food
        self.just_picked_food = False # Reset flag for this tick

        # Determine the next grid cell to move to
        new_pos_int = self._choose_move()

        moved = False
        if new_pos_int and new_pos_int != old_pos_int:
            # --- Execute Move ---
            self.pos = new_pos_int
            sim.update_entity_position(self, old_pos_int, new_pos_int) # Update simulation tracking
            # Update direction based on the move made
            self.last_move_direction = (new_pos_int[0] - old_pos_int[0], new_pos_int[1] - old_pos_int[1])
            self._update_path_history(new_pos_int) # Add to history
            self.stuck_timer = 0 # Reset stuck timer as movement occurred
            moved = True
        elif new_pos_int == old_pos_int:
            # Ant chose to stay in the same cell (or _choose_move returned current pos)
            self.stuck_timer += 1
            self.last_move_info += "(Move->Same)"
            # No change in direction if didn't move to a new cell
            # self.last_move_direction = (0, 0) # Or keep old direction? Keep old seems better.
        else: # new_pos_int is None (truly blocked)
            self.stuck_timer += 1
            self.last_move_info += "(NoChoice)"
            self.last_move_direction = (0, 0) # Indicate no definite direction

        # --- 7. Post-Movement Actions (Pheromones, State Changes, Food) ---
        current_pos_int = self.pos # Use updated position for actions
        nest_pos_int = sim.nest_pos
        is_near_nest = distance_sq(current_pos_int, nest_pos_int) <= NEST_RADIUS**2

        # --- Action: Pick up Food ---
        if self.state in [AntState.SEARCHING, AntState.HUNTING] and self.carry_amount == 0:
            # Check for food at the *new* current position
            found_food_type = None
            food_amount = 0.0
            try:
                foods = grid.food[current_pos_int[0], current_pos_int[1]]
                # Prioritize sugar if both exist? Or based on need? Simple check for now.
                if foods[FoodType.SUGAR.value] > 0.1:
                    found_food_type = FoodType.SUGAR
                    food_amount = foods[FoodType.SUGAR.value]
                elif foods[FoodType.PROTEIN.value] > 0.1:
                    found_food_type = FoodType.PROTEIN
                    food_amount = foods[FoodType.PROTEIN.value]
            except IndexError:
                pass # Should not happen with valid pos, but safety

            if found_food_type:
                # Ant found food and is able to carry it
                pickup_amount = min(self.max_capacity, food_amount)
                if pickup_amount > 0.01: # Ensure a meaningful amount is picked up
                    # Update ant's carry status
                    self.carry_amount = pickup_amount
                    self.carry_type = found_food_type
                    # Remove food from grid
                    food_idx = found_food_type.value
                    try:
                        grid.food[current_pos_int[0], current_pos_int[1], food_idx] = max(0, food_amount - pickup_amount)
                    except IndexError: pass
                    # Drop pheromones at the source
                    grid.add_pheromone(current_pos_int, P_FOOD_AT_SOURCE, "food", food_type=found_food_type)
                    # Drop recruitment pheromone if the source is rich
                    if food_amount >= RICH_FOOD_THRESHOLD:
                        grid.add_pheromone(current_pos_int, P_RECRUIT_FOOD, "recruitment")
                    # Switch state to returning
                    self._switch_state(AntState.RETURNING_TO_NEST,
                                       f"Picked {found_food_type.name[:1]}({pickup_amount:.1f})")
                    self.just_picked_food = True # Set flag for next tick's pheromone logic
                    self.target_prey = None # Stop hunting if picked up food

        # --- Action: Drop Pheromones while Searching (Negative) ---
        elif moved and self.state == AntState.SEARCHING and not is_near_nest:
             # Drop negative pheromone at the *previous* location if searching and moved away from nest area
             if is_valid_pos(old_pos_int, sim.grid_width, sim.grid_height):
                  grid.add_pheromone(old_pos_int, P_NEGATIVE_SEARCH, "negative")

        # --- Action: Drop Off Food / Drop Pheromones while Returning ---
        elif self.state == AntState.RETURNING_TO_NEST:
            if is_near_nest:  # Check if inside the core nest radius for drop-off
                # --- Drop off Food ---
                # (Code für Drop-off bleibt gleich) ...
                dropped_amount = self.carry_amount
                type_dropped = self.carry_type
                if dropped_amount > 0 and type_dropped:
                    if type_dropped == FoodType.SUGAR:
                        sim.colony_food_storage_sugar += dropped_amount
                    elif type_dropped == FoodType.PROTEIN:
                        sim.colony_food_storage_protein += dropped_amount
                    self.carry_amount = 0
                    self.carry_type = None

                # --- Switch State after Drop-off ---
                # (Code für State Switch bleibt gleich) ...
                next_state = AntState.SEARCHING  # Default for workers
                state_reason = "Dropped->"
                if self.caste == AntCaste.WORKER:
                    sugar_crit = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD
                    protein_crit = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD
                    if sugar_crit or protein_crit:
                        state_reason += "SEARCH(Need!)"
                    else:
                        state_reason += "SEARCH"
                elif self.caste == AntCaste.SOLDIER:
                    next_state = AntState.PATROLLING  # Soldiers usually return to patrol
                    state_reason += "PATROL"
                self._switch_state(next_state, state_reason)


            elif moved and not local_just_picked:
                # --- Drop Trail Pheromones while Returning ---
                # Drop at the *previous* location after moving
                # <<< ÄNDERUNG: Größerer Radius für Pheromon-Stopp >>>
                # Stoppe das Ablegen von Trail-Pheromonen schon etwas *weiter* vom Nest entfernt.
                # Erhöhe den Radius von (NEST_RADIUS - 1)**2 auf z.B. (NEST_RADIUS + 2)**2
                stop_pheromone_radius_sq = (NEST_RADIUS + 2) ** 2  # Neuer, größerer Radius
                # <<< ENDE ÄNDERUNG >>>

                if is_valid_pos(old_pos_int, sim.grid_width, sim.grid_height) and \
                        distance_sq(old_pos_int, nest_pos_int) > stop_pheromone_radius_sq:  # <<< Geänderte Bedingung
                    # Drop home pheromone trail
                    grid.add_pheromone(old_pos_int, P_HOME_RETURNING, "home")
                    # Drop food pheromone trail if carrying food
                    if self.carry_amount > 0 and self.carry_type:
                        grid.add_pheromone(old_pos_int, P_FOOD_RETURNING_TRAIL, "food", food_type=self.carry_type)

        # --- 8. Stuck Detection and Escape ---
        # Check if stuck timer exceeds threshold and not already escaping
        if self.stuck_timer >= WORKER_STUCK_THRESHOLD and self.state != AntState.ESCAPING:
            # Check if being stuck is "legitimate" (e.g., fighting an enemy)
            is_fighting = False
            for p_int in neighbors_int: # Use neighbors including center
                if sim.get_enemy_at(p_int):
                    is_fighting = True
                    break
            # Add check for hunting adjacent prey? Maybe allow escaping even then?

            # Only enter ESCAPING state if not actively engaged in combat
            if not is_fighting:
                self._switch_state(AntState.ESCAPING, "Stuck!")
                self.escape_timer = WORKER_ESCAPE_DURATION # Set duration for escape state
                # Note: stuck_timer reset happens in _switch_state

    def attack(self, target):
        """
        Performs an attack on a target entity (Enemy or Prey).

        Deals damage equal to the ant's `attack_power` to the target.
        Also triggers a visual attack indicator in the simulation.

        Args:
            target: The Enemy or Prey object to attack.
        """
        sim = self.simulation # Get simulation instance for context

        # Check if the target is a valid type, has a take_damage method, and is alive
        if isinstance(target, (Enemy, Prey)) and hasattr(target, 'take_damage') and target.hp > 0:
            target_pos = target.pos # Store target position for indicator
            # Call the target's method to inflict damage
            target.take_damage(self.attack_power, self) # Pass self as attacker

            # Add a visual indicator for the attack event
            sim.add_attack_indicator(self.pos, target_pos, ATTACK_INDICATOR_COLOR_ANT)

            # --- Speed Boost Removed ---
            # The following lines related to speed boost on attack were removed
            # as per the refactoring goal to eliminate speed multipliers.
            # self.speed_boost_timer = ANT_SPEED_BOOST_DURATION
            # self.speed_boost_multiplier = ANT_SPEED_BOOST_MULTIPLIER

    def take_damage(self, amount: float, attacker):
        """
        Handles the ant receiving damage from an attacker.

        Reduces the ant's HP. If the ant survives, it drops alarm and
        recruitment pheromones to signal the attack to nearby colony members.
        Marks the ant for removal if HP drops to zero.

        Args:
            amount: The amount of damage received.
            attacker: The entity that dealt the damage (currently unused, but available).
        """
        # Ignore damage if already dying or dead
        if self.is_dying or self.hp <= 0:
            return

        self.hp -= amount
        grid = self.simulation.grid # Get grid reference
        pos_int = self.pos          # Get current position

        if self.hp > 0:
            # --- Ant survived the hit ---
            # Drop alarm pheromone to signal immediate danger
            grid.add_pheromone(pos_int, P_ALARM_FIGHT, "alarm") # Use fight amount

            # Drop recruitment pheromone to call for help
            # Soldiers drop a stronger signal when damaged
            recruit_amount = P_RECRUIT_DAMAGE_SOLDIER if self.caste == AntCaste.SOLDIER else P_RECRUIT_DAMAGE
            grid.add_pheromone(pos_int, recruit_amount, "recruitment")
        else:
            # --- Ant died ---
            self.hp = 0 # Ensure HP doesn't go negative
            self.is_dying = True # Mark for removal in the main simulation loop
            self.last_move_info = "Killed" # Update status info
            # Optional: Could drop a final negative pheromone upon death here.
            # grid.add_pheromone(pos_int, 50.0, "negative") # Example

    def _update_visible_enemies(self):
        """
        Updates the list `self.visible_enemies` with Enemy objects currently
        within the ant's visual range (`ANT_VISUAL_RANGE`).

        Uses the spatial grid for efficient querying of nearby enemies.
        Filters for living enemies.
        """
        sim = self.simulation
        current_pos_int = self.pos # Ant's grid position
        self.visible_enemies.clear() # Reset the list for the current tick

        # Optimization: Skip if there are no enemies present in the simulation
        if not sim.enemies:
            return

        # Define visual range in pixels for the spatial grid query
        visual_range_px = ANT_VISUAL_RANGE * sim.cell_size
        # Calculate the center pixel position of the ant's grid cell
        center_px = (int(current_pos_int[0] * sim.cell_size + sim.cell_size / 2),
                     int(current_pos_int[1] * sim.cell_size + sim.cell_size / 2))

        # Query the spatial grid for enemies within the visual pixel radius
        # The spatial grid function performs the necessary distance checks.
        potential_enemies = sim.spatial_grid.get_nearby_entities(center_px, visual_range_px, Enemy)

        # Filter the potential enemies to include only those that are alive.
        # Line-of-Sight Check (Optional):
        # If obstacles should block vision, a raycasting check from the ant's position
        # to each potential_enemy.pos could be inserted here. This adds complexity.
        # For now, we assume clear line of sight if within range.
        for enemy in potential_enemies:
            if enemy.hp > 0: # Check if the enemy is alive
                 self.visible_enemies.append(enemy)

        # Optional Sorting:
        # If targeting logic elsewhere prioritizes the absolute closest visible enemy,
        # sorting the list here might be useful.
        self.visible_enemies.sort(key=lambda e: distance_sq(current_pos_int, e.pos))

# --- Queen Class ---
class Queen:
    """
    Represents the queen ant, the core of the colony.

    Responsible for laying eggs to grow the colony's population. Her health is
    critical, and her egg-laying rate can adapt based on the colony's food reserves.
    She does not move or fight under normal circumstances but can be attacked.
    """

    def __init__(self, pos_grid: tuple, sim):
        """
        Initializes the Queen.

        Args:
            pos_grid: The initial (x, y) grid coordinates for the queen.
            sim: Reference to the main AntSimulation object.
        """
        self.pos = tuple(map(int, pos_grid)) # Queen's fixed grid position
        self.simulation = sim
        self.hp = float(QUEEN_HP)         # Current Hit Points
        self.max_hp = float(QUEEN_HP)     # Maximum Hit Points
        self.age = 0                      # Age in simulation ticks
        self.color = QUEEN_COLOR

        # --- Egg Laying ---
        # Timer tracking progress towards the next egg-laying attempt.
        self.egg_lay_timer_progress = 0.0
        # Current interval (in ticks) between egg-laying attempts.
        # Starts at the base rate but can increase if food is low.
        self.egg_lay_interval_ticks = float(QUEEN_EGG_LAY_RATE)

        # --- Static Attributes (Not typically used by Queen) ---
        self.attack_power = 0             # Queen does not attack
        self.carry_amount = 0             # Queen does not carry food
        self.is_dying = False             # Flag for removal process

    def update(self):
        """
        Updates the queen's state for one simulation tick.

        Handles aging and the adaptive egg-laying process based on food storage.
        """
        sim = self.simulation

        # If already marked for death, do nothing further
        if self.is_dying:
            return

        # --- Aging ---
        self.age += 1 # Increment age by one tick

        # --- Adaptive Egg-Laying Rate Adjustment ---
        # Adjust the time required between egg lays based on food availability.
        # If food is critically low, the interval increases (slower laying).
        # If food is abundant, the interval slowly decreases back towards the base rate.

        # Define thresholds for food scarcity check (relative to CRITICAL_FOOD_THRESHOLD)
        food_low_threshold_factor = 1.0    # Threshold for 'low' food
        food_very_low_threshold_factor = 0.2 # Threshold for 'very low' food

        # Check current food levels against thresholds
        sugar_low = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD * food_low_threshold_factor
        protein_low = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * food_low_threshold_factor
        sugar_very_low = sim.colony_food_storage_sugar < CRITICAL_FOOD_THRESHOLD * food_very_low_threshold_factor
        protein_very_low = sim.colony_food_storage_protein < CRITICAL_FOOD_THRESHOLD * food_very_low_threshold_factor

        # Define parameters for adjusting the lay interval
        base_rate = float(QUEEN_EGG_LAY_RATE) # Target minimum interval
        max_slowdown_factor = 4.0             # Maximum multiplier for the interval (e.g., 4x slower)
        # Factor to increase interval (slow down laying) when food is low
        slowdown_increment = 1.05             # Multiplicative increase (e.g., 5% slower per check)
        # Factor to decrease interval (speed up laying) when food is sufficient
        speedup_decrement = 0.98              # Multiplicative decrease (e.g., 2% faster per check)

        # Adjust interval based on food levels
        if sugar_very_low or protein_very_low:
            # Food critically low: Increase interval significantly (much slower laying)
            target_interval = self.egg_lay_interval_ticks * (slowdown_increment * 1.5) # Faster slowdown
            self.egg_lay_interval_ticks = min(base_rate * max_slowdown_factor, target_interval)
        elif sugar_low or protein_low:
            # Food low: Increase interval moderately (slower laying)
            target_interval = self.egg_lay_interval_ticks * slowdown_increment
            self.egg_lay_interval_ticks = min(base_rate * max_slowdown_factor, target_interval)
        else:
            # Sufficient food: Gradually decrease interval back towards the base rate
            target_interval = self.egg_lay_interval_ticks * speedup_decrement
            self.egg_lay_interval_ticks = max(base_rate, target_interval) # Don't go faster than base rate

        # --- Egg Laying Process ---
        self.egg_lay_timer_progress += 1 # Increment progress by one tick

        # Check if enough time has passed for an egg-laying attempt (using the current adaptive interval)
        if self.egg_lay_timer_progress >= self.egg_lay_interval_ticks:
            self.egg_lay_timer_progress = 0 # Reset timer for next attempt (simple reset)
            # Optional: self.egg_lay_timer_progress %= self.egg_lay_interval_ticks # Preserve overshoot

            # --- Pre-checks before laying ---
            # 1. Check maximum ant population limit
            if len(sim.ants) >= MAX_ANTS:
                # print("DEBUG: Max ants reached. Queen skips egg laying.") # Optional debug
                return # Skip laying if colony is full

            # 2. Check if food is critically low (safety check to prevent consuming last resources)
            if sugar_very_low or protein_very_low:
                 # print("DEBUG: Queen avoids laying egg - food critically low.") # Optional Debug
                 return # Skip laying

            # 3. Check if colony has enough resources *specifically for this egg*
            needed_s = QUEEN_FOOD_PER_EGG_SUGAR
            needed_p = QUEEN_FOOD_PER_EGG_PROTEIN
            can_lay_this_egg = (sim.colony_food_storage_sugar >= needed_s and
                                sim.colony_food_storage_protein >= needed_p)

            if can_lay_this_egg:
                # --- Lay the Egg ---
                # Consume resources from colony storage
                sim.colony_food_storage_sugar -= needed_s
                sim.colony_food_storage_protein -= needed_p

                # Decide the caste of the new egg based on colony needs
                caste = self._decide_caste()

                # Find a suitable position within the nest to place the egg
                egg_pos = self._find_egg_position()

                # If a valid position was found, create and add the egg brood item
                if egg_pos:
                    current_tick_int = int(sim.ticks) # Get current integer tick
                    egg = BroodItem(BroodStage.EGG, caste, egg_pos, current_tick_int, sim)
                    sim.add_brood(egg)
                # else: # Optional debug if placing fails
                    # print("Warning: Queen could not find position to lay egg.")

            # else: # Optional debug if specific resources lacking
                # print("DEBUG: Queen cannot lay egg - insufficient resources for this egg.")
                pass

    def _decide_caste(self) -> AntCaste:
        """
        Decides whether the next egg laid should be a worker or a soldier.

        Compares the current ratio of soldiers (including developing pupae/larvae)
        in the colony to the `QUEEN_SOLDIER_RATIO_TARGET`. Aims to maintain the
        target ratio by adjusting the probability of laying a soldier egg.

        Returns:
            The chosen AntCaste (WORKER or SOLDIER) for the new egg.
        """
        sim = self.simulation
        soldier_count = 0
        worker_count = 0

        # Count existing adult ants by caste
        for ant in sim.ants:
            if ant.caste == AntCaste.SOLDIER:
                soldier_count += 1
            elif ant.caste == AntCaste.WORKER:
                worker_count += 1

        # Count developing brood (larvae and pupae) by caste
        # This anticipates future population composition.
        for brood_item in sim.brood:
            # Only count Larvae and Pupae, as Eggs haven't consumed significant resources yet
            # and their caste determination might be less critical for immediate ratio balancing.
            if brood_item.stage in [BroodStage.LARVA, BroodStage.PUPA]:
                if brood_item.caste == AntCaste.SOLDIER:
                    soldier_count += 1
                elif brood_item.caste == AntCaste.WORKER:
                    worker_count += 1

        # Calculate the current effective soldier ratio
        total_population = soldier_count + worker_count
        current_ratio = 0.0
        if total_population > 0:
            current_ratio = soldier_count / total_population

        target_ratio = QUEEN_SOLDIER_RATIO_TARGET

        # --- Determine Probability based on Ratio ---
        # If soldier ratio is significantly below target, increase chance of laying a soldier
        if current_ratio < target_ratio * 0.8: # Example: If ratio is < 80% of target
            soldier_probability = 0.65 # High probability
        # If soldier ratio is slightly below target
        elif current_ratio < target_ratio:
             soldier_probability = 0.30 # Moderate probability
        # If soldier ratio is at or above target
        else:
             soldier_probability = 0.04 # Low base probability (allows occasional soldiers even if ratio is met)

        # Make the probabilistic choice
        if random.random() < soldier_probability:
            return AntCaste.SOLDIER
        else:
            return AntCaste.WORKER

    def _find_egg_position(self) -> tuple | None:
        """
        Finds a suitable random, empty cell within the nest radius to place a new egg.

        Attempts to find a random unoccupied spot first. If that fails after
        several attempts, it falls back to checking immediate neighbors of the queen.
        Ensures the chosen spot is not an obstacle and not the queen's own position.

        Returns:
            A valid (x, y) grid coordinate tuple if a spot is found, otherwise None.
        """
        sim = self.simulation
        nest_center_int = sim.nest_pos # Use the central nest position from simulation
        nest_radius_sq = NEST_RADIUS ** 2
        max_random_attempts = 25 # Number of tries to find a random spot

        # --- Attempt 1: Find a random valid spot within the nest radius ---
        for _ in range(max_random_attempts):
            # Generate random offset relative to the nest center
            # Ensure offset stays within the square bounding the nest radius
            offset_x = rnd(-NEST_RADIUS, NEST_RADIUS)
            offset_y = rnd(-NEST_RADIUS, NEST_RADIUS)

            # Calculate potential position
            potential_pos = (nest_center_int[0] + offset_x, nest_center_int[1] + offset_y)

            # --- Check validity of the potential position ---
            # 1. Is it within the circular nest radius?
            if distance_sq(potential_pos, nest_center_int) > nest_radius_sq:
                continue # Outside circular nest area

            # 2. Is it within grid bounds and not an obstacle? (uses helper function)
            if not is_valid_pos(potential_pos, sim.grid_width, sim.grid_height) or \
               sim.grid.is_obstacle(potential_pos):
                continue # Invalid grid position or obstacle

            # 3. Is it the queen's own position?
            if potential_pos == self.pos:
                continue # Cannot place on queen's spot

            # 4. Is the spot currently occupied by other brood?
            # Check the simulation's brood position dictionary.
            if potential_pos in sim.brood_positions and sim.brood_positions[potential_pos]:
                 # Allow some stacking? For now, prefer empty spots.
                 continue # Spot occupied by other brood

            # --- Found a suitable random spot ---
            return potential_pos

        # --- Attempt 2 (Fallback): Check immediate neighbors of the Queen ---
        # This is a fallback if finding a random empty spot failed.
        # print("Warning: Queen falling back to neighbor check for egg position.") # Optional debug
        possible_spots = get_neighbors(self.pos, sim.grid_width, sim.grid_height)
        valid_neighbor_spots = []
        for spot in possible_spots:
             # Check obstacle and queen's position
             if not sim.grid.is_obstacle(spot) and spot != self.pos:
                  # Check for brood occupation
                  if not (spot in sim.brood_positions and sim.brood_positions[spot]):
                       valid_neighbor_spots.append(spot) # Add free neighbor

        if valid_neighbor_spots:
            # Choose randomly among the valid, unoccupied neighbors
            return random.choice(valid_neighbor_spots)
        # else: # If even neighbors are blocked/occupied (very unlikely)
             # print("Warning: Queen could not find any valid neighbor spot for egg.") # Optional debug


        # --- Failed to find any suitable position ---
        return None

    def take_damage(self, amount: float, attacker):
        """
        Handles the queen receiving damage from an attacker.

        Reduces the queen's HP. If the queen survives, she drops very strong
        alarm and recruitment pheromones due to the high threat level.
        If HP drops to zero, marks the queen for removal, triggering the
        end of the simulation run.

        Args:
            amount: The amount of damage received.
            attacker: The entity that dealt the damage (currently unused).
        """
        # Ignore damage if already dying or dead
        if self.is_dying or self.hp <= 0:
            return

        self.hp -= amount
        grid = self.simulation.grid # Get grid reference
        pos_int = self.pos          # Queen's position

        if self.hp > 0:
            # --- Queen Survived ---
            # Queen taking damage is a critical event, release strong signals.
            # Drop strong alarm pheromone
            grid.add_pheromone(pos_int, P_ALARM_FIGHT * 8, "alarm") # Significantly amplified signal
            # Drop strong recruitment pheromone
            grid.add_pheromone(pos_int, P_RECRUIT_DAMAGE * 8, "recruitment") # Significantly amplified signal
        else:
            # --- Queen Died ---
            self.hp = 0 # Ensure HP is not negative
            self.is_dying = True # Mark for removal
            # The actual simulation end logic is handled in AntSimulation.kill_queen
            # based on this flag or direct HP check.
            # print(f"Queen marked as dying at {self.pos}") # Optional debug

    def draw(self, surface: pygame.Surface):
        """Draws the queen ant onto the given surface, resembling other ants but larger."""
        sim = self.simulation
        cs = sim.cell_size
        # Calculate center pixel position (basis for relative positioning)
        pos_px = (int(self.pos[0] * cs + cs / 2),
                  int(self.pos[1] * cs + cs / 2))

        # --- Queen Size Factors (Analog zu Ant Attributes) ---
        # Königin ist größer, also kleinerer size_factor
        queen_size_factor = 1.3 # Kleinere Zahl = größere Ameise im Verhältnis zur Zelle
        # Kopf ist relativ zum Körper vielleicht etwas kleiner als bei Soldaten
        queen_head_size_factor = 0.35

        # --- Calculate Body Part Sizes ---
        # Overall size of the queen
        queen_size = max(3, int(cs / queen_size_factor)) # Größer als normale Ameise
        # Calculate sizes of body parts relative to queen_size
        head_size = int(queen_size * queen_head_size_factor)
        thorax_size = int(queen_size * 0.25) # Etwas größerer Thorax als Standardameise
        abdomen_size = int(queen_size * 0.8) # Deutlich größeres Abdomen

        # --- Define Fixed Orientation (Queen faces up) ---
        move_dir_x, move_dir_y = 0, -1

        # --- Calculate Body Part Positions (Relative to Thorax Center = pos_px) ---
        thorax_center = pos_px

        # Abdomen is behind (downwards)
        # Größerer Offset wegen des größeren Abdomens
        abdomen_offset = int(queen_size * 0.6) # Weiter nach hinten versetzt
        abdomen_center = (thorax_center[0] - int(move_dir_x * abdomen_offset),
                          thorax_center[1] - int(move_dir_y * abdomen_offset))

        # Head is in front (upwards)
        head_offset = int(thorax_size * 0.7) # Kopf direkt vor Thorax
        head_center = (thorax_center[0] + int(move_dir_x * head_offset),
                       thorax_center[1] + int(move_dir_y * head_offset))

        # --- Draw Abdomen (Large and Elongated) ---
        abdomen_width = max(2, int(abdomen_size * 0.7))
        abdomen_height = max(3, int(abdomen_size * 1.1)) # Etwas breiter als hoch
        abdomen_rect = pygame.Rect(abdomen_center[0] - abdomen_width // 2,
                                   abdomen_center[1] - abdomen_height // 2,
                                   abdomen_width, abdomen_height)
        # Verwende die Hauptfarbe der Königin
        pygame.draw.ellipse(surface, self.color, abdomen_rect)

        # --- Draw Thorax ---
        thorax_width = max(1, int(thorax_size * 1.1))
        thorax_height = max(1, int(thorax_size * 0.9))
        thorax_rect = pygame.Rect(thorax_center[0] - thorax_width // 2,
                                  thorax_center[1] - thorax_height // 2,
                                  thorax_width, thorax_height)
        # Etwas dunklerer Thorax? Oder gleiche Farbe. Nehmen wir etwas dunkler.
        thorax_color = tuple(max(0, c - 20) for c in self.color[:3])
        pygame.draw.ellipse(surface, thorax_color, thorax_rect)

        # --- Draw Head ---
        head_radius = max(1, int(head_size / 2))
        # Dunklerer Kopf
        head_color = tuple(max(0, c - 40) for c in self.color[:3])
        pygame.draw.circle(surface, head_color, head_center, head_radius)

        # --- Draw Antennae (wie bei Ant.draw) ---
        antenna_base_pos = head_center
        antenna_length = int(head_size * 1.3) # Längere Fühler
        antenna_angle_offset = 0.6 # Weiter gespreizt

        # Berechne Fühler-Endpunkte basierend auf Ausrichtung
        # Linker Fühler
        antenna_left_angle = math.atan2(move_dir_y, move_dir_x) + antenna_angle_offset
        antenna_left_end = (antenna_base_pos[0] + int(antenna_length * math.cos(antenna_left_angle)),
                            antenna_base_pos[1] + int(antenna_length * math.sin(antenna_left_angle)))
        # Rechter Fühler
        antenna_right_angle = math.atan2(move_dir_y, move_dir_x) - antenna_angle_offset
        antenna_right_end = (antenna_base_pos[0] + int(antenna_length * math.cos(antenna_right_angle)),
                             antenna_base_pos[1] + int(antenna_length * math.sin(antenna_right_angle)))

        # Zeichne Fühlerlinien
        antenna_color = head_color # Gleiche Farbe wie Kopf
        pygame.draw.line(surface, antenna_color, antenna_base_pos, antenna_left_end, 1)
        pygame.draw.line(surface, antenna_color, antenna_base_pos, antenna_right_end, 1)

        # --- Outline (Optional, aber gut für Sichtbarkeit) ---
        outline_color = (50, 50, 50) # Dunkelgrau
        pygame.draw.ellipse(surface, outline_color, abdomen_rect, 1)
        pygame.draw.ellipse(surface, outline_color, thorax_rect, 1)
        pygame.draw.circle(surface, outline_color, head_center, head_radius, 1)


# --- Enemy Class ---
class Enemy:
    """
    Represents an enemy entity that poses a threat to the ant colony.

    Enemies move around the grid, potentially attracted towards the nest.
    They attack nearby ants or the queen on sight. Upon death, they may leave
    behind some food resources.
    """
    def __init__(self, pos_grid: tuple, sim):
        """
        Initializes a new enemy.

        Args:
            pos_grid: The initial (x, y) grid coordinates.
            sim: Reference to the main AntSimulation object.
        """
        self.pos = tuple(map(int, pos_grid)) # Current grid coordinates (x, y)
        self.simulation = sim
        self.hp = float(ENEMY_HP)           # Current Hit Points
        self.max_hp = float(ENEMY_HP)       # Maximum Hit Points
        self.attack_power = ENEMY_ATTACK    # Damage dealt per attack
        self.color = ENEMY_COLOR

        # --- Movement ---
        # Timer determining when the next move attempt occurs.
        # Initialized randomly to stagger enemy movement.
        self.move_delay_timer = rnd(0, ENEMY_MOVE_DELAY)

        # --- State/Interaction ---
        self.is_dying = False # Flag to prevent multiple death processing

    def update(self):
        """
        Updates the enemy's state for one simulation tick.

        Handles attacking nearby ants/queen or moving.
        """
        sim = self.simulation

        # If already marked for death, do nothing further
        if self.is_dying:
            return

        # --- 1. Attack Logic ---
        current_pos_int = self.pos
        # Check neighbors (including current cell) for potential targets
        neighbors_int = get_neighbors(current_pos_int, sim.grid_width, sim.grid_height, include_center=True)
        target_ant = None    # Potential ant target
        target_queen = None  # Potential queen target

        # Scan neighbors for queen or ants
        for p_int in neighbors_int:
            # Use simulation's method which checks for both ants and queen
            entity_at_pos = sim.get_ant_at(p_int) # Returns Ant or Queen object if present
            if entity_at_pos and entity_at_pos.hp > 0:
                # Prioritize attacking the Queen
                if isinstance(entity_at_pos, Queen):
                    target_queen = entity_at_pos
                    break # Found queen, attack her immediately
                # If it's not the queen, store the first ant found
                elif target_ant is None and isinstance(entity_at_pos, Ant):
                    target_ant = entity_at_pos

        # Choose the target: Queen has highest priority
        chosen_target = target_queen if target_queen else target_ant

        # If a target was found in range, attack and skip movement for this tick
        if chosen_target:
            self.attack(chosen_target)
            # print(f"DEBUG: Enemy at {self.pos} attacks {type(chosen_target).__name__} at {chosen_target.pos}") # Optional debug
            return # Attacked, end turn

        # --- 2. Movement Logic (if no attack occurred) ---
        self.move_delay_timer -= 1 # Decrement timer by 1 each tick
        if self.move_delay_timer > 0:
            return # Not time to move yet

        # Reset timer for the next move attempt
        self.move_delay_timer += ENEMY_MOVE_DELAY

        # Find valid moves: Check neighboring cells
        possible_moves_int = get_neighbors(current_pos_int, sim.grid_width, sim.grid_height)
        valid_moves_int = []
        for m_int in possible_moves_int:
            # Check for obstacles, other enemies, ants (incl. queen), and prey
            if (not sim.grid.is_obstacle(m_int) and
                    not sim.is_enemy_at(m_int, exclude_self=self) and
                    not sim.is_ant_at(m_int) and # Checks ants and queen
                    not sim.is_prey_at(m_int)):
                valid_moves_int.append(m_int)

        # If valid moves exist, choose one
        if valid_moves_int:
            chosen_move_int = None
            nest_pos_int = sim.nest_pos # Get nest coordinates

            # --- Move Selection: Nest Attraction or Random ---
            # Small probability to move towards the nest
            if random.random() < ENEMY_NEST_ATTRACTION:
                best_nest_move = None
                min_dist_sq_to_nest = distance_sq(current_pos_int, nest_pos_int)
                # Find the valid move that gets closest to the nest
                for move in valid_moves_int:
                    d_sq = distance_sq(move, nest_pos_int)
                    if d_sq < min_dist_sq_to_nest:
                        min_dist_sq_to_nest = d_sq
                        best_nest_move = move
                # Choose the best nest-ward move if found, otherwise random among valid
                chosen_move_int = best_nest_move if best_nest_move else random.choice(valid_moves_int)
            else:
                # Default behavior: choose a random valid move
                chosen_move_int = random.choice(valid_moves_int)

            # --- Execute the move ---
            if chosen_move_int and chosen_move_int != current_pos_int:
                old_pos = self.pos
                self.pos = chosen_move_int
                # IMPORTANT: Update simulation's position tracking and spatial grid
                sim.update_entity_position(self, old_pos, self.pos)

    def attack(self, target):
        """
        Performs an attack on a target ant or queen.

        Args:
            target: The Ant or Queen object to attack.
        """
        sim = self.simulation # Get simulation instance

        # Check type and method existence for safety, and ensure target is alive
        if isinstance(target, (Ant, Queen)) and hasattr(target, 'take_damage') and target.hp > 0:
            target_pos = target.pos # Store target position for indicator
            # Call target's method to inflict damage
            target.take_damage(self.attack_power, self) # Pass self as attacker

            # Add a visual indicator for the attack event
            sim.add_attack_indicator(self.pos, target_pos, ATTACK_INDICATOR_COLOR_ENEMY)

    def take_damage(self, amount: float, attacker):
        """
        Handles the enemy receiving damage from an attacker.

        Reduces HP. If the enemy survives, it drops alarm pheromone.
        Marks the enemy for removal if HP drops to zero.

        Args:
            amount: The amount of damage received.
            attacker: The entity that dealt the damage (currently unused).
        """
         # Ignore damage if already dying or dead
        if self.is_dying or self.hp <= 0:
            return

        self.hp -= amount
        grid = self.simulation.grid # Get grid reference
        pos_int = self.pos          # Enemy's position

        if self.hp > 0:
            # --- Enemy Survived ---
            # Drop alarm pheromone when hit to signal danger nearby
            grid.add_pheromone(pos_int, P_ALARM_FIGHT, "alarm")
        else:
            # --- Enemy Died ---
            self.hp = 0 # Ensure HP is not negative
            self.is_dying = True # Mark for removal in the main simulation loop
            # print(f"Enemy marked as dying at {self.pos}") # Optional debug

    def draw(self, surface: pygame.Surface):
        """
        Draws the enemy onto the specified Pygame surface as a spider-like shape.

        Args:
            surface: The Pygame surface to draw on.
        """
        sim = self.simulation
        # Basic validation
        if not is_valid_pos(self.pos, sim.grid_width, sim.grid_height):
            return

        cs = sim.cell_size
        # Calculate center pixel position for drawing
        pos_px = (int(self.pos[0] * cs + cs / 2),
                  int(self.pos[1] * cs + cs / 2))

        # --- Body (Slightly Elongated Ellipse) ---
        body_width = max(2, int(cs * 0.45))
        body_height = max(3, int(cs * 0.6))
        body_rect = pygame.Rect(pos_px[0] - body_width // 2,
                                pos_px[1] - body_height // 2,
                                body_width, body_height)
        pygame.draw.ellipse(surface, self.color, body_rect)

        # --- Legs (8 Thin Lines) ---
        num_legs = 8
        leg_length = cs * 0.45 # Adjusted leg length slightly
        leg_color = tuple(max(0, c - 60) for c in self.color) # Darker color
        leg_thickness = max(1, cs // 10) # Scale thickness slightly with cell size

        # Angles for legs, distributed around the body
        # Slightly irregular spacing for a more organic look
        base_angles = [
            math.pi * 0.20, math.pi * 0.45, math.pi * 0.55, math.pi * 0.80, # One side (top-right quadrant first)
            math.pi * 1.20, math.pi * 1.45, math.pi * 1.55, math.pi * 1.80  # Other side (bottom-left quadrant last)
        ]

        # Offset attach points slightly towards the center top/bottom for visual appeal
        attach_offset_y = body_height * 0.15

        for i, angle in enumerate(base_angles):
            # Alternate attach points vertically for better visual separation
            attach_y = pos_px[1] - attach_offset_y if i < 4 else pos_px[1] + attach_offset_y
            attach_point = (pos_px[0], attach_y)

            # Add slight randomness to angle for less rigid appearance
            angle += rnd_uniform(-0.08, 0.08)

            # Calculate end point of the leg
            end_x = attach_point[0] + leg_length * math.cos(angle)
            end_y = attach_point[1] + leg_length * math.sin(angle)

            # Draw the leg line
            pygame.draw.line(surface, leg_color, attach_point, (int(end_x), int(end_y)), leg_thickness)

        # --- Outline the Body ---
        pygame.draw.ellipse(surface, (0, 0, 0), body_rect, 1) # Black 1px outline

# --- Main Simulation Class ---

class AntSimulation:
    """
    Manages the overall ant colony simulation.

    Handles initialization, the main game loop, entity management (ants, queen,
    enemies, prey, brood), grid updates (food, pheromones), user input,
    drawing, UI elements (buttons, debug info, legend), and optional network
    streaming.
    """

    def __init__(self, log_filename=None):
        """
        Initializes the Ant Simulation environment.

        Sets up Pygame, calculates screen and grid dimensions, initializes fonts,
        creates simulation objects (grid, entities), prepares UI elements, and
        starts the simulation state.

        Args:
            log_filename (str | None): The path to the CSV file for logging
                                       simulation statistics. If None or if
                                       ENABLE_SIMULATION_LOGGING is False,
                                       logging is disabled.
        """
        print("AntSimulation: Initializing...")
        self.app_running = True          # Flag controlling the main application loop
        self.simulation_running = False  # Flag controlling the active simulation run loop

        # --- Logging Setup ---
        self.log_filename = log_filename if ENABLE_SIMULATION_LOGGING else None
        self.logging_interval = LOGGING_INTERVAL_TICKS # Ticks between log entries
        print(f"AntSimulation: Logging {'enabled (' + str(self.log_filename) + ')' if self.log_filename else 'disabled'}.")

        # --- Pygame and Display Initialization ---
        print("AntSimulation: Initializing Pygame and display...")
        try:
            pygame.init()
            if not pygame.display.get_init():
                raise RuntimeError("Pygame display module failed to initialize.")
            # Attempt to center the game window on the screen
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            pygame.display.set_caption("Ant Simulation")
            print("AntSimulation: Pygame initialized.")

            # Determine Screen/Window Size based on configuration constants
            screen_info = pygame.display.Info()
            monitor_width, monitor_height = screen_info.current_w, screen_info.current_h
            print(f"AntSimulation: Detected Monitor Size: {monitor_width}x{monitor_height}")

            # Choose display mode: Fullscreen or Windowed
            display_flags = pygame.DOUBLEBUF | pygame.HWSURFACE # Hardware acceleration flags
            if USE_FULLSCREEN:
                self.screen_width = monitor_width
                self.screen_height = monitor_height
                display_flags |= pygame.FULLSCREEN
                print("AntSimulation: Using Fullscreen Mode.")
            else:
                # Calculate window size (default, percentage, or capped by monitor)
                target_w, target_h = DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
                if USE_SCREEN_PERCENT is not None and 0.1 <= USE_SCREEN_PERCENT <= 1.0:
                    target_w = int(monitor_width * USE_SCREEN_PERCENT)
                    target_h = int(monitor_height * USE_SCREEN_PERCENT)
                    print(f"AntSimulation: Using {USE_SCREEN_PERCENT*100:.0f}% of screen: {target_w}x{target_h}")
                else:
                    print(f"AntSimulation: Using Default Window Size: {target_w}x{target_h}")
                # Ensure window dimensions do not exceed monitor dimensions
                self.screen_width = min(target_w, monitor_width)
                self.screen_height = min(target_h, monitor_height)
                print("AntSimulation: Using Windowed Mode.")

            # Set the Pygame display mode
            print("AntSimulation: Setting display mode...")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), display_flags)
            actual_width, actual_height = self.screen.get_size()
            print(f"AntSimulation: Display Mode Set: {actual_width}x{actual_height}")
            # Update dimensions if Pygame adjusted them (e.g., under constraints)
            self.screen_width, self.screen_height = actual_width, actual_height

        except Exception as e:
            print(f"FATAL: Pygame/Display initialization failed: {e}")
            traceback.print_exc()
            self.app_running = False # Prevent further initialization or running
            print("AntSimulation: Initialization aborted (Pygame/Display Error).")
            return # Cannot continue

        # --- Calculate Grid Dimensions based on Screen and Cell Size ---
        print("AntSimulation: Calculating grid dimensions...")
        self.cell_size = CELL_SIZE
        # Ensure grid dimensions are at least 1x1
        self.grid_width = max(1, self.screen_width // self.cell_size)
        self.grid_height = max(1, self.screen_height // self.cell_size)
        # Calculate the pixel dimensions covered exactly by the grid
        self.world_width_px = self.grid_width * self.cell_size
        self.world_height_px = self.grid_height * self.cell_size

        # Adjust screen surface size if calculated grid is smaller than requested window
        # This ensures the drawn area matches the grid dimensions perfectly.
        if self.world_width_px < self.screen_width or self.world_height_px < self.screen_height:
            print(f"AntSimulation: Adjusting screen surface to grid dimensions: {self.world_width_px}x{self.world_height_px}")
            try:
                self.screen = pygame.display.set_mode((self.world_width_px, self.world_height_px), display_flags)
                # Update screen dimensions after resizing
                self.screen_width, self.screen_height = self.world_width_px, self.world_height_px
            except Exception as e:
                print(f"FATAL: Failed to resize screen to grid dimensions: {e}")
                self.app_running = False
                print("AntSimulation: Initialization aborted (Screen Resize Error).")
                return

        # --- Calculate Nest Position (Center of Grid) ---
        self.nest_pos = (self.grid_width // 2, self.grid_height // 2)
        print(f"AntSimulation: Grid={self.grid_width}x{self.grid_height}, Cell={self.cell_size}px, Nest={self.nest_pos}")

        # --- Initialize Fonts (Scaled based on screen height) ---
        print("AntSimulation: Initializing fonts...")
        self.font = None
        self.debug_font = None
        self.legend_font = None
        self._init_fonts() # Method handles font loading and scaling
        # _init_fonts sets self.app_running to False on failure
        if not self.app_running:
            print("AntSimulation: Font initialization failed.")
            print("AntSimulation: Initialization aborted (Font Error).")
            return # Cannot continue without fonts
        print("AntSimulation: Fonts initialized.")

        # --- Simulation State Variables ---
        print("AntSimulation: Initializing simulation state variables...")
        self.clock = pygame.time.Clock() # Pygame clock for FPS control
        # Target FPS for simulation updates and rendering << REMOVED Speed Multipliers
        self.target_fps = TARGET_FPS
        self.grid = WorldGrid(self.grid_width, self.grid_height) # World grid instance
        self.end_game_reason = ""       # Stores reason for simulation end (e.g., "Queen died")
        self.colony_generation = 0      # Counter for simulation restarts
        self.ticks = 0                  # Simulation time steps elapsed (integer) << CHANGED to int

        # Soldier patrol radius (squared for efficient distance checks)
        self.soldier_patrol_radius_sq = (NEST_RADIUS * SOLDIER_PATROL_RADIUS_MULTIPLIER)**2

        # --- Entity Management ---
        self.ants: list[Ant] = []       # List holding all active Ant objects
        self.enemies: list[Enemy] = []  # List holding all active Enemy objects
        self.brood: list[BroodItem] = [] # List holding all active BroodItem objects
        self.prey: list[Prey] = []      # List holding all active Prey objects
        self.queen: Queen | None = None # Holds the single Queen object

        # --- Position Lookups (Dictionaries for fast entity checking at a grid cell) ---
        # Key: (x, y) grid tuple, Value: Ant/Enemy/Prey object
        self.ant_positions: dict[tuple, Ant] = {} # Includes Queen if checking this dict directly
        self.enemy_positions: dict[tuple, Enemy] = {}
        self.prey_positions: dict[tuple, Prey] = {}
        # Key: (x, y) grid tuple, Value: List of BroodItem objects at that cell
        self.brood_positions: dict[tuple, list[BroodItem]] = {}

        # --- Optimized Ant Position Tracking (for potential future use with NumPy) ---
        # Array to store ant positions directly, indexed by ant.index
        self.ant_positions_array = np.full((MAX_ANTS, 2), -1, dtype=np.int16)
        # Dictionary mapping Ant object instances to their index in the array
        self.ant_indices: dict[Ant, int] = {}
        # Counter for assigning the next available index in the array
        self.next_ant_index = 0

        # List to store recent attack events for visual indicators
        self.recent_attacks: list[dict] = []

        # --- Colony Resources ---
        self.colony_food_storage_sugar = 0.0
        self.colony_food_storage_protein = 0.0

        # --- Timers (Tick-based) ---
        self.enemy_spawn_timer = 0            # Ticks accumulated towards next enemy spawn
        self.enemy_spawn_interval = ENEMY_SPAWN_RATE # Ticks between spawn attempts
        self.prey_spawn_timer = 0             # Ticks accumulated towards next prey spawn
        self.prey_spawn_interval = PREY_SPAWN_RATE   # Ticks between spawn attempts
        self.food_replenish_timer = 0         # Ticks accumulated towards next food replenish
        # Replenish rate adjusted dynamically based on map size during reset
        self.food_replenish_interval = FOOD_REPLENISH_RATE

        print("AntSimulation: Simulation state variables initialized.")

        # --- UI State ---
        print("AntSimulation: Initializing UI state...")
        self.show_debug_info = True       # Flag to display debug overlay
        self.show_legend = False          # Flag to display legend overlay
        self.show_pheromones = False      # Flag to render pheromone layer
        self.buttons = self._create_buttons() # Create UI button objects
        print("AntSimulation: UI state initialized.")

        # --- Drawing Surfaces & Caches ---
        print("AntSimulation: Initializing drawing surfaces...")
        # Surface for static elements (background color, obstacles) - drawn once per reset
        self.static_background_surface = pygame.Surface((self.world_width_px, self.world_height_px))
        # Surface for caching the pheromone layer - updated periodically
        self.pheromone_surface = pygame.Surface((self.world_width_px, self.world_height_px), pygame.SRCALPHA)
        self.pheromone_surface.fill((0, 0, 0, 0)) # Start transparent
        # Tick counter for throttling pheromone surface updates
        self.last_pheromone_update_tick = -100 # Initialize to force first update

        # Surface to hold the latest complete frame for network streaming
        self.latest_frame_surface: pygame.Surface | None = None
        # Dedicated RNG for consistent food dot patterns across frames
        self.food_dot_rng = random.Random()
        print("AntSimulation: Drawing surfaces initialized.")

        # --- Spatial Grid for Collision/Proximity Detection ---
        print("AntSimulation: Initializing spatial grid...")
        self.spatial_grid = SpatialGrid(self.world_width_px, self.world_height_px, self.cell_size)
        print("AntSimulation: Spatial grid initialized.")

        # --- Prepare Static Background ---
        # This draws the initial obstacles onto the static surface. Needs to be done
        # after grid dimensions are known but before the first draw call.
        print("AntSimulation: Preparing static background...")
        self._prepare_static_background()
        print("AntSimulation: Static background prepared.")


        # --- Start Optional Network Stream ---
        print("AntSimulation: Starting streaming server (if enabled)...")
        self._start_streaming_server_if_enabled()
        print("AntSimulation: Streaming server handled.")

        # --- Initial Simulation Reset ---
        # This populates the simulation with the queen, initial ants, food etc.
        # and sets `self.simulation_running` to True if successful.
        print("AntSimulation: Performing initial simulation reset...")
        if self.app_running: # Only reset if initialization hasn't failed already
            self._reset_simulation()
            if not self.simulation_running:
                print("ERROR: Simulation reset failed to start simulation.")
                self.app_running = False # Ensure app doesn't proceed if reset fails
            else:
                print("AntSimulation: Simulation reset successful.")
        else:
            print("AntSimulation: Skipping simulation reset because app_running was already False.")

        print(f"AntSimulation: Initialization complete. App Running: {self.app_running}, Sim Running: {self.simulation_running}")

    def _init_fonts(self):
        """
        Initializes Pygame fonts used for UI text (debug info, buttons, legend).

        Scales font sizes based on the screen height relative to a reference height
        (`REFERENCE_HEIGHT_FOR_SCALING`) to maintain readability across different
        resolutions. Attempts to load system fonts first, falling back to Pygame's
        default font if necessary. Sets `self.app_running` to False if font
        initialization fails.
        """
        # Check if the Pygame font module is available
        if not pygame.font.get_init():
            print("FATAL: Pygame font module not initialized.")
            self.app_running = False # Cannot run without fonts
            return

        # Calculate scaling factor based on current height vs reference height
        # Ensures fonts scale reasonably with window size.
        scale_factor = self.screen_height / REFERENCE_HEIGHT_FOR_SCALING
        # Clamp the scale factor to prevent excessively large or small fonts
        scale_factor = max(0.7, min(1.5, scale_factor)) # Adjust clamps as needed

        # Calculate scaled font sizes (ensure minimum size of 8 pixels)
        font_size = max(8, int(BASE_FONT_SIZE * scale_factor))
        debug_font_size = max(8, int(BASE_DEBUG_FONT_SIZE * scale_factor))
        legend_font_size = max(8, int(BASE_LEGEND_FONT_SIZE * scale_factor))
        print(f"AntSimulation: Font scaling factor: {scale_factor:.2f} -> Sizes: Main={font_size}, Debug={debug_font_size}, Legend={legend_font_size}")

        try:
            # Attempt to load preferred system fonts first
            try:
                # Use common sans-serif for general UI and monospace for debug info
                self.font = pygame.font.SysFont("sans", font_size)
                self.debug_font = pygame.font.SysFont("monospace", debug_font_size)
                self.legend_font = pygame.font.SysFont("sans", legend_font_size)
                print("AntSimulation: Using scaled system 'sans' and 'monospace' fonts.")
            except Exception:
                # Fallback if system fonts are not found or fail to load
                print("AntSimulation: System fonts not found or failed. Trying default font.")
                # Use Pygame's built-in default font with the calculated scaled sizes
                self.font = pygame.font.Font(None, font_size)
                self.debug_font = pygame.font.Font(None, debug_font_size)
                self.legend_font = pygame.font.Font(None, legend_font_size)
                print("AntSimulation: Using scaled Pygame default font.")

            # Final check: Ensure all required fonts were successfully loaded
            if not self.font or not self.debug_font or not self.legend_font:
                raise RuntimeError("Font loading failed even with fallback.")

        except Exception as e:
            # Catch any error during font loading or initialization
            print(f"FATAL: Font initialization failed: {e}. Cannot render text.")
            self.font = None
            self.debug_font = None
            self.legend_font = None
            self.app_running = False # Critical failure, cannot continue

    def _start_streaming_server_if_enabled(self):
        """
        Starts the Flask network streaming server in a separate thread if enabled.

        Configures Flask routes for serving the HTML viewer page and the MJPEG
        video stream. Creates and starts a daemon thread to run the Flask server,
        allowing the main simulation loop to continue unblocked.
        """
        global streaming_app, streaming_thread, stop_streaming_event # Access global vars

        # Check if streaming is enabled in config and if Flask library is available
        if not ENABLE_NETWORK_STREAM or not Flask:
            if ENABLE_NETWORK_STREAM and not Flask:
                print("WARNING: ENABLE_NETWORK_STREAM=True, but Flask library not found. Streaming disabled.")
            return # Do nothing if disabled or Flask not installed

        print("AntSimulation: Setting up Flask streaming server...")
        # --- Flask App Setup (defined within method to access simulation dimensions) ---
        streaming_app = Flask(__name__) # Create Flask app instance
        stop_streaming_event.clear() # Ensure the stop event is clear before starting

        # Capture current simulation world dimensions for the HTML template context
        # These dimensions tell the browser how large the video feed image should be.
        template_width = self.world_width_px
        template_height = self.world_height_px

        # --- Define Flask Routes ---
        @streaming_app.route('/')
        def index():
            """Serves the simple HTML page that displays the video stream."""
            # Pass the captured world dimensions to the HTML template
            return render_template_string(HTML_TEMPLATE, width=template_width, height=template_height)

        @streaming_app.route('/video_feed')
        def video_feed():
            """Route that provides the Motion JPEG (MJPEG) video stream."""
            # Uses the `stream_frames` generator function to yield individual JPEG frames.
            # The 'multipart/x-mixed-replace' mimetype allows the browser to continuously
            # update the image element with new frames received from the stream.
            return Response(stream_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        # --- End Flask App Setup ---

        # --- Start Flask Server in a Daemon Thread ---
        print(f"AntSimulation: Starting streaming thread (Host: {STREAMING_HOST}, Port: {STREAMING_PORT})...")
        # Create the thread targeting the `run_server` function.
        streaming_thread = threading.Thread(
            target=run_server,
            args=(streaming_app, STREAMING_HOST, STREAMING_PORT),
            # Set as daemon thread: ensures the thread exits automatically when the main program exits,
            # even if the server hasn't fully shut down internally.
            daemon=True
        )
        streaming_thread.start() # Start the server thread

    def _stop_streaming_server(self):
        """
        Signals the Flask streaming server thread to stop.

        Sets the `stop_streaming_event` which is checked by the `stream_frames`
        generator and potentially by the `run_server` loop (though clean Flask
        shutdown from another thread is complex). Attempts to join the thread
        briefly to allow for cleanup.
        """
        global streaming_thread, stop_streaming_event # Access global vars

        # Check if the streaming thread exists and is currently running
        if streaming_thread and streaming_thread.is_alive():
            print("AntSimulation: Stopping Flask server thread...")
            # Signal the stream_frames generator and potentially run_server to stop
            stop_streaming_event.set()

            # Attempt to wait for the thread to finish.
            # Stopping a Flask server cleanly from another thread can be tricky.
            # The daemon flag ensures it exits with the main app, but join()
            # gives it a chance to shut down more gracefully if possible.
            # A short timeout prevents blocking the main thread indefinitely.
            streaming_thread.join(timeout=1.0) # Wait up to 1 second

            if streaming_thread.is_alive():
                 print("AntSimulation: Warning - Streaming thread did not stop within timeout.")
            else:
                 print("AntSimulation: Streaming thread stopped.")
        # Clear globals for safety, though daemon thread exit should handle this
        streaming_thread = None
        streaming_app = None

    def _reset_simulation(self):
        """
        Resets the simulation state to start a new colony generation.

        Clears all existing entities (ants, enemies, brood, prey), resets colony
        resources, timers, and the world grid (placing new obstacles and food).
        Spawns the initial queen and ants. Sets the simulation state to running.
        """
        self.colony_generation += 1
        print(f"\nAntSimulation: Resetting simulation (Generation {self.colony_generation})...")

        # --- Reset Core Simulation State ---
        self.ticks = 0                  # Reset simulation time to zero
        self.end_game_reason = ""       # Clear previous end reason

        # --- Clear Entity Lists and Lookups ---
        # Clear lists first
        self.ants.clear()
        self.enemies.clear()
        self.brood.clear()
        self.prey.clear()
        # Clear position lookup dictionaries
        self.ant_positions.clear()
        self.enemy_positions.clear()
        self.prey_positions.clear()
        self.brood_positions.clear()
        # Clear optimized ant position tracking
        self.ant_positions_array.fill(-1) # Reset array values
        self.ant_indices.clear()
        self.next_ant_index = 0
        # Clear queen reference
        self.queen = None
        # Clear attack indicators
        self.recent_attacks.clear()
        # Clear spatial grid (implicitly handled by removing/re-adding entities)
        self.spatial_grid.grid.clear() # Explicitly clear the grid dict


        # --- Reset Colony Resources ---
        self.colony_food_storage_sugar = INITIAL_COLONY_FOOD_SUGAR
        self.colony_food_storage_protein = INITIAL_COLONY_FOOD_PROTEIN

        # --- Reset Timers ---
        self.enemy_spawn_timer = 0
        self.prey_spawn_timer = 0
        self.food_replenish_timer = 0

        # --- Reset and Regenerate World Grid ---
        # This places new obstacles and initial food clusters relative to the nest.
        self.grid.reset(self.nest_pos) # Pass nest position for placement logic
        print("AntSimulation: World grid reset.")

        # --- Redraw Static Background ---
        # Obstacles might have changed, so redraw them onto the background surface.
        self._prepare_static_background()
        print("AntSimulation: Static background redrawn.")

        # --- Reset Pheromone Cache Surface ---
        # Clear the cached surface to prevent showing old pheromones.
        self.pheromone_surface.fill((0, 0, 0, 0)) # Fill with transparent
        self.last_pheromone_update_tick = -100 # Force redraw on the first relevant frame
        print("AntSimulation: Pheromone cache cleared.")

        # --- Spawn Initial Entities ---
        # Place the queen and initial workers/soldiers.
        print("AntSimulation: Spawning initial entities...")
        if not self._spawn_initial_entities():
            # Critical error if initial placement fails (e.g., cannot place queen)
            print("CRITICAL ERROR during simulation reset (entity spawn). Cannot continue.")
            self.simulation_running = False
            self.app_running = False # Signal main app loop to stop too
            self.end_game_reason = "Initial Spawn Error"
            return # Abort reset
        print("AntSimulation: Initial entities spawned.")

        # --- Adjust Timed Event Intervals (Optional, based on map size) ---
        # Example: Scale food replenishment interval based on grid area relative to default size.
        # Larger maps might replenish slightly less often.
        try:
            map_area_factor = math.sqrt(self.grid_width * self.grid_height) / math.sqrt(150 * 80) # Example default size
            self.food_replenish_interval = int(FOOD_REPLENISH_RATE * map_area_factor)
            # Apply similar scaling to enemy/prey spawns if desired
            self.enemy_spawn_interval = int(ENEMY_SPAWN_RATE * map_area_factor)
            self.prey_spawn_interval = int(PREY_SPAWN_RATE * map_area_factor)
            print(f"AntSimulation: Intervals adjusted for map size (Factor: {map_area_factor:.2f})")
            print(f"  Food Replenish: {self.food_replenish_interval} ticks")
            print(f"  Enemy Spawn: {self.enemy_spawn_interval} ticks")
            print(f"  Prey Spawn: {self.prey_spawn_interval} ticks")
        except Exception as e:
             print(f"Warning: Could not scale intervals by map size: {e}")
             self.food_replenish_interval = FOOD_REPLENISH_RATE # Use default
             self.enemy_spawn_interval = ENEMY_SPAWN_RATE
             self.prey_spawn_interval = PREY_SPAWN_RATE

        # --- Start the Simulation ---
        # Set the flag to indicate the simulation run is active.
        self.simulation_running = True
        print(f"AntSimulation: Generation {self.colony_generation} started (Target FPS: {self.target_fps}).")

    def _prepare_static_background(self):
        """
        Draws static elements (background color, obstacles) onto a dedicated surface.

        This surface (`self.static_background_surface`) is blitted onto the main
        screen at the beginning of each draw cycle, avoiding the need to redraw
        static obstacles every frame. Obstacles are drawn with slight color
        variation for a more textured look.
        """
        # 1. Fill the surface with the base map background color
        self.static_background_surface.fill(MAP_BG_COLOR)

        # 2. Draw Obstacles
        cs = self.cell_size # Local alias for cell size

        # Get coordinates where obstacles exist from the boolean grid
        # `np.argwhere` returns an array of [row, col] indices where the condition is true.
        # Note: NumPy array indexing might be (row, col), but our grid logic uses (x, y) / (col, row).
        # Ensure consistency: `self.obstacles` is indexed [x, y] or [col, row]. Argwhere on a
        # (width, height) array will give [[x1, y1], [x2, y2], ...].
        obstacle_coords = np.argwhere(self.grid.obstacles)

        # Define base obstacle color and variation range
        base_r, base_g, base_b = OBSTACLE_COLOR
        var = OBSTACLE_COLOR_VARIATION

        # Iterate through each obstacle cell coordinate
        drawn_count = 0
        for x, y in obstacle_coords:
            # Calculate a slightly varied color for this specific obstacle cell
            r = max(0, min(255, base_r + rnd(-var, var))) # Clamp color values
            g = max(0, min(255, base_g + rnd(-var, var)))
            b = max(0, min(255, base_b + rnd(-var, var)))
            cell_color = (r, g, b)

            # Calculate the pixel rectangle for this grid cell
            rect = (x * cs, y * cs, cs, cs)

            # Draw the rectangle onto the static background surface
            try:
                 pygame.draw.rect(self.static_background_surface, cell_color, rect)
                 drawn_count += 1
            except Exception as e:
                 print(f"Error drawing obstacle rect at {(x, y)}: {e}") # Log error if drawing fails

        print(f"AntSimulation: Prepared static background with {drawn_count} obstacle cells.")

    def _create_buttons(self) -> list[dict]:
        """
        Creates the dictionary representations for UI buttons.

        Calculates button positions based on screen width to center them horizontally
        near the top of the screen. Defines text, action strings, and optional
        keyboard shortcuts for each button.

        Returns:
            A list of dictionaries, where each dictionary represents a button
            and contains its 'rect' (pygame.Rect), 'text', 'action' string,
            and 'key' (pygame key constant or None). Returns an empty list if
            fonts are not initialized.
        """
        buttons = []
        # Cannot create buttons without a loaded font for size calculation and rendering
        if not self.font:
            print("Warning: Cannot create buttons - font not loaded.")
            return buttons

        # Define button dimensions and spacing
        # Scale height based on the main UI font size, ensure minimum size
        button_h = max(20, int(self.font.get_height() * 1.5))
        # Set width relative to height for a reasonable aspect ratio
        button_w = max(60, int(button_h * 3.5))
        margin = 5 # Pixel margin between adjacent buttons

        # Define button properties: text displayed, action string triggered on click,
        # and optional associated keyboard shortcut.
        # << REMOVED Speed Buttons >>
        button_definitions = [
            {"text": "Stats", "action": "toggle_debug", "key": pygame.K_d},
            {"text": "Legend", "action": "toggle_legend", "key": pygame.K_l},
            {"text": "Pheromones", "action": "toggle_pheromones", "key": pygame.K_p},
            {"text": "Restart", "action": "restart", "key": pygame.K_r}, # Added R key
            {"text": "Quit", "action": "quit", "key": pygame.K_ESCAPE},
        ]

        # Calculate total width required for all buttons and margins between them
        num_buttons = len(button_definitions)
        total_buttons_width = num_buttons * button_w + (num_buttons - 1) * margin

        # Calculate starting X position to center the row of buttons horizontally
        # Ensure world_width_px is used, as screen_width might be larger if adjusted.
        start_x = (self.world_width_px - total_buttons_width) // 2
        # Place buttons near the top margin
        start_y = margin

        # Create button dictionaries with calculated positions
        for i, button_def in enumerate(button_definitions):
            # Calculate X position for the current button
            button_x = start_x + i * (button_w + margin)
            # Create the Pygame Rect object for position and collision detection
            rect = pygame.Rect(button_x, start_y, button_w, button_h)
            # Append the button dictionary to the list
            buttons.append({
                "rect": rect,
                "text": button_def["text"],
                "action": button_def["action"],
                "key": button_def.get("key", None) # Use .get() for safety if "key" is missing
            })
            # print(f"  Button '{button_def['text']}' created at {rect}") # Debug

        print(f"AntSimulation: Created {len(buttons)} UI buttons.")
        return buttons

    def _spawn_initial_entities(self) -> bool:
        """
        Spawns the initial set of entities for a new simulation run.

        Places the queen, initial ants around her, and initial enemies and prey
        at appropriate locations on the grid, avoiding obstacles and ensuring
        valid starting positions.

        Returns:
            True if all critical initial entities (especially the Queen) could be
            placed successfully, False otherwise.
        """
        print("AntSimulation: Spawning initial entities...")

        # --- 1. Spawn the Queen ---
        # Find a valid spot near the calculated nest center. This is critical.
        queen_pos = self._find_valid_queen_pos()
        if queen_pos:
            self.queen = Queen(queen_pos, self)
            # Manually add queen to ant_positions for consistency in checks like is_ant_at
            # Queen is not added to self.ants list or spatial grid generally.
            self.ant_positions[self.queen.pos] = self.queen
            print(f"  Queen placed successfully at {self.queen.pos}.")
        else:
            # If the queen cannot be placed, the simulation cannot proceed.
            print("CRITICAL ERROR: Cannot place Queen near nest center. Aborting spawn.")
            return False

        # --- 2. Spawn Initial Ants near the Queen ---
        spawned_ants = 0
        max_ant_spawn_attempts = INITIAL_ANTS * 25 # Allow many attempts per ant

        for _ in range(max_ant_spawn_attempts):
            # Stop if the required number of ants have been spawned
            if spawned_ants >= INITIAL_ANTS:
                break

            # Attempt to find a position near the queen
            # Spawn within a radius slightly larger than the nest radius
            radius_offset = NEST_RADIUS + 1
            angle = rnd_uniform(0, 2 * math.pi)
            dist = rnd_uniform(0, radius_offset)
            # Calculate offset position relative to queen's position
            try:
                 ox = int(dist * math.cos(angle))
                 oy = int(dist * math.sin(angle))
                 pos_grid = (self.queen.pos[0] + ox, self.queen.pos[1] + oy)
            except TypeError: # Queen's pos might be invalid if init failed, though caught earlier
                 print("Error calculating initial ant position - queen position invalid?")
                 continue # Skip this attempt

            # Decide caste based on initial target ratio
            caste = AntCaste.SOLDIER if random.random() < QUEEN_SOLDIER_RATIO_TARGET else AntCaste.WORKER

            # Use the simulation's add_ant method, which handles validation and adding to structures
            if self.add_ant(pos_grid, caste):
                spawned_ants += 1
            # No explicit 'attempts += 1' needed, loop runs max_ant_spawn_attempts total tries

        if spawned_ants < INITIAL_ANTS:
            print(f"  Warning: Spawned only {spawned_ants}/{INITIAL_ANTS} initial ants after {max_ant_spawn_attempts} attempts.")
        else:
            print(f"  Spawned {spawned_ants} initial ants successfully.")

        # --- 3. Spawn Initial Enemies ---
        enemies_spawned = 0
        # Spawn one enemy at a time using the dedicated spawn method
        for _ in range(INITIAL_ENEMIES):
            if self.spawn_enemy(): # spawn_enemy handles placement logic and adding
                enemies_spawned += 1

        if enemies_spawned < INITIAL_ENEMIES:
            print(f"  Warning: Spawned only {enemies_spawned}/{INITIAL_ENEMIES} initial enemies.")
        else:
            print(f"  Spawned {enemies_spawned} initial enemies successfully.")

        # --- 4. Spawn Initial Prey ---
        prey_spawned = 0
        # Spawn one prey item at a time using the dedicated spawn method
        for _ in range(INITIAL_PREY):
            if self.spawn_prey(): # spawn_prey handles placement logic and adding
                prey_spawned += 1

        if prey_spawned < INITIAL_PREY:
            print(f"  Warning: Spawned only {prey_spawned}/{INITIAL_PREY} initial prey.")
        else:
            print(f"  Spawned {prey_spawned} initial prey successfully.")

        print("AntSimulation: Initial entity spawning complete.")
        # Return True assuming queen placement succeeded (otherwise returned False earlier)
        return True

    def _find_valid_queen_pos(self) -> tuple | None:
        """
        Finds a suitable, non-obstacle position for the Queen near the nest center.

        Starts by checking the exact nest center. If blocked, it checks immediate
        neighbors, then expands the search outwards in rings until a valid spot
        is found or a maximum search radius is exceeded.

        Returns:
            An (x, y) grid coordinate tuple if a valid position is found,
            otherwise None.
        """
        base_pos_int = self.nest_pos # Start at the ideal center

        # --- Check 1: Exact Nest Center ---
        # Check if the center is within bounds (should be) and not an obstacle
        if is_valid_pos(base_pos_int, self.grid_width, self.grid_height) and \
           not self.grid.is_obstacle(base_pos_int):
            # print(f"  Queen position found at exact nest center: {base_pos_int}") # Debug
            return base_pos_int

        # --- Check 2: Immediate Neighbors ---
        # print(f"  Nest center {base_pos_int} is blocked. Checking neighbors...") # Debug
        neighbors = get_neighbors(base_pos_int, self.grid_width, self.grid_height)
        random.shuffle(neighbors) # Check neighbors in random order
        for p_int in neighbors:
            # Check only for obstacles, assuming neighbors are within bounds (checked by get_neighbors)
            if not self.grid.is_obstacle(p_int):
                # print(f"  Queen position found at neighbor: {p_int}") # Debug
                return p_int # Found a valid neighbor

        # --- Check 3: Expanding Search Rings ---
        # If immediate neighbors are also blocked, search in rings further out.
        max_search_radius = 5 # Limit how far out to search
        # print(f"  Neighbors also blocked. Searching outwards (max radius {max_search_radius})...") # Debug
        for r in range(2, max_search_radius + 1):
            # print(f"    Checking ring radius {r}...") # Debug
            perimeter_cells = []
            # Iterate over the bounding box of the ring
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # Check if the point is on the perimeter of the square defined by radius r
                    if abs(dx) == r or abs(dy) == r:
                        # Calculate the potential position
                        p_int = (base_pos_int[0] + dx, base_pos_int[1] + dy)
                        # Check if the position is valid (within grid) and not an obstacle
                        if is_valid_pos(p_int, self.grid_width, self.grid_height) and \
                           not self.grid.is_obstacle(p_int):
                            perimeter_cells.append(p_int)

            # If valid spots were found in this ring, choose one randomly and return
            if perimeter_cells:
                chosen_pos = random.choice(perimeter_cells)
                # print(f"  Queen position found at radius {r}: {chosen_pos}") # Debug
                return chosen_pos

        # --- Failure Case ---
        # If no valid spot is found within the maximum search radius
        print(f"CRITICAL: Could not find any valid spot for Queen within radius {max_search_radius} of nest center {base_pos_int}.")
        return None

    # --- Entity Management Methods ---

    def add_entity(self, entity, entity_list: list, position_dict: dict, add_to_spatial_grid: bool = True):
        """
        Generic helper method to add an entity to the simulation lists and lookups.

        Checks if the entity's position is valid (within bounds, not an obstacle,
        not occupied by critical entities like the queen or potentially other types).
        Adds the entity to the specified list, position dictionary, and optionally
        to the spatial grid.

        Args:
            entity: The entity object to add (e.g., Ant, Enemy, Prey, BroodItem).
                    Must have a `pos` attribute (tuple grid coordinates).
            entity_list: The main list for this entity type (e.g., self.ants).
            position_dict: The dictionary for position lookup (e.g., self.ant_positions).
                           For Brood, this should be self.brood_positions (handles lists).
            add_to_spatial_grid: If True, add the entity to the spatial grid
                                 for proximity checks. Defaults to True.

        Returns:
            True if the entity was successfully added, False otherwise.
        """
        # Ensure position is integer tuple
        pos_int = tuple(map(int, entity.pos))

        # --- Basic Validity Checks ---
        if not is_valid_pos(pos_int, self.grid_width, self.grid_height):
            # print(f"Debug: Add entity failed - invalid position {pos_int} for {type(entity).__name__}") # Optional debug
            return False
        if self.grid.is_obstacle(pos_int):
            # print(f"Debug: Add entity failed - obstacle at {pos_int} for {type(entity).__name__}") # Optional debug
            return False
        if self.queen and pos_int == self.queen.pos and not isinstance(entity, Queen): # Allow placing the queen herself
             # print(f"Debug: Add entity failed - queen at {pos_int} for {type(entity).__name__}") # Optional debug
             return False

        # --- Check for Collision with other Entities ---
        # Specific checks based on the type of entity being added.
        # Avoid adding an ant where another ant/enemy/prey exists.
        if isinstance(entity, Ant):
            if self.is_ant_at(pos_int) or self.is_enemy_at(pos_int) or self.is_prey_at(pos_int):
                # print(f"Debug: Add ant failed - collision at {pos_int}") # Optional debug
                return False
        # Avoid adding enemy where another ant/enemy/prey exists.
        elif isinstance(entity, Enemy):
             if self.is_ant_at(pos_int) or self.is_enemy_at(pos_int) or self.is_prey_at(pos_int):
                 # print(f"Debug: Add enemy failed - collision at {pos_int}") # Optional debug
                 return False
        # Avoid adding prey where another ant/enemy/prey exists.
        elif isinstance(entity, Prey):
             if self.is_ant_at(pos_int) or self.is_enemy_at(pos_int) or self.is_prey_at(pos_int):
                 # print(f"Debug: Add prey failed - collision at {pos_int}") # Optional debug
                 return False
        # Brood can often share a cell (handled by list in dict), specific checks might be needed if desired.

        # --- Add Entity to Structures ---
        try:
            # Add to the main entity list
            entity_list.append(entity)

            # Add to the position lookup dictionary
            # Special handling for brood (list per cell) vs single entity per cell
            if position_dict is self.brood_positions:
                # Brood uses a list for each position
                if pos_int not in position_dict:
                    position_dict[pos_int] = []
                position_dict[pos_int].append(entity)
            else:
                # Other types assume one entity per cell in the dict
                position_dict[pos_int] = entity

            # Add to the spatial grid if requested
            if add_to_spatial_grid:
                 # Spatial grid uses pixel coordinates
                 pixel_pos = (pos_int[0] * self.cell_size + self.cell_size // 2,
                              pos_int[1] * self.cell_size + self.cell_size // 2)
                 # The entity object needs a 'pos' attribute containing pixel coordinates
                 # for the spatial grid. Let's assume entity.pos is grid coords for now,
                 # and spatial grid works with that or converts internally.
                 # *** Correction: Spatial Grid expects pixel coordinates. ***
                 # *** This generic add_entity might be problematic if entity.pos isn't pixel coords. ***
                 # *** Let's refine this in specific add methods (add_ant etc.) ***
                 # For now, let's comment out the direct spatial grid add here and handle it
                 # in the specific add_ant, add_enemy, add_prey methods.
                 # self.spatial_grid.add_entity(entity)
                 pass # Handle spatial grid add in specific methods

            return True # Successfully added

        except Exception as e:
            # Catch potential errors during list/dict modification
            print(f"ERROR: Exception adding entity {type(entity).__name__} at {pos_int}: {e}")
            # Attempt to rollback if possible (difficult with lists/dicts directly)
            # Best practice might be to check more thoroughly before modifying.
            # For now, just report failure.
            return False

    def add_ant(self, pos_grid: tuple, caste: AntCaste) -> bool:
        """
        Adds a new ant of the specified caste to the simulation at the given grid position.

        Performs extensive validation checks for position validity, obstacles,
        collisions with other entities, and colony population limits. Adds the
        ant to all relevant tracking structures (list, dictionary, position array,
        spatial grid).

        Args:
            pos_grid: The target (x, y) grid coordinates for the new ant.
            caste: The AntCaste (WORKER or SOLDIER) for the new ant.

        Returns:
            True if the ant was successfully added, False otherwise.
        """
        # Ensure integer grid coordinates
        pos_int = tuple(map(int, pos_grid))

        # --- Validation Checks ---
        # 1. Basic Validity (Bounds, Obstacle)
        if not is_valid_pos(pos_int, self.grid_width, self.grid_height):
            # print(f"Debug: Add ant failed - invalid position {pos_int}") # Optional
            return False
        if self.grid.is_obstacle(pos_int):
            # print(f"Debug: Add ant failed - obstacle at {pos_int}") # Optional
            return False

        # 2. Collision Check (Ants, Queen, Enemies, Prey)
        if self.is_ant_at(pos_int): # Checks other ants and queen
            # print(f"Debug: Add ant failed - collision (ant/queen) at {pos_int}") # Optional
            return False
        if self.is_enemy_at(pos_int):
            # print(f"Debug: Add ant failed - collision (enemy) at {pos_int}") # Optional
            return False
        if self.is_prey_at(pos_int):
            # print(f"Debug: Add ant failed - collision (prey) at {pos_int}") # Optional
            return False

        # 3. Population Limit Check
        if len(self.ants) >= MAX_ANTS:
            # print(f"Debug: Add ant failed - MAX_ANTS ({MAX_ANTS}) reached.") # Optional
            return False # Cannot add ant if colony is full

        # --- Create and Add Ant ---
        try:
            # Create the ant instance
            ant = Ant(pos_int, self, caste)

            # Assign index for optimized position tracking
            # Find the next available index slot (important if ants died)
            current_index = self.next_ant_index
            # Check if the calculated next index is actually free, otherwise search
            while self.ant_positions_array[current_index, 0] != -1: # Check if x-coord is -1 (unused)
                current_index += 1
                if current_index >= MAX_ANTS:
                    # This should ideally not happen if len(self.ants) check passed, but safety first.
                    print(f"ERROR: No free slot found in ant_positions_array despite len(ants) < MAX_ANTS!")
                    return False # Array is unexpectedly full

            # Assign the found index to the ant and update tracking structures
            ant.index = current_index
            self.ant_positions_array[current_index] = pos_int # Store grid coords
            self.ant_indices[ant] = current_index
            # Update the hint for the *next* potential free slot
            self.next_ant_index = current_index + 1

            # Add to standard list and position dictionary
            self.ants.append(ant)
            self.ant_positions[pos_int] = ant

            # Add to spatial grid (assuming spatial grid handles grid coordinates now)
            # If SpatialGrid still requires pixel coords, conversion is needed here:
            # pixel_pos = (pos_int[0]*self.cell_size + self.cell_size//2, pos_int[1]*self.cell_size + self.cell_size//2)
            # original_pos = ant.pos
            # ant.pos = pixel_pos # Temporarily set pixel pos
            # self.spatial_grid.add_entity(ant)
            # ant.pos = original_pos # Restore grid pos
            # Assuming SpatialGrid is updated:
            self.spatial_grid.add_entity(ant)


            return True # Successfully added the ant

        except Exception as e:
            print(f"ERROR: Exception during add_ant at {pos_int} for caste {caste.name}: {e}")
            traceback.print_exc()
            # Attempt cleanup if partially added? Difficult.
            return False

    def add_brood(self, brood_item: BroodItem) -> bool:
        """
        Adds a new brood item (egg, larva, or pupa) to the simulation.

        Checks for valid position (within grid, not obstacle). Brood items can
        typically occupy the same cell as other brood items. Adds the item to
        the main brood list and the `brood_positions` dictionary (which maps
        a position to a *list* of brood items at that cell).

        Args:
            brood_item: The BroodItem object to add.

        Returns:
            True if the brood item was successfully added, False otherwise.
        """
        # Ensure integer grid coordinates from the brood item
        pos_int = tuple(map(int, brood_item.pos))

        # --- Validation Checks ---
        # 1. Basic Validity (Bounds, Obstacle)
        if not is_valid_pos(pos_int, self.grid_width, self.grid_height):
            # print(f"Debug: Add brood failed - invalid position {pos_int}") # Optional
            return False
        if self.grid.is_obstacle(pos_int):
            # print(f"Debug: Add brood failed - obstacle at {pos_int}") # Optional
            return False
        # Note: We typically DON'T check for collision with ants/enemies/prey here,
        # assuming brood exists peacefully underneath mobile entities within the nest.
        # Collision checks with *other brood* are implicitly handled by using a list
        # in the brood_positions dictionary.

        # --- Add Brood Item ---
        try:
            # Add to the main list of all brood items
            self.brood.append(brood_item)

            # Add to the position lookup dictionary (list per cell)
            # If this position doesn't have a list yet, create one
            if pos_int not in self.brood_positions:
                self.brood_positions[pos_int] = []
            # Append the new item to the list for this grid cell
            self.brood_positions[pos_int].append(brood_item)

            # Brood items are typically static and not added to the spatial grid,
            # which is primarily for optimizing checks between mobile entities.
            # If brood needed proximity checks (e.g., for tending), they might
            # be added, but currently they are not.

            return True # Successfully added

        except Exception as e:
            print(f"ERROR: Exception during add_brood at {pos_int}: {e}")
            # Attempt rollback? Difficult.
            return False

    def remove_entity(self, entity, entity_list: list, position_dict: dict):
        """
        Generic helper to remove an entity from its main list and position lookup dictionary.

        Handles potential errors if the entity or its position is not found in the
        respective structures (e.g., if removed by another process concurrently).
        Specifically handles the list-based structure of `brood_positions`.

        Note: This method does *not* handle removal from the spatial grid or
              the optimized ant position array; those are managed by the specific
              `kill_*` methods (like `kill_ant`).

        Args:
            entity: The entity object to remove. Must have a `pos` attribute.
            entity_list: The main list holding entities of this type (e.g., self.ants).
            position_dict: The dictionary used for position lookup
                           (e.g., self.ant_positions, self.brood_positions).
        """
        if entity is None:
             # print("Warning: Attempted to remove a None entity.") # Optional debug
             return

        entity_type_name = type(entity).__name__ # For logging

        # --- 1. Remove from the main list ---
        try:
            if entity in entity_list:
                entity_list.remove(entity)
            # else: # Optional debug if not found in list
                # print(f"Debug: Entity {entity_type_name} (pos {entity.pos}) not found in main list during removal.")
        except ValueError:
            # Occurs if entity was already removed from the list somehow
            # print(f"Warning: ValueError removing {entity_type_name} (pos {entity.pos}) from list (already removed?).") # Optional debug
            pass
        except Exception as e:
             print(f"ERROR: Unexpected exception removing {entity_type_name} (pos {entity.pos}) from list: {e}")


        # --- 2. Remove from the position dictionary ---
        try:
            pos_int = tuple(map(int, entity.pos)) # Get position

            # Special handling for brood_positions (list per cell)
            if position_dict is self.brood_positions:
                if pos_int in position_dict:
                    # Check if the specific item exists in the list at this position
                    if entity in position_dict[pos_int]:
                        position_dict[pos_int].remove(entity)
                        # If the list becomes empty after removal, delete the position key
                        if not position_dict[pos_int]:
                            del position_dict[pos_int]
                    # else: # Optional debug if item not in list at the key
                        # print(f"Debug: BroodItem {entity} not found in list at {pos_int} during removal.")
                # else: # Optional debug if pos not in dict
                    # print(f"Debug: Position {pos_int} not found in brood_positions dict during removal.")

            # Handling for single-entity-per-cell dictionaries (ants, enemies, prey)
            else:
                # Check if the position exists as a key and if the value is the entity we intend to remove
                if pos_int in position_dict and position_dict[pos_int] == entity:
                    del position_dict[pos_int]
                # else: # Optional debug if key missing or entity mismatch
                    # if pos_int not in position_dict:
                    #     print(f"Debug: Position {pos_int} not found in {entity_type_name}_positions dict during removal.")
                    # elif position_dict.get(pos_int) != entity:
                    #     print(f"Debug: Entity mismatch at {pos_int} in {entity_type_name}_positions dict during removal (Expected: {entity}, Found: {position_dict.get(pos_int)}).")


        except (KeyError, ValueError, TypeError) as e:
            # KeyError: Position not found in dict.
            # ValueError: Brood item not found in list during remove().
            # TypeError: Problem with entity.pos.
            # print(f"Warning: Exception ({type(e).__name__}) removing {entity_type_name} (pos {getattr(entity, 'pos', 'N/A')}) from position dict: {e}") # Optional debug
            pass
        except Exception as e:
             print(f"ERROR: Unexpected exception removing {entity_type_name} (pos {getattr(entity, 'pos', 'N/A')}) from position dict: {e}")

    def update_entity_position(self, entity, old_pos_grid: tuple, new_pos_grid: tuple):
        """
        Updates the position tracking structures when an entity moves between grid cells.

        Handles updates for:
        - Position dictionary (e.g., self.ant_positions).
        - Optimized ant position array (self.ant_positions_array) if the entity is an Ant.
        - Spatial grid.

        Args:
            entity: The entity object that moved.
            old_pos_grid: The previous (x, y) grid coordinates tuple.
            new_pos_grid: The new (x, y) grid coordinates tuple.
        """
        # Choose the correct position dictionary based on entity type
        pos_dict = None
        is_ant = isinstance(entity, Ant)
        is_enemy = isinstance(entity, Enemy)
        is_prey = isinstance(entity, Prey)

        if is_ant:
            pos_dict = self.ant_positions
        elif is_enemy:
            pos_dict = self.enemy_positions
        elif is_prey:
            pos_dict = self.prey_positions
        # Brood doesn't move, so no update needed for brood_positions here

        # --- Update Position Dictionary ---
        if pos_dict is not None:
            # Remove from the old position key, ensuring the entity matches
            if old_pos_grid in pos_dict and pos_dict[old_pos_grid] == entity:
                del pos_dict[old_pos_grid]
            # Add to the new position key
            pos_dict[new_pos_grid] = entity

        # --- Update Optimized Ant Position Array (if applicable) ---
        if is_ant:
            ant_index = getattr(entity, 'index', -1) # Get ant's index safely
            if ant_index >= 0 and ant_index < MAX_ANTS:
                # Update the position stored at the ant's index in the NumPy array
                self.ant_positions_array[ant_index] = new_pos_grid
            # else: # Optional debug for invalid index
                # print(f"Warning: Invalid index {ant_index} for ant {entity} during position update.")

        # --- Update Spatial Grid ---
        # Spatial grid tracks entities for proximity checks.
        # Note: Ensure SpatialGrid uses grid coordinates OR convert here if it uses pixels.
        # Assuming SpatialGrid is updated to work with grid coordinates:
        self.spatial_grid.update_entity_position(entity, old_pos_grid)
        # If SpatialGrid requires pixel coordinates:
        # old_pos_px = (old_pos_grid[0]*self.cell_size + self.cell_size//2, old_pos_grid[1]*self.cell_size + self.cell_size//2)
        # self.spatial_grid.update_entity_position(entity, old_pos_px)

    def spawn_enemy(self) -> bool:
        """
        Spawns a new enemy at a random, valid location on the grid.

        Attempts to place the enemy far from the nest and avoids obstacles
        and cells already occupied by other entities.

        Returns:
            True if an enemy was successfully spawned and added, False otherwise.
        """
        max_spawn_tries = 80 # Number of attempts to find a suitable spawn location
        nest_pos_int = self.nest_pos

        # Define spawn area constraints: minimum distance from the nest center
        # Spawn enemies further out than initial food clusters.
        min_dist_sq_from_nest = (MIN_FOOD_DIST_FROM_NEST + 5)**2

        for _ in range(max_spawn_tries):
            # Choose random grid coordinates
            pos_grid = (rnd(0, self.grid_width - 1), rnd(0, self.grid_height - 1))

            # --- Check Spawn Conditions ---
            # 1. Distance from Nest
            if distance_sq(pos_grid, nest_pos_int) <= min_dist_sq_from_nest:
                continue # Too close to nest, try again

            # 2. Obstacle Check
            if self.grid.is_obstacle(pos_grid):
                continue # Cannot spawn on obstacle

            # 3. Collision Check (Ants, Queen, Other Enemies, Prey)
            if self.is_enemy_at(pos_grid) or \
               self.is_ant_at(pos_grid) or \
               self.is_prey_at(pos_grid):
                continue # Spot occupied, try again

            # --- Found Valid Spot: Create and Add Enemy ---
            try:
                enemy = Enemy(pos_grid, self)
                # Add to the main list
                self.enemies.append(enemy)
                # Add to the position dictionary
                self.enemy_positions[pos_grid] = enemy
                # Add to the spatial grid
                self.spatial_grid.add_entity(enemy)
                # print(f"  Spawned enemy at {pos_grid}") # Optional debug
                return True # Successfully spawned

            except Exception as e:
                print(f"ERROR: Exception during enemy creation/adding at {pos_grid}: {e}")
                return False # Indicate failure if creation/adding fails

        # Failed to find a suitable spot after all attempts
        # print("Warning: Failed to spawn enemy after max tries.") # Optional debug
        return False

    def spawn_prey(self) -> bool:
        """
        Spawns a new prey item at a random, valid location on the grid.

        Attempts to place the prey away from the immediate nest area and avoids
        obstacles and cells already occupied by other entities. Prey can typically
        spawn closer to the nest than enemies.

        Returns:
            True if prey was successfully spawned and added, False otherwise.
        """
        max_spawn_tries = 70 # Number of attempts to find a suitable location
        nest_pos_int = self.nest_pos

        # Define spawn area constraints: minimum distance from nest
        # Prey can spawn closer than enemies, but potentially not right in the nest center.
        min_dist_sq_from_nest = (MIN_FOOD_DIST_FROM_NEST - 10)**2
        # Ensure minimum distance isn't negative if default config is very small
        min_dist_sq_from_nest = max(4, min_dist_sq_from_nest) # E.g., min radius of 2 cells

        for _ in range(max_spawn_tries):
            # Choose random grid coordinates
            pos_grid = (rnd(0, self.grid_width - 1), rnd(0, self.grid_height - 1))

            # --- Check Spawn Conditions ---
            # 1. Distance from Nest
            if distance_sq(pos_grid, nest_pos_int) <= min_dist_sq_from_nest:
                continue # Too close, try again

            # 2. Obstacle Check
            if self.grid.is_obstacle(pos_grid):
                continue

            # 3. Collision Check (Ants, Queen, Enemies, Other Prey)
            if self.is_prey_at(pos_grid) or \
               self.is_enemy_at(pos_grid) or \
               self.is_ant_at(pos_grid):
                continue # Spot occupied, try again

            # --- Found Valid Spot: Create and Add Prey ---
            try:
                prey_item = Prey(pos_grid, self)
                # Add to the main list
                self.prey.append(prey_item)
                # Add to the position dictionary
                self.prey_positions[pos_grid] = prey_item
                # Add to the spatial grid
                self.spatial_grid.add_entity(prey_item)
                # print(f"  Spawned prey at {pos_grid}") # Optional debug
                return True # Successfully spawned

            except Exception as e:
                print(f"ERROR: Exception during prey creation/adding at {pos_grid}: {e}")
                return False # Indicate failure

        # Failed to find a suitable spot after all attempts
        # print("Warning: Failed to spawn prey after max tries.") # Optional debug
        return False

    # --- Kill Methods ---

    def kill_ant(self, ant_to_remove: Ant, reason: str = "unknown"):
        """
        Removes a dead or invalid ant from the simulation.

        Handles removal from all relevant tracking structures: the main ant list,
        the position dictionary, the optimized position array, the index lookup,
        and the spatial grid. Also updates the `next_ant_index` hint.

        Args:
            ant_to_remove: The Ant object instance to remove.
            reason: A string describing why the ant is being removed (for logging/debug).
        """
        # print(f"DEBUG: Killing Ant {id(ant_to_remove)} at {ant_to_remove.pos}, Reason: {reason}") # Optional detailed debug

        # --- Remove from standard list and position dictionary ---
        # Uses the generic remove_entity helper for these structures.
        self.remove_entity(ant_to_remove, self.ants, self.ant_positions)

        # --- Remove from Optimized Position Array and Index Tracking ---
        ant_index = getattr(ant_to_remove, 'index', -1) # Safely get the ant's index
        if ant_index >= 0 and ant_index < MAX_ANTS:
            # Mark the position in the array as unused (-1)
            self.ant_positions_array[ant_index] = [-1, -1]

            # Remove the ant from the index lookup dictionary
            if ant_to_remove in self.ant_indices:
                del self.ant_indices[ant_to_remove]

            # Update the hint for the next free index. If the removed ant's index
            # was lower than the current hint, it becomes the new hint.
            self.next_ant_index = min(self.next_ant_index, ant_index)

            # Invalidate the index on the ant object itself
            ant_to_remove.index = -1
        # else: # Optional debug for invalid index during removal
            # print(f"Warning: Invalid or missing index ({ant_index}) for ant being removed at {ant_to_remove.pos}.")

        # --- Remove from Spatial Grid ---
        # Ensure removal from spatial grid happens *before* potentially modifying
        # the ant's state further or losing its position reference.
        self.spatial_grid.remove_entity(ant_to_remove)

        # --- Final State ---
        # Ensure the ant is marked as not alive if this wasn't already set
        ant_to_remove.hp = 0
        ant_to_remove.is_dying = True

        # Optional: Drop negative pheromone or small amount of food on death?
        # Example:
        # grid = self.grid
        # pos_int = ant_to_remove.pos
        # if is_valid_pos(pos_int, self.grid_width, self.grid_height) and not grid.is_obstacle(pos_int):
        #     grid.add_pheromone(pos_int, 25.0, "negative") # Small negative signal

    def kill_enemy(self, enemy_to_remove: Enemy):
        """
        Removes a dead enemy from the simulation and adds food resources to the grid.

        Handles removal from the main enemy list, position dictionary, and spatial grid.
        Adds specified amounts of sugar and protein food resources to the grid cell
        where the enemy died.

        Args:
            enemy_to_remove: The Enemy object instance to remove.
        """
        # print(f"DEBUG: Killing Enemy {id(enemy_to_remove)} at {enemy_to_remove.pos}") # Optional debug
        # Get position before removing entity from structures
        pos_int = tuple(map(int, enemy_to_remove.pos))

        # --- Remove from standard list and position dictionary ---
        self.remove_entity(enemy_to_remove, self.enemies, self.enemy_positions)

        # --- Remove from Spatial Grid ---
        self.spatial_grid.remove_entity(enemy_to_remove)

        # --- Add Food Resources at Death Location ---
        # Check if the death position is valid and not an obstacle
        if is_valid_pos(pos_int, self.grid_width, self.grid_height) and \
           not self.grid.is_obstacle(pos_int):
            fx, fy = pos_int
            grid = self.grid
            s_idx = FoodType.SUGAR.value
            p_idx = FoodType.PROTEIN.value
            try:
                # Add sugar reward, clamped to the maximum food per cell
                current_sugar = grid.food[fx, fy, s_idx]
                grid.food[fx, fy, s_idx] = min(MAX_FOOD_PER_CELL,
                                               current_sugar + ENEMY_TO_FOOD_ON_DEATH_SUGAR)

                # Add protein reward, clamped to the maximum food per cell
                current_protein = grid.food[fx, fy, p_idx]
                grid.food[fx, fy, p_idx] = min(MAX_FOOD_PER_CELL,
                                               current_protein + ENEMY_TO_FOOD_ON_DEATH_PROTEIN)
                # print(f"  Added food S:{ENEMY_TO_FOOD_ON_DEATH_SUGAR}, P:{ENEMY_TO_FOOD_ON_DEATH_PROTEIN} at {pos_int}") # Debug
            except IndexError:
                # This should not happen if is_valid_pos passed, but safety first.
                print(f"Warning: IndexError adding food from dead enemy at {pos_int}")
            except Exception as e:
                 print(f"ERROR: Exception adding food from dead enemy at {pos_int}: {e}")

        # --- Final State Update ---
        # Ensure the removed enemy object reflects its dead state.
        enemy_to_remove.hp = 0
        enemy_to_remove.is_dying = True

    def kill_prey(self, prey_to_remove: Prey):
        """
        Removes dead prey from the simulation and adds protein resource to the grid.

        Handles removal from the main prey list, position dictionary, and spatial grid.
        Adds a specified amount of protein food resource to the grid cell where
        the prey died.

        Args:
            prey_to_remove: The Prey object instance to remove.
        """
        # print(f"DEBUG: Killing Prey {id(prey_to_remove)} at {prey_to_remove.pos}") # Optional debug
        # Get position before removing entity from structures
        pos_int = tuple(map(int, prey_to_remove.pos))

        # --- Remove from standard list and position dictionary ---
        self.remove_entity(prey_to_remove, self.prey, self.prey_positions)

        # --- Remove from Spatial Grid ---
        self.spatial_grid.remove_entity(prey_to_remove)

        # --- Add Protein Resource at Death Location ---
        # Check if the death position is valid and not an obstacle
        if is_valid_pos(pos_int, self.grid_width, self.grid_height) and \
           not self.grid.is_obstacle(pos_int):
            fx, fy = pos_int
            grid = self.grid
            p_idx = FoodType.PROTEIN.value # Index for protein in the food array
            try:
                # Add protein reward, clamped to the maximum food per cell
                current_protein = grid.food[fx, fy, p_idx]
                grid.food[fx, fy, p_idx] = min(MAX_FOOD_PER_CELL,
                                               current_protein + PROTEIN_ON_DEATH)
                # print(f"  Added food P:{PROTEIN_ON_DEATH} at {pos_int} from dead prey.") # Debug
            except IndexError:
                # Safety check, should not happen if is_valid_pos passed.
                print(f"Warning: IndexError adding protein from dead prey at {pos_int}")
            except Exception as e:
                 print(f"ERROR: Exception adding protein from dead prey at {pos_int}: {e}")

        # --- Final State Update ---
        # Ensure the removed prey object reflects its dead state.
        prey_to_remove.hp = 0
        prey_to_remove.is_dying = True

    def kill_queen(self, queen_to_remove: Queen):
        """
        Handles the queen's death, which signifies the end of the current simulation run.

        Sets the simulation state to not running and records the reason for the game end.

        Args:
            queen_to_remove: The Queen object instance that has died.
        """
        # Check if the queen being removed is the currently active queen
        if self.queen == queen_to_remove and self.queen is not None:
            queen_pos = self.queen.pos # Get position before setting queen to None

            print(f"\n--- QUEEN DIED at {queen_pos} (Tick {self.ticks}, Generation {self.colony_generation}) ---")
            # Log final stats for this generation
            print(f"    Final Food - Sugar: {self.colony_food_storage_sugar:.1f}, Protein: {self.colony_food_storage_protein:.1f}")
            print(f"    Final Ants: {len(self.ants)}, Final Brood: {len(self.brood)}")
            print(f"    Final Enemies: {len(self.enemies)}, Final Prey: {len(self.prey)}")

            # --- Update Simulation State ---
            self.queen = None # Remove the queen reference
            self.simulation_running = False # Stop the current simulation run loop
            self.end_game_reason = "Queen Died" # Set reason for end dialog

            # Note: The queen object might still exist temporarily if referenced elsewhere,
            # but setting self.queen = None and simulation_running = False are the key actions.
            # The object will eventually be garbage collected if not held elsewhere.
            # Ensure the queen object itself reflects its state:
            queen_to_remove.hp = 0
            queen_to_remove.is_dying = True

            # Remove queen from ant_positions lookup if she was added there
            if queen_pos in self.ant_positions and isinstance(self.ant_positions[queen_pos], Queen):
                del self.ant_positions[queen_pos]

        # else: # Optional: Handle cases where queen is already None or mismatch
            # print(f"Warning: kill_queen called but self.queen is already None or mismatch.")
            # self.simulation_running = False # Ensure sim stops anyway if called unexpectedly

    # --- Position Query Methods ---

    def is_ant_at(self, pos_grid: tuple, exclude_self: Ant | Queen | None = None) -> bool:
        """
        Checks if an ant or the queen occupies the given grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.
            exclude_self: Optional. If provided, this specific Ant or Queen
                          instance will be ignored during the check (useful for
                          preventing an entity from blocking itself).

        Returns:
            True if an ant or the queen (not matching exclude_self) is found
            at the position, False otherwise.
        """
        pos_int = tuple(map(int, pos_grid)) # Ensure integer coordinates

        # Check Queen first (as she's not in the main ants list/dict usually)
        if self.queen and self.queen.pos == pos_int and self.queen is not exclude_self:
            return True

        # Check the ant position dictionary
        # This is generally O(1) on average.
        ant = self.ant_positions.get(pos_int)
        # Return True if an ant exists at the position and it's not the excluded one
        return ant is not None and ant is not exclude_self

    def get_ant_at(self, pos_grid: tuple) -> Ant | Queen | None:
        """
        Returns the ant or queen instance at the specified grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.

        Returns:
            The Ant or Queen object found at the position, or None if the cell
            is empty or contains a different entity type.
        """
        pos_int = tuple(map(int, pos_grid))

        # Check Queen first
        if self.queen and self.queen.pos == pos_int:
            return self.queen

        # Check the ant position dictionary
        # Note: Using the dictionary is likely faster for single-cell lookups than
        # querying the spatial grid, which involves coordinate conversion and iteration.
        return self.ant_positions.get(pos_int, None) # Return ant or None if not found

    def is_enemy_at(self, pos_grid: tuple, exclude_self: Enemy | None = None) -> bool:
        """
        Checks if an enemy occupies the given grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.
            exclude_self: Optional. If provided, this specific Enemy instance
                          will be ignored.

        Returns:
            True if an enemy (not matching exclude_self) is found, False otherwise.
        """
        pos_int = tuple(map(int, pos_grid))
        enemy = self.enemy_positions.get(pos_int)
        return enemy is not None and enemy is not exclude_self

    def get_enemy_at(self, pos_grid: tuple) -> Enemy | None:
        """
        Returns the enemy instance at the specified grid position.

        Uses the spatial grid for potentially faster lookups if many enemies
        are clustered, although direct dictionary lookup is also efficient.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.

        Returns:
            The Enemy object found at the position, or None if the cell is empty
            or contains a different entity type.
        """
        # Direct dictionary lookup is likely sufficient and simpler here.
        pos_int = tuple(map(int, pos_grid))
        return self.enemy_positions.get(pos_int, None)

        # --- Alternative using Spatial Grid (more complex, potentially faster if dict is huge) ---
        # pos_int = tuple(map(int, pos_grid))
        # center_px = (pos_int[0] * self.cell_size + self.cell_size // 2,
        #              pos_int[1] * self.cell_size + self.cell_size // 2)
        # # Search a tiny radius around the center pixel
        # nearby_enemies = self.spatial_grid.get_nearby_entities(center_px, 1, Enemy)
        # for enemy in nearby_enemies:
        #     # Check if the found enemy's grid position matches the query
        #     if tuple(map(int, enemy.pos)) == pos_int:
        #         return enemy
        # return None # Not found

    def is_prey_at(self, pos_grid: tuple, exclude_self: Prey | None = None) -> bool:
        """
        Checks if prey occupies the given grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.
            exclude_self: Optional. If provided, this specific Prey instance
                          will be ignored.

        Returns:
            True if prey (not matching exclude_self) is found, False otherwise.
        """
        pos_int = tuple(map(int, pos_grid))
        prey_item = self.prey_positions.get(pos_int)
        return prey_item is not None and prey_item is not exclude_self

    def get_prey_at(self, pos_grid: tuple) -> Prey | None:
        """
        Returns the prey instance at the specified grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.

        Returns:
            The Prey object found at the position, or None if the cell is empty
            or contains a different entity type.
        """
        # Direct dictionary lookup is efficient for single cell checks.
        pos_int = tuple(map(int, pos_grid))
        return self.prey_positions.get(pos_int, None)

    def get_brood_positions(self) -> set[tuple]:
        """
        Returns a set of all grid positions currently containing any brood items.

        Returns:
            A set containing (x, y) grid coordinate tuples where brood exists.
        """
        # Return the keys from the brood_positions dictionary, converted to a set.
        # This efficiently gives unique positions with brood.
        return set(self.brood_positions.keys())

    def find_nearby_prey(self, pos_grid: tuple, radius_sq: float) -> list[Prey]:
        """
        Finds all living prey items within a specified squared radius of a position.

        Uses the spatial grid for efficient initial filtering, then performs
        precise distance checks.

        Args:
            pos_grid: The center (x, y) grid coordinates for the search.
            radius_sq: The squared radius (in grid units) to search within.

        Returns:
            A list of living Prey objects found within the radius.
        """
        nearby_living_prey = []
        pos_int = tuple(map(int, pos_grid))

        # Convert grid radius to pixel radius for spatial grid query
        radius_px = (radius_sq**0.5) * self.cell_size
        center_px = (pos_int[0] * self.cell_size + self.cell_size // 2,
                     pos_int[1] * self.cell_size + self.cell_size // 2)

        # Query spatial grid for potentially nearby prey
        nearby_entities = self.spatial_grid.get_nearby_entities(center_px, radius_px, Prey)

        # Filter results: check actual distance (grid units) and if prey is alive
        for p in nearby_entities:
            # Ensure prey still exists in the main list and is alive
            # Check distance using grid coordinates for accuracy with radius_sq
            if p in self.prey and p.hp > 0 and distance_sq(pos_int, p.pos) <= radius_sq:
                nearby_living_prey.append(p)

        return nearby_living_prey

    # --- Position Query Methods ---

    def is_ant_at(self, pos_grid: tuple, exclude_self: Ant | Queen | None = None) -> bool:
        """
        Checks if an ant or the queen occupies the given grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.
            exclude_self: Optional. If provided, this specific Ant or Queen
                          instance will be ignored during the check (useful for
                          preventing an entity from blocking itself).

        Returns:
            True if an ant or the queen (not matching exclude_self) is found
            at the position, False otherwise.
        """
        pos_int = tuple(map(int, pos_grid)) # Ensure integer coordinates

        # Check Queen first (as she's not in the main ants list/dict usually)
        if self.queen and self.queen.pos == pos_int and self.queen is not exclude_self:
            return True

        # Check the ant position dictionary
        # This is generally O(1) on average.
        ant = self.ant_positions.get(pos_int)
        # Return True if an ant exists at the position and it's not the excluded one
        return ant is not None and ant is not exclude_self

    def get_ant_at(self, pos_grid: tuple) -> Ant | Queen | None:
        """
        Returns the ant or queen instance at the specified grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.

        Returns:
            The Ant or Queen object found at the position, or None if the cell
            is empty or contains a different entity type.
        """
        pos_int = tuple(map(int, pos_grid))

        # Check Queen first
        if self.queen and self.queen.pos == pos_int:
            return self.queen

        # Check the ant position dictionary
        # Note: Using the dictionary is likely faster for single-cell lookups than
        # querying the spatial grid, which involves coordinate conversion and iteration.
        return self.ant_positions.get(pos_int, None) # Return ant or None if not found

    def is_enemy_at(self, pos_grid: tuple, exclude_self: Enemy | None = None) -> bool:
        """
        Checks if an enemy occupies the given grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.
            exclude_self: Optional. If provided, this specific Enemy instance
                          will be ignored.

        Returns:
            True if an enemy (not matching exclude_self) is found, False otherwise.
        """
        pos_int = tuple(map(int, pos_grid))
        enemy = self.enemy_positions.get(pos_int)
        return enemy is not None and enemy is not exclude_self

    def get_enemy_at(self, pos_grid: tuple) -> Enemy | None:
        """
        Returns the enemy instance at the specified grid position.

        Uses the spatial grid for potentially faster lookups if many enemies
        are clustered, although direct dictionary lookup is also efficient.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.

        Returns:
            The Enemy object found at the position, or None if the cell is empty
            or contains a different entity type.
        """
        # Direct dictionary lookup is likely sufficient and simpler here.
        pos_int = tuple(map(int, pos_grid))
        return self.enemy_positions.get(pos_int, None)

        # --- Alternative using Spatial Grid (more complex, potentially faster if dict is huge) ---
        # pos_int = tuple(map(int, pos_grid))
        # center_px = (pos_int[0] * self.cell_size + self.cell_size // 2,
        #              pos_int[1] * self.cell_size + self.cell_size // 2)
        # # Search a tiny radius around the center pixel
        # nearby_enemies = self.spatial_grid.get_nearby_entities(center_px, 1, Enemy)
        # for enemy in nearby_enemies:
        #     # Check if the found enemy's grid position matches the query
        #     if tuple(map(int, enemy.pos)) == pos_int:
        #         return enemy
        # return None # Not found

    def is_prey_at(self, pos_grid: tuple, exclude_self: Prey | None = None) -> bool:
        """
        Checks if prey occupies the given grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.
            exclude_self: Optional. If provided, this specific Prey instance
                          will be ignored.

        Returns:
            True if prey (not matching exclude_self) is found, False otherwise.
        """
        pos_int = tuple(map(int, pos_grid))
        prey_item = self.prey_positions.get(pos_int)
        return prey_item is not None and prey_item is not exclude_self

    def get_prey_at(self, pos_grid: tuple) -> Prey | None:
        """
        Returns the prey instance at the specified grid position.

        Args:
            pos_grid: The (x, y) grid coordinate tuple to check.

        Returns:
            The Prey object found at the position, or None if the cell is empty
            or contains a different entity type.
        """
        # Direct dictionary lookup is efficient for single cell checks.
        pos_int = tuple(map(int, pos_grid))
        return self.prey_positions.get(pos_int, None)

    def get_brood_positions(self) -> set[tuple]:
        """
        Returns a set of all grid positions currently containing any brood items.

        Returns:
            A set containing (x, y) grid coordinate tuples where brood exists.
        """
        # Return the keys from the brood_positions dictionary, converted to a set.
        # This efficiently gives unique positions with brood.
        return set(self.brood_positions.keys())

    def find_nearby_prey(self, pos_grid: tuple, radius_sq: float) -> list[Prey]:
        """
        Finds all living prey items within a specified squared radius of a position.

        Uses the spatial grid for efficient initial filtering, then performs
        precise distance checks.

        Args:
            pos_grid: The center (x, y) grid coordinates for the search.
            radius_sq: The squared radius (in grid units) to search within.

        Returns:
            A list of living Prey objects found within the radius.
        """
        nearby_living_prey = []
        pos_int = tuple(map(int, pos_grid))

        # Convert grid radius to pixel radius for spatial grid query
        radius_px = (radius_sq**0.5) * self.cell_size
        center_px = (pos_int[0] * self.cell_size + self.cell_size // 2,
                     pos_int[1] * self.cell_size + self.cell_size // 2)

        # Query spatial grid for potentially nearby prey
        nearby_entities = self.spatial_grid.get_nearby_entities(center_px, radius_px, Prey)

        # Filter results: check actual distance (grid units) and if prey is alive
        for p in nearby_entities:
            # Ensure prey still exists in the main list and is alive
            # Check distance using grid coordinates for accuracy with radius_sq
            if p in self.prey and p.hp > 0 and distance_sq(pos_int, p.pos) <= radius_sq:
                nearby_living_prey.append(p)

        return nearby_living_prey

    def update(self):
        """
        Performs a single step (tick) of the simulation.

        Updates all active entities (queen, ants, enemies, prey, brood),
        handles entity spawning and death, updates pheromones, checks for
        game end conditions, and manages timed events like food replenishment.
        """
        global latest_frame_bytes # For network streaming frame capture

        # --- 1. Increment Simulation Time ---
        # Simulation proceeds one tick at a time.
        self.ticks += 1
        current_tick = self.ticks # Use integer tick for comparisons and logging

        # --- 2. Update Queen ---
        if self.queen:
            self.queen.update() # Queen's update now assumes 1 tick per call
            # Check if the queen died during her update (e.g., starvation if implemented)
            # The primary death check is in step 5, but early exit is possible.
            if not self.simulation_running:
                 print(f"DEBUG: Simulation stopped during Queen update at tick {current_tick}.")
                 return # Queen death stops the simulation run

        # --- 3. Update Brood ---
        hatched_pupae = []
        # Iterate over a copy, as brood list can change during iteration (hatching)
        for item in list(self.brood):
            # Check if item still exists (might have been removed if invalid?)
            if item in self.brood:
                hatch_signal = item.update(current_tick) # Pass current tick for age/feeding checks
                if hatch_signal: # update() returns self if hatched
                    hatched_pupae.append(hatch_signal)

        # --- Handle Hatched Pupae (Spawn New Ants) ---
        for pupa in hatched_pupae:
            # Double check pupa still exists before removing/spawning
            if pupa in self.brood:
                # Remove the pupa item first
                self.remove_entity(pupa, self.brood, self.brood_positions)
                # Attempt to spawn the new ant at/near the pupa's location
                self._spawn_hatched_ant(pupa.caste, pupa.pos)

        # --- 4. Update Mobile Entities (Ants, Enemies, Prey) ---
        # Update copies and shuffle order to avoid bias in interaction sequences.
        # Use list() to create shallow copies.
        ants_to_update = list(self.ants)
        random.shuffle(ants_to_update)
        enemies_to_update = list(self.enemies)
        random.shuffle(enemies_to_update)
        prey_to_update = list(self.prey)
        random.shuffle(prey_to_update)

        # Update Ants
        for ant in ants_to_update:
            # Check if ant still exists and is alive before updating
            # (might have died from earlier interaction in this same tick)
            if ant in self.ants and not ant.is_dying:
                try:
                    ant.update() # Ant update now assumes 1 tick per call
                except Exception as e:
                    print(f"ERROR: Exception during Ant {id(ant)} update at {ant.pos}: {e}")
                    traceback.print_exc()
                    ant.is_dying = True # Mark problematic ant for removal

        # Update Enemies
        for enemy in enemies_to_update:
            if enemy in self.enemies and not enemy.is_dying:
                 try:
                    enemy.update()
                 except Exception as e:
                    print(f"ERROR: Exception during Enemy {id(enemy)} update at {enemy.pos}: {e}")
                    traceback.print_exc()
                    enemy.is_dying = True

        # Update Prey
        for p in prey_to_update:
            if p in self.prey and not p.is_dying:
                 try:
                    p.update()
                 except Exception as e:
                    print(f"ERROR: Exception during Prey {id(p)} update at {p.pos}: {e}")
                    traceback.print_exc()
                    p.is_dying = True

        # --- 5. Entity Cleanup (Remove Dead/Invalid Entities) ---
        # Collect entities marked for death (`is_dying` flag is set)
        # Also remove entities that somehow ended up on an obstacle.
        ants_to_remove = [a for a in self.ants if a.is_dying or self.grid.is_obstacle(a.pos)]
        enemies_to_remove = [e for e in self.enemies if e.is_dying or self.grid.is_obstacle(e.pos)]
        prey_to_remove = [p for p in self.prey if p.is_dying or self.grid.is_obstacle(p.pos)]
        # Brood removal happens during hatching or if they fail update checks (e.g. starvation if added)

        # Remove collected entities using the specific kill methods
        for a in ants_to_remove: self.kill_ant(a, a.last_move_info if a.is_dying else "On Obstacle")
        for e in enemies_to_remove: self.kill_enemy(e)
        for p in prey_to_remove: self.kill_prey(p)

        # Final check for the Queen's status after all updates and removals
        if self.queen and (self.queen.is_dying or self.grid.is_obstacle(self.queen.pos)):
            self.kill_queen(self.queen) # This will set simulation_running = False

        # Exit update step immediately if the queen died
        if not self.simulation_running:
            print(f"DEBUG: Simulation stopped during entity cleanup at tick {current_tick}.")
            return

        # --- 6. Update Grid Systems (Pheromones) ---
        # Decay and diffusion now happen once per tick.
        try:
            self.grid.update_pheromones()
        except Exception as e:
             print(f"ERROR: Exception during pheromone update: {e}")
             traceback.print_exc() # Continue simulation if possible

        # --- 7. Spawning Timers and Events ---
        # Enemy Spawning
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= self.enemy_spawn_interval:
            self.enemy_spawn_timer = 0 # Reset timer
            # Limit total enemies based on initial count (e.g., max 6 times initial)
            if len(self.enemies) < INITIAL_ENEMIES * 6:
                self.spawn_enemy() # Attempt to spawn one enemy

        # Prey Spawning
        self.prey_spawn_timer += 1
        if self.prey_spawn_timer >= self.prey_spawn_interval:
            self.prey_spawn_timer = 0 # Reset timer
            max_prey = INITIAL_PREY * 3 # Limit total prey
            if len(self.prey) < max_prey:
                self.spawn_prey() # Attempt to spawn one prey

        # Food Replenishment
        self.food_replenish_timer += 1
        if self.food_replenish_timer >= self.food_replenish_interval:
            self.food_replenish_timer = 0 # Reset timer
            self.grid.replenish_food(self.nest_pos) # Add new food clusters


        # --- 8. Logging ---
        # Check if logging is enabled and the interval is met
        if self.log_filename and (current_tick % self.logging_interval == 0):
            try:
                # Gather current simulation statistics
                ant_c = len(self.ants)
                w_c = sum(1 for a in self.ants if a.caste == AntCaste.WORKER)
                s_c = ant_c - w_c # Soldier count is total minus workers
                brood_c = len(self.brood)
                e_c = sum(1 for b in self.brood if b.stage == BroodStage.EGG)
                l_c = sum(1 for b in self.brood if b.stage == BroodStage.LARVA)
                p_c = brood_c - e_c - l_c # Pupa count
                enemy_c = len(self.enemies)
                prey_c = len(self.prey)

                # Format the log entry as a CSV row string
                log_line = (
                    f"{current_tick},{self.colony_generation},"
                    f"{self.colony_food_storage_sugar:.2f},{self.colony_food_storage_protein:.2f},"
                    f"{ant_c},{w_c},{s_c},"
                    f"{e_c},{l_c},{p_c},"
                    f"{enemy_c},{prey_c}\n"
                )

                # Append the log entry to the file
                # Using 'with open' ensures the file is closed properly even if errors occur.
                # Opening in append mode ('a') adds to the end or creates the file if it doesn't exist.
                with open(self.log_filename, 'a', encoding='utf-8') as f: # Specify encoding
                    f.write(log_line)

            except Exception as e:
                print(f"ERROR writing to log file '{self.log_filename}': {e}")
                # Optional: Disable logging for the rest of this run upon error
                # self.log_filename = None

        # --- 9. Frame Capture for Network Streaming ---
        # This needs to happen *after* all updates but *before* drawing usually,
        # or right after drawing if the drawn frame is what needs to be streamed.
        # Currently placed here, assuming `self.draw()` will capture the frame later.
        # If streaming is enabled, the `draw` method will capture `self.screen`
        # into `self.latest_frame_surface`. This surface is then processed here (or in draw).

        # Processing the captured frame for streaming (can also be done in `draw`)
        if ENABLE_NETWORK_STREAM and Flask and streaming_thread and streaming_thread.is_alive():
            if self.latest_frame_surface: # Check if draw() captured a surface
                try:
                    frame_buffer = io.BytesIO() # In-memory buffer for JPEG data
                    # Save the captured surface to the buffer
                    pygame.image.save(self.latest_frame_surface, frame_buffer, ".jpg")
                    frame_buffer.seek(0) # Rewind buffer to the beginning
                    # Update the globally shared frame bytes under lock
                    with latest_frame_lock:
                        latest_frame_bytes = frame_buffer.read()
                    self.latest_frame_surface = None # Clear the captured surface reference
                except pygame.error as e:
                    print(f"Pygame error during frame capture/save for streaming: {e}")
                    with latest_frame_lock: latest_frame_bytes = None # Ensure no stale data
                except Exception as e:
                    print(f"Error processing frame for streaming: {e}")
                    with latest_frame_lock: latest_frame_bytes = None
            # else: # Frame wasn't captured in draw() this cycle
                 # pass

    def _spawn_hatched_ant(self, caste: AntCaste, pupa_pos_grid: tuple):
        """
        Attempts to spawn a newly hatched ant at or near the pupa's last position.

        Tries the exact pupa position first. If blocked, checks immediate neighbors.
        If still blocked, attempts a few random spots near the nest center as a last resort.
        Logs a warning if spawning ultimately fails.

        Args:
            caste: The AntCaste of the ant to be spawned.
            pupa_pos_grid: The (x, y) grid coordinates where the pupa was located.

        Returns:
            True if the ant was successfully spawned, False otherwise.
        """
        # --- Attempt 1: Spawn at the exact pupa position ---
        if self.add_ant(pupa_pos_grid, caste):
            return True # Success!

        # --- Attempt 2: Try immediate neighbors ---
        # print(f"Debug: Pupa pos {pupa_pos_grid} blocked for hatch. Trying neighbors...") # Optional debug
        neighbors = get_neighbors(pupa_pos_grid, self.grid_width, self.grid_height)
        random.shuffle(neighbors) # Check neighbors in random order
        for pos_grid in neighbors:
            if self.add_ant(pos_grid, caste):
                # print(f"  Spawned hatched ant at neighbor {pos_grid}.") # Optional debug
                return True # Success!

        # --- Attempt 3 (Fallback): Try random spots near nest center ---
        # print(f"Warning: Neighbors also blocked for hatch at {pupa_pos_grid}. Trying random near nest...") # Optional debug
        # Determine base position for random placement (queen pos or nest center)
        if self.queen:
            base_pos = self.queen.pos
        else:
            base_pos = self.nest_pos # Fallback if queen somehow died between brood update and here

        max_fallback_attempts = 15
        for _ in range(max_fallback_attempts):
            # Random offset within the nest radius (excluding boundary maybe)
            ox = rnd(-(NEST_RADIUS - 1), NEST_RADIUS - 1)
            oy = rnd(-(NEST_RADIUS - 1), NEST_RADIUS - 1)
            pos_grid = (base_pos[0] + ox, base_pos[1] + oy)

            # Attempt to add the ant at the random position
            if self.add_ant(pos_grid, caste):
                 # print(f"  Spawned hatched ant at random nest pos {pos_grid}.") # Optional debug
                 return True # Success!

        # --- Failure Case ---
        # If all attempts failed to place the hatched ant
        print(f"ERROR: Failed to spawn hatched {caste.name} near {pupa_pos_grid} after all attempts.")
        return False

    # --- Drawing Methods ---

    def draw_debug_info(self):
        """
        Draws the overlay displaying simulation statistics and mouse hover information.

        Renders text using the debug font onto the main screen surface. Information
        includes tick count, FPS, entity counts (total, castes, brood stages),
        colony resources, and detailed information about the grid cell and any
        entities under the mouse cursor.
        """
        # Check if the debug font is loaded and debug info is enabled
        if not self.debug_font or not self.show_debug_info:
            return

        # --- Gather Simulation Statistics ---
        try:
            ant_c = len(self.ants)
            enemy_c = len(self.enemies)
            brood_c = len(self.brood)
            prey_c = len(self.prey)
            food_s = self.colony_food_storage_sugar
            food_p = self.colony_food_storage_protein
            tick_display = self.ticks # Now an integer
            fps = self.clock.get_fps()

            # Caste & Brood Stage Breakdown
            w_c = sum(1 for a in self.ants if a.caste == AntCaste.WORKER)
            s_c = ant_c - w_c # Soldier count
            e_c = sum(1 for b in self.brood if b.stage == BroodStage.EGG)
            l_c = sum(1 for b in self.brood if b.stage == BroodStage.LARVA)
            p_c = brood_c - e_c - l_c # Pupa count
        except Exception as e:
            print(f"Error gathering stats for debug info: {e}")
            # Display error message instead of stats if gathering fails
            texts = ["Error gathering stats!"]
        else:
            # --- Format Text Lines for Display ---
            # << REMOVED Speed Multiplier Line >>
            texts = [
                f"Generation: {self.colony_generation}",
                f"Tick: {tick_display} | FPS: {fps:.0f} (Target: {self.target_fps})",
                # speed_text removed
                f"Ants: {ant_c} (W:{w_c} S:{s_c}) / {MAX_ANTS}", # Show max ants limit
                f"Brood: {brood_c} (E:{e_c} L:{l_c} P:{p_c})",
                f"Enemies: {enemy_c}",
                f"Prey: {prey_c}",
                f"Food S: {food_s:.1f}",
                f"Food P: {food_p:.1f}"
            ]

        # --- Render Top-Left Statistics ---
        # Calculate starting position below UI buttons
        button_bottom_y = 0
        if self.buttons:
            try:
                button_bottom_y = self.buttons[0]['rect'].bottom
            except (IndexError, KeyError):
                button_bottom_y = 10 # Fallback if buttons list is empty or malformed
        y_start = button_bottom_y + 5 # Start 5 pixels below buttons
        line_height = self.debug_font.get_height() + 2 # Spacing between lines
        text_color = (240, 240, 240) # White/Light Grey

        for i, txt in enumerate(texts):
            try:
                # Render text surface
                surf = self.debug_font.render(txt, True, text_color)
                # Blit onto the main screen
                self.screen.blit(surf, (5, y_start + i * line_height))
            except Exception as e:
                # Print error but attempt to continue drawing other lines
                print(f"ERROR: Debug Font render error (line '{txt}'): {e}")

        # --- Mouse Hover Information ---
        try:
            # Get current mouse coordinates
            mx, my = pygame.mouse.get_pos()
            # Convert pixel coordinates to grid coordinates
            gx = mx // self.cell_size
            gy = my // self.cell_size
            hover_pos_grid = (gx, gy) # Grid cell under cursor

            # Check if the mouse cursor is within the simulation grid bounds
            if is_valid_pos(hover_pos_grid, self.grid_width, self.grid_height):
                hover_lines = [] # List to store lines of text for hover info
                grid = self.grid # Local reference to the world grid

                # --- Entity Information at Hover Position ---
                # Check for Ant/Queen, Enemy, or Prey at the cell
                # Use get_* methods which handle lookups efficiently
                entity = self.get_ant_at(hover_pos_grid) or \
                         self.get_enemy_at(hover_pos_grid) or \
                         self.get_prey_at(hover_pos_grid)

                if entity:
                    entity_pos_int = tuple(map(int, entity.pos)) # Ensure integer pos for display
                    hp_str = f"HP:{entity.hp:.0f}/{entity.max_hp:.0f}" if hasattr(entity, 'hp') else ""
                    # Format entity-specific details
                    if isinstance(entity, Queen):
                        hover_lines.append(f"QUEEN @{entity_pos_int}")
                        hover_lines.append(f"{hp_str} Age:{entity.age:.0f}")
                    elif isinstance(entity, Ant):
                        carry_str = f"C:{entity.carry_amount:.1f}({entity.carry_type.name if entity.carry_type else '-'})"
                        age_str = f"Age:{entity.age}/{entity.max_age_ticks}" # Show current/max age
                        move_str = f"Mv:{entity.last_move_info[:28]}" # Limit length of move info
                        hover_lines.append(f"{entity.caste.name} @{entity_pos_int}")
                        hover_lines.append(f"S:{entity.state.name} {hp_str}")
                        hover_lines.extend([carry_str, age_str, move_str])
                    elif isinstance(entity, Enemy):
                        hover_lines.append(f"ENEMY @{entity_pos_int}")
                        hover_lines.append(f"{hp_str}")
                    elif isinstance(entity, Prey):
                        hover_lines.append(f"PREY @{entity_pos_int}")
                        hover_lines.append(f"{hp_str}")

                # --- Brood Information at Hover Position ---
                brood_at_pos = self.brood_positions.get(hover_pos_grid, [])
                if brood_at_pos:
                    hover_lines.append(f"Brood:{len(brood_at_pos)} @{hover_pos_grid}")
                    # Show details for the first few brood items if list is long
                    for b in brood_at_pos[:3]: # Limit display to 3 items
                        prog_str = f"{b.progress_timer:.0f}/{b.duration}"
                        hover_lines.append(f"-{b.stage.name[:1]}({b.caste.name[:1]}) P:{prog_str}")

                # --- Cell Information (Obstacle, Food, Pheromones) ---
                is_obs = grid.is_obstacle(hover_pos_grid) # Use grid method for check
                obs_txt = " OBSTACLE" if is_obs else ""
                hover_lines.append(f"Cell: {hover_pos_grid}{obs_txt}")

                # Display food and pheromone levels only if not an obstacle
                if not is_obs:
                    try:
                        # Food levels
                        foods = grid.food[gx, gy] # Direct access after bounds check
                        food_s_lvl = foods[FoodType.SUGAR.value]
                        food_p_lvl = foods[FoodType.PROTEIN.value]
                        hover_lines.append(f"Food S:{food_s_lvl:.1f} P:{food_p_lvl:.1f}")

                        # Pheromone levels
                        ph_home = grid.get_pheromone(hover_pos_grid, "home")
                        ph_food_s = grid.get_pheromone(hover_pos_grid, "food", FoodType.SUGAR)
                        ph_food_p = grid.get_pheromone(hover_pos_grid, "food", FoodType.PROTEIN)
                        ph_alarm = grid.get_pheromone(hover_pos_grid, "alarm")
                        ph_neg = grid.get_pheromone(hover_pos_grid, "negative")
                        ph_rec = grid.get_pheromone(hover_pos_grid, "recruitment")
                        # Format pheromone lines for readability
                        ph1 = f"Ph H:{ph_home:.0f} FS:{ph_food_s:.0f} FP:{ph_food_p:.0f}"
                        ph2 = f"Ph A:{ph_alarm:.0f} N:{ph_neg:.0f} R:{ph_rec:.0f}"
                        hover_lines.extend([ph1, ph2])
                    except IndexError:
                        hover_lines.append("Error reading cell data") # Should not happen if is_valid_pos passed
                    except Exception as e:
                         hover_lines.append(f"Error cell data: {e}")

                # --- Render Hover Text (Bottom-Left) ---
                hover_color = (255, 255, 0) # Yellow
                # Calculate starting Y position from the bottom edge of the screen
                # Use world_height_px as the screen might be larger
                hover_y_start = self.world_height_px - (len(hover_lines) * line_height) - 5
                for i, line in enumerate(hover_lines):
                    # Render each line of hover text
                    surf = self.debug_font.render(line, True, hover_color)
                    # Blit onto the main screen
                    self.screen.blit(surf, (5, hover_y_start + i * line_height))

        except Exception as e:
            # Catch potential errors during hover info gathering/rendering (e.g., font issues)
            print(f"ERROR: Exception drawing hover info: {e}")
            # Optionally draw an error message at hover location
            try:
                 surf = self.debug_font.render("Hover Error!", True, (255,0,0))
                 self.screen.blit(surf, (5, self.world_height_px - line_height - 5))
            except Exception: pass # Ignore errors during error message rendering

    def _draw_legend(self):
        """
        Draws the simulation legend overlay if `self.show_legend` is True.

        Displays color swatches and labels for various entities, resources,
        and pheromones defined in the simulation constants. Positions the
        legend box relative to the top-right corner of the simulation area,
        below the UI buttons.
        """
        # Check if the legend font is loaded and legend is enabled
        if not self.legend_font or not self.show_legend:
            return

        # Define items to be included in the legend: (Label Text, Color Tuple or None for Titles)
        # Uses colors defined in the configuration constants section.
        legend_items = [
            ("--- Entities ---", None),      # Section Title
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
            ("", None),                     # Spacer
            ("--- Brood ---", None),         # Section Title
            ("Egg", EGG_COLOR),             # Use full RGBA, draw handles alpha
            ("Larva", LARVA_COLOR),
            ("Pupa", PUPA_COLOR),
            ("", None),                     # Spacer
            ("--- Resources ---", None),     # Section Title
            ("Food (Sugar)", FOOD_COLORS[FoodType.SUGAR]),
            ("Food (Protein)", FOOD_COLORS[FoodType.PROTEIN]),
            ("Obstacle", OBSTACLE_COLOR),   # Added Obstacle color
            ("", None),                     # Spacer
            ("--- Pheromones ---", None),    # Section Title
            ("Home Trail", PHEROMONE_HOME_COLOR),
            ("Food S Trail", PHEROMONE_FOOD_SUGAR_COLOR),
            ("Food P Trail", PHEROMONE_FOOD_PROTEIN_COLOR),
            ("Alarm Signal", PHEROMONE_ALARM_COLOR),
            ("Negative Trail", PHEROMONE_NEGATIVE_COLOR),
            ("Recruit Signal", PHEROMONE_RECRUITMENT_COLOR),
        ]

        # --- Calculate Legend Box Positioning and Sizing ---
        padding = 6 # Internal padding within the legend box
        line_height = self.legend_font.get_height() + padding // 2 # Calculate line height based on font
        swatch_size = self.legend_font.get_height() # Size of the color square next to text

        # Estimate required width based on the longest text label and swatch size
        max_text_width = 0
        for text, _ in legend_items:
            if text: # Ensure text is not None or empty
                try:
                    text_width = self.legend_font.size(text)[0]
                    max_text_width = max(max_text_width, text_width)
                except Exception:
                    pass # Ignore font errors during width calculation

        # Calculate total width and height needed for the legend box
        legend_width = max(120, swatch_size + max_text_width + 3 * padding) # Ensure minimum width
        legend_height = len(legend_items) * line_height + padding # Total height based on items

        # Position the legend box: Top-right corner, below buttons
        # Use world_width_px for positioning relative to the simulation area edge
        start_x = self.world_width_px - legend_width - 10 # 10px from right edge
        # Determine Y position below buttons, or default if buttons aren't present
        button_bottom_y = 10 # Default margin from top
        if self.buttons:
            try:
                button_bottom_y = self.buttons[0]['rect'].bottom + 10 # 10px below buttons
            except (IndexError, KeyError): pass # Use default if error
        start_y = button_bottom_y

        # --- Draw Legend Box ---
        try:
            # Create a dedicated surface for the legend with transparency support
            legend_surf = pygame.Surface((legend_width, legend_height), pygame.SRCALPHA)
            # Fill with semi-transparent background color
            legend_surf.fill(LEGEND_BG_COLOR)

            # --- Draw Legend Items onto the Legend Surface ---
            current_y = padding # Start drawing below top padding
            for text, color in legend_items:
                text_x_offset = padding # Default starting X for text

                # Draw color swatch if a color is provided for this item
                if color and isinstance(color, tuple) and len(color) >= 3:
                    swatch_rect = pygame.Rect(padding, current_y + 1, swatch_size, swatch_size)
                    # Use only RGB for swatch color, ignore alpha for solid block
                    draw_color = color[:3]
                    try:
                        pygame.draw.rect(legend_surf, draw_color, swatch_rect)
                    except TypeError: # Handle potential color format errors
                         pygame.draw.rect(legend_surf, (255,0,255), swatch_rect) # Draw Magenta on error
                    # Indent the text to appear after the swatch
                    text_x_offset += swatch_size + padding // 2
                elif color is None and text.startswith("---"):
                     # Center-align titles/separators slightly
                     text_x_offset = padding * 2
                # else: Title or spacer, draw text normally starting at default offset


                # Render and blit the text label
                if text: # Only draw if text is not empty
                    try:
                        # Render using legend font and color
                        text_surface = self.legend_font.render(text, True, LEGEND_TEXT_COLOR)
                        # Blit onto the legend surface at calculated position
                        legend_surf.blit(text_surface, (text_x_offset, current_y))
                    except Exception as e:
                        # Log font rendering errors but continue drawing other items
                        print(f"ERROR: Legend Font render error ('{text}'): {e}")

                # Move drawing position down for the next line
                current_y += line_height

            # --- Blit the Completed Legend Surface onto the Main Screen ---
            self.screen.blit(legend_surf, (start_x, start_y))

        except pygame.error as e:
            # Catch Pygame-specific errors during surface creation or drawing
            print(f"ERROR: Pygame error creating or drawing legend surface: {e}")
        except Exception as e:
            # Catch any other unexpected errors during legend drawing
            print(f"ERROR: Unexpected error drawing legend: {e}")

    # --- Drawing Methods ---

    def draw(self):
        """
        Draws all elements of the current simulation state onto the main screen.

        This is the main drawing routine called once per frame. It orchestrates
        the drawing of the grid, entities (brood, queen, ants, enemies, prey),
        UI overlays (debug info, legend, buttons), and captures the frame for
        network streaming if enabled.
        """
        # --- 1. Draw Grid Elements (Background, Food, Pheromones, Nest Area) ---
        try:
            self._draw_grid()
        except Exception as e:
            print(f"ERROR: Exception in _draw_grid: {e}")
            traceback.print_exc() # Continue if possible, but log error

        # --- 2. Draw Static/Less Dynamic Entities ---
        try:
            self._draw_brood() # Brood items change appearance slowly
            self._draw_queen() # Queen is static unless attacked
        except Exception as e:
            print(f"ERROR: Exception drawing brood/queen: {e}")
            traceback.print_exc()

        # --- 3. Draw Mobile Entities ---
        # Draw enemies, prey, and ants on top of the grid and static elements.
        # The individual draw methods handle invalid positions.
        try:
            self._draw_queen()
            self._draw_enemies()
            self._draw_prey()
            self._draw_ants() # Ants drawn last to appear on top
        except Exception as e:
            print(f"ERROR: Exception drawing mobile entities: {e}")
            traceback.print_exc()

        # --- 4. Draw Visual Effects ---
        try:
            # Draw attack indicators (lines/flashes for recent attacks)
            self._draw_attack_indicators()
        except Exception as e:
            print(f"ERROR: Exception drawing attack indicators: {e}")
            traceback.print_exc()

        # --- 5. Draw UI Overlays ---
        try:
            # --- KORREKTUR: Sicherstellen, dass diese Zeilen NICHT auskommentiert sind ---
            # Draw debug statistics overlay if enabled (method checks internal flag)
            self.draw_debug_info()
            # Draw legend overlay if enabled (method checks internal flag)
            self._draw_legend()
            # --- ENDE KORREKTUR ---

            # Draw UI buttons last so they are on top of overlays (usually desired)
            self._draw_buttons()
        except Exception as e:
            print(f"ERROR: Exception drawing UI elements: {e}")
            traceback.print_exc()

        # --- 6. Frame Capture for Network Streaming ---
        # If streaming is enabled, capture a *copy* of the fully drawn screen
        if ENABLE_NETWORK_STREAM and Flask:
            try:
                self.latest_frame_surface = self.screen.copy()
            except pygame.error as e:
                print(f"Error copying screen surface for streaming: {e}")
                self.latest_frame_surface = None
        else:
            self.latest_frame_surface = None

        # --- 7. Final Display Update ---
        # Flip the Pygame display buffer to show the newly drawn frame
        try:
            pygame.display.flip()
        except pygame.error as e:
            print(f"ERROR: Pygame error during display flip: {e}")
            self.app_running = False

    def _draw_grid(self):
        """
        Draws the grid background, food items, pheromones, and nest area highlight.

        Blits the pre-rendered static background (map color, obstacles).
        Draws food using a stable dot pattern based on amount and type.
        Draws the pheromone overlay, updating a cached surface periodically
        if pheromones are currently visible. Draws a visual highlight for the nest area.
        """
        cs = self.cell_size  # Local alias for cell size

        # --- 1. Blit Static Background (Map Color & Obstacles) ---
        try:
            # Draw the pre-rendered surface containing the background color and obstacles.
            self.screen.blit(self.static_background_surface, (0, 0))
        except pygame.error as e:
            print(f"ERROR: Failed to blit static background: {e}")
            # Attempting to continue might lead to more errors, but try.
        except Exception as e:
            print(f"ERROR: Unexpected error blitting static background: {e}")

        # --- 2. Draw Pheromones (using cached surface) ---
        # Pheromones are drawn onto a separate transparent surface (`pheromone_surface`)
        # which is updated periodically to improve performance.

        if self.show_pheromones:
            current_int_tick = self.ticks # Get current tick
            # Check if enough ticks passed since last update OR if toggle forces redraw (handled by setting last_tick=-100)
            # Update pheromone surface every few ticks (e.g., 3) when visible.
            pheromone_update_interval = 3
            needs_ph_update = (current_int_tick - self.last_pheromone_update_tick >= pheromone_update_interval)

            # --- 2a. Update the Pheromone Cache Surface if Needed ---
            if needs_ph_update:
                self.last_pheromone_update_tick = current_int_tick # Update time first
                # Clear the surface completely before redrawing all pheromones
                self.pheromone_surface.fill((0, 0, 0, 0))

                # Define pheromone types, their colors, source arrays, and max values
                ph_info = {
                    "home": (PHEROMONE_HOME_COLOR, self.grid.pheromones_home, PHEROMONE_MAX),
                    "food_sugar": (PHEROMONE_FOOD_SUGAR_COLOR, self.grid.pheromones_food_sugar, PHEROMONE_MAX),
                    "food_protein": (PHEROMONE_FOOD_PROTEIN_COLOR, self.grid.pheromones_food_protein, PHEROMONE_MAX),
                    "alarm": (PHEROMONE_ALARM_COLOR, self.grid.pheromones_alarm, PHEROMONE_MAX),
                    "negative": (PHEROMONE_NEGATIVE_COLOR, self.grid.pheromones_negative, PHEROMONE_MAX),
                    "recruitment": (PHEROMONE_RECRUITMENT_COLOR, self.grid.pheromones_recruitment, RECRUITMENT_PHEROMONE_MAX),
                }
                min_alpha_for_draw = 5 # Minimum calculated alpha to actually draw the rect

                # Iterate through each pheromone type to draw it onto the cache surface
                for ph_type, (base_col, arr, current_max) in ph_info.items():
                    try:
                        # Normalize strength relative to a fraction of max for visual scaling
                        # Adjust divisor (e.g., current_max / 2.5) to control intensity appearance
                        norm_divisor = max(current_max / 2.5, 1.0)
                        # Find coordinates where pheromone strength exceeds the drawing threshold
                        nz_coords = np.argwhere(arr > MIN_PHEROMONE_DRAW_THRESHOLD)

                        # Draw a semi-transparent rectangle for each cell with sufficient pheromone
                        for x, y in nz_coords:
                             # Bounds check (redundant due to argwhere, but safe)
                            if not (0 <= x < self.grid_width and 0 <= y < self.grid_height): continue
                            try:
                                val = arr[x, y] # Get pheromone value
                                # Normalize value for alpha calculation
                                norm_val = normalize(val, norm_divisor)
                                # Get base alpha from color tuple, default to 255 if not specified
                                alpha_base = base_col[3] if len(base_col) > 3 else 255
                                # Calculate final alpha, clamping between 0 and 255
                                alpha = min(max(int(norm_val * alpha_base), 0), 255)

                                # Only draw if alpha is above the minimum threshold
                                if alpha >= min_alpha_for_draw:
                                    # Create color tuple with calculated alpha
                                    color = (*base_col[:3], alpha)
                                    # Define rectangle coordinates and size
                                    rect_coords = (x * cs, y * cs, cs, cs)
                                    # Draw the rectangle onto the pheromone cache surface
                                    pygame.draw.rect(self.pheromone_surface, color, rect_coords)
                            # Catch errors during drawing for a single cell
                            except IndexError: continue
                            except (ValueError, TypeError) as e: continue
                    # Catch errors during processing of a whole pheromone type
                    except (pygame.error, ValueError, Exception) as e:
                        print(f"ERROR: Exception updating pheromone surface for '{ph_type}': {e}")

            # --- 2b. Blit the Cached Pheromone Surface onto the Main Screen ---
            try:
                # Draw the (potentially updated) pheromone surface over the background/obstacles.
                self.screen.blit(self.pheromone_surface, (0, 0))
            except Exception as e:
                print(f"ERROR: Failed to blit pheromone surface: {e}")

        # --- 3. Draw Food (Stable Dot Pattern) ---
        try:
            # Calculate total food per cell to efficiently find cells with any food
            food_totals = np.sum(self.grid.food, axis=2)
            # Minimum total food required in a cell to attempt drawing dots
            min_food_for_dot_check = 0.01
            # Get coordinates of cells containing food above the threshold
            food_nz_coords = np.argwhere(food_totals > min_food_for_dot_check)

            # Get indices and colors for food types
            s_idx = FoodType.SUGAR.value
            p_idx = FoodType.PROTEIN.value
            s_col = FOOD_COLORS[FoodType.SUGAR]
            p_col = FOOD_COLORS[FoodType.PROTEIN]
            dot_radius = max(1, FOOD_DOT_RADIUS) # Ensure radius is at least 1 pixel

            # Use the dedicated RNG for food dots to ensure patterns are consistent
            # unless the food amounts change significantly.
            food_rng = self.food_dot_rng # Use the simulation's dedicated RNG instance
            half_cs = cs / 2.0 # Pre-calculate half cell size

            for x, y in food_nz_coords:
                # Bounds check (redundant but safe)
                if not (0 <= x < self.grid_width and 0 <= y < self.grid_height): continue
                try:
                    foods = self.grid.food[x, y]
                    s = foods[s_idx] # Sugar amount
                    p = foods[p_idx] # Protein amount
                    total = s + p
                    if total < min_food_for_dot_check: continue # Skip if negligible amount

                    # Calculate number of dots based on total food, clamped by max dots per cell
                    num_dots = max(1, min(FOOD_MAX_DOTS_PER_CELL, int(total * FOOD_DOTS_PER_UNIT)))

                    # Determine mixed color based on the ratio of sugar to protein
                    sr = s / total if total > 0 else 0.5 # Sugar ratio
                    pr = 1.0 - sr                     # Protein ratio
                    color_mixed = (int(s_col[0] * sr + p_col[0] * pr),
                                   int(s_col[1] * sr + p_col[1] * pr),
                                   int(s_col[2] * sr + p_col[2] * pr))
                    # Clamp mixed color components to valid RGB range [0, 255]
                    dot_color = tuple(max(0, min(255, c)) for c in color_mixed)

                    # Seed the RNG based on the cell's coordinates for consistent patterns
                    # Use a combination of x and y to generate a unique integer seed per cell.
                    cell_seed = int(x * self.grid_height + y) # Simple seeding approach
                    food_rng.seed(cell_seed)

                    # Calculate drawing boundaries within the cell
                    cell_x_start = x * cs
                    cell_y_start = y * cs
                    # Define margins within the cell to draw dots, ensuring dots don't overlap edges perfectly
                    min_draw_offset = dot_radius
                    max_draw_offset = cs - dot_radius

                    # Draw the calculated number of dots
                    for _ in range(num_dots):
                         # Generate random position within the cell's drawing boundaries
                         # Check if drawing area is valid (cell size > 2*radius)
                        if max_draw_offset > min_draw_offset:
                            dot_x = cell_x_start + food_rng.uniform(min_draw_offset, max_draw_offset)
                            dot_y = cell_y_start + food_rng.uniform(min_draw_offset, max_draw_offset)
                        else: # If cell is too small, just draw in center
                            dot_x = cell_x_start + half_cs
                            dot_y = cell_y_start + half_cs

                        # Draw the dot circle onto the main screen
                        pygame.draw.circle(self.screen, dot_color, (int(dot_x), int(dot_y)), dot_radius)

                # Catch errors during drawing for a specific cell
                except IndexError: continue
                except (ValueError, TypeError) as e:
                     print(f"ERROR: Value/Type Error drawing food at {(x, y)}: Color={dot_color}, Error={e}")
                     continue
        # Catch errors during the overall food drawing process
        except Exception as e:
            print(f"ERROR: General exception during food grid drawing: {e}")
            traceback.print_exc()

        # --- 4. Draw Nest Area Highlight ---
        # Draw a subtle visual indicator for the nest area boundaries.
        nest_radius_grid = NEST_RADIUS
        nx, ny = self.nest_pos # Nest center grid coordinates
        # Calculate pixel center and radius
        center_x_px = int(nx * cs + cs / 2)
        center_y_px = int(ny * cs + cs / 2)
        radius_px = nest_radius_grid * cs

        try:
            # Ensure radius is positive before attempting to draw
            if radius_px <= 0: raise ValueError("Nest radius non-positive")
            # Create a temporary surface for the circle to handle alpha transparency correctly
            surf_size = int(radius_px * 2)
            if surf_size <= 0: raise ValueError("Nest highlight surface size non-positive")

            nest_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA) # Use SRCALPHA for transparency
            nest_surf.fill((0, 0, 0, 0)) # Fill with fully transparent

            # Draw the semi-transparent circle onto the temporary surface
            # Color: Light grey, semi-transparent (e.g., alpha 35)
            highlight_color = (100, 100, 100, 35)
            pygame.draw.circle(nest_surf, highlight_color,
                               (surf_size // 2, surf_size // 2), # Center of the temp surface
                               radius_px) # Radius in pixels

            # Blit the temporary surface onto the main screen, centered at the nest position
            blit_pos = (center_x_px - surf_size // 2, center_y_px - surf_size // 2)
            self.screen.blit(nest_surf, blit_pos)

        except (ValueError, pygame.error, OverflowError) as e:
            # Silently ignore failures to draw the nest highlight (non-critical)
            # print(f"WARN: Could not draw nest highlight: {e}") # Optional Debug
            pass

    def _draw_brood(self):
        """
        Draws all active brood items (eggs, larvae, pupae) onto the screen.

        Iterates through the simulation's brood list and calls the `draw`
        method of each individual BroodItem object. Performs basic checks
        to ensure the item still exists and has a valid position before drawing.
        """
        # Iterate over a shallow copy of the list for safety, in case the original
        # list is modified during drawing (e.g., by hatching in the same frame, though unlikely here).
        for item in list(self.brood):
            try:
                # Check if the item still exists in the main brood list and has a valid grid position
                if item in self.brood and is_valid_pos(item.pos, self.grid_width, self.grid_height):
                    # Call the brood item's own draw method, passing the main screen surface
                    item.draw(self.screen)
            except Exception as e:
                # Catch potential errors within the brood item's draw method
                print(f"ERROR: Failed to draw brood item {item} at {getattr(item, 'pos', 'N/A')}: {e}")
                # Attempt to continue drawing other items

    def _draw_queen(self):

        if self.queen and is_valid_pos(self.queen.pos, self.grid_width, self.grid_height):
            self.queen.draw(self.screen)

    def _draw_ants(self):
        """
        Draws all active worker and soldier ants onto the screen.

        Iterates through the simulation's ants list and calls the `draw`
        method of each individual Ant object. Performs basic checks before drawing.
        """
        # Iterate over a shallow copy of the ants list for safety during drawing
        for ant in list(self.ants):
            try:
                # Check if the ant object still exists in the main list
                # (it might have died and been removed in the same update cycle before drawing)
                # Also check if its position is valid on the grid.
                if ant in self.ants and is_valid_pos(ant.pos, self.grid_width, self.grid_height):
                    # Call the ant's own draw method, which handles orientation, color based on state, etc.
                    ant.draw(self.screen)
            except Exception as e:
                # Catch potential errors within the individual ant's draw method
                ant_id = getattr(ant, 'index', id(ant)) # Get index if available, else object id
                print(f"ERROR: Failed to draw ant {ant_id} at {getattr(ant, 'pos', 'N/A')}: {e}")
                # Attempt to continue drawing other ants

    def _draw_enemies(self):
        """
        Draws all active enemy entities onto the screen.

        Iterates through the simulation's enemies list and calls the `draw`
        method of each individual Enemy object. Performs basic checks before drawing.
        """
        # Iterate over a shallow copy of the enemies list for safety
        for enemy in list(self.enemies):
            try:
                # Check if the enemy object still exists and has a valid grid position
                if enemy in self.enemies and is_valid_pos(enemy.pos, self.grid_width, self.grid_height):
                    # Call the enemy's own draw method
                    enemy.draw(self.screen)
            except Exception as e:
                # Catch potential errors within the individual enemy's draw method
                print(f"ERROR: Failed to draw enemy {id(enemy)} at {getattr(enemy, 'pos', 'N/A')}: {e}")
                # Attempt to continue drawing other enemies

    def _draw_prey(self):
        """
        Draws all active prey entities onto the screen.

        Iterates through the simulation's prey list and calls the `draw`
        method of each individual Prey object. Performs basic checks before drawing.
        """
        # Iterate over a shallow copy of the prey list for safety
        for prey_item in list(self.prey):
            try:
                # Check if the prey object still exists and has a valid grid position
                if prey_item in self.prey and is_valid_pos(prey_item.pos, self.grid_width, self.grid_height):
                    # Call the prey's own draw method
                    prey_item.draw(self.screen)
            except Exception as e:
                # Catch potential errors within the individual prey's draw method
                print(f"ERROR: Failed to draw prey {id(prey_item)} at {getattr(prey_item, 'pos', 'N/A')}: {e}")
                # Attempt to continue drawing other prey items

    def _draw_buttons(self):
        """
        Draws the UI buttons onto the screen.

        Iterates through the `self.buttons` list (created by `_create_buttons`).
        Changes button background color on mouse hover. Renders button text.
        """
        # Check if font and button list are initialized
        if not self.font or not self.buttons:
            return

        try:
            # Get current mouse position for hover detection
            mouse_pos = pygame.mouse.get_pos()

            # Iterate through each button defined in the list
            for button in self.buttons:
                rect = button["rect"]
                text = button["text"]
                action = button["action"] # Not used for drawing, but available

                # Determine button background color based on mouse hover state
                is_hovered = rect.collidepoint(mouse_pos)
                color = BUTTON_HOVER_COLOR if is_hovered else BUTTON_COLOR

                # Draw the button background rectangle with rounded corners
                pygame.draw.rect(self.screen, color, rect, border_radius=3)

                # --- Render and Draw Button Text ---
                try:
                    # Render the text using the main UI font and color
                    text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
                    # Get the rectangle of the rendered text surface
                    text_rect = text_surf.get_rect()
                    # Center the text rectangle within the button rectangle
                    text_rect.center = rect.center
                    # Blit the text surface onto the main screen at the centered position
                    self.screen.blit(text_surf, text_rect)
                except Exception as e:
                    # Log errors during text rendering for a specific button, but continue
                    print(f"ERROR: Button font render error ('{text}'): {e}")

        except Exception as e:
            # Catch unexpected errors during the overall button drawing process
            print(f"ERROR: Exception drawing UI buttons: {e}")

    def handle_events(self) -> str | None:
        """
        Processes Pygame events for user input.

        Handles quitting the application, keyboard shortcuts corresponding to
        UI buttons, and mouse clicks on UI buttons.

        Returns:
            A string indicating a significant action ("quit_app", "sim_stop",
            "ui_action") or None if no significant event requiring special
            handling occurred.
        """
        for event in pygame.event.get():
            # --- Quit Event (Window Close Button) ---
            if event.type == pygame.QUIT:
                self.simulation_running = False # Stop simulation loop
                self.app_running = False      # Stop application loop
                self.end_game_reason = "Window Closed"
                print("Event: Quit requested (Window Close).")
                return "quit_app" # Signal immediate application exit

            # --- Key Presses ---
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key corresponds to a defined button shortcut
                key_action = None
                for button in self.buttons:
                    # Check if button has a key defined and if it matches the event key
                    if button.get("key") is not None and button["key"] == event.key:
                        key_action = button["action"]
                        break # Found matching key action

                if key_action:
                    print(f"Event: Key '{pygame.key.name(event.key)}' triggered action '{key_action}'.")
                    # Handle the action associated with the key press
                    action_result = self._handle_button_click(key_action)
                    # Map the action result to return signals
                    if key_action == "quit": return "quit_app"
                    if key_action == "restart": return "sim_stop" # Signal sim stop for restart dialog
                    return "ui_action" # Generic signal for UI changes (debug, legend, pheromones)

                # Handle keys not directly bound to buttons (if any)
                # Example: A general pause key (currently not implemented)
                # if event.key == pygame.K_SPACE:
                #     # Toggle pause logic here
                #     print("Event: Pause toggled (Spacebar).")
                #     return "ui_action"

            # --- Mouse Click ---
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check only for left mouse button clicks
                if event.button == 1:
                    mouse_pos = event.pos # Get click coordinates
                    # Check if the click collided with any UI button
                    for button in self.buttons:
                        if button["rect"].collidepoint(mouse_pos):
                            action = button["action"]
                            print(f"Event: Mouse click on button '{button['text']}' triggered action '{action}'.")
                            # Handle the action associated with the button click
                            action_result = self._handle_button_click(action)
                            # Map the action result to return signals
                            if action == "quit": return "quit_app"
                            if action == "restart": return "sim_stop" # Signal sim stop for restart dialog
                            return "ui_action" # Generic UI action

        # No significant event requiring loop interruption occurred
        return None

    def _handle_button_click(self, action: str):
        """
        Handles the logic associated with clicking a UI button or pressing its shortcut key.

        Modifies the simulation state based on the provided action string.

        Args:
            action: A string identifying the action to perform (e.g.,
                    "toggle_debug", "restart", "quit"). Matches the 'action'
                    defined in `_create_buttons`.
        """
        print(f"Action triggered: {action}") # Log the action being handled

        if action == "toggle_debug":
            self.show_debug_info = not self.show_debug_info
            print(f"  Debug info {'enabled' if self.show_debug_info else 'disabled'}.")
        elif action == "toggle_legend":
            self.show_legend = not self.show_legend
            print(f"  Legend {'enabled' if self.show_legend else 'disabled'}.")
        elif action == "toggle_pheromones":
            self.show_pheromones = not self.show_pheromones
            print(f"  Pheromone display {'enabled' if self.show_pheromones else 'disabled'}.")
            if not self.show_pheromones:
                # Clear the pheromone cache surface immediately when turning off
                # to prevent showing the last frame briefly if turned back on quickly.
                self.pheromone_surface.fill((0, 0, 0, 0))
            else:
                # Force the pheromone cache to redraw on the next frame when turning on
                # by invalidating the last update tick.
                self.last_pheromone_update_tick = -100
        elif action == "restart":
            # Set flags to stop the current simulation run, triggering the end dialog.
            self.simulation_running = False
            self.end_game_reason = "Restart Button/Key"
            print("  Restart requested.")
        elif action == "quit":
            # Set flags to stop both the simulation and the main application loop.
            self.simulation_running = False
            self.app_running = False
            self.end_game_reason = "Quit Button/Key"
            print("  Quit requested.")
        # --- Speed control actions removed ---
        # elif action == "speed_down": ...
        # elif action == "speed_up": ...
        else:
            # Log a warning if an unrecognized action string is received.
            print(f"Warning: Unknown button action received: '{action}'")

        # This function primarily modifies state flags. The return value isn't
        # strictly necessary for the current event handling logic but could be used.
        # For now, just return None implicitly.

    def _show_end_game_dialog(self, auto_restart_enabled: bool = False):
        """
        Displays a dialog box when a simulation run ends (e.g., Queen dies).

        Offers the user options to restart the simulation or quit the application.
        Includes an optional auto-restart timer, typically triggered after the
        Queen's death.

        Args:
            auto_restart_enabled: If True, enables the countdown timer for
                                  automatic simulation restart. Defaults to False.

        Returns:
            A string indicating the user's choice: "restart" or "quit".
        """
        # Ensure necessary fonts are loaded before attempting to render text
        if not self.font:
            print("ERROR: Cannot display end game dialog - font not loaded.")
            # If fonts failed, we cannot show the dialog, force quit the app.
            self.app_running = False
            return "quit"

        # --- Auto-Restart Timer Setup ---
        auto_restart_start_time_ms = None
        auto_restart_delay_ms = AUTO_RESTART_DELAY_SECONDS * 1000 # Convert seconds to ms
        if auto_restart_enabled:
            auto_restart_start_time_ms = pygame.time.get_ticks() # Get current time in ms
            print(f"AntSimulation: Auto-restart timer started ({AUTO_RESTART_DELAY_SECONDS}s)...")

        # --- Dialog Box Configuration ---
        # Define dimensions and center position on the screen
        dialog_w, dialog_h = 350, 210 # Width, Height in pixels (increased height for timer)
        # Use world dimensions for centering, as screen might be larger
        dialog_x = (self.world_width_px - dialog_w) // 2
        dialog_y = (self.world_height_px - dialog_h) // 2

        # --- Button Configuration within Dialog ---
        btn_w, btn_h = 120, 40 # Button dimensions
        btn_margin = 30      # Horizontal margin between buttons
        # Calculate Y position for buttons near the bottom of the dialog
        btn_y = dialog_y + dialog_h - btn_h - 25 # Position relative to dialog bottom
        # Calculate X positions to center the two buttons horizontally
        total_btn_width = btn_w * 2 + btn_margin
        btn_restart_x = dialog_x + (dialog_w - total_btn_width) // 2
        btn_quit_x = btn_restart_x + btn_w + btn_margin
        # Create Rect objects for button collision detection and drawing
        restart_rect = pygame.Rect(btn_restart_x, btn_y, btn_w, btn_h)
        quit_rect = pygame.Rect(btn_quit_x, btn_y, btn_w, btn_h)

        # --- Text Content and Colors ---
        text_color = (240, 240, 240)    # Light grey/white for main text
        timer_color = (255, 255, 100)   # Yellow for timer text
        title_text = f"Generation {self.colony_generation} Ended"
        reason_text = f"Reason: {self.end_game_reason}"

        # --- Background Overlay ---
        # Create a semi-transparent surface to dim the background simulation view
        overlay = pygame.Surface((self.world_width_px, self.world_height_px), pygame.SRCALPHA)
        overlay.fill(END_DIALOG_BG_COLOR) # Use configured semi-transparent black

        # --- Dialog Event Loop ---
        # This loop runs while waiting for user input (Restart/Quit) or timer expiry.
        waiting_for_choice = True
        while waiting_for_choice and self.app_running:
            current_time_ms = pygame.time.get_ticks() # Get current time for timer check
            mouse_pos = pygame.mouse.get_pos()      # Get mouse position for hover effects

            # --- Auto-Restart Timer Check ---
            if auto_restart_start_time_ms is not None:
                elapsed_ms = current_time_ms - auto_restart_start_time_ms
                if elapsed_ms >= auto_restart_delay_ms:
                    print("AntSimulation: Auto-restart timer expired. Restarting simulation.")
                    return "restart" # Force restart when timer expires

            # --- Event Handling within Dialog ---
            for event in pygame.event.get():
                # Handle Quit event (window close)
                if event.type == pygame.QUIT:
                    self.app_running = False
                    waiting_for_choice = False
                    return "quit" # Exit app directly
                # Handle Keyboard Input (ESC to Quit)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.app_running = False
                    waiting_for_choice = False
                    return "quit" # Exit app on ESC
                # Handle Mouse Clicks on Buttons
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                    if restart_rect.collidepoint(mouse_pos):
                        print("AntSimulation: Restart chosen by user.")
                        return "restart" # Signal simulation restart
                    if quit_rect.collidepoint(mouse_pos):
                        print("AntSimulation: Quit chosen by user.")
                        self.app_running = False # Signal application quit
                        return "quit"

            # --- Drawing the Dialog ---
            # 1. Draw the background dimming overlay
            self.screen.blit(overlay, (0, 0))
            # 2. Draw the solid dialog box background
            dialog_bg_color = (40, 40, 80) # Dark blue
            pygame.draw.rect(self.screen, dialog_bg_color, (dialog_x, dialog_y, dialog_w, dialog_h), border_radius=6)

            # 3. Render and Draw Text Content
            try:
                # Title Text (Centered near top)
                title_surf = self.font.render(title_text, True, text_color)
                title_rect = title_surf.get_rect(center=(dialog_x + dialog_w // 2, dialog_y + 35))
                self.screen.blit(title_surf, title_rect)
                # Reason Text (Centered below title)
                reason_surf = self.font.render(reason_text, True, text_color)
                reason_rect = reason_surf.get_rect(center=(dialog_x + dialog_w // 2, dialog_y + 70))
                self.screen.blit(reason_surf, reason_rect)

                # Auto-Restart Timer Text (if applicable)
                if auto_restart_start_time_ms is not None:
                    remaining_ms = max(0, auto_restart_delay_ms - elapsed_ms)
                    remaining_sec = remaining_ms / 1000.0
                    timer_text = f"Auto-Restart in {remaining_sec:.1f}s"
                    timer_surf = self.font.render(timer_text, True, timer_color)
                    timer_rect = timer_surf.get_rect(center=(dialog_x + dialog_w // 2, dialog_y + 115))
                    self.screen.blit(timer_surf, timer_rect)

            except Exception as e:
                print(f"ERROR: Dialog text render error: {e}") # Log font rendering errors

            # 4. Draw Buttons (with hover effect)
            # Restart Button
            r_color = BUTTON_HOVER_COLOR if restart_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, r_color, restart_rect, border_radius=4)
            try:
                 r_text_surf = self.font.render("Restart", True, BUTTON_TEXT_COLOR) # Corrected text
                 r_text_rect = r_text_surf.get_rect(center=restart_rect.center)
                 self.screen.blit(r_text_surf, r_text_rect)
            except Exception as e: print(f"ERROR: Restart button text render error: {e}")
            # Quit Button
            q_color = BUTTON_HOVER_COLOR if quit_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, q_color, quit_rect, border_radius=4)
            try:
                 q_text_surf = self.font.render("Quit", True, BUTTON_TEXT_COLOR) # Corrected text
                 q_text_rect = q_text_surf.get_rect(center=quit_rect.center)
                 self.screen.blit(q_text_surf, q_text_rect)
            except Exception as e: print(f"ERROR: Quit button text render error: {e}")

            # Update the display to show the drawn dialog
            pygame.display.flip()
            # Control frame rate for the dialog loop (lower FPS is acceptable)
            self.clock.tick(30)

        # Loop exited without making a choice (e.g., app_running became False via external signal)
        # Default to quitting the application in this case.
        print("AntSimulation: End game dialog loop exited, defaulting to quit.")
        return "quit"

    def add_attack_indicator(self, attacker_pos_grid: tuple, target_pos_grid: tuple, color: tuple):
        """
        Registers a recent attack event to be visualized temporarily.

        Stores the positions and color associated with an attack, along with a
        timer, in the `self.recent_attacks` list. This list is processed by
        `_draw_attack_indicators` to create visual feedback.

        Args:
            attacker_pos_grid: The (x, y) grid coordinates of the attacker.
            target_pos_grid: The (x, y) grid coordinates of the target.
            color: The RGBA color tuple to use for drawing the indicator.
        """
        # Validate positions before adding indicator
        if not is_valid_pos(attacker_pos_grid, self.grid_width, self.grid_height) or \
           not is_valid_pos(target_pos_grid, self.grid_width, self.grid_height):
            # print(f"Warning: Ignoring attack indicator with invalid position(s): {attacker_pos_grid} -> {target_pos_grid}") # Optional debug
            return # Ignore attacks involving invalid grid positions

        # Create a dictionary to store indicator information
        indicator = {
            "attacker_pos": attacker_pos_grid,
            "target_pos": target_pos_grid,
            "color": color,
            "timer": float(ATTACK_INDICATOR_DURATION_TICKS) # Start timer (use float for smoother decay if needed)
        }
        # Append the indicator dictionary to the list
        self.recent_attacks.append(indicator)

    def _draw_attack_indicators(self):
        """
        Draws visual indicators (e.g., lines) for recent attack events.

        Iterates through the `self.recent_attacks` list. For each active
        indicator (timer > 0), it draws a line between the attacker and target
        positions using the specified color, fading the indicator out as its
        timer decreases. Removes expired indicators.
        """
        # Skip drawing if the list is empty
        if not self.recent_attacks:
            return

        cs = self.cell_size       # Cell size in pixels
        cs_half = cs / 2.0      # Half cell size for centering calculations
        indices_to_remove = [] # List to track expired indicators

        # Iterate backwards through the list for safe removal while iterating
        for i in range(len(self.recent_attacks) - 1, -1, -1):
            indicator = self.recent_attacks[i]

            # --- Update Timer ---
            # Decrement timer by 1 each frame (since simulation runs 1 tick per frame now)
            indicator["timer"] -= 1.0

            # --- Check for Expiry ---
            if indicator["timer"] <= 0:
                indices_to_remove.append(i) # Mark for removal
                continue # Skip drawing expired indicator

            # --- Calculate Pixel Positions ---
            try:
                attacker_px = (int(indicator["attacker_pos"][0] * cs + cs_half),
                               int(indicator["attacker_pos"][1] * cs + cs_half))
                target_px = (int(indicator["target_pos"][0] * cs + cs_half),
                             int(indicator["target_pos"][1] * cs + cs_half))
            except (TypeError, IndexError):
                 indices_to_remove.append(i) # Remove if positions are invalid
                 continue

            # --- Calculate Alpha for Fade-Out Effect ---
            base_alpha = indicator["color"][3] # Get base alpha from the RGBA color tuple
            # Calculate current alpha based on the proportion of time remaining
            time_fraction = indicator["timer"] / ATTACK_INDICATOR_DURATION_TICKS
            current_alpha = max(0, min(255, int(base_alpha * time_fraction)))

            # --- Draw the Indicator (if sufficiently visible) ---
            min_draw_alpha = 10 # Don't draw if almost fully faded
            if current_alpha >= min_draw_alpha:
                # Create the color tuple with the calculated faded alpha
                line_color = (*indicator["color"][:3], current_alpha)
                try:
                    # Draw a line connecting attacker and target centers
                    line_thickness = 1 # Thickness of the indicator line
                    pygame.draw.line(self.screen, line_color, attacker_px, target_px, line_thickness)
                    # Optional: Draw small circles at ends for emphasis
                    # pygame.draw.circle(self.screen, line_color, attacker_px, 3)
                    pygame.draw.circle(self.screen, line_color, target_px, 3)
                except (TypeError, ValueError) as e:
                    # Catch errors related to color format or drawing parameters
                    print(f"Warning: Error drawing attack indicator: {e}, Color: {line_color}")
                    indices_to_remove.append(i) # Remove problematic indicator


        # --- Remove Expired Indicators ---
        # Remove marked indicators from the list efficiently
        if indices_to_remove:
            # Sort indices in ascending order to remove correctly after reversing iteration
            indices_to_remove.sort()
            # Remove elements by index, starting from the highest index to avoid shifting issues
            for index in reversed(indices_to_remove):
                 # Basic range check before deleting
                 if 0 <= index < len(self.recent_attacks):
                     del self.recent_attacks[index]
                 # else: # Debugging for out-of-range index
                     # print(f"Warning: Index {index} out of range for recent_attacks removal (len={len(self.recent_attacks)})")

    # This method was present but its calls seem to have been commented out
    # in the scoring/selection logic. If detailed per-decision logging is
    # desired, the calls within ant movement methods need to be re-enabled
    # and the log_data structure defined carefully. The main `update` method
    # currently handles periodic summary logging.
    def log_ant_decision(self, log_data: dict):
        """
        (Potentially Unused) Processes and logs detailed data about factors
        influencing an individual ant's movement decision.

        Configured to append data to the main simulation log file if file logging
        is active. Requires careful definition of the `log_data` dictionary
        structure and calls from within ant scoring/selection methods.

        Args:
            log_data: A dictionary containing key-value pairs representing
                      decision factors (e.g., scores, pheromones, chosen move).
        """
        # Check if the main log file is configured
        if self.log_filename:
            try:
                # Open file in append mode, ensuring it's closed afterwards
                with open(self.log_filename, "a", encoding='utf-8') as f:
                    # Simple conversion to CSV: Join dictionary values with commas.
                    # Assumes a flat structure and appropriate string conversion.
                    # Important: This might mix detailed decision logs with summary logs
                    # if appended to the same file without distinct headers/formats.
                    # Consider logging to a separate file if this detail is needed.
                    values_str = [str(v) for v in log_data.values()] # Ensure string conversion
                    csv_line = ",".join(values_str) + "\n"
                    f.write(csv_line)
            except Exception as e:
                print(f"ERROR writing detailed ant decision log: {e}")
                # Avoid potential infinite loops if logging itself causes errors.
                # Consider disabling this specific logging after the first error.
        # else: # If file logging is disabled
            # Alternative: Store logs in memory (can consume significant RAM)
            # if not hasattr(self, 'decision_log_memory'): self.decision_log_memory = []
            # self.decision_log_memory.append(log_data)
            pass # Currently does nothing if file logging is off

    def run(self):
        """
        Main application loop for the ant simulation.

        Handles the overall execution flow, including managing individual
        simulation runs (generations), displaying the end-game dialog,
        processing user choices (restart/quit), and controlling the frame rate.
        """
        print("Starting Ant Simulation...")
        print("Controls: [D] Stats | [L] Legend | [P] Pheromones | [R] Restart | [ESC] Quit")

        # --- Network Streaming Information ---
        if ENABLE_NETWORK_STREAM and Flask:
            # Attempt to guess the accessible IP address for the stream URL
            hostname = STREAMING_HOST if STREAMING_HOST != "0.0.0.0" else "localhost"
            try:
                # Try to get the local network IP (requires network connection)
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80)) # Connect to external server (doesn't send data)
                local_ip = s.getsockname()[0]
                s.close()
                hostname = local_ip # Use local IP if found
            except Exception:
                pass # Stick with default hostname if IP fetch fails
            print(f"INFO: Network stream *may be* available at http://{hostname}:{STREAMING_PORT}")
            if STREAMING_HOST == "0.0.0.0":
                print("      (Host 0.0.0.0 means access may be possible from other devices on your network)")

        # --- Main Application Loop ---
        # This loop continues as long as the application is intended to run (`self.app_running`).
        # It allows for multiple simulation runs (generations) via restarts.
        while self.app_running:

            # --- Inner Simulation Loop ---
            # This loop runs for a single simulation generation, continuing as long as
            # `self.simulation_running` is true (e.g., queen is alive) and the app hasn't quit.
            while self.simulation_running and self.app_running:

                # 1. Process User Input (Keyboard/Mouse Events)
                # Handles quit requests, UI toggles, restart requests.
                event_action = self.handle_events()

                # Handle actions that affect the simulation or application loops
                if event_action == "quit_app":
                    # Ensure both loops terminate if quit is requested
                    self.app_running = False
                    break # Exit inner simulation loop immediately
                if event_action == "sim_stop":
                    # Stop the current simulation run to show end/restart dialog
                    self.simulation_running = False # Will exit inner loop naturally
                    break # Exit inner simulation loop

                # 2. Update Simulation State (Perform one tick)
                # This advances the simulation logic by one step.
                try:
                    self.update()
                except Exception as e:
                    print("\n--- FATAL ERROR DURING SIMULATION UPDATE ---")
                    print(f"Tick: {self.ticks}")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    print("------------------------------------------")
                    # Stop the simulation and potentially the app on critical update errors
                    self.simulation_running = False
                    self.app_running = False # Consider making this optional
                    self.end_game_reason = "Runtime Error"
                    break # Exit inner simulation loop

                # 3. Draw the Current Frame
                # Renders the simulation state onto the screen.
                try:
                     self.draw()
                except Exception as e:
                     print("\n--- FATAL ERROR DURING SIMULATION DRAW ---")
                     print(f"Tick: {self.ticks}")
                     print(f"Error: {e}")
                     traceback.print_exc()
                     print("------------------------------------------")
                     # Stop the simulation and potentially the app on critical draw errors
                     self.simulation_running = False
                     self.app_running = False # Consider making this optional
                     self.end_game_reason = "Drawing Error"
                     break # Exit inner simulation loop


                # 4. Control Frame Rate
                # Limits the loop speed to the target FPS. Also yields CPU time.
                self.clock.tick(self.target_fps)
            # --- End Inner Simulation Loop ---

            # Check if the application should exit completely (e.g., quit requested)
            if not self.app_running:
                break # Exit outer application loop

            # --- Show End Game/Restart Dialog ---
            # This code is reached when the inner simulation loop ends (e.g., queen died, restart button)
            # but the application itself hasn't quit yet.

            # Set a default reason if none was provided (e.g., manual stop without reason)
            if not self.end_game_reason:
                self.end_game_reason = "Simulation Ended"

            # Determine if auto-restart should be enabled (e.g., only after Queen dies)
            enable_auto_restart = (self.end_game_reason == "Queen Died")

            # Display the dialog and wait for user choice ("restart" or "quit")
            choice = self._show_end_game_dialog(auto_restart_enabled=enable_auto_restart)

            # Handle the user's choice from the dialog
            if choice == "restart":
                # Reset the simulation state to start a new generation
                self._reset_simulation()
                # The simulation_running flag is set within _reset_simulation if successful.
                # The outer while loop will then re-enter the inner simulation loop.
            elif choice == "quit":
                # User chose Quit in the dialog, or dialog loop exited abnormally.
                self.app_running = False # Signal outer application loop to terminate

        # --- Application Exit ---
        print("AntSimulation: Exiting application.")
        # Attempt to stop the network streaming thread gracefully if it's running
        self._stop_streaming_server()
        # Quit Pygame modules
        try:
            pygame.quit()
            print("AntSimulation: Pygame shut down gracefully.")
        except Exception as e:
            print(f"ERROR: Exception during Pygame quit: {e}")

# --- Main Execution Block ---

# This block executes only when the script is run directly (not imported as a module).
if __name__ == "__main__":
    print("\n--- Ant Simulation Startup ---")

    # --- Basic Dependency Checks and Info ---
    print("Performing basic checks...")
    try:
        print(f"  Pygame Version: {pygame.version.ver}")
    except NameError:
        print("FATAL: Pygame library not found or import failed. Please install Pygame.")
        exit() # Cannot run without Pygame
    try:
        print(f"  NumPy Version: {np.__version__}")
    except NameError:
        print("FATAL: NumPy library not found or import failed. Please install NumPy.")
        exit() # Cannot run without NumPy
    try:
        # SciPy is used for pheromone diffusion
        import scipy
        print(f"  SciPy Version: {scipy.__version__}")
    except ImportError:
        print("FATAL: SciPy library not found or import failed. Please install SciPy.")
        exit() # Cannot run without SciPy (for diffusion)
    except NameError:
         print("FATAL: SciPy name not defined after import? Check installation.")
         exit()

    # Check for optional Flask library (for network streaming)
    if Flask:
        # Flask is imported, check version if possible (may not have __version__)
        flask_version = getattr(Flask, '__version__', 'Unknown')
        print(f"  Flask Version: {flask_version} (Network Streaming available: {ENABLE_NETWORK_STREAM})")
    else:
        print("  Flask library not found (Network Streaming disabled).")
    print("Basic checks complete.")

    # --- Setup Logging File ---
    log_file_to_pass = None # Variable to hold the log filename if logging is enabled
    if ENABLE_SIMULATION_LOGGING:
        log_file_to_pass = SIMULATION_LOG_FILE
        print(f"Attempting to prepare log file: {log_file_to_pass}")
        try:
            # Check if file exists and is empty to decide whether to write the header
            file_exists = os.path.exists(log_file_to_pass)
            # Consider file empty if it doesn't exist or has size 0
            is_empty = not file_exists or os.path.getsize(log_file_to_pass) == 0

            # Open file in append mode ('a'). Creates the file if it doesn't exist.
            # Use utf-8 encoding for broader compatibility.
            with open(log_file_to_pass, 'a', encoding='utf-8') as f:
                if is_empty:
                    print(f"  Log file is new or empty. Writing header.")
                    f.write(LOG_HEADER)
                else:
                    print(f"  Log file exists. Appending data.")
            print(f"Logging enabled. Data will be appended to: {log_file_to_pass}")
        except Exception as e:
            print(f"ERROR preparing log file '{log_file_to_pass}': {e}")
            print("Disabling logging for this session due to error.")
            log_file_to_pass = None # Disable logging if file preparation failed
            # Force the global flag off too, just in case it's used elsewhere directly
            ENABLE_SIMULATION_LOGGING = False
    else:
        print("Simulation logging is disabled by configuration (ENABLE_SIMULATION_LOGGING=False).")

    # --- Initialize Simulation Class ---
    initialization_success = False
    simulation_instance = None # Variable to hold the simulation object

    # Profiling setup (kept disabled as per original code)
    ENABLE_PROFILING = False
    profiler = None
    if ENABLE_PROFILING:
        try:
            import cProfile
            import pstats
            print("Profiling enabled.")
            profiler = cProfile.Profile()
        except ImportError:
            print("Warning: cProfile module not found. Profiling disabled.")
            ENABLE_PROFILING = False

    try:
        print("Attempting to initialize AntSimulation class...")
        # Pass the configured log filename (or None) to the constructor
        simulation_instance = AntSimulation(log_filename=log_file_to_pass)

        # Check the `app_running` flag set by the constructor to confirm success
        if simulation_instance.app_running:
            initialization_success = True
            print("AntSimulation initialized successfully.")
        else:
            # Initialization failed internally (e.g., Pygame, Font error)
            # Error messages should have been printed by the constructor.
            print("ERROR: AntSimulation initialization failed (check logs above).")

    except Exception as e:
        # Catch any unexpected errors during the AntSimulation constructor call
        print("\n--- FATAL ERROR DURING SIMULATION INITIALIZATION ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print("-------------------------------------------------")
        initialization_success = False # Ensure flag reflects failure

    # --- Run Simulation if Initialization Succeeded ---
    if initialization_success and simulation_instance:
        print("\nStarting simulation run...")
        try:
            # --- Start Profiling (if enabled) ---
            if ENABLE_PROFILING and profiler:
                print("Starting profiler...")
                profiler.enable()

            # --- Execute the Main Simulation Loop ---
            simulation_instance.run()

            # --- Stop Profiling (if enabled) ---
            if ENABLE_PROFILING and profiler:
                profiler.disable()
                print("Profiling finished. Processing results...")
                # Sort results by cumulative time and print top 30 functions
                stats = pstats.Stats(profiler).sort_stats('cumulative')
                stats.print_stats(30)
                # Optionally save full stats to a file
                # stats_filename = "antsim_profile.prof"
                # stats.dump_stats(stats_filename)
                # print(f"Full profiling stats saved to {stats_filename}")

        except Exception as e:
            # Catch unexpected errors during the simulation run itself
            print("\n--- FATAL ERROR DURING SIMULATION RUN ---")
            print(f"Error: {e}")
            traceback.print_exc()
            # Attempt to quit Pygame gracefully even after runtime error
            try:
                pygame.quit()
            except Exception as pqe:
                print(f"Error during Pygame quit after runtime error: {pqe}")
            # Prevent console window from closing immediately on Windows
            input("Press Enter to Exit after runtime error.")

    elif not simulation_instance:
         print("\nERROR: Simulation instance was not created. Cannot run.")
         input("Press Enter to Exit.")
    else:
         print("\nSimulation initialization failed. Cannot run.")
         # Attempt graceful Pygame quit if it was initialized partially
         if pygame.display.get_init():
              pygame.quit()
         input("Press Enter to Exit.")


    # --- End of Script ---
    print("\n--- Ant Simulation script finished ---")
