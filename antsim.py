# -*- coding: utf-8 -*-
import pygame
import random
import math
import time
from enum import Enum
import numpy as np

# --- Configuration Constants ---

# World & Grid
GRID_WIDTH = 150
GRID_HEIGHT = 100
CELL_SIZE = 8
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE
MAP_BG_COLOR = (20, 20, 10)

# Nest
NEST_POS = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
NEST_RADIUS = 5

# Food
INITIAL_FOOD_CLUSTERS = 2
FOOD_PER_CLUSTER = 500
FOOD_CLUSTER_RADIUS = 4
MIN_FOOD_DIST_FROM_NEST = 20
MAX_FOOD_PER_CELL = 100.0
FOOD_COLOR = (255, 255, 0)

# Pheromones
PHEROMONE_MAX = 1000.0
PHEROMONE_DECAY = 0.997
PHEROMONE_DIFFUSION_RATE = 0.05

# Weights
W_HOME_PHEROMONE_RETURN = 45.0
W_FOOD_PHEROMONE_SEARCH = 40.0 # Keep this increased from last change
W_HOME_PHEROMONE_SEARCH = 10.0  # *** CHANGE: No longer avoid home pheromones when searching ***
W_ALARM_PHEROMONE = -30.0
W_NEST_DIRECTION_RETURN = 80.0
W_PERSISTENCE = 1.5
W_AVOID_HISTORY = -1000.0
W_RANDOM_NOISE = 0.2
W_AVOID_FOOD_RETURN = -50.0

# Probabilistic Choice Params
PROBABILISTIC_CHOICE_TEMP = 1.0
MIN_SCORE_FOR_PROB_CHOICE = 0.01

# Pheromone Drops
P_FOOD_SEARCHING = 0.0      # Increased searching trail strength
P_HOME_RETURNING = 100.0     # Keep home trail strong
P_FOOD_AT_SOURCE = 500.0    # Significantly increased food signal at source
P_FOOD_AT_NEST = 0.0
P_ALARM_FIGHT = 100.0

# Ant Params
INITIAL_WORKERS = 200
QUEEN_HP = 500
WORKER_HP = 50
WORKER_ATTACK = 5
WORKER_CAPACITY = 1
WORKER_MAX_AGE_MEAN = 8000
WORKER_MAX_AGE_STDDEV = 1000
WORKER_MOVE_DELAY = 0
WORKER_PATH_HISTORY_LENGTH = 10 # How many *ticks* back history is remembered
WORKER_STUCK_THRESHOLD = 60
WORKER_ESCAPE_DURATION = 30
QUEEN_EGG_RATE = 50

# Enemy Params
INITIAL_ENEMIES = 3
ENEMY_HP = 80
ENEMY_ATTACK = 8
ENEMY_MOVE_DELAY = 3
ENEMY_SPAWN_RATE = 300
ENEMY_TO_FOOD_ON_DEATH = 50.0

# Colors
QUEEN_COLOR = (255, 0, 255)
WORKER_SEARCH_COLOR = (0, 150, 255)
WORKER_RETURN_COLOR = (0, 255, 100)
WORKER_ESCAPE_COLOR = (255, 165, 0)
ENEMY_COLOR = (200, 0, 0)
PHEROMONE_HOME_COLOR = (0, 0, 255, 150)
PHEROMONE_FOOD_COLOR = (0, 255, 0, 150)
PHEROMONE_ALARM_COLOR = (255, 0, 0, 180)


class AntState(Enum):
    """Enum to define the state of an ant."""
    SEARCHING = 1
    RETURNING_TO_NEST = 2
    ESCAPING = 3


# --- Helper Functions ---

def is_valid(pos):
    """Check if a position (x, y) is within the grid boundaries."""
    if not isinstance(pos, (tuple, list)) or len(pos) != 2:
        return False
    x, y = pos
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def get_neighbors(pos, include_center=False):
    """Get valid neighbor coordinates for a given position."""
    if not is_valid(pos):
        return []
    x, y = pos
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0 and not include_center:
                continue
            n_pos = (x + dx, y + dy)
            if is_valid(n_pos):
                neighbors.append(n_pos)
    return neighbors


def distance_sq(pos1, pos2):
    """Calculate the squared Euclidean distance between two points."""
    # Return infinity if any position is invalid to prevent errors
    if not pos1 or not pos2 or not is_valid(pos1) or not is_valid(pos2):
        return float('inf')
    try:
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
    except (TypeError, IndexError):
        # Handle potential issues if pos1/pos2 are not valid coords
        return float('inf')


def normalize(value, max_val):
    """Normalize a value to the range [0, 1]."""
    if max_val <= 0:
        return 0.0
    return min(1.0, max(0.0, float(value) / float(max_val)))


# --- Grid Class ---
# (WorldGrid class remains unchanged from the previous version)
class WorldGrid:
    """Manages the simulation grid including food and pheromones."""

    def __init__(self):
        """Initialize the grid arrays."""
        self.food = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=float)
        self.pheromones_home = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=float
        )
        self.pheromones_food = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=float
        )
        self.pheromones_alarm = np.zeros(
            (GRID_WIDTH, GRID_HEIGHT), dtype=float
        )

    def place_food_clusters(self):
        """Place initial food clusters on the grid."""
        for _ in range(INITIAL_FOOD_CLUSTERS):
            attempts = 0
            cx = cy = 0  # Initialize cluster center coordinates
            # Try to find a spot away from the nest
            while attempts < 100:
                cx = random.randint(0, GRID_WIDTH - 1)
                cy = random.randint(0, GRID_HEIGHT - 1)
                if distance_sq((cx, cy), NEST_POS) > MIN_FOOD_DIST_FROM_NEST**2:
                    break  # Found a suitable spot
                attempts += 1
            else:
                # Fallback if no suitable spot found after many attempts
                cx = random.randint(0, GRID_WIDTH - 1)
                cy = random.randint(0, GRID_HEIGHT - 1)

            # Distribute food around the center using Gaussian distribution
            added_food = 0
            for _ in range(int(FOOD_PER_CLUSTER * 2)): # Try more placements than needed
                if added_food >= FOOD_PER_CLUSTER:
                    break
                fx = cx + int(random.gauss(0, FOOD_CLUSTER_RADIUS / 2))
                fy = cy + int(random.gauss(0, FOOD_CLUSTER_RADIUS / 2))

                if is_valid((fx, fy)):
                    amount = random.uniform(0.5, 1.0) * (MAX_FOOD_PER_CELL / 10)
                    self.food[fx, fy] += amount
                    added_food += amount

            # Clamp food amount in the cluster area
            min_x = max(0, cx - int(FOOD_CLUSTER_RADIUS * 2))
            max_x = min(GRID_WIDTH, cx + int(FOOD_CLUSTER_RADIUS * 2))
            min_y = max(0, cy - int(FOOD_CLUSTER_RADIUS * 2))
            max_y = min(GRID_HEIGHT, cy + int(FOOD_CLUSTER_RADIUS * 2))
            if min_x < max_x and min_y < max_y: # Ensure valid slice
                np.clip(
                    self.food[min_x:max_x, min_y:max_y], 0,
                    MAX_FOOD_PER_CELL,
                    out=self.food[min_x:max_x, min_y:max_y]
                )

    def get_pheromone(self, pos, type='home'):
        """Get the pheromone level at a specific position."""
        if not is_valid(pos): return 0.0
        x, y = pos
        try:
            if type == 'home': return self.pheromones_home[x, y]
            elif type == 'food': return self.pheromones_food[x, y]
            elif type == 'alarm': return self.pheromones_alarm[x, y]
        except IndexError: return 0.0
        return 0.0

    def add_pheromone(self, pos, amount, type='home'):
        """Add pheromone to a specific position, clamping at max value."""
        if not is_valid(pos) or amount <= 0: return
        x, y = pos
        try:
            target_array = None
            if type == 'home': target_array = self.pheromones_home
            elif type == 'food': target_array = self.pheromones_food
            elif type == 'alarm': target_array = self.pheromones_alarm

            if target_array is not None:
                target_array[x, y] = min(target_array[x, y] + amount, PHEROMONE_MAX)
        except IndexError: pass

    def update_pheromones(self):
        """Apply decay and diffusion to all pheromone grids."""
        self.pheromones_home *= PHEROMONE_DECAY
        self.pheromones_food *= PHEROMONE_DECAY
        self.pheromones_alarm *= PHEROMONE_DECAY

        if PHEROMONE_DIFFUSION_RATE > 0:
            rate = PHEROMONE_DIFFUSION_RATE
            for ph_array in [self.pheromones_home, self.pheromones_food, self.pheromones_alarm]:
                padded = np.pad(ph_array, 1, mode='constant')
                neighbors = (
                    padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                    padded[1:-1, :-2] +                    padded[1:-1, 2:] +
                    padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
                )
                ph_array[:] = ph_array * (1 - rate) + (neighbors / 8.0) * rate

        min_ph = 0.01
        np.clip(self.pheromones_home, 0, PHEROMONE_MAX, out=self.pheromones_home)
        self.pheromones_home[self.pheromones_home < min_ph] = 0
        np.clip(self.pheromones_food, 0, PHEROMONE_MAX, out=self.pheromones_food)
        self.pheromones_food[self.pheromones_food < min_ph] = 0
        np.clip(self.pheromones_alarm, 0, PHEROMONE_MAX, out=self.pheromones_alarm)
        self.pheromones_alarm[self.pheromones_alarm < min_ph] = 0

    def draw(self, surface):
        """Draw grid elements (food, pheromones, nest) onto the surface."""
        ph_home_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ph_food_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ph_alarm_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        min_draw_ph = 0.5

        home_nz = np.argwhere(self.pheromones_home > min_draw_ph)
        food_nz = np.argwhere(self.pheromones_food > min_draw_ph)
        alarm_nz = np.argwhere(self.pheromones_alarm > min_draw_ph)

        for x, y in home_nz:
            ph_h = self.pheromones_home[x, y]
            alpha = int(normalize(ph_h, PHEROMONE_MAX / 5) * PHEROMONE_HOME_COLOR[3])
            if alpha > 3:
                color = (*PHEROMONE_HOME_COLOR[:3], alpha)
                pygame.draw.rect(ph_home_surf, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for x, y in food_nz:
            ph_f = self.pheromones_food[x, y]
            alpha = int(normalize(ph_f, PHEROMONE_MAX / 5) * PHEROMONE_FOOD_COLOR[3])
            if alpha > 3:
                color = (*PHEROMONE_FOOD_COLOR[:3], alpha)
                pygame.draw.rect(ph_food_surf, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for x, y in alarm_nz:
            ph_a = self.pheromones_alarm[x, y]
            alpha = int(normalize(ph_a, PHEROMONE_MAX / 3) * PHEROMONE_ALARM_COLOR[3])
            if alpha > 5:
                color = (*PHEROMONE_ALARM_COLOR[:3], alpha)
                pygame.draw.rect(ph_alarm_surf, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        surface.blit(ph_home_surf, (0, 0))
        surface.blit(ph_food_surf, (0, 0))
        surface.blit(ph_alarm_surf, (0, 0))

        min_draw_food = 0.1
        nz_food_x, nz_food_y = np.where(self.food > min_draw_food)
        for x, y in zip(nz_food_x, nz_food_y):
            food_amount = self.food[x, y]
            intensity = int(normalize(food_amount, MAX_FOOD_PER_CELL) * 200) + 55
            color = (min(255, intensity), min(255, intensity), 0)
            rect = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)

        nest_rect_coords = (
            (NEST_POS[0] - NEST_RADIUS) * CELL_SIZE, (NEST_POS[1] - NEST_RADIUS) * CELL_SIZE,
            (NEST_RADIUS * 2 + 1) * CELL_SIZE, (NEST_RADIUS * 2 + 1) * CELL_SIZE
        )
        nest_rect = pygame.Rect(nest_rect_coords)
        nest_surf = pygame.Surface((nest_rect.width, nest_rect.height), pygame.SRCALPHA)
        nest_surf.fill((100, 100, 100, 30))
        surface.blit(nest_surf, nest_rect.topleft)

# --- Entity Classes ---

class Ant:
    """Represents a worker ant in the simulation."""

    def __init__(self, pos, simulation):
        """Initialize ant properties."""
        self.pos = tuple(pos) # Ensure position is a tuple
        self.simulation = simulation
        self.hp = WORKER_HP
        self.max_hp = WORKER_HP
        self.attack_power = WORKER_ATTACK
        self.state = AntState.SEARCHING
        self.carry_amount = 0
        self.max_capacity = WORKER_CAPACITY
        self.age = 0
        self.max_age = int(
            random.gauss(WORKER_MAX_AGE_MEAN, WORKER_MAX_AGE_STDDEV)
        )
        # Use a list of tuples for history (more efficient check)
        self.path_history = [] # Stores positions tuples visited recently
        self.history_timestamps = [] # Stores corresponding timestamps
        self.move_delay_timer = 0
        self.last_move_direction = (0, 0)
        self.stuck_timer = 0
        self.escape_timer = 0
        self.last_move_info = "Init"
        self.just_picked_food = False

    def _update_path_history(self, new_pos):
        """Add current position to history and trim old entries based on time."""
        current_time = self.simulation.ticks
        new_pos_tuple = tuple(new_pos)

        # Add new entry
        self.path_history.append(new_pos_tuple)
        self.history_timestamps.append(current_time)

        # Trim old entries based on WORKER_PATH_HISTORY_LENGTH (time duration)
        cutoff_time = current_time - WORKER_PATH_HISTORY_LENGTH
        # Find the index of the first entry that is recent enough
        valid_index = 0
        while valid_index < len(self.history_timestamps) and self.history_timestamps[valid_index] < cutoff_time:
            valid_index += 1

        # Keep only the recent part of the history
        self.path_history = self.path_history[valid_index:]
        self.history_timestamps = self.history_timestamps[valid_index:]


    def _is_in_history(self, pos):
        """Check if a position is in the recent path history."""
        # Efficiently check if the position tuple exists in the list
        return tuple(pos) in self.path_history

    def _clear_path_history(self):
        """Clears the path history."""
        self.path_history = []
        self.history_timestamps = []


    def _choose_move(self):
        """Determine the next move based on state, pheromones, and goals."""
        current_pos = self.pos
        valid_neighbors = []

        # Phase 1: Identify Valid Moves
        potential_neighbors = get_neighbors(current_pos)
        if not potential_neighbors:
            self.last_move_info = "No neighbors!"
            return None

        q_pos = self.simulation.queen.pos if self.simulation.queen else None
        for n_pos in potential_neighbors:
            # Exclude neighbors in recent history OR occupied by the static queen
            if not self._is_in_history(n_pos) and n_pos != q_pos:
                valid_neighbors.append(n_pos)

        if not valid_neighbors:
            self.last_move_info = "Blocked by history/queen"
            # Simple fallback: try moving to the oldest history spot if truly blocked
            if self.path_history and self.path_history[0] in potential_neighbors and self.path_history[0] != q_pos:
                 # print(f"Ant {id(self)} forcing move to oldest history: {self.path_history[0]}")
                 return self.path_history[0]
            return None # Completely stuck

        # Phase 2: High Priority Actions
        if self.state == AntState.SEARCHING:
            for n_pos in valid_neighbors:
                if self.simulation.grid.food[n_pos[0], n_pos[1]] > 0.1:
                    self.last_move_info = f"Adjacent Food -> {n_pos}"
                    return n_pos
        elif self.state == AntState.RETURNING_TO_NEST:
            for n_pos in valid_neighbors:
                if distance_sq(n_pos, NEST_POS) <= 1: # Directly adjacent or at center
                    self.last_move_info = f"Adjacent Nest -> {n_pos}"
                    return n_pos # Go straight into nest center/adj cell

        # Phase 2.5: Handle Escaping State
        if self.state == AntState.ESCAPING:
            chosen_move = random.choice(valid_neighbors)
            self.last_move_info = f"Escaping -> {chosen_move}"
            return chosen_move

        # --- Phase 3 & 4: Score Calculation ---
        move_scores = {}
        dist_sq_now = distance_sq(current_pos, NEST_POS) # Calculate current distance once

        for n_pos in valid_neighbors:
            score = 0.0
            home_ph = self.simulation.grid.get_pheromone(n_pos, 'home')
            food_ph = self.simulation.grid.get_pheromone(n_pos, 'food')
            alarm_ph = self.simulation.grid.get_pheromone(n_pos, 'alarm')
            dist_sq_next = distance_sq(n_pos, NEST_POS)

            # Base score contributions based on state
            if self.state == AntState.RETURNING_TO_NEST:
                score += home_ph * W_HOME_PHEROMONE_RETURN
                # Strong bonus for getting closer to the nest
                if dist_sq_next < dist_sq_now:
                    score += W_NEST_DIRECTION_RETURN
                # Optional: Add slight penalty for moving away?
                # elif dist_sq_next > dist_sq_now:
                #     score -= W_NEST_DIRECTION_RETURN / 4 # Smaller penalty

            elif self.state == AntState.SEARCHING:
                score += food_ph * W_FOOD_PHEROMONE_SEARCH
                score += home_ph * W_HOME_PHEROMONE_SEARCH # Follow reverse trail slightly

            # General score modifiers
            score += alarm_ph * W_ALARM_PHEROMONE # Avoid alarm
            move_direction = (n_pos[0] - current_pos[0], n_pos[1] - current_pos[1])
            if move_direction == self.last_move_direction:
                score += W_PERSISTENCE # Persistence bonus

            score += random.uniform(-W_RANDOM_NOISE, W_RANDOM_NOISE) # Random noise

            move_scores[n_pos] = score

        # --- Phase 5: Final Selection (State-Dependent) ---
        if not move_scores:
            self.last_move_info = "No moves scored"
            return random.choice(valid_neighbors) if valid_neighbors else None

        # *** MODIFIED SELECTION LOGIC FOR RETURNING ***
        if self.state == AntState.RETURNING_TO_NEST:
            # --- STRONGLY Prefer moves that get closer to the nest ---
            closer_neighbors = {}
            other_neighbors = {}

            for n_pos, score in move_scores.items():
                if distance_sq(n_pos, NEST_POS) < dist_sq_now:
                    closer_neighbors[n_pos] = score
                else:
                    other_neighbors[n_pos] = score

            target_selection_pool = {}
            if closer_neighbors:
                # If there are moves towards the nest, ONLY consider those
                target_selection_pool = closer_neighbors
                selection_type = "Closer"
            elif other_neighbors:
                 # Otherwise (e.g., right next to nest or must move sideways), consider others
                target_selection_pool = other_neighbors
                selection_type = "Other"
            else:
                 # Should not happen if valid_neighbors existed
                 self.last_move_info = "Return: No moves in pools?"
                 return random.choice(valid_neighbors) if valid_neighbors else None


            # --- Deterministic Choice within the chosen pool ---
            best_score = -float('inf')
            best_moves = []
            for n_pos, score in target_selection_pool.items():
                if score > best_score:
                    best_score = score
                    best_moves = [n_pos]
                elif score == best_score:
                    best_moves.append(n_pos)

            if not best_moves:
                self.last_move_info = f"Return({selection_type}): No best move found?"
                # Fallback to random choice from the pool, or original valid neighbors
                fallback_pool = list(target_selection_pool.keys()) or valid_neighbors
                return random.choice(fallback_pool) if fallback_pool else None


            # Tie-breaking: Prefer highest home pheromone, then random
            if len(best_moves) == 1:
                chosen_move = best_moves[0]
                self.last_move_info = f"R({selection_type})Best->{chosen_move} (S:{best_score:.1f})"
            else:
                best_moves.sort(key=lambda p: self.simulation.grid.get_pheromone(p, 'home'), reverse=True)
                max_ph_tied = self.simulation.grid.get_pheromone(best_moves[0], 'home')
                top_pheromone_moves = [p for p in best_moves if self.simulation.grid.get_pheromone(p, 'home') == max_ph_tied]
                chosen_move = random.choice(top_pheromone_moves)
                self.last_move_info = f"R({selection_type})TieBrk->{chosen_move} (S:{best_score:.1f})"
            return chosen_move


        elif self.state == AntState.SEARCHING:
            # --- Probabilistic Choice for Searching Ants ---
            population = list(move_scores.keys())
            raw_scores = list(move_scores.values())
            min_score = min(raw_scores) if raw_scores else 0
            shift = abs(min_score) + 0.1
            weights = [max(MIN_SCORE_FOR_PROB_CHOICE, (s + shift) ** PROBABILISTIC_CHOICE_TEMP) for s in raw_scores]

            total_weight = sum(weights)
            if total_weight <= 0:
                self.last_move_info = f"Search: Low weights (Sum:{total_weight:.2f})"
                return random.choice(valid_neighbors) if valid_neighbors else None

            try:
                chosen_move = random.choices(population=population, weights=weights, k=1)[0]
                chosen_score = move_scores.get(chosen_move, -999)
                self.last_move_info = f"SearchProb->{chosen_move} (S:{chosen_score:.1f})"
                return chosen_move
            except Exception as e:
                print(f"Error during random.choices (Search): {e} - Pop: {population}, Weights: {weights}")
                self.last_move_info = "SearchProb Error"
                return random.choice(valid_neighbors) if valid_neighbors else None
        else:
            # Fallback for unhandled states
            self.last_move_info = f"Invalid state in move choice: {self.state}"
            return random.choice(valid_neighbors) if valid_neighbors else None

    def update(self):
        """Update the ant's state, position, and interactions."""
        self.age += 1
        if self.hp <= 0 or self.age > self.max_age:
            self.simulation.kill_ant(self)
            return

        if self.state == AntState.ESCAPING:
            self.escape_timer -= 1
            if self.escape_timer <= 0:
                self.state = AntState.SEARCHING
                self.stuck_timer = 0
                self._clear_path_history() # Clear history after escaping

        # --- Combat ---
        enemies_nearby = []
        for n_pos in get_neighbors(self.pos, include_center=True):
            enemy = self.simulation.get_enemy_at(n_pos)
            if enemy: enemies_nearby.append(enemy)

        if enemies_nearby:
            target_enemy = random.choice(enemies_nearby)
            self.attack(target_enemy)
            self.simulation.grid.add_pheromone(self.pos, P_ALARM_FIGHT, 'alarm')
            self.stuck_timer = 0
            self.last_move_info = f"Fighting {target_enemy.pos}"
            return # Skip move if fighting

        # --- Movement Delay ---
        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return

        # --- Movement ---
        old_pos = self.pos
        local_just_picked_food = self.just_picked_food
        self.just_picked_food = False

        new_pos_candidate = self._choose_move()
        moved = False

        if new_pos_candidate and tuple(new_pos_candidate) != old_pos:
            target_pos = tuple(new_pos_candidate)
            if not self.simulation.is_ant_at(target_pos, exclude_self=self):
                move_dir = (target_pos[0] - old_pos[0], target_pos[1] - old_pos[1])
                if abs(move_dir[0]) <= 1 and abs(move_dir[1]) <= 1:
                    self.pos = target_pos
                    self.last_move_direction = move_dir
                    self._update_path_history(target_pos) # Update history *after* successful move
                    self.move_delay_timer = WORKER_MOVE_DELAY
                    self.stuck_timer = 0
                    moved = True
                else:
                    self.stuck_timer += 1
                    self.last_move_info += "(InvalidStep!)"
            else:
                self.stuck_timer += 1
                self.last_move_info += "(BlockedAnt!)"
        else:
            self.stuck_timer += 1
            if new_pos_candidate is None: self.last_move_info += "(NoChoice)"
            else: self.last_move_info += "(StayedPut)"

        # --- Post-Movement Actions & State ---
        current_cell_food = 0
        if is_valid(self.pos):
            try: current_cell_food = self.simulation.grid.food[self.pos[0], self.pos[1]]
            except IndexError: pass
        is_near_nest = distance_sq(self.pos, NEST_POS) <= (NEST_RADIUS)**2 # Check within radius squared


        # --- State transitions & Pheromone drops ---
        if self.state == AntState.SEARCHING:
            # Pickup food
            if (self.carry_amount < self.max_capacity and
                    current_cell_food > 0.1 and is_valid(self.pos)):
                pickup_amount = min(self.max_capacity - self.carry_amount, current_cell_food)
                self.carry_amount += pickup_amount
                self.simulation.grid.food[self.pos[0], self.pos[1]] = max(0, current_cell_food - pickup_amount)
                self.simulation.grid.add_pheromone(self.pos, P_FOOD_AT_SOURCE, 'food')
                self.state = AntState.RETURNING_TO_NEST
                self._clear_path_history() # *** Clear history ON STATE CHANGE to returning ***
                self.last_move_info = f"PickedFood({pickup_amount:.1f})"
                self.just_picked_food = True # Set flag

            # Drop searching trail
            elif moved and is_valid(old_pos):
                self.simulation.grid.add_pheromone(old_pos, P_FOOD_SEARCHING, 'food')

        elif self.state == AntState.RETURNING_TO_NEST:
            # Arrived at nest
            if is_near_nest:
                dropped_amount = self.carry_amount
                if dropped_amount > 0:
                    self.simulation.food_collected += dropped_amount
                    self.carry_amount = 0
                self.state = AntState.SEARCHING
                self._clear_path_history() # *** Clear history ON STATE CHANGE to searching ***
                self.last_move_info = f"DroppedFood({dropped_amount:.1f})" if dropped_amount > 0 else "NestEmptyHand"

            # Drop home trail
            elif moved and is_valid(old_pos):
                 # Drop strong home trail, UNLESS food was *just* picked up
                if not local_just_picked_food:
                    self.simulation.grid.add_pheromone(old_pos, P_HOME_RETURNING, 'home')

        # --- Check if Stuck ---
        if (self.stuck_timer >= WORKER_STUCK_THRESHOLD and
                self.state != AntState.ESCAPING and not enemies_nearby):
            self.state = AntState.ESCAPING
            self.escape_timer = WORKER_ESCAPE_DURATION
            self.stuck_timer = 0
            self._clear_path_history() # Clear history when starting to escape
            self.last_move_info = "Stuck->Escaping"

    def attack(self, target_enemy):
        """Attack a target enemy."""
        damage = self.attack_power
        target_enemy.take_damage(damage, self)

    def take_damage(self, amount, attacker):
        """Process damage taken."""
        self.hp -= amount

    def draw(self, surface):
        """Draw the ant on the simulation surface."""
        draw_pos = (
            int(self.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos[1] * CELL_SIZE + CELL_SIZE / 2)
        )
        radius = int(CELL_SIZE / 2.5)
        color_map = {
            AntState.SEARCHING: WORKER_SEARCH_COLOR,
            AntState.RETURNING_TO_NEST: WORKER_RETURN_COLOR,
            AntState.ESCAPING: WORKER_ESCAPE_COLOR
        }
        color = color_map.get(self.state, (200, 200, 200))
        pygame.draw.circle(surface, color, draw_pos, radius)
        if self.carry_amount > 0:
            pygame.draw.circle(surface, FOOD_COLOR, draw_pos, int(radius * 0.6))

# --- Queen Class ---
# (Queen class remains unchanged from the previous version)
class Queen(Ant):
    """Represents the queen ant, responsible for laying eggs."""
    def __init__(self, pos, sim):
        super().__init__(pos, sim)
        self.hp = QUEEN_HP
        self.max_hp = QUEEN_HP
        self.max_age = float('inf')
        self.egg_timer = 0
        self.color = QUEEN_COLOR
        self.state = None
        self.attack_power = 0
        self.pos = tuple(map(int, pos))

    def update(self):
        if self.hp <= 0:
            self.simulation.kill_queen(self)
            return
        self.age += 1
        self.egg_timer += 1
        if self.egg_timer >= QUEEN_EGG_RATE:
            self.egg_timer = 0
            spawn_attempts = 0
            spawned = False
            while spawn_attempts < 10:
                ox = random.randint(-2, 2)
                oy = random.randint(-2, 2)
                if ox == 0 and oy == 0:
                    spawn_attempts += 1
                    continue
                spawn_pos = (self.pos[0] + ox, self.pos[1] + oy)
                if self.simulation.add_ant(spawn_pos):
                    spawned = True
                    break
                spawn_attempts += 1

    def take_damage(self, amount, attacker):
        self.hp -= amount
        self.simulation.grid.add_pheromone(self.pos, P_ALARM_FIGHT * 2, 'alarm')

    def draw(self, surface):
        draw_pos = (
            int(self.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos[1] * CELL_SIZE + CELL_SIZE / 2)
        )
        radius = int(CELL_SIZE / 1.8)
        pygame.draw.circle(surface, self.color, draw_pos, radius)
        pygame.draw.circle(surface, (255, 255, 255), draw_pos, radius, 1)

# --- Enemy Class ---
# (Enemy class remains unchanged from the previous version)
class Enemy:
    """Represents an enemy entity."""
    def __init__(self, pos, sim):
        self.pos = tuple(pos)
        self.simulation = sim
        self.hp = ENEMY_HP
        self.max_hp = ENEMY_HP
        self.attack_power = ENEMY_ATTACK
        self.move_delay_timer = 0
        self.color = ENEMY_COLOR

    def update(self):
        if self.hp <= 0:
            self.simulation.kill_enemy(self)
            return

        ants_nearby = []
        for n_pos in get_neighbors(self.pos, include_center=True):
            ant = self.simulation.get_ant_at(n_pos)
            if ant: ants_nearby.append(ant)

        if ants_nearby:
            target_ant = random.choice(ants_nearby)
            self.attack(target_ant)
            return

        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return

        possible_moves = get_neighbors(self.pos)
        valid_moves = [
            m for m in possible_moves
            if not self.simulation.is_enemy_at(m, exclude_self=self) and
               not self.simulation.is_ant_at(m)
        ]
        if valid_moves:
            self.pos = tuple(random.choice(valid_moves))
            self.move_delay_timer = ENEMY_MOVE_DELAY

    def attack(self, target_ant):
        target_ant.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        self.hp -= amount

    def draw(self, surface):
        draw_pos = (
            int(self.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos[1] * CELL_SIZE + CELL_SIZE / 2)
        )
        radius = int(CELL_SIZE / 2)
        pygame.draw.circle(surface, self.color, draw_pos, radius)
        pygame.draw.circle(surface, (0, 0, 0), draw_pos, radius + 1, 1)

# --- Main Simulation Class ---
# (AntSimulation class remains mostly unchanged,
# using the updated Ant, Queen, Enemy, WorldGrid classes)
class AntSimulation:
    """Manages the overall simulation loop, entities, and drawing."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ant Simulation")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("sans", 18)
        except Exception:
            self.font = pygame.font.Font(None, 22)
        self.running = True
        self.ticks = 0
        self.grid = WorldGrid()
        self.grid.place_food_clusters()
        self.ants = []
        self.enemies = []
        self.queen = None
        self.food_collected = 0.0
        self.enemy_spawn_timer = 0
        self._spawn_initial_entities()
        self.show_debug_info = False

    def _spawn_initial_entities(self):
        queen_pos = tuple(map(int, NEST_POS))
        if is_valid(queen_pos):
             self.queen = Queen(queen_pos, self)
        else:
            print(f"Error: Invalid NEST_POS {NEST_POS}. Cannot spawn queen.")
            self.running = False
            return

        for _ in range(INITIAL_ENEMIES): self.spawn_enemy()

        spawned_count = 0
        spawn_attempts = 0
        max_attempts = INITIAL_WORKERS * 5
        while spawned_count < INITIAL_WORKERS and spawn_attempts < max_attempts:
            ox = random.randint(-NEST_RADIUS, NEST_RADIUS)
            oy = random.randint(-NEST_RADIUS, NEST_RADIUS)
            spawn_pos = (NEST_POS[0] + ox, NEST_POS[1] + oy)
            if self.add_ant(spawn_pos): spawned_count += 1
            spawn_attempts += 1
        while spawned_count < INITIAL_WORKERS and spawn_attempts < max_attempts + 5:
             if self.add_ant(NEST_POS): spawned_count += 1
             spawn_attempts += 1
        # if spawned_count < INITIAL_WORKERS: print(f"Warning: Only spawned {spawned_count}/{INITIAL_WORKERS} initial workers.")

    def add_ant(self, pos):
        pos_tuple = tuple(map(int, pos))
        if not is_valid(pos_tuple): return False
        if not self.is_ant_at(pos_tuple) and not self.is_enemy_at(pos_tuple):
            self.ants.append(Ant(pos_tuple, self))
            return True
        return False

    def spawn_enemy(self):
        tries = 0
        while tries < 50:
            ex = random.randint(0, GRID_WIDTH - 1)
            ey = random.randint(0, GRID_HEIGHT - 1)
            spawn_pos = (ex, ey)
            if (is_valid(spawn_pos) and
                    distance_sq(spawn_pos, NEST_POS) > (MIN_FOOD_DIST_FROM_NEST // 2)**2 and
                    not self.is_enemy_at(spawn_pos) and
                    not self.is_ant_at(spawn_pos)):
                self.enemies.append(Enemy(spawn_pos, self))
                return True
            tries += 1
        return False

    def kill_ant(self, ant_to_remove):
        try: self.ants.remove(ant_to_remove)
        except ValueError: pass

    def kill_enemy(self, enemy_to_remove):
        try:
            if is_valid(enemy_to_remove.pos):
                fx, fy = enemy_to_remove.pos
                self.grid.food[fx, fy] = min(MAX_FOOD_PER_CELL, self.grid.food[fx, fy] + ENEMY_TO_FOOD_ON_DEATH)
            self.enemies.remove(enemy_to_remove)
        except ValueError: pass

    def kill_queen(self, queen_to_remove):
        print(f"--- SIMULATION END: QUEEN DIED AT TICK {self.ticks} ---")
        self.queen = None
        self.running = False

    def is_ant_at(self, pos, exclude_self=None):
        pos_tuple = tuple(pos)
        if (self.queen and self.queen.pos == pos_tuple and exclude_self != self.queen): return True
        for ant in self.ants:
            if ant is exclude_self: continue
            if ant.pos == pos_tuple: return True
        return False

    def get_ant_at(self, pos):
        pos_tuple = tuple(pos)
        if self.queen and self.queen.pos == pos_tuple: return self.queen
        for ant in self.ants:
            if ant.pos == pos_tuple: return ant
        return None

    def is_enemy_at(self, pos, exclude_self=None):
        pos_tuple = tuple(pos)
        for e in self.enemies:
             if e is exclude_self: continue
             if e.pos == pos_tuple: return True
        return False

    def get_enemy_at(self, pos):
        pos_tuple = tuple(pos)
        for e in self.enemies:
            if e.pos == pos_tuple: return e
        return None

    def update(self):
        if not self.running: return
        self.ticks += 1

        if self.queen:
            self.queen.update()
            if not self.running: return

        current_ants = list(self.ants)
        for ant in current_ants:
            if ant in self.ants: ant.update()

        current_enemies = list(self.enemies)
        for enemy in current_enemies:
            if enemy in self.enemies: enemy.update()

        self.grid.update_pheromones()

        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= ENEMY_SPAWN_RATE:
            self.enemy_spawn_timer = 0
            self.spawn_enemy()

    def draw_debug_info(self):
        if not self.font: return
        ant_c = len(self.ants)
        enemy_c = len(self.enemies)
        food_c = int(self.food_collected)
        tick_c = self.ticks
        fps = self.clock.get_fps()

        texts = [f"Ants: {ant_c}", f"Enemies: {enemy_c}", f"Food: {food_c}", f"Tick: {tick_c}"]
        y_pos = 5
        for i, text in enumerate(texts):
            try:
                text_surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (5, y_pos + i * 16))
            except Exception as e: pass # Ignore font rendering errors

        fps_color = (0, 255, 0) if fps > 25 else (255, 255, 0) if fps > 15 else (255, 0, 0)
        try:
            fps_surface = self.font.render(f"FPS: {fps:.0f}", True, fps_color)
            fps_rect = fps_surface.get_rect(topright=(WIDTH - 5, 5))
            self.screen.blit(fps_surface, fps_rect)
        except Exception as e: pass

        try:
            mx, my = pygame.mouse.get_pos()
            gx, gy = mx // CELL_SIZE, my // CELL_SIZE
            if is_valid((gx, gy)):
                entity = self.get_ant_at((gx, gy)) or self.get_enemy_at((gx, gy))
                debug_lines = []
                if entity:
                    if isinstance(entity, Queen): debug_lines = [f"QUEEN", f"HP: {entity.hp}/{entity.max_hp}"]
                    elif isinstance(entity, Ant):
                        state_char = entity.state.name[0] if entity.state else '?'
                        debug_lines = [f"Ant", f"S:{state_char} HP:{entity.hp} A:{entity.age}",
                                       f"C:{entity.carry_amount:.1f} Stk:{entity.stuck_timer}",
                                       f"Mv:{entity.last_move_info[:20]}"]
                    elif isinstance(entity, Enemy): debug_lines = [f"ENEMY", f"HP: {entity.hp}/{entity.max_hp}"]

                home_ph = self.grid.get_pheromone((gx, gy), 'home')
                food_ph = self.grid.get_pheromone((gx, gy), 'food')
                alarm_ph = self.grid.get_pheromone((gx, gy), 'alarm')
                grid_food = self.grid.food[gx, gy]
                debug_lines.extend([f"Cell:({gx},{gy}) Fd:{grid_food:.1f}",
                                    f"Ph: H:{home_ph:.0f} F:{food_ph:.0f} A:{alarm_ph:.0f}"])

                y_offset = HEIGHT - (len(debug_lines) * 16) - 5
                for i, line in enumerate(debug_lines):
                    line_surface = self.font.render(line, True, (255, 255, 0))
                    self.screen.blit(line_surface, (5, y_offset + i * 16))
        except Exception as e: pass

    def draw(self):
        self.screen.fill(MAP_BG_COLOR)
        self.grid.draw(self.screen)
        if self.queen: self.queen.draw(self.screen)
        ants_to_draw = list(self.ants)
        enemies_to_draw = list(self.enemies)
        for ant in ants_to_draw: ant.draw(self.screen)
        for enemy in enemies_to_draw: enemy.draw(self.screen)
        if self.show_debug_info: self.draw_debug_info()
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: self.running = False
                if event.key == pygame.K_d:
                    self.show_debug_info = not self.show_debug_info
                    # print(f"Debug info {'ON' if self.show_debug_info else 'OFF'}")

    def run(self):
        TARGET_FPS = 60
        print("Starting Ant Simulation...")
        print("Press 'D' to toggle debug info.")
        print("Press 'ESC' to quit.")
        while self.running:
            self.handle_events()
            if self.running: self.update()
            self.draw()
            self.clock.tick(TARGET_FPS)
        print(f"Simulation ended after {self.ticks} ticks.")
        pygame.quit()

# --- Start Simulation ---
if __name__ == '__main__':
    try: import numpy
    except ImportError:
        print("\n--- ERROR: NumPy module not found! ---")
        print("Please install NumPy (e.g., 'pip install numpy')")
        input("Press Enter to exit."); exit()
    try:
        import pygame
        if not pygame.get_init(): pygame.init()
        if not pygame.font.get_init(): pygame.font.init()
    except ImportError: print("\n--- ERROR: Pygame not found! ---"); input("Press Enter to exit."); exit()
    except Exception as e: print(f"\n--- ERROR: Pygame init failed! {e}---"); input("Press Enter to exit."); exit()

    simulation = AntSimulation()
    simulation.run()
    print("Simulation process finished.")