# -*- coding: utf-8 -*-
import pygame
import random
import math
import time
from enum import Enum
import numpy as np

# --- Configuration Constants ---

# World & Grid
GRID_WIDTH = 80
GRID_HEIGHT = 100
CELL_SIZE = 8
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE
MAP_BG_COLOR = (20, 20, 10)

# Nest
NEST_POS = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
NEST_RADIUS = 5

# Food
INITIAL_FOOD_CLUSTERS = 5
FOOD_PER_CLUSTER = 1000
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
W_FOOD_PHEROMONE_SEARCH = 20.0
W_HOME_PHEROMONE_SEARCH = 3.0
W_ALARM_PHEROMONE = 30.0
W_NEST_DIRECTION_RETURN = 15.0
W_PERSISTENCE = 1.5
W_AVOID_HISTORY = -1000.0
W_RANDOM_NOISE = 0.4
W_AVOID_FOOD_RETURN = -50.0  # Penalty for returning near food

# Probabilistic Choice Params
PROBABILISTIC_CHOICE_TEMP = 1.0
MIN_SCORE_FOR_PROB_CHOICE = 0.01

# Pheromone Drops
P_FOOD_SEARCHING = 0.5
P_HOME_RETURNING = 700.0
P_FOOD_AT_SOURCE = 60.0
P_FOOD_AT_NEST = 0.0
P_ALARM_FIGHT = 100.0

# Ant Params
INITIAL_WORKERS = 75
QUEEN_HP = 500
WORKER_HP = 50
WORKER_ATTACK = 5
WORKER_CAPACITY = 1
WORKER_MAX_AGE_MEAN = 8000
WORKER_MAX_AGE_STDDEV = 1000
WORKER_MOVE_DELAY = 0
WORKER_PATH_HISTORY_LENGTH = 10
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
    if not is_valid(pos1) or not is_valid(pos2):
        return float('inf')
    return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2


def normalize(value, max_val):
    """Normalize a value to the range [0, 1]."""
    if max_val <= 0:
        return 0.0
    return min(1.0, max(0.0, float(value) / float(max_val)))


# --- Grid Class ---

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
                # Fallback if no suitable spot found
                cx = random.randint(0, GRID_WIDTH - 1)
                cy = random.randint(0, GRID_HEIGHT - 1)

            # Distribute food around the center
            added_food = 0
            for _ in range(int(FOOD_PER_CLUSTER * 2)):
                if added_food >= FOOD_PER_CLUSTER:
                    break
                fx = cx + int(random.gauss(0, FOOD_CLUSTER_RADIUS / 2))
                fy = cy + int(random.gauss(0, FOOD_CLUSTER_RADIUS / 2))
                if is_valid((fx, fy)):
                    amount = random.uniform(
                        0.5, 1.0
                    ) * (MAX_FOOD_PER_CELL / 10)
                    self.food[fx, fy] += amount
                    added_food += amount

            # Clamp food amount in the cluster area
            min_x = max(0, cx - int(FOOD_CLUSTER_RADIUS * 2))
            max_x = min(GRID_WIDTH, cx + int(FOOD_CLUSTER_RADIUS * 2))
            min_y = max(0, cy - int(FOOD_CLUSTER_RADIUS * 2))
            max_y = min(GRID_HEIGHT, cy + int(FOOD_CLUSTER_RADIUS * 2))
            if min_x < max_x and min_y < max_y:
                np.clip(
                    self.food[min_x:max_x, min_y:max_y], 0,
                    MAX_FOOD_PER_CELL,
                    out=self.food[min_x:max_x, min_y:max_y]
                )

    def get_pheromone(self, pos, type='home'):
        """Get the pheromone level at a specific position."""
        if not is_valid(pos):
            return 0.0
        x, y = pos
        try:
            if type == 'home':
                return self.pheromones_home[x, y]
            elif type == 'food':
                return self.pheromones_food[x, y]
            elif type == 'alarm':
                return self.pheromones_alarm[x, y]
        except IndexError:
            return 0.0  # Should not happen with is_valid check
        return 0.0  # Unknown type

    def add_pheromone(self, pos, amount, type='home'):
        """Add pheromone to a specific position, clamping at max value."""
        if not is_valid(pos) or amount <= 0:
            return
        x, y = pos
        try:
            target_array = None
            if type == 'home':
                target_array = self.pheromones_home
            elif type == 'food':
                target_array = self.pheromones_food
            elif type == 'alarm':
                target_array = self.pheromones_alarm

            if target_array is not None:
                target_array[x, y] = min(
                    target_array[x, y] + amount, PHEROMONE_MAX
                )
        except IndexError:
            pass  # Ignore potential index errors

    def update_pheromones(self):
        """Apply decay and diffusion to all pheromone grids."""
        # 1. Decay
        self.pheromones_home *= PHEROMONE_DECAY
        self.pheromones_food *= PHEROMONE_DECAY
        self.pheromones_alarm *= PHEROMONE_DECAY

        # 2. Diffusion (using NumPy slicing and padding)
        if PHEROMONE_DIFFUSION_RATE > 0:
            rate = PHEROMONE_DIFFUSION_RATE
            for ph_array in [
                    self.pheromones_home,
                    self.pheromones_food,
                    self.pheromones_alarm]:
                padded = np.pad(ph_array, 1, mode='constant')
                # Sum neighbors using slices
                neighbors = (
                    padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                    padded[1:-1, :-2] + padded[1:-1, 2:] +
                    padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
                )
                # Apply diffusion formula (update array in place)
                ph_array[:] = ph_array * (1 - rate) + (neighbors / 8.0) * rate

        # 3. Clamp final values and remove low noise
        min_ph = 0.01
        np.clip(
            self.pheromones_home, 0, PHEROMONE_MAX,
            out=self.pheromones_home
        )
        self.pheromones_home[self.pheromones_home < min_ph] = 0
        np.clip(
            self.pheromones_food, 0, PHEROMONE_MAX,
            out=self.pheromones_food
        )
        self.pheromones_food[self.pheromones_food < min_ph] = 0
        np.clip(
            self.pheromones_alarm, 0, PHEROMONE_MAX,
            out=self.pheromones_alarm
        )
        self.pheromones_alarm[self.pheromones_alarm < min_ph] = 0

    def draw(self, surface):
        """Draw grid elements (food, pheromones, nest) onto the surface."""
        # Pheromone layers (drawn first)
        ph_home_surf = pygame.Surface(
            (WIDTH, HEIGHT), pygame.SRCALPHA
        )
        ph_food_surf = pygame.Surface(
            (WIDTH, HEIGHT), pygame.SRCALPHA
        )
        ph_alarm_surf = pygame.Surface(
            (WIDTH, HEIGHT), pygame.SRCALPHA
        )
        min_draw_ph = 0.5  # Visual threshold

        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # Home Pheromone
                ph_h = self.pheromones_home[x, y]
                if ph_h > min_draw_ph:
                    alpha = int(
                        normalize(ph_h, PHEROMONE_MAX / 5) *
                        PHEROMONE_HOME_COLOR[3]
                    )
                    if alpha > 3:
                        color = (*PHEROMONE_HOME_COLOR[:3], alpha)
                        pygame.draw.rect(ph_home_surf, color, rect)
                # Food Pheromone
                ph_f = self.pheromones_food[x, y]
                if ph_f > min_draw_ph:
                    alpha = int(
                        normalize(ph_f, PHEROMONE_MAX / 5) *
                        PHEROMONE_FOOD_COLOR[3]
                    )
                    if alpha > 3:
                        color = (*PHEROMONE_FOOD_COLOR[:3], alpha)
                        pygame.draw.rect(ph_food_surf, color, rect)
                # Alarm Pheromone
                ph_a = self.pheromones_alarm[x, y]
                if ph_a > min_draw_ph:
                    alpha = int(
                        normalize(ph_a, PHEROMONE_MAX / 3) *
                        PHEROMONE_ALARM_COLOR[3]
                    )
                    if alpha > 5:
                        color = (*PHEROMONE_ALARM_COLOR[:3], alpha)
                        pygame.draw.rect(ph_alarm_surf, color, rect)

        surface.blit(ph_home_surf, (0, 0))
        surface.blit(ph_food_surf, (0, 0))
        surface.blit(ph_alarm_surf, (0, 0))

        # Food
        min_draw_food = 0.1
        nz_food_x, nz_food_y = np.where(self.food > min_draw_food)
        for x, y in zip(nz_food_x, nz_food_y):
            food_amount = self.food[x, y]
            intensity = int(
                normalize(food_amount, MAX_FOOD_PER_CELL) * 200
            ) + 55
            color = (min(255, intensity), min(255, intensity), 0)
            rect = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)

        # Nest Area Overlay
        nest_rect_coords = (
            (NEST_POS[0] - NEST_RADIUS) * CELL_SIZE,
            (NEST_POS[1] - NEST_RADIUS) * CELL_SIZE,
            (NEST_RADIUS * 2 + 1) * CELL_SIZE,
            (NEST_RADIUS * 2 + 1) * CELL_SIZE
        )
        nest_rect = pygame.Rect(nest_rect_coords)
        nest_surf = pygame.Surface(
            (nest_rect.width, nest_rect.height), pygame.SRCALPHA
        )
        nest_surf.fill((100, 100, 100, 30))  # Semi-transparent gray
        surface.blit(nest_surf, nest_rect.topleft)


# --- Entity Classes ---

class Ant:
    """Represents a worker ant in the simulation."""

    def __init__(self, pos, simulation):
        """Initialize ant properties."""
        self.pos = tuple(pos)
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
        self.path_history = []
        self.move_delay_timer = 0
        self.last_move_direction = (0, 0)
        self.stuck_timer = 0
        self.escape_timer = 0
        self.last_move_info = "Init"
        # Flag to delay pheromone drop right after picking food
        self.just_picked_food = False

    def _update_path_history(self, new_pos):
        """Add current position to history and trim old entries."""
        current_time = self.simulation.ticks
        new_pos_tuple = tuple(new_pos)
        self.path_history.append((new_pos_tuple, current_time))
        cutoff_time = current_time - WORKER_PATH_HISTORY_LENGTH
        # Keep only recent history
        self.path_history = [
            (p, t) for p, t in self.path_history if t >= cutoff_time
        ]

    def _is_in_history(self, pos):
        """Check if a position is in the recent path history."""
        pos_tuple = tuple(pos)
        cutoff_time = self.simulation.ticks - WORKER_PATH_HISTORY_LENGTH
        for hist_pos, timestamp in self.path_history:
            if hist_pos == pos_tuple and timestamp >= cutoff_time:
                return True  # Found in recent history
        return False  # Not found

    def _choose_move(self):
        """Determine the next move based on state, pheromones, and goals."""
        current_pos = self.pos
        valid_neighbors = []

        # Phase 1: Identify Valid Moves (Filter blocked/history)
        for n_pos in get_neighbors(current_pos):
            if not self._is_in_history(n_pos):
                # Add obstacle check here if needed
                valid_neighbors.append(n_pos)

        if not valid_neighbors:
            self.last_move_info = "No valid non-history neighbors"
            return None

        # Phase 2: High Priority Actions
        if self.state == AntState.SEARCHING:
            for n_pos in valid_neighbors:
                if self.simulation.grid.food[n_pos[0], n_pos[1]] > 0.1:
                    self.last_move_info = f"Adjacent Food at {n_pos}"
                    return n_pos
        elif self.state == AntState.RETURNING_TO_NEST:
            # Simplified: Check if any valid neighbor IS the nest center or very close
            for n_pos in valid_neighbors:
                if distance_sq(n_pos, NEST_POS) <= 1:  # Allow reaching center tile or immediate neighbors
                    self.last_move_info = f"Adjacent Nest at {n_pos}"
                    return n_pos

        # Phase 2.5: Handle Escaping State
        if self.state == AntState.ESCAPING:
            # Keep simplified escape: Random valid move
            chosen_move = random.choice(valid_neighbors)
            self.last_move_info = f"Escaping randomly to {chosen_move}"
            return chosen_move

        # --- Phase 3 & 4: Score Calculation ---
        move_scores = {}
        for n_pos in valid_neighbors:
            score = 0.0
            home_ph = self.simulation.grid.get_pheromone(n_pos, 'home')
            food_ph = self.simulation.grid.get_pheromone(n_pos, 'food')
            alarm_ph = self.simulation.grid.get_pheromone(n_pos, 'alarm')

            score += alarm_ph * W_ALARM_PHEROMONE

            if self.state == AntState.RETURNING_TO_NEST:
                score += home_ph * W_HOME_PHEROMONE_RETURN  # Stronger weight
                dist_sq_now = distance_sq(current_pos, NEST_POS)
                dist_sq_next = distance_sq(n_pos, NEST_POS)
                if dist_sq_next < dist_sq_now:  # Check includes validity via distance_sq
                    score += W_NEST_DIRECTION_RETURN  # Stronger weight

            elif self.state == AntState.SEARCHING:
                score += food_ph * W_FOOD_PHEROMONE_SEARCH
                score += home_ph * W_HOME_PHEROMONE_SEARCH  # Weak guidance remains

            # Persistence bonus
            move_direction = (n_pos[0] - current_pos[0], n_pos[1] - current_pos[1])
            if move_direction == self.last_move_direction: score += W_PERSISTENCE

            # Random noise
            score += random.uniform(-W_RANDOM_NOISE, W_RANDOM_NOISE)

            move_scores[n_pos] = score

        # --- Phase 5: Final Selection (State-Dependent) ---
        if not move_scores:
            self.last_move_info = "No moves scored"
            return random.choice(valid_neighbors) if valid_neighbors else None  # Fallback

        # *** MODIFIED SELECTION LOGIC ***
        if self.state == AntState.RETURNING_TO_NEST:
            # --- Deterministic Choice for Returning Ants ---
            best_score = -float('inf')
            best_moves = []  # In case of ties

            for n_pos, score in move_scores.items():
                if score > best_score:
                    best_score = score
                    best_moves = [n_pos]
                elif score == best_score:
                    best_moves.append(n_pos)

            if not best_moves:  # Should not happen if move_scores has items
                self.last_move_info = "Return: No best move found?"
                return random.choice(valid_neighbors) if valid_neighbors else None

            # Tie-breaking for deterministic: Prefer highest home pheromone, then random
            if len(best_moves) == 1:
                chosen_move = best_moves[0]
                self.last_move_info = f"ReturnBest->{chosen_move} (Score:{best_score:.1f})"
            else:
                # Sort tied moves by home pheromone level (descending)
                best_moves.sort(key=lambda p: self.simulation.grid.get_pheromone(p, 'home'), reverse=True)
                # Choose the one with the highest pheromone after sorting
                chosen_move = best_moves[0]
                # Optional: Random choice among the top-pheromone ones if still tied?
                # chosen_move = random.choice([p for p in best_moves if self.simulation.grid.get_pheromone(p, 'home') == self.simulation.grid.get_pheromone(best_moves[0], 'home')])
                self.last_move_info = f"ReturnTieBreak->{chosen_move} (Score:{best_score:.1f})"
            return chosen_move

        elif self.state == AntState.SEARCHING:
            # --- Probabilistic Choice for Searching Ants (as before) ---
            population = list(move_scores.keys())
            raw_scores = list(move_scores.values())
            weights = []
            min_score = min(raw_scores) if raw_scores else 0
            shift = abs(min_score) + 0.1  # Ensure positive base for weighting
            for score in raw_scores:
                adjusted_score = (score + shift) ** PROBABILISTIC_CHOICE_TEMP
                weights.append(max(MIN_SCORE_FOR_PROB_CHOICE, adjusted_score))

            total_weight = sum(weights)
            if total_weight <= 0:
                self.last_move_info = f"Search: All scores low (Weight:{total_weight:.2f})"
                return random.choice(valid_neighbors) if valid_neighbors else None

            try:
                chosen_move = random.choices(population=population, weights=weights, k=1)[0]
                chosen_score = move_scores.get(chosen_move, -999)
                self.last_move_info = f"SearchProb->{chosen_move} (Score:{chosen_score:.1f})"
                return chosen_move
            except Exception as e:
                print(f"Error during random.choices (Search): {e} - Weights: {weights}")
                self.last_move_info = "SearchProb Error"
                return random.choice(valid_neighbors) if valid_neighbors else None
        else:
            # Fallback for any other state (should be ESCAPING, handled earlier)
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

        # Check for enemies before moving
        enemies_nearby = []
        for n_pos in get_neighbors(self.pos):
            enemy = self.simulation.get_enemy_at(n_pos)
            if enemy:
                enemies_nearby.append(enemy)
        if enemies_nearby:
            target_enemy = random.choice(enemies_nearby)
            self.attack(target_enemy)
            self.simulation.grid.add_pheromone(
                self.pos, P_ALARM_FIGHT, 'alarm'
            )
            self.stuck_timer = 0
            self.last_move_info = "Fighting"
            return  # Skip move if fighting

        # Movement delay
        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return

        # Remember previous state for pheromone logic
        local_just_picked_food = self.just_picked_food
        self.just_picked_food = False  # Reset flag for this tick

        # Choose and attempt move
        old_pos = self.pos
        new_pos = self._choose_move()
        moved = False

        if new_pos and tuple(new_pos) != old_pos:
            target_pos = tuple(new_pos)
            # Final collision check before executing move
            if not self.simulation.is_ant_at(target_pos, exclude_self=self):
                move_dir = (target_pos[0]-old_pos[0], target_pos[1]-old_pos[1])
                # Ensure it's a valid single step (should be from choose_move)
                if abs(move_dir[0]) <= 1 and abs(move_dir[1]) <= 1:
                    self.last_move_direction = move_dir
                    self.pos = target_pos  # Update position
                    self._update_path_history(target_pos)
                    self.move_delay_timer = WORKER_MOVE_DELAY
                    self.stuck_timer = 0  # Reset stuck timer
                    moved = True
                else:
                    # This indicates an issue in _choose_move if reached
                    self.stuck_timer += 1
                    self.last_move_info += "(InvalidStep!)"
            else:  # Blocked by another ant
                self.stuck_timer += 1
                self.last_move_info += "(Blocked!)"
        else:  # No move chosen or suggested staying put
            self.stuck_timer += 1
            if new_pos is None:
                self.last_move_info += "(NoMoveChoice)"

        # Update state and actions based on NEW position
        current_cell_food = 0
        is_valid_pos = is_valid(self.pos)
        if is_valid_pos:
            try:
                current_cell_food = self.simulation.grid.food[
                    self.pos[0], self.pos[1]
                ]
            except IndexError:
                pass  # Should be handled by is_valid
        is_near_nest = distance_sq(self.pos, NEST_POS) <= (NEST_RADIUS - 1)**2

        # --- State transitions & pheromone drops ---
        if self.state == AntState.SEARCHING:
            # Pickup food if possible
            if (self.carry_amount < self.max_capacity and
                    current_cell_food > 0.1 and is_valid_pos):
                pickup = min(
                    self.max_capacity - self.carry_amount, current_cell_food
                )
                self.carry_amount += pickup
                self.simulation.grid.food[self.pos[0], self.pos[1]] -= pickup
                self.simulation.grid.add_pheromone(
                    self.pos, P_FOOD_AT_SOURCE, 'food'
                )
                self.state = AntState.RETURNING_TO_NEST
                self.path_history = []  # Clear history for new goal
                self.last_move_info = "PickedFood"
                self.just_picked_food = True  # Set flag
            # Drop weak searching trail if moved
            elif moved and is_valid(old_pos):
                self.simulation.grid.add_pheromone(
                    old_pos, P_FOOD_SEARCHING, 'food'
                )

        elif self.state == AntState.RETURNING_TO_NEST:
            # Arrived at nest
            if is_near_nest:
                if self.carry_amount > 0:
                    self.simulation.food_collected += self.carry_amount
                    self.carry_amount = 0
                # Switch state regardless of whether food was carried
                self.state = AntState.SEARCHING
                self.path_history = []
                self.last_move_info = "DroppedFood" if self.carry_amount > 0 else "NestEmpty"
            # On the way home
            elif moved and is_valid(old_pos):
                # Drop strong home trail, unless just picked up food
                if not local_just_picked_food:
                    self.simulation.grid.add_pheromone(
                        old_pos, P_HOME_RETURNING, 'home'
                    )

        # Check if stuck
        if (self.stuck_timer >= WORKER_STUCK_THRESHOLD and
                self.state != AntState.ESCAPING and not enemies_nearby):
            self.state = AntState.ESCAPING
            self.escape_timer = WORKER_ESCAPE_DURATION
            self.stuck_timer = 0

    def attack(self, target_enemy):
        """Attack a target enemy."""
        damage = self.attack_power
        target_enemy.take_damage(damage, self)

    def take_damage(self, amount, attacker):
        """Process damage taken."""
        self.hp -= amount
        # Could add logic here (e.g., chance to enter ESCAPE state)

    def draw(self, surface):
        """Draw the ant on the simulation surface."""
        draw_pos = (
            int(self.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos[1] * CELL_SIZE + CELL_SIZE / 2)
        )
        radius = int(CELL_SIZE / 2.5)
        # Color based on state
        color_map = {
            AntState.SEARCHING: WORKER_SEARCH_COLOR,
            AntState.RETURNING_TO_NEST: WORKER_RETURN_COLOR,
            AntState.ESCAPING: WORKER_ESCAPE_COLOR
        }
        color = color_map.get(self.state, (200, 200, 200))  # Default grey
        pygame.draw.circle(surface, color, draw_pos, radius)
        # Indicate carrying food
        if self.carry_amount > 0:
            pygame.draw.circle(
                surface, FOOD_COLOR, draw_pos, int(radius * 0.6)
            )


class Queen(Ant):
    """Represents the queen ant, responsible for laying eggs."""

    def __init__(self, pos, sim):
        super().__init__(pos, sim)
        self.hp = QUEEN_HP
        self.max_hp = QUEEN_HP
        self.max_age = float('inf')  # Immortal
        self.egg_timer = 0
        self.color = QUEEN_COLOR
        self.state = None  # Queen is static

    def update(self):
        """Update queen: lay eggs periodically."""
        if self.hp <= 0:
            self.simulation.kill_queen(self)
            return
        self.age += 1
        self.egg_timer += 1
        if self.egg_timer >= QUEEN_EGG_RATE:
            self.egg_timer = 0
            spawn_attempts = 0
            while spawn_attempts < 10:
                ox = random.randint(-2, 2)
                oy = random.randint(-2, 2)
                sp = (int(self.pos[0] + ox), int(self.pos[1] + oy))
                # Check validity and occupation (exclude self for checking)
                if (is_valid(sp) and
                        not self.simulation.is_ant_at(sp, exclude_self=self) and
                        not self.simulation.is_enemy_at(sp)):
                    if self.simulation.add_ant(sp):
                        break  # Successfully spawned
                spawn_attempts += 1

    def draw(self, surface):
        """Draw the queen (larger, different color)."""
        draw_pos = (
            int(self.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos[1] * CELL_SIZE + CELL_SIZE / 2)
        )
        radius = int(CELL_SIZE / 1.8)  # Larger than workers
        pygame.draw.circle(surface, self.color, draw_pos, radius)
        pygame.draw.circle(surface, (255, 255, 255), draw_pos, radius, 1) # White outline


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
        """Update enemy: attack nearby ants or move randomly."""
        if self.hp <= 0:
            self.simulation.kill_enemy(self)
            return

        # Check for ants nearby to attack
        ants_nearby = []
        for n_pos in get_neighbors(self.pos):
            ant = self.simulation.get_ant_at(n_pos) # Check includes queen
            if ant:
                ants_nearby.append(ant)
        if ants_nearby:
            self.attack(random.choice(ants_nearby))
            return  # Skip move if attacking

        # Movement logic
        if self.move_delay_timer > 0:
            self.move_delay_timer -= 1
            return

        possible_moves = get_neighbors(self.pos)
        # Find valid empty cells
        valid_moves = [
            m for m in possible_moves
            if not self.simulation.is_enemy_at(m) and
               not self.simulation.is_ant_at(m) # Check both ants and enemies
        ]
        if valid_moves:
            self.pos = tuple(random.choice(valid_moves))
            self.move_delay_timer = ENEMY_MOVE_DELAY

    def attack(self, target_ant):
        """Attack a target ant."""
        target_ant.take_damage(self.attack_power, self)

    def take_damage(self, amount, attacker):
        """Process damage taken."""
        self.hp -= amount

    def draw(self, surface):
        """Draw the enemy."""
        draw_pos = (
            int(self.pos[0] * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos[1] * CELL_SIZE + CELL_SIZE / 2)
        )
        radius = int(CELL_SIZE / 2)
        pygame.draw.circle(surface, self.color, draw_pos, radius)
        pygame.draw.circle(surface, (0, 0, 0), draw_pos, radius + 1, 1) # Outline


# --- Main Simulation Class ---

class AntSimulation:
    """Manages the overall simulation loop, entities, and drawing."""

    def __init__(self):
        """Initialize Pygame, simulation state, and entities."""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ameisen Sim v8 - PEP8")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("sans", 20) # Smaller font
        except Exception:
            self.font = pygame.font.Font(None, 24) # Pygame default if sans fails
        self.running = True
        self.ticks = 0
        self.grid = WorldGrid()
        self.grid.place_food_clusters()
        self.ants = []
        self.enemies = []
        self.queen = None
        self.food_collected = 0
        self.enemy_spawn_timer = 0
        self._spawn_initial_entities()
        self.show_debug_info = False

    def _spawn_initial_entities(self):
        """Create the queen, initial workers, and initial enemies."""
        queen_pos = tuple(map(int, NEST_POS))
        self.queen = Queen(queen_pos, self)

        # Spawn enemies first to occupy some space potentially
        for _ in range(INITIAL_ENEMIES):
            self.spawn_enemy()

        # Spawn initial workers, trying around the nest
        spawn_attempts = 0
        while len(self.ants) < INITIAL_WORKERS and spawn_attempts < INITIAL_WORKERS * 3:
            ox = random.randint(-NEST_RADIUS, NEST_RADIUS)
            oy = random.randint(-NEST_RADIUS, NEST_RADIUS)
            sp = (int(NEST_POS[0] + ox), int(NEST_POS[1] + oy))
            self.add_ant(sp) # add_ant checks validity and occupation
            spawn_attempts += 1
        # Add ants directly at nest center if needed
        while len(self.ants) < INITIAL_WORKERS:
             if not self.add_ant(NEST_POS): # Try nest center
                 # If nest center is blocked, maybe stop or log error
                 break


    def add_ant(self, pos):
        """Add a new worker ant if the position is valid and unoccupied."""
        pt = tuple(map(int, pos))
        if not is_valid(pt):
            return False
        q_pos = self.queen.pos if self.queen else None
        # Check if occupied by queen, other ants, or enemies
        if not self.is_ant_at(pt) and not self.is_enemy_at(pt) and pt != q_pos:
            self.ants.append(Ant(pt, self))
            return True
        return False

    def spawn_enemy(self):
        """Spawn a new enemy at a random valid location."""
        tries = 0
        while tries < 50:
            ex = random.randint(0, GRID_WIDTH - 1)
            ey = random.randint(0, GRID_HEIGHT - 1)
            sp = (ex, ey)
            q_pos = self.queen.pos if self.queen else None
            # Check distance from nest and if position is free
            if (distance_sq(sp, NEST_POS) > (MIN_FOOD_DIST_FROM_NEST // 2)**2 and
                    not self.is_enemy_at(sp) and
                    not self.is_ant_at(sp) and
                    sp != q_pos):
                self.enemies.append(Enemy(sp, self))
                return True
            tries += 1
        return False # Failed to find a spot

    def kill_ant(self, ant_to_remove):
        """Safely remove an ant from the simulation."""
        try:
            self.ants.remove(ant_to_remove)
        except ValueError:
            pass # Ant might already be removed

    def kill_enemy(self, enemy_to_remove):
        """Safely remove an enemy and drop food."""
        try:
            if is_valid(enemy_to_remove.pos):
                fx, fy = enemy_to_remove.pos
                self.grid.food[fx, fy] = min(
                    MAX_FOOD_PER_CELL,
                    self.grid.food[fx, fy] + ENEMY_TO_FOOD_ON_DEATH
                )
            self.enemies.remove(enemy_to_remove)
        except ValueError:
            pass # Enemy might already be removed

    def kill_queen(self, queen_to_remove):
        """Handle the queen's death (end simulation)."""
        print("--- SIMULATION END: QUEEN DIED ---")
        self.queen = None
        self.running = False

    def is_ant_at(self, pos, exclude_self=None):
        """Check if an ant (worker or queen) is at a given position."""
        pos_tuple = tuple(pos)
        # Check queen (blocks unless excluded)
        if (self.queen and self.queen.pos == pos_tuple and
                exclude_self != self.queen):
            return True
        # Check workers
        for ant in self.ants:
            if ant is exclude_self:
                continue  # Don't compare ant to itself
            if ant.pos == pos_tuple:
                return True
        return False # Position is free of ants

    def get_ant_at(self, pos):
        """Get the specific ant object at a position, or None."""
        pt = tuple(pos)
        if self.queen and self.queen.pos == pt:
            return self.queen
        for ant in self.ants:
            if ant.pos == pt:
                return ant
        return None

    def is_enemy_at(self, pos):
        """Check if an enemy is at a given position."""
        pt = tuple(pos)
        for e in self.enemies:
            if e.pos == pt:
                return True
        return False

    def get_enemy_at(self, pos):
        """Get the specific enemy object at a position, or None."""
        pt = tuple(pos)
        for e in self.enemies:
            if e.pos == pt:
                return e
        return None

    def update(self):
        """Run one simulation tick: update entities and grid."""
        self.ticks += 1
        if not self.running:
            return

        # Update Queen
        if self.queen:
            self.queen.update()

        # Update Workers and Enemies (use list copies for safe iteration)
        for ant in list(self.ants):
            if ant in self.ants:  # Check if still alive
                ant.update()
        for enemy in list(self.enemies):
            if enemy in self.enemies: # Check if still alive
                enemy.update()

        # Update pheromones (decay, diffusion)
        self.grid.update_pheromones()

        # Spawn new enemies periodically
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= ENEMY_SPAWN_RATE:
            self.enemy_spawn_timer = 0
            self.spawn_enemy()

    def draw_debug_info(self):
        """Render debug information onto the screen."""
        ant_c = len(self.ants)
        enemy_c = len(self.enemies)
        food_c = int(self.food_collected)
        tick_c = self.ticks
        fps = self.clock.get_fps()

        # Basic simulation stats
        texts = [f"A:{ant_c}", f"F:{enemy_c}", f"S:{food_c}", f"T:{tick_c}"]
        y_pos = 5
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (5, y_pos + i * 18))

        # FPS display
        fps_color = (0, 255, 0) if fps > 25 else (255, 255, 0) if fps > 15 else (255, 0, 0)
        fps_surface = self.font.render(f"{fps:.0f}", True, fps_color)
        self.screen.blit(fps_surface, (WIDTH - 35, 5))

        # Info for ant under mouse cursor
        try:
            mx, my = pygame.mouse.get_pos()
            gx, gy = mx // CELL_SIZE, my // CELL_SIZE
            if is_valid((gx, gy)):
                ant = self.get_ant_at((gx, gy))
                # Display worker ant details
                if ant and isinstance(ant, Ant) and not isinstance(ant, Queen):
                    state_char = ant.state.name[0] if ant.state else '?'
                    debug_lines = [
                        f"S:{state_char}", f"HP:{ant.hp}", f"A:{ant.age}",
                        f"C:{ant.carry_amount}", f"St:{ant.stuck_timer}",
                        f"M:{ant.last_move_info[:15]}" # Truncate long move info
                    ]
                    y_offset = HEIGHT - (len(debug_lines) * 18) - 5
                    for i, line in enumerate(debug_lines):
                        line_surface = self.font.render(line, True, (255, 255, 0))
                        self.screen.blit(line_surface, (5, y_offset + i * 18))

                # Display pheromone levels at cursor
                home_ph = self.grid.get_pheromone((gx, gy), 'home')
                food_ph = self.grid.get_pheromone((gx, gy), 'food')
                alarm_ph = self.grid.get_pheromone((gx, gy), 'alarm')
                ph_h_surf = self.font.render(
                    f"H:{home_ph:.0f}", True, PHEROMONE_HOME_COLOR
                )
                ph_f_surf = self.font.render(
                    f"F:{food_ph:.0f}", True, PHEROMONE_FOOD_COLOR
                )
                ph_a_surf = self.font.render(
                    f"A:{alarm_ph:.0f}", True, PHEROMONE_ALARM_COLOR
                )
                self.screen.blit(ph_h_surf, (mx + 8, my - 10))
                self.screen.blit(ph_f_surf, (mx + 8, my + 2))
                self.screen.blit(ph_a_surf, (mx + 8, my + 14))
        except Exception:
            pass # Ignore errors during debug drawing

    def draw(self):
        """Draw the entire simulation state to the screen."""
        self.screen.fill(MAP_BG_COLOR)
        self.grid.draw(self.screen)
        if self.queen:
            self.queen.draw(self.screen)
        # Draw entities efficiently
        for ant in self.ants: ant.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        # Overlay debug info if enabled
        if self.show_debug_info:
            self.draw_debug_info()
        pygame.display.flip()

    def handle_events(self):
        """Process user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_d: # Toggle debug info
                    self.show_debug_info = not self.show_debug_info

    def run(self):
        """Start and manage the main simulation loop."""
        TARGET_FPS = 60
        while self.running:
            self.handle_events()    # Check for input
            if self.running:        # Re-check if events caused quit
                self.update()       # Update simulation state
            self.draw()             # Render the screen
            self.clock.tick(TARGET_FPS) # Maintain frame rate
        pygame.quit()


# --- Start Simulation ---
if __name__ == '__main__':
    try:
        import numpy
    except ImportError:
        print("FEHLER: NumPy Modul nicht gefunden!")
        print("Bitte installiere NumPy (z.B. ueber Pydroid 3 -> Pip -> numpy).")
        exit()

    print("Starte Ameisen Simulation...")
    simulation = AntSimulation()
    simulation.run()
    print("Simulation beendet.")