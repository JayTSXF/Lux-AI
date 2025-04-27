import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

class LuxAI_S3_Env(gym.Env):
    """
    Custom environment for Lux AI Season 3 Challenge.
    Implements core mechanics: factories (spawn points), units, energy collection,
    fog of war, movement, sapping, relic point generation, and scoring&#8203;:contentReference[oaicite:20]{index=20}&#8203;:contentReference[oaicite:21]{index=21}.
    This environment supports two-player self-play and is compatible with CleanRL's PPO.
    """
    def __init__(self, map_size=24, max_steps=100, spawn_rate=3, fog_of_war=True):
        super().__init__()
        # Game parameters (could be randomized within ranges as in official game)
        self.map_size = map_size
        self.max_steps = max_steps
        self.spawn_rate = spawn_rate
        self.fog_of_war = fog_of_war
        # Unit parameters (using default values from Lux S3 specs)
        self.max_units = 16
        self.init_unit_energy = 100
        self.max_unit_energy = 400
        self.min_unit_energy = 0
        self.unit_move_cost = 2         # cost to move (except staying)&#8203;:contentReference[oaicite:22]{index=22}
        self.unit_sap_cost = 10        # energy cost to perform a sap&#8203;:contentReference[oaicite:23]{index=23}
        self.unit_sap_range = 4        # sap range (Chebyshev distance)&#8203;:contentReference[oaicite:24]{index=24}
        self.unit_sap_dropoff = 0.5    # dropoff factor for adjacent sap effect&#8203;:contentReference[oaicite:25]{index=25}
        self.unit_sensor_range = 2     # sensor range for vision&#8203;:contentReference[oaicite:26]{index=26}
        self.nebula_vision_reduction = 1  # vision reduction per nebula tile&#8203;:contentReference[oaicite:27]{index=27}
        self.nebula_energy_reduction = 0  # energy reduction per step on nebula&#8203;:contentReference[oaicite:28]{index=28}
        self.unit_energy_void_factor = 0.125  # factor for void field strength&#8203;:contentReference[oaicite:29]{index=29}
        # Map representation
        # We'll use numeric codes for tile types:
        # 0 = empty, 1 = asteroid, 2 = nebula, 3 = energy node, 4 = relic node
        self.map_tiles = None
        self.energy_field = None   # 2D array of energy values per tile
        self.relic_mask = None     # 2D boolean array of which tiles yield relic points when occupied
        # Spawn locations for two teams (corners of the map)
        self.spawn_locs = {"player_0": (0, 0), "player_1": (map_size-1, map_size-1)}
        # Unit state: each team has an array of unit info (id index)
        # We'll store for each unit id: [alive_flag, x, y, energy]
        self.units = {
            "player_0": np.zeros((self.max_units, 4), dtype=np.int32),
            "player_1": np.zeros((self.max_units, 4), dtype=np.int32)
        }
        # Points for each team
        self.points = {"player_0": 0, "player_1": 0}
        # Step counter
        self.step_count = 0
        # For exploration bonus, track tiles seen by each player
        self.discovered = {"player_0": np.zeros((map_size, map_size), dtype=bool),
                           "player_1": np.zeros((map_size, map_size), dtype=bool)}
        # Observation and action spaces
        # Observation space: Dict of {player_0: {"map": ..., "units": ...}, player_1: {...}}
        # Map: (map_size, map_size, channels), Units: (max_units, 4)
        self.obs_channels = 6  # asteroid, nebula, energy, relic_node, friendly_unit, enemy_unit
        # Define observation sub-spaces
        map_obs_space = gym.spaces.Box(low=0, high=1,
                                   shape=(self.map_size, self.map_size, self.obs_channels),
                                   dtype=np.float32)
        # Units features: [x_norm, y_norm, energy_norm, alive_flag]
        units_obs_space = gym.spaces.Box(low=0, high=1,
                                     shape=(self.max_units, 4),
                                     dtype=np.float32)
        player_obs_space = gym.spaces.Dict({"map": map_obs_space, "units": units_obs_space})
        self.observation_space = spaces.Dict({"player_0": player_obs_space, "player_1": player_obs_space})
        # Action space: Dict of {player_0: MultiDiscrete([...]), player_1: MultiDiscrete([...])}
        # Each player's action is a vector of length max_units (one action per unit).
        # 0-4 = move (0: stay, 1: up, 2: right, 3: down, 4: left)
        # 5+ = sap target relative offset (dx, dy) within range.
        sap_target_count = (2 * self.unit_sap_range + 1) ** 2
        total_actions_per_unit = 5 + sap_target_count
        self.action_space = spaces.Dict({
            "player_0": spaces.MultiDiscrete([total_actions_per_unit] * self.max_units),
            "player_1": spaces.MultiDiscrete([total_actions_per_unit] * self.max_units)
        })
        # Random number generator
        self._np_random, _ = gym.utils.seeding.np_random()
        # Initialize environment state
        self.reset()

    def reset(self):
        """Reset the environment to the start of a new match (episode)."""
        # Reset step count and points
        self.step_count = 0
        self.points = {"player_0": 0, "player_1": 0}
        # Reset discovered tiles for fog of war exploration
        self.discovered = {"player_0": np.zeros((self.map_size, self.map_size), dtype=bool),
                           "player_1": np.zeros((self.map_size, self.map_size), dtype=bool)}
        # Initialize map tiles and fields
        self.map_tiles = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        self.energy_field = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.relic_mask = np.zeros((self.map_size, self.map_size), dtype=bool)
        # Generate terrain and objects symmetrically
        # 1. Asteroids (impassable)&#8203;:contentReference[oaicite:30]{index=30}
        placed = set()
        num_asteroids = 10
        for _ in range(num_asteroids):
            x = self._np_random.integers(0, self.map_size)
            y = self._np_random.integers(0, self.map_size)
            sym = (self.map_size - 1 - x, self.map_size - 1 - y)
            if (x, y) in placed or (x, y) == sym:
                continue
            if (x, y) in (self.spawn_locs["player_0"], self.spawn_locs["player_1"]):
                continue  # don't block spawn
            placed.add((x, y)); placed.add(sym)
            self.map_tiles[x, y] = 1  # asteroid
            self.map_tiles[sym[0], sym[1]] = 1
        # 2. Nebula (passable, vision/energy reduction)&#8203;:contentReference[oaicite:31]{index=31}&#8203;:contentReference[oaicite:32]{index=32}
        placed = set()
        num_nebula = 10
        for _ in range(num_nebula):
            x = self._np_random.integers(0, self.map_size)
            y = self._np_random.integers(0, self.map_size)
            sym = (self.map_size - 1 - x, self.map_size - 1 - y)
            if self.map_tiles[x, y] != 0 or (x, y) in placed or (x, y) == sym:
                continue
            if (x, y) in (self.spawn_locs["player_0"], self.spawn_locs["player_1"]):
                continue
            placed.add((x, y)); placed.add(sym)
            self.map_tiles[x, y] = 2  # nebula
            self.map_tiles[sym[0], sym[1]] = 2
        # 3. Energy Nodes (provide energy field)&#8203;:contentReference[oaicite:33]{index=33}
        energy_node_positions = []
        num_energy = 3
        for _ in range(num_energy):
            x = self._np_random.integers(0, self.map_size)
            y = self._np_random.integers(0, self.map_size)
            sym = (self.map_size - 1 - x, self.map_size - 1 - y)
            if self.map_tiles[x, y] != 0 or (x, y) == sym:
                continue
            if (x, y) in (self.spawn_locs["player_0"], self.spawn_locs["player_1"]):
                continue
            self.map_tiles[x, y] = 3; self.map_tiles[sym[0], sym[1]] = 3
            energy_node_positions.append((x, y)); energy_node_positions.append(sym)
        # Precompute energy field values from nodes (simple distance-based function)
        for (nx, ny) in energy_node_positions:
            max_e = 10  # maximum energy value at the node
            for i in range(self.map_size):
                for j in range(self.map_size):
                    dist = abs(i - nx) + abs(j - ny)  # Manhattan distance for simplicity
                    contrib = max(0, max_e - dist)
                    self.energy_field[i, j] += contrib
        self.energy_field = np.clip(self.energy_field, 0, 20)
        # 4. Relic Nodes (hidden scoring tiles)&#8203;:contentReference[oaicite:34]{index=34}
        x = self._np_random.integers(5, self.map_size-5)
        y = self._np_random.integers(5, self.map_size-5)
        sym = (self.map_size - 1 - x, self.map_size - 1 - y)
        if self.map_tiles[x, y] == 0 and (x, y) != sym:
            self.map_tiles[x, y] = 4; self.map_tiles[sym[0], sym[1]] = 4
            # Generate a random 5x5 mask of scoring tiles around each relic node&#8203;:contentReference[oaicite:35]{index=35}
            mask_size = 5
            mask = self._np_random.random((mask_size, mask_size)) < 0.2  # 20% chance each
            cx, cy = x, y; sx, sy = sym
            for dx in range(-mask_size//2, mask_size//2+1):
                for dy in range(-mask_size//2, mask_size//2+1):
                    if mask[dx + mask_size//2, dy + mask_size//2]:
                        tx, ty = cx+dx, cy+dy
                        tx2, ty2 = sx+dx, sy+dy
                        if 0 <= tx < self.map_size and 0 <= ty < self.map_size:
                            self.relic_mask[tx, ty] = True
                        if 0 <= tx2 < self.map_size and 0 <= ty2 < self.map_size:
                            self.relic_mask[tx2, ty2] = True
        # Initialize units (all dead to start, then spawn initial units)
        for pid in self.units:
            self.units[pid][:] = np.array([0, 0, 0, 0])
        # Spawn one unit for each team at the spawn locations
        for pid, (sx, sy) in self.spawn_locs.items():
            uid = 0
            # Ensure spawn tile not asteroid
            if self.map_tiles[sx, sy] == 1:
                self.map_tiles[sx, sy] = 0
            self.units[pid][uid] = [1, sx, sy, self.init_unit_energy]
        # Return initial observations for both players
        return self._get_observation()

    def _get_observation(self):
        """
        Compute the observation for both players with fog of war applied.
        Each observation contains a 'map' (spatial features) and a 'units' array of per-unit stats.
        """
        obs = {}
        for pid in ["player_0", "player_1"]:
            visible = np.zeros((self.map_size, self.map_size), dtype=bool)
            # Compute team vision mask&#8203;:contentReference[oaicite:36]{index=36}&#8203;:contentReference[oaicite:37]{index=37}
            for alive, ux, uy, energy in self.units[pid]:
                if alive == 0:
                    continue
                # Vision contribution from this unit
                for dx in range(-self.unit_sensor_range, self.unit_sensor_range + 1):
                    for dy in range(-self.unit_sensor_range, self.unit_sensor_range + 1):
                        tx, ty = ux + dx, uy + dy
                        if 0 <= tx < self.map_size and 0 <= ty < self.map_size:
                            # Vision power decreases with distance
                            vis_power = 1 + self.unit_sensor_range - max(abs(dx), abs(dy))
                            if dx == 0 and dy == 0:
                                vis_power += 10  # ensure unit sees itself&#8203;:contentReference[oaicite:38]{index=38}
                            if self.map_tiles[tx, ty] == 2:  # nebula tile causes vision reduction
                                vis_power -= self.nebula_vision_reduction
                            if vis_power > 0:
                                visible[tx, ty] = True
            # Mark newly discovered tiles for exploration bonus
            new_visible = visible & ~self.discovered[pid]
            self.discovered[pid] |= visible
            new_tiles_count = np.sum(new_visible)
            if not hasattr(self, "new_tiles_observed"):
                self.new_tiles_observed = {}
            self.new_tiles_observed[pid] = int(new_tiles_count)
            # Build map feature channels
            max_energy_val = 20.0
            asteroid_chan = (self.map_tiles == 1) & visible
            nebula_chan = (self.map_tiles == 2) & visible
            energy_chan = np.where(visible, self.energy_field / max_energy_val, 0)
            relic_chan = (self.map_tiles == 4) & visible
            friendly_chan = np.zeros((self.map_size, self.map_size), dtype=float)
            enemy_chan = np.zeros((self.map_size, self.map_size), dtype=float)
            # Friendly units presence
            for alive, ux, uy, energy in self.units[pid]:
                if alive == 1:
                    friendly_chan[ux, uy] += 1
            friendly_chan = np.clip(friendly_chan, 0, 1)
            # Enemy units presence (only if visible)
            opp = "player_1" if pid == "player_0" else "player_0"
            for alive, ex, ey, e_energy in self.units[opp]:
                if alive == 1 and visible[ex, ey]:
                    enemy_chan[ex, ey] += 1
            enemy_chan = np.clip(enemy_chan, 0, 1)
            # Stack channels into an array (H x W x C)
            map_channels = np.stack([
                asteroid_chan.astype(float),
                nebula_chan.astype(float),
                energy_chan.astype(float),
                relic_chan.astype(float),
                friendly_chan.astype(float),
                enemy_chan.astype(float)
            ], axis=-1)
            # Build per-unit feature array
            units_arr = np.zeros((self.max_units, 4), dtype=float)
            for uid in range(self.max_units):
                alive, ux, uy, energy = self.units[pid][uid]
                if alive == 1:
                    units_arr[uid, 0] = ux / (self.map_size - 1)  # x position (normalized)
                    units_arr[uid, 1] = uy / (self.map_size - 1)  # y position (normalized)
                    units_arr[uid, 2] = energy / self.max_unit_energy  # energy (normalized)
                    units_arr[uid, 3] = 1.0  # alive flag
                else:
                    units_arr[uid, 3] = 0.0
            # Convert to torch tensors for efficient usage in PyTorch-based PPO
            obs[pid] = {
                "map": torch.from_numpy(map_channels).float(),
                "units": torch.from_numpy(units_arr).float()
            }
        return obs

    def step(self, actions):
        """
        Execute one time step with actions from both players.
        `actions` is a dict: {"player_0": action_array, "player_1": action_array},
        where each action_array is length `max_units`. Returns (obs, rewards, done, info).
        """
        rewards = {"player_0": 0.0, "player_1": 0.0}
        wasted_energy = {"player_0": 0, "player_1": 0}   # energy spent on ineffective actions
        invalid_actions = {"player_0": 0, "player_1": 0}  # actions that could not be executed
        # Store energy after move (before sap) for collision/void calculations
        energy_after_move = {"player_0": [], "player_1": []}
        # Phase 1: Movement for both players&#8203;:contentReference[oaicite:39]{index=39}
        for pid in ["player_0", "player_1"]:
            acts = actions[pid]
            # Convert actions to list for iteration (accept torch or numpy inputs)
            if isinstance(acts, np.ndarray):
                acts = acts.tolist()
            elif isinstance(acts, torch.Tensor):
                acts = acts.cpu().numpy().tolist()
            for uid, action in enumerate(acts):
                alive, x, y, energy = self.units[pid][uid]
                if alive == 0:
                    continue  # skip if no unit
                action = int(action)
                if action < 5:
                    # Movement action (0: stay, 1: up, 2: right, 3: down, 4: left)
                    dx = dy = 0
                    if action == 1: dx = -1   # up (decrease x)
                    elif action == 2: dy = 1  # right (increase y)
                    elif action == 3: dx = 1   # down (increase x)
                    elif action == 4: dy = -1  # left (decrease y)
                    if action == 0:
                        # stay still (no cost)
                        pass
                    elif energy >= self.unit_move_cost:
                        new_x, new_y = x + dx, y + dy
                        if new_x < 0 or new_x >= self.map_size or new_y < 0 or new_y >= self.map_size:
                            # Move off map: no movement, but energy consumed&#8203;:contentReference[oaicite:40]{index=40}
                            self.units[pid][uid, 3] = energy - self.unit_move_cost
                            wasted_energy[pid] += self.unit_move_cost
                        elif self.map_tiles[new_x, new_y] == 1:
                            # Move into asteroid: blocked, energy consumed&#8203;:contentReference[oaicite:41]{index=41}
                            self.units[pid][uid, 3] = energy - self.unit_move_cost
                            wasted_energy[pid] += self.unit_move_cost
                        else:
                            # Valid move (including moving into empty, nebula, or any tile that is not asteroid)
                            self.units[pid][uid, 1] = new_x
                            self.units[pid][uid, 2] = new_y
                            self.units[pid][uid, 3] = energy - self.unit_move_cost
                    else:
                        # Not enough energy to move – action invalid (no effect)
                        invalid_actions[pid] += 1
                else:
                    # Sap action will be processed in next phase
                    pass
            # Record energies after all moves (before saps) for collision checks
            energy_after_move[pid] = [e for (_, _, _, e) in self.units[pid]]
        # Phase 2: Sapping actions&#8203;:contentReference[oaicite:42]{index=42}
        for pid in ["player_0", "player_1"]:
            opp = "player_1" if pid == "player_0" else "player_0"
            acts = actions[pid]
            if isinstance(acts, np.ndarray):
                acts = acts.tolist()
            elif isinstance(acts, torch.Tensor):
                acts = acts.cpu().numpy().tolist()
            for uid, action in enumerate(acts):
                alive, x, y, energy = self.units[pid][uid]
                if alive == 0 or action < 5:
                    continue  # skip if unit is dead or not a sap action
                offset = action - 5
                side = 2 * self.unit_sap_range + 1
                dx = offset // side - self.unit_sap_range
                dy = offset % side - self.unit_sap_range
                if energy >= self.unit_sap_cost:
                    # Enough energy to perform sap
                    self.units[pid][uid, 3] = energy - self.unit_sap_cost  # spend energy
                    target_x, target_y = x + dx, y + dy
                    # Direct sap on target tile (within range square)&#8203;:contentReference[oaicite:43]{index=43}
                    if 0 <= target_x < self.map_size and 0 <= target_y < self.map_size:
                        for eid, (e_alive, ex, ey, e_energy) in enumerate(self.units[opp]):
                            if e_alive == 1 and ex == target_x and ey == target_y:
                                # Enemy on target tile loses full sap cost energy
                                self.units[opp][eid, 3] = e_energy - self.unit_sap_cost
                    # Splash sap on adjacent 8 tiles (drop-off)&#8203;:contentReference[oaicite:44]{index=44}
                    sap_drop = int(self.unit_sap_cost * self.unit_sap_dropoff)
                    if 0 <= target_x < self.map_size and 0 <= target_y < self.map_size:
                        for dx2 in [-1, 0, 1]:
                            for dy2 in [-1, 0, 1]:
                                if dx2 == 0 and dy2 == 0:
                                    continue
                                nx, ny = target_x + dx2, target_y + dy2
                                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                                    for eid, (e_alive, ex, ey, e_energy) in enumerate(self.units[opp]):
                                        if e_alive == 1 and ex == nx and ey == ny:
                                            # Enemy adjacent to target loses reduced energy
                                            self.units[opp][eid, 3] = e_energy - sap_drop
                    # If target is out of bounds (sap into space), or no enemies hit, count as waste
                    if not (0 <= target_x < self.map_size and 0 <= target_y < self.map_size):
                        wasted_energy[pid] += self.unit_sap_cost
                    else:
                        # Check if any enemy was within the 3x3 sap area
                        hit = False
                        for eid, (e_alive, ex, ey, e_energy) in enumerate(self.units[opp]):
                            if e_alive == 1 and abs(ex - (x+dx)) <= 1 and abs(ey - (y+dy)) <= 1:
                                hit = True
                                break
                        if not hit:
                            wasted_energy[pid] += self.unit_sap_cost
                else:
                    # Not enough energy to sap – invalid action
                    invalid_actions[pid] += 1
        # Phase 3: Collisions (resolve units on same tile)&#8203;:contentReference[oaicite:45]{index=45}
        removed = {"player_0": [False]*self.max_units, "player_1": [False]*self.max_units}
        # Map positions to lists of units from each team
        pos_to_units = {}
        for pid in ["player_0", "player_1"]:
            for uid, (alive, ux, uy, energy) in enumerate(self.units[pid]):
                if alive == 1:
                    pos = (ux, uy)
                    pos_to_units.setdefault(pos, {"player_0": [], "player_1": []})
                    pos_to_units[pos][pid].append(uid)
        for pos, teams in pos_to_units.items():
            if teams["player_0"] and teams["player_1"]:
                # Both teams have units on this tile, determine outcome by total energy&#8203;:contentReference[oaicite:46]{index=46}
                total_p0 = sum(energy_after_move["player_0"][uid] for uid in teams["player_0"])
                total_p1 = sum(energy_after_move["player_1"][uid] for uid in teams["player_1"])
                if total_p0 > total_p1:
                    # player_0 wins: remove all player_1 units on this tile
                    for uid in teams["player_1"]:
                        self.units["player_1"][uid, 0] = 0
                        removed["player_1"][uid] = True
                elif total_p1 > total_p0:
                    # player_1 wins
                    for uid in teams["player_0"]:
                        self.units["player_0"][uid, 0] = 0
                        removed["player_0"][uid] = True
                else:
                    # tie: all units on this tile are removed
                    for uid in teams["player_0"]:
                        self.units["player_0"][uid, 0] = 0
                        removed["player_0"][uid] = True
                    for uid in teams["player_1"]:
                        self.units["player_1"][uid, 0] = 0
                        removed["player_1"][uid] = True
        # Energy Void Fields (passive sapping around each unit)&#8203;:contentReference[oaicite:47]{index=47}
        void_map_p0 = np.zeros((self.map_size, self.map_size), dtype=int)
        void_map_p1 = np.zeros((self.map_size, self.map_size), dtype=int)
        # Each surviving unit contributes to enemy void map on adjacent tiles
        for uid, (alive, x, y, energy) in enumerate(self.units["player_0"]):
            if alive == 1 and not removed["player_0"][uid]:
                e = energy_after_move["player_0"][uid]
                void_strength = int(e * self.unit_energy_void_factor)  # energy contribution&#8203;:contentReference[oaicite:48]{index=48}
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:  # cardinal neighbors
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        void_map_p0[nx, ny] += void_strength
        for uid, (alive, x, y, energy) in enumerate(self.units["player_1"]):
            if alive == 1 and not removed["player_1"][uid]:
                e = energy_after_move["player_1"][uid]
                void_strength = int(e * self.unit_energy_void_factor)
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        void_map_p1[nx, ny] += void_strength
        # Apply void damage to each unit from opposing team's void map
        for pid, opp in [("player_0", "player_1"), ("player_1", "player_0")]:
            for uid, (alive, x, y, energy) in enumerate(self.units[pid]):
                if alive == 1 and not removed[pid][uid]:
                    # total void strength from opponent affecting this tile
                    V = void_map_p1[x, y] if pid == "player_0" else void_map_p0[x, y]
                    # number of same-team units on this tile (share damage)&#8203;:contentReference[oaicite:49]{index=49}
                    count = sum(1 for (a, ux, uy, en) in self.units[pid] if a == 1 and ux == x and uy == y)
                    if count > 0:
                        damage = V // count
                        self.units[pid][uid, 3] = energy - damage
        # Remove units with energy < 0 (killed by sap or void)&#8203;:contentReference[oaicite:50]{index=50}
        for pid in ["player_0", "player_1"]:
            for uid, (alive, x, y, energy) in enumerate(self.units[pid]):
                if alive == 1 and energy < 0:
                    self.units[pid][uid, 0] = 0
        # Phase 4: Environment effects – energy recharge and nebula drain&#8203;:contentReference[oaicite:51]{index=51}&#8203;:contentReference[oaicite:52]{index=52}
        for pid in ["player_0", "player_1"]:
            for uid, (alive, x, y, energy) in enumerate(self.units[pid]):
                if alive == 1:
                    # Nebula tile causes energy reduction (not below 0)&#8203;:contentReference[oaicite:53]{index=53}
                    if self.map_tiles[x, y] == 2:
                        energy = max(0, energy - self.nebula_energy_reduction)
                    # Energy field recharge from energy nodes&#8203;:contentReference[oaicite:54]{index=54}
                    energy += int(self.energy_field[x, y])
                    # Cap energy at max limit
                    if energy > self.max_unit_energy:
                        energy = self.max_unit_energy
                    self.units[pid][uid, 3] = energy
        # Phase 5: Spawn new units at factories (spawn_rate timing)
        if (self.step_count + 1) % self.spawn_rate == 0:
            for pid in ["player_0", "player_1"]:
                # spawn at most one unit per team if there's capacity
                for uid, (alive, _, _, _) in enumerate(self.units[pid]):
                    if alive == 0:
                        sx, sy = self.spawn_locs[pid]
                        if self.map_tiles[sx, sy] == 1:  # clear asteroid if somehow present
                            self.map_tiles[sx, sy] = 0
                        self.units[pid][uid] = [1, sx, sy, self.init_unit_energy]
                        break
        # (Phase 6: vision updates already handled in observation, Phase 7: object drift not implemented)
        # Phase 8: Relic point scoring&#8203;:contentReference[oaicite:55]{index=55}
        points_gained = {"player_0": 0, "player_1": 0}
        for pid in ["player_0", "player_1"]:
            positions = set()
            for uid, (alive, x, y, energy) in enumerate(self.units[pid]):
                if alive == 1:
                    pos = (x, y)
                    if pos in positions:
                        continue
                    positions.add(pos)
                    if self.relic_mask[x, y]:
                        points_gained[pid] += 1
                        # Each unique scoring tile yields at most one point per turn&#8203;:contentReference[oaicite:56]{index=56}
            self.points[pid] += points_gained[pid]
        # Calculate rewards for this step
        for pid in ["player_0", "player_1"]:
            # Relic points collected
            rewards[pid] += points_gained[pid]
            # Energy collected from nodes (sum of energy field values gained by all units, scaled)
            energy_gain = 0
            for alive, x, y, e in self.units[pid]:
                if alive == 1:
                    energy_gain += int(self.energy_field[x, y])
            rewards[pid] += energy_gain * 0.01
            # Penalty for wasted energy on ineffective actions
            rewards[pid] -= wasted_energy[pid] * 0.01
            # Penalty for invalid actions (very small)
            rewards[pid] -= invalid_actions[pid] * 0.01
        # Exploration bonus for newly discovered tiles
        if hasattr(self, "new_tiles_observed"):
            for pid in ["player_0", "player_1"]:
                new_tiles = self.new_tiles_observed.get(pid, 0)
                rewards[pid] += new_tiles * 0.01
        # Advance turn
        self.step_count += 1
        done = False
        if self.step_count >= self.max_steps:
            done = True
        # Get next observations
        obs = self._get_observation()
        # Info: include current score and winner at end
        info = {"points": self.points.copy()}
        if done:
            if self.points["player_0"] > self.points["player_1"]:
                info["winner"] = "player_0"
            elif self.points["player_1"] > self.points["player_0"]:
                info["winner"] = "player_1"
            else:
                # Tie-breaker: higher total energy wins, otherwise tie
                total_e0 = sum(e for (_, _, _, e) in self.units["player_0"])
                total_e1 = sum(e for (_, _, _, e) in self.units["player_1"])
                if total_e0 > total_e1:
                    info["winner"] = "player_0"
                elif total_e1 > total_e0:
                    info["winner"] = "player_1"
                else:
                    info["winner"] = "tie"
        return obs, rewards, done, info

    def render(self, mode='human'):
        """Simple textual rendering of the map state."""
        if mode != 'human':
            return
        symbols = {0: '.', 1: '#', 2: '~', 3: 'E', 4: 'R'}
        # Overlay units on map
        render_map = [[symbols[self.map_tiles[i, j]] for j in range(self.map_size)] for i in range(self.map_size)]
        # Mark units by team (0 or 1)
        for pid, symbol in zip(["player_0", "player_1"], ['0', '1']):
            for alive, x, y, energy in self.units[pid]:
                if alive == 1:
                    render_map[x][y] = symbol
        output = "\n".join("".join(row) for row in render_map)
        print(output)
        print(f"Points: P0={self.points['player_0']}  P1={self.points['player_1']}")

