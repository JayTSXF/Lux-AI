import sys
import gym
from gym import spaces
import numpy as np

# Import global constants and helper functions from base
from base import Global, ActionType, SPACE_SIZE, get_opposite

# Define constants: number of teams, maximum number of units, maximum number of relic nodes
NUM_TEAMS = 2
MAX_UNITS = Global.MAX_UNITS
MAX_RELIC_NODES = Global.MAX_RELIC_NODES

class PPOGameEnv(gym.Env):
    """
    PPOGameEnv simulates an environment that closely mimics the real competition setting and meets the following requirements:
    
    1. Observation Data Computation Modifications:
       - Each allied unit has its own independent sensor mask (calculated by compute_unit_vision(unit)),
         and get_unit_obs(unit) constructs a local observation in a fixed JSON-like format.
       - The global observation returned to the agent is the union (logical "OR") of the sensor masks 
         from all allied units, preserving the fixed observation format required in the competition.
    
    2. Reward Function Optimization:
       The environment state is updated based on the actions and returns (observation, reward, done, info).
       The modified reward logic is as follows:
          1. Each unit computes its own unit_reward individually.
          2. If a movement action causes the unit to move off the map or the target tile is an Asteroid, 
             the action is deemed invalid and unit_reward is decreased by 0.2.
          3. Sap Action:
             - Check if a relic is visible in the unit’s local observation (in relic_nodes_mask);
             - If visible, count the number of enemy units in the unit’s 8-neighborhood. 
               If the count is >= 2, then the sap reward equals +1.0 multiplied by the number of enemy units; otherwise, subtract 2.0.
             - If no relic is visible, subtract 2.0.
          4. Non-sap Actions:
             - After a successful move, check if the unit is located at a potential point within any relic configuration:
                  * If this is the first visit to the potential point, add +2.0 to unit_reward and mark it as visited;
                  * If the potential point has not yet yielded a team point, then increase self.score by 1, 
                    add +3.0 to unit_reward, and mark it as team_points_space;
                  * If the unit is already on team_points_space, then award +3.0 reward per turn.
             - If the unit is on an energy node (energy == Global.MAX_ENERGY_PER_TILE), add +0.2 to unit_reward;
             - If the unit is on a Nebula (tile_type == 1), subtract 0.2 from unit_reward;
             - If after moving the unit overlaps with an enemy unit and the enemy's energy is lower, 
               then for each such enemy unit, add +1.0 reward.
          5. Global Exploration Reward: For each new tile discovered by the combined vision of all allied units, add +0.1 reward.
          6. At the end of each step, the final reward is computed as: (points reward * 0.5) + (rule-based reward * 0.5).
    
    3. Enemy Unit Strategy:
       - After spawning, enemy units do not act proactively; their positions are updated only every 20 steps 
         by shifting the map to the right by 1 tile, making them passive opponents.
       - This design is primarily for early-stage debugging, with more proactive adversarial strategies to be introduced later.
    """
    
    def __init__(self):
        super(PPOGameEnv, self).__init__()
        
        # Modify the action space: each unit makes independent decisions (action values range from 0 to 5)
        self.action_space = spaces.MultiDiscrete([len(ActionType)] * MAX_UNITS)
        
        # The observation space remains unchanged
        self.observation_space = spaces.Dict({
            "units_position": spaces.Box(
                low=0,
                high=SPACE_SIZE - 1,
                shape=(NUM_TEAMS, MAX_UNITS, 2),
                dtype=np.int32
            ),
            "units_energy": spaces.Box(
                low=0,
                high=400,  # Maximum unit energy is 400
                shape=(NUM_TEAMS, MAX_UNITS, 1),
                dtype=np.int32
            ),
            "units_mask": spaces.Box(
                low=0,
                high=1,
                shape=(NUM_TEAMS, MAX_UNITS),
                dtype=np.int8
            ),
            "sensor_mask": spaces.Box(
                low=0,
                high=1,
                shape=(SPACE_SIZE, SPACE_SIZE),
                dtype=np.int8
            ),
            "map_features_tile_type": spaces.Box(
                low=-1,
                high=2,
                shape=(SPACE_SIZE, SPACE_SIZE),
                dtype=np.int8
            ),
            "map_features_energy": spaces.Box(
                low=-1,
                high=Global.MAX_ENERGY_PER_TILE,
                shape=(SPACE_SIZE, SPACE_SIZE),
                dtype=np.int8
            ),
            "relic_nodes_mask": spaces.Box(
                low=0,
                high=1,
                shape=(MAX_RELIC_NODES,),
                dtype=np.int8
            ),
            "relic_nodes": spaces.Box(
                low=-1,
                high=SPACE_SIZE - 1,
                shape=(MAX_RELIC_NODES, 2),
                dtype=np.int32
            ),
            "team_points": spaces.Box(
                low=0,
                high=1000,
                shape=(NUM_TEAMS,),
                dtype=np.int32
            ),
            "team_wins": spaces.Box(
                low=0,
                high=1000,
                shape=(NUM_TEAMS,),
                dtype=np.int32
            ),
            "steps": spaces.Box(
                low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "match_steps": spaces.Box(
                low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "remainingOverageTime": spaces.Box(
                low=0, high=1000, shape=(1,), dtype=np.int32
            ),
            "env_cfg_map_width": spaces.Box(
                low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32
            ),
            "env_cfg_map_height": spaces.Box(
                low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32
            ),
            "env_cfg_max_steps_in_match": spaces.Box(
                low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_move_cost": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_sap_cost": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_sap_range": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            )
        })
        
        self.max_steps = Global.MAX_STEPS_IN_MATCH
        self.current_step = 0

        # Full map state: map tiles, relic markers, energy map
        self.tile_map = None     # -1 unknown, 0 empty, 1 Nebula, 2 Asteroid
        self.relic_map = None    # Relic existence marker, 1 indicates presence
        self.energy_map = None   # Energy value for each tile
        
        # Unit state: lists of allied and enemy units, each unit is represented as a dictionary {"x": int, "y": int, "energy": int}
        self.team_units = []    # Allied units
        self.enemy_units = []   # Enemy units
        
        # Spawn points: allies spawn at top-left, enemies spawn at bottom-right
        self.team_spawn = (0, 0)
        self.enemy_spawn = (SPACE_SIZE - 1, SPACE_SIZE - 1)
        
        # Exploration record: a boolean array for the full map, recording the tiles seen by the combined vision of allied units (recorded only once globally)
        self.visited = None
        
        # Team score (allied score)
        self.score = 0
        
        # Environment configuration parameters (env_cfg)
        self.env_cfg = {
            "map_width": SPACE_SIZE,
            "map_height": SPACE_SIZE,
            "max_steps_in_match": Global.MAX_STEPS_IN_MATCH,
            "unit_move_cost": Global.UNIT_MOVE_COST,
            "unit_sap_cost": Global.UNIT_SAP_COST if hasattr(Global, "UNIT_SAP_COST") else 30,
            "unit_sap_range": Global.UNIT_SAP_RANGE,
        }
        
        # New: for rewards related to relic configurations
        self.relic_configurations = []   # list of (center_x, center_y, mask (5x5 boolean))
        self.potential_visited = None      # Record for the whole map, shape (SPACE_SIZE, SPACE_SIZE)
        self.team_points_space = None      # Record for the whole map indicating which tiles have already contributed a team point

        self._init_state()

    def _init_state(self):
        """Initialize full map state, units, and records"""
        num_tiles = SPACE_SIZE * SPACE_SIZE
        
        # Initialize tile_map: randomly set parts to Nebula (1) or Asteroid (2)
        self.tile_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_nebula = int(num_tiles * 0.1)
        num_asteroid = int(num_tiles * 0.1)
        indices = np.random.choice(num_tiles, num_nebula + num_asteroid, replace=False)
        flat_tiles = self.tile_map.flatten()
        flat_tiles[indices[:num_nebula]] = 1
        flat_tiles[indices[num_nebula:]] = 2
        self.tile_map = flat_tiles.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # Initialize relic_map: randomly select 3 positions and set them to 1 (indicating a relic exists)
        self.relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        relic_indices = np.random.choice(num_tiles, 3, replace=False)
        flat_relic = self.relic_map.flatten()
        flat_relic[relic_indices] = 1
        self.relic_map = flat_relic.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # Initialize energy_map: randomly generate 2 energy nodes, set their value to MAX_ENERGY_PER_TILE, rest are 0
        self.energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_energy_nodes = 2
        indices_energy = np.random.choice(num_tiles, num_energy_nodes, replace=False)
        flat_energy = self.energy_map.flatten()
        for idx in indices_energy:
            flat_energy[idx] = Global.MAX_ENERGY_PER_TILE
        self.energy_map = flat_energy.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # Initialize allied units: initially generate 1 unit at the allied spawn point
        self.team_units = []
        spawn_x, spawn_y = self.team_spawn
        self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})
        
        # Initialize enemy units: initially generate 1 unit at the enemy spawn point
        self.enemy_units = []
        spawn_x_e, spawn_y_e = self.enemy_spawn
        self.enemy_units.append({"x": spawn_x_e, "y": spawn_y_e, "energy": 100})
        
        # Initialize exploration record: create a full map boolean array, marking tiles seen by the combined vision of allied units
        self.visited = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        union_mask = self.get_global_sensor_mask()
        self.visited = union_mask.copy()
        
        # Initialize team score
        self.score = 0
        
        # New: Initialize relic configurations and potential point records
        self.relic_configurations = []
        relic_coords = np.argwhere(self.relic_map == 1)
        for (y, x) in relic_coords:
            # Generate a 5x5 mask, randomly select 8 cells to be True (for training, consider selecting 10 to avoid too sparse rewards)
            mask = np.zeros((5,5), dtype=bool)
            indices = np.random.choice(25, 8, replace=False)
            mask_flat = mask.flatten()
            mask_flat[indices] = True
            mask = mask_flat.reshape((5,5))
            self.relic_configurations.append((x, y, mask))
        self.potential_visited = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        self.team_points_space = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        
        self.current_step = 0

    def compute_unit_vision(self, unit):
        """
        Calculate an independent sensor mask based on the given unit's position.
        The range is determined by the unit's sensor range (Chebyshev distance), and the contribution 
        is reduced for Nebula tiles.
        No wrapping is applied; only tiles within the map are considered.
        Returns a boolean matrix with shape (SPACE_SIZE, SPACE_SIZE).
        """
        sensor_range = Global.UNIT_SENSOR_RANGE
        nebula_reduction = 2
        vision = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        x, y = unit["x"], unit["y"]
        for dy in range(-sensor_range, sensor_range + 1):
            for dx in range(-sensor_range, sensor_range + 1):
                new_x = x + dx
                new_y = y + dy
                if not (0 <= new_x < SPACE_SIZE and 0 <= new_y < SPACE_SIZE):
                    continue
                contrib = sensor_range + 1 - max(abs(dx), abs(dy))
                if self.tile_map[new_y, new_x] == 1:
                    contrib -= nebula_reduction
                vision[new_y, new_x] += contrib
        return vision > 0

    def get_global_sensor_mask(self):
        """
        Return the union (logical OR) of the sensor masks of all allied units.
        """
        mask = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        for unit in self.team_units:
            mask |= self.compute_unit_vision(unit)
        return mask

    def get_unit_obs(self, unit):
        """
        Construct a local observation dictionary based on the independent sensor mask of the given unit,
        in the fixed JSON format as required by the competition.
        Only the area visible to the unit is used for filtering.
        """
        sensor_mask = self.compute_unit_vision(unit)
        map_tile_type = np.where(sensor_mask, self.tile_map, -1)
        map_energy = np.where(sensor_mask, self.energy_map, -1)
        map_features = {"tile_type": map_tile_type, "energy": map_energy}
        sensor_mask_int = sensor_mask.astype(np.int8)
        
        # Construct unit information, filtering allied and enemy units using the unit's sensor mask
        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)
        for i, u in enumerate(self.team_units):
            ux, uy = u["x"], u["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = np.array([ux, uy])
                units_energy[0, i] = u["energy"]
                units_mask[0, i] = 1
        for i, u in enumerate(self.enemy_units):
            ux, uy = u["x"], u["y"]
            if sensor_mask[uy, ux]:
                units_position[1, i] = np.array([ux, uy])
                units_energy[1, i] = u["energy"]
                units_mask[1, i] = 1
        units = {"position": units_position, "energy": units_energy}
        
        # Construct relic_nodes information: only show relic coordinates within the sensor_mask
        relic_coords = np.argwhere(self.relic_map == 1)
        relic_nodes = np.full((MAX_RELIC_NODES, 2), -1, dtype=np.int32)
        relic_nodes_mask = np.zeros(MAX_RELIC_NODES, dtype=np.int8)
        idx = 0
        for (ry, rx) in relic_coords:
            if idx >= MAX_RELIC_NODES:
                break
            if sensor_mask[ry, rx]:
                relic_nodes[idx] = np.array([rx, ry])
                relic_nodes_mask[idx] = 1
            else:
                relic_nodes[idx] = np.array([-1, -1])
                relic_nodes_mask[idx] = 0
            idx += 1
        
        team_points = np.array([self.score, 0], dtype=np.int32)
        team_wins = np.array([0, 0], dtype=np.int32)
        steps = self.current_step
        match_steps = self.current_step
        
        obs = {
            "units": units,
            "units_mask": units_mask,
            "sensor_mask": sensor_mask_int,
            "map_features": map_features,
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": steps,
            "match_steps": match_steps
        }
        observation = {
            "obs": obs,
            "remainingOverageTime": 60,
            "player": "player_0",
            "info": {"env_cfg": self.env_cfg}
        }
        return observation

    def get_obs(self):
        """
        Return the flattened global observation dictionary, ensuring that all keys exactly match the observation_space.
        """
        sensor_mask = self.get_global_sensor_mask()
        sensor_mask_int = sensor_mask.astype(np.int8)
        
        map_features_tile_type = np.where(sensor_mask, self.tile_map, -1)
        map_features_energy = np.where(sensor_mask, self.energy_map, -1)
        
        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)

        # Allied units
        for i, unit in enumerate(self.team_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = np.array([ux, uy])
                units_energy[0, i] = unit["energy"]
                units_mask[0, i] = 1
        # Enemy units
        for i, unit in enumerate(self.enemy_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[1, i] = np.array([ux, uy])
                units_energy[1, i] = unit["energy"]
                units_mask[1, i] = 1
                
        relic_coords = np.argwhere(self.relic_map == 1)
        relic_nodes = np.full((MAX_RELIC_NODES, 2), -1, dtype=np.int32)
        relic_nodes_mask = np.zeros((MAX_RELIC_NODES,), dtype=np.int8)
        idx = 0
        for (ry, rx) in relic_coords:
            if idx >= MAX_RELIC_NODES:
                break
            if sensor_mask[ry, rx]:
                relic_nodes[idx] = np.array([rx, ry])
                relic_nodes_mask[idx] = 1
            else:
                relic_nodes[idx] = np.array([-1, -1])
                relic_nodes_mask[idx] = 0
            idx += 1
    
        team_points = np.array([self.score, 0], dtype=np.int32)
        team_wins = np.array([0, 0], dtype=np.int32)
        steps = np.array([self.current_step], dtype=np.int32)
        match_steps = np.array([self.current_step], dtype=np.int32)
        remainingOverageTime = np.array([60], dtype=np.int32)
        
        env_cfg_map_width = np.array([self.env_cfg["map_width"]], dtype=np.int32)
        env_cfg_map_height = np.array([self.env_cfg["map_height"]], dtype=np.int32)
        env_cfg_max_steps_in_match = np.array([self.env_cfg["max_steps_in_match"]], dtype=np.int32)
        env_cfg_unit_move_cost = np.array([self.env_cfg["unit_move_cost"]], dtype=np.int32)
        env_cfg_unit_sap_cost = np.array([self.env_cfg["unit_sap_cost"]], dtype=np.int32)
        env_cfg_unit_sap_range = np.array([self.env_cfg["unit_sap_range"]], dtype=np.int32)
        
        flat_obs = {
            "units_position": units_position,
            "units_energy": units_energy,
            "units_mask": units_mask,
            "sensor_mask": sensor_mask_int,
            "map_features_tile_type": map_features_tile_type,
            "map_features_energy": map_features_energy,
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": steps,
            "match_steps": match_steps,
            "remainingOverageTime": remainingOverageTime,
            "env_cfg_map_width": env_cfg_map_width,
            "env_cfg_map_height": env_cfg_map_height,
            "env_cfg_max_steps_in_match": env_cfg_max_steps_in_match,
            "env_cfg_unit_move_cost": env_cfg_unit_move_cost,
            "env_cfg_unit_sap_cost": env_cfg_unit_sap_cost,
            "env_cfg_unit_sap_range": env_cfg_unit_sap_range
        }
        
        return flat_obs
    
    def reset(self):
        """
        Reset the environment state and return the initial flattened observation data.
        """
        self._init_state()
        return self.get_obs()

    def _spawn_unit(self, team):
        """Spawn a new unit: allied or enemy, with an initial energy of 100, spawning at their respective spawn points"""
        if team == 0:
            spawn_x, spawn_y = self.team_spawn
            self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})
        elif team == 1:
            spawn_x, spawn_y = self.enemy_spawn
            self.enemy_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})

    def step(self, actions):
        """
        Update the environment state based on the actions and return (observation, reward, done, info).
        Modified reward logic:
          1. Each unit computes its own unit_reward individually.
          2. If a movement action causes the unit to move off the map or the target tile is an Asteroid, 
             the action is deemed invalid and unit_reward is reduced by 0.2.
          3. Sap Action:
             - Check if a relic is visible in the unit's local observation (relic_nodes_mask);
             - If visible, count the number of enemy units in the unit's 8-neighborhood; if the count is >=2, 
               the sap reward equals +1.0 times the number of enemy units; otherwise, subtract 2.0;
             - If no relic is visible, subtract 2.0.
          4. Non-sap Actions:
             - After a successful move, check if the unit is located at a potential point within any relic configuration:
                  * If this is the first visit to the potential point, add +2.0 to unit_reward and mark it as visited;
                  * If the potential point has not yet yielded a team point, then increase self.score by 1, 
                    add +3.0 to unit_reward, and mark it as team_points_space;
                  * If already on team_points_space, then award +3.0 per turn;
             - If the unit is on an energy node (energy == Global.MAX_ENERGY_PER_TILE), add +0.2 to unit_reward;
             - If the unit is on a Nebula (tile_type == 1), subtract 0.2 from unit_reward;
             - If after moving the unit overlaps with an enemy unit and the enemy's energy is lower than the unit's, 
               then for each such enemy unit, add +1.0 reward.
          5. Global Exploration Reward: For every new tile discovered by the combined vision of all allied units, add +0.1 reward per tile.
          6. At the end of each step, the final reward is computed as: total_reward * 0.5 + score_increase * 0.5.
          7. Every 3 steps, spawn a new unit (if the maximum units limit has not been reached); 
             every 20 steps, shift the entire map, relic map, and energy map, as well as enemy unit positions 
             (shifting enemy units to the right by 1 tile with boundary checks).
        """
        prev_score = self.score
        
        self.current_step += 1
        total_reward = 0.0

        # Process each allied unit
        for idx, unit in enumerate(self.team_units):
            unit_reward = 0.0
            act = actions[idx]
            action_enum = ActionType(act)
            # print(f"Unit {idx} action: {action_enum}", file=sys.stderr)
            
            # Obtain the local observation for the unit
            unit_obs = self.get_unit_obs(unit)
            
            # If the action is sap
            if action_enum == ActionType.sap:
                # Check if a relic is visible in the local observation
                if np.any(unit_obs["obs"]["relic_nodes_mask"] == 1):
                    # Count the number of enemy units in the unit's 8-neighborhood
                    enemy_count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx = unit["x"] + dx
                            ny = unit["y"] + dy
                            if not (0 <= nx < SPACE_SIZE and 0 <= ny < SPACE_SIZE):
                                continue
                            for enemy in self.enemy_units:
                                if enemy["x"] == nx and enemy["y"] == ny:
                                    enemy_count += 1
                    if enemy_count >= 2:
                        unit_reward += 2.0 * enemy_count #1.0 (UPDATED)
                    else:
                        unit_reward -= 0.5 #1.0 (UPDATED)
                else:
                    unit_reward -= 0.5 #1.0 (UPDATED)
                # Sap action does not change the position
            else:
                # Compute movement direction
                if action_enum in [ActionType.up, ActionType.right, ActionType.down, ActionType.left]:
                    dx, dy = action_enum.to_direction()
                else:
                    dx, dy = (0, 0)
                new_x = unit["x"] + dx
                new_y = unit["y"] + dy
                # Check boundaries and obstacles
                if not (0 <= new_x < SPACE_SIZE and 0 <= new_y < SPACE_SIZE):
                    new_x, new_y = unit["x"], unit["y"]
                    unit_reward -= 0.4  # Out of bounds. Was 0.2 (UPDATED)
                elif self.tile_map[new_y, new_x] == 2:
                    new_x, new_y = unit["x"], unit["y"]
                    unit_reward -= 0.4  # Hit an Asteroid. Was 0.2 (UPDATED)
                else:
                    # Successful move
                    unit["x"], unit["y"] = new_x, new_y
                    #UPDATE: Extra penalty if unit is near the edge (to encourage staying more central)
                    edge_margin = 3  # for example, consider positions closer than 3 tiles to the edge as less desirable 
                    if new_x < edge_margin or new_x >= SPACE_SIZE - edge_margin or new_y < edge_margin or new_y >= SPACE_SIZE - edge_margin:
                        unit_reward -= 0.2  # penalty for being too close to the edge

                
                # Obtain updated local observation after moving
                unit_obs = self.get_unit_obs(unit)
                
                # Check for relic configuration rewards: iterate over all relic configurations
                # and determine if the unit is within the configuration (with boundary considerations)
                for (rx, ry, mask) in self.relic_configurations:
                    # Relic configuration area: center (rx, ry) ±2
                    # If the unit is within the range [rx-2, rx+2] and [ry-2, ry+2]
                    if rx - 2 <= unit["x"] <= rx + 2 and ry - 2 <= unit["y"] <= ry + 2:
                        # Calculate the index in the configuration mask
                        ix = unit["x"] - rx + 2
                        iy = unit["y"] - ry + 2
                        # Check if the index is within the mask range (with boundary considerations)
                        if 0 <= ix < 5 and 0 <= iy < 5:
                            # If the potential point has not been visited, then reward +2.0 and mark as visited
                            if not mask[iy, ix]:
                                if not self.potential_visited[unit["y"], unit["x"]]:
                                    unit_reward += 2.0 #Was 1.5 (UPDATED)
                                    self.potential_visited[unit["y"], unit["x"]] = True
                            # If the potential point has actually contributed to team points, then reward +3.0
                            else:
                                # If this point has not yet generated a team point, then increase team score and reward +3.0,
                                # and mark it as team_points_space
                                if not self.team_points_space[unit["y"], unit["x"]]:
                                    self.score += 1
                                    unit_reward += 3.0
                                    self.team_points_space[unit["y"], unit["x"]] = True
                                else:
                                    # If already on team_points_space, then reward +3.0 per turn
                                    self.score += 1
                                    unit_reward += 3.0
                # Energy node reward
                if unit_obs["obs"]["map_features"]["energy"][unit["y"], unit["x"]] == Global.MAX_ENERGY_PER_TILE:
                    unit_reward += 0.2
                # Nebula penalty
                if unit_obs["obs"]["map_features"]["tile_type"][unit["y"], unit["x"]] == 1:
                    unit_reward -= 0.2
                # Attack action: if the unit overlaps with an enemy unit and the enemy's energy is lower, 
                # then reward +1.0 for each eligible enemy unit
                for enemy in self.enemy_units:
                    if enemy["x"] == unit["x"] and enemy["y"] == unit["y"]:
                        if enemy["energy"] < unit["energy"]:
                            unit_reward += 1.0
            total_reward += unit_reward
            # print("################################", file=sys.stderr)
            # print("step:", self.current_step)
            # print("")
            # print(total_reward, file=sys.stderr)

        # Global exploration reward: based on the newly discovered tiles in the combined vision of allied units
        union_mask = self.get_global_sensor_mask()
        new_tiles = union_mask & (~self.visited)
        num_new = np.sum(new_tiles)
        if num_new > 0:
            total_reward += 0.2 * num_new
        self.visited[new_tiles] = True

        # Every 3 steps, spawn a new unit (if the maximum number of units is not reached)
        if self.current_step % 3 == 0:
            if len(self.team_units) < MAX_UNITS:
                self._spawn_unit(team=0)
            if len(self.enemy_units) < MAX_UNITS:
                self._spawn_unit(team=1)

        # Every 20 steps, shift the entire map, relic map, and energy map, as well as enemy unit positions 
        # (shift right by 1 tile, with boundary checks for enemy units)
        if self.current_step % 20 == 0:
            # Here, np.roll is used to keep the internal map data unchanged, 
            # but for enemy units we perform boundary checking
            self.tile_map = np.roll(self.tile_map, shift=1, axis=1)
            self.relic_map = np.roll(self.relic_map, shift=1, axis=1)
            self.energy_map = np.roll(self.energy_map, shift=1, axis=1)
            for enemy in self.enemy_units:
                new_ex = enemy["x"] + 1
                if new_ex >= SPACE_SIZE:
                    new_ex = enemy["x"]  # Remain unchanged
                enemy["x"] = new_ex

        # At the end of the step, calculate the increase in self.score
        score_increase = self.score - prev_score
    
        # Combine the total reward: total_reward * 0.5 + score_increase * 0.5
        final_reward = total_reward * 0.5 + score_increase * 0.5
        # final_reward = score_increase
        
        done = self.current_step >= self.max_steps
        # done = self.current_step >= 200
        info = {"score": self.score, "step": self.current_step}
        return self.get_obs(), final_reward, done, info

    def render(self, mode='human'):
        display = self.tile_map.astype(str).copy()
        for unit in self.team_units:
            display[unit["y"], unit["x"]] = 'A'
        print("Step:", self.current_step)
        print(display)
