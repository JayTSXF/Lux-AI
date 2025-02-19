import sys
import numpy as np
import os
from stable_baselines3 import PPO
from lux.utils import direction_to  
from base import Global, warp_point 

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.env_cfg = env_cfg
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        self.model_path = os.path.join(os.path.dirname(__file__), "model/ppo_game_env_model.zip")

        # Load PPO model
        if os.path.exists(self.model_path):
            self.ppo_model = PPO.load(self.model_path)
        else:
            self.ppo_model = None

    def compute_team_vision(self, tile_map, agent_positions):
        sensor_range = Global.UNIT_SENSOR_RANGE
        nebula_reduction = 2  
        vision = np.zeros(tile_map.shape, dtype=np.float32)
        for (x, y) in agent_positions:
            for dy in range(-sensor_range, sensor_range + 1):
                for dx in range(-sensor_range, sensor_range + 1):
                    new_x, new_y = warp_point(x + dx, y + dy)
                    contrib = sensor_range + 1 - max(abs(dx), abs(dy))
                    if tile_map[new_y, new_x] == 1:
                        contrib -= nebula_reduction
                    vision[new_y, new_x] += contrib
        visible_mask = vision > 0
        return visible_mask

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])          
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        opp_unit_positions = np.array(obs["units"]["position"][self.opp_team_id]) 
        unit_energys = np.array(obs["units"]["energy"][self.team_id])     
        observed_relic_node_positions = np.array(obs["relic_nodes"])       
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])      
        
        map_height = self.env_cfg["map_height"]
        map_width = self.env_cfg["map_width"]
        sensor_mask = np.array(obs["sensor_mask"]) 
        tile_type_obs = np.array(obs["map_features"]["tile_type"])  
        tile_map = np.where(sensor_mask, tile_type_obs, -1)
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=np.int32)
        available_unit_ids = np.where(unit_mask)[0]
        

        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]    
            unit_energy = unit_energys[unit_id]    
            
            obs_grid = np.zeros((map_height, map_width, 3), dtype=np.float32)
            
            obs_grid[..., 0] = tile_map
            
            relic_map = np.zeros((map_height, map_width), dtype=np.int8)
            for i in range(len(observed_relic_node_positions)):
                if observed_relic_nodes_mask[i]:
                    x, y = observed_relic_node_positions[i]
                    x, y = int(x), int(y)
                    if 0 <= x < map_width and 0 <= y < map_height:
                        if sensor_mask[y, x]:
                            relic_map[y, x] = 1
            obs_grid[..., 1] = relic_map
            
            agent_layer = np.zeros((map_height, map_width), dtype=np.int8)
            unit_x = int(unit_pos[0])
            unit_y = int(unit_pos[1])
            if 0 <= unit_x < map_width and 0 <= unit_y < map_height and sensor_mask[unit_y, unit_x]:
                agent_layer[unit_y, unit_x] = 1
            obs_grid[..., 2] = agent_layer
            
            state = obs_grid[np.newaxis, ...]
            action, _ = self.ppo_model.predict(state, deterministic=True)
            action = int(action)
            actions[unit_id] = [action, 0, 0]
        
        return actions