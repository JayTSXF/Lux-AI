import gym
from gym import spaces
import numpy as np

from base import Global, ActionType, SPACE_SIZE, warp_point, get_opposite

class PPOGameEnv(gym.Env):
    
    def __init__(self):
        super(PPOGameEnv, self).__init__()
        
        self.obs_shape = (SPACE_SIZE, SPACE_SIZE, 3)
        self.observation_space = spaces.Box(low=-1, high=2, shape=self.obs_shape, dtype=np.int8)
        
        self.action_space = spaces.Discrete(len(ActionType))
        
        self.max_steps = Global.MAX_STEPS_IN_MATCH
        self.current_step = 0
        
        self.tile_map = None     
        self.relic_map = None    
        self.agent_position = None  
        
        self.agent_x = None
        self.agent_y = None
        
        # 用于统计得分
        self.score = 0
        
        self._init_state()

    def _init_state(self):
    
        self.tile_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_tiles = SPACE_SIZE * SPACE_SIZE
        num_nebula = int(num_tiles * 0.1)     
        num_asteroid = int(num_tiles * 0.05)   
        indices = np.random.choice(num_tiles, num_nebula + num_asteroid, replace=False)
        flat_tiles = self.tile_map.flatten()
        flat_tiles[indices[:num_nebula]] = 1
        flat_tiles[indices[num_nebula:]] = 2
        self.tile_map = flat_tiles.reshape((SPACE_SIZE, SPACE_SIZE))
        
        self.relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        relic_indices = np.random.choice(num_tiles, 3, replace=False)
        flat_relic = self.relic_map.flatten()
        flat_relic[relic_indices] = 1
        self.relic_map = flat_relic.reshape((SPACE_SIZE, SPACE_SIZE))
        
        self.agent_position = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        self.agent_x, self.agent_y = SPACE_SIZE // 2, SPACE_SIZE // 2
        self.agent_position[self.agent_y, self.agent_x] = 1
        
        self.current_step = 0
        self.score = 0

    def compute_team_vision(self):
        sensor_range = Global.UNIT_SENSOR_RANGE
        nebula_reduction = 2  
        vision = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        
        unit_positions = np.argwhere(self.agent_position == 1)  
        for (y, x) in unit_positions:
            for dy in range(-sensor_range, sensor_range + 1):
                for dx in range(-sensor_range, sensor_range + 1):
                    new_x, new_y = warp_point(x + dx, y + dy)
                    contrib = sensor_range + 1 - max(abs(dx), abs(dy))
                    if self.tile_map[new_y, new_x] == 1:
                        contrib -= nebula_reduction
                    vision[new_y, new_x] += contrib
        visible_mask = vision > 0
        return visible_mask

    def _get_obs(self):
        visible_mask = self.compute_team_vision()
        obs_tile = np.where(visible_mask, self.tile_map, -1)
        obs_relic = np.where(visible_mask, self.relic_map, 0)
        obs_agent = np.where(visible_mask, self.agent_position, 0)
        
        obs = np.stack([obs_tile, obs_relic, obs_agent], axis=-1)
        return obs

    def reset(self):
        self._init_state()
        return self._get_obs()
    
    def step(self, action):
    
        self.current_step += 1
        reward = 0.0
        
        action_enum = ActionType(action)
        
        dx, dy = 0, 0
        if action_enum in [ActionType.up, ActionType.right, ActionType.down, ActionType.left]:
            dx, dy = action_enum.to_direction()
        elif action_enum == ActionType.center:
            dx, dy = 0, 0
        elif action_enum == ActionType.sap:
            reward -= 0.1  
        
        if action_enum != ActionType.sap:
            new_x, new_y = warp_point(self.agent_x + dx, self.agent_y + dy)
            if self.tile_map[new_y, new_x] == 2:
                reward -= 1.0 
            else:
                self.agent_position[:, :] = 0
                self.agent_x, self.agent_y = new_x, new_y
                self.agent_position[self.agent_y, self.agent_x] = 1
        
        if self.relic_map[self.agent_y, self.agent_x] == 1:
            reward += 1.0
            self.score += 1
            self.relic_map[self.agent_y, self.agent_x] = 0
            opp_x, opp_y = get_opposite(self.agent_x, self.agent_y)
            self.relic_map[opp_y, opp_x] = 0
        
        if (self.current_step % 20) == 0:
            self.tile_map = np.roll(self.tile_map, shift=1, axis=1)
            self.relic_map = np.roll(self.relic_map, shift=1, axis=1)
        
        done = self.current_step >= self.max_steps
        info = {"score": self.score, "step": self.current_step}
        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        display = self.tile_map.astype(str).copy()
        display[self.agent_y, self.agent_x] = 'A'
        print("Step:", self.current_step)
        print(display)