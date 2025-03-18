import sys
import gym
from gym import spaces
import numpy as np

# 导入 base 中的全局常量和辅助函数
from base import Global, ActionType, SPACE_SIZE, get_opposite

# 定义常量：队伍数、最大单位数、最大遗迹节点数
NUM_TEAMS = 2
MAX_UNITS = Global.MAX_UNITS
MAX_RELIC_NODES = Global.MAX_RELIC_NODES

class PPOGameEnv(gym.Env):
    """
    PPOGameEnv 模拟环境尽可能还原真实比赛环境，并满足以下要求：
    
    1. 观察数据计算修改：
       - 每个己方单位均有自己独立的 sensor mask（由 compute_unit_vision(unit) 计算），
         并由 get_unit_obs(unit) 构造出符合固定格式的局部观察（字典形式）。
       - 返回给代理的全局观察则采用所有己方单位 sensor mask 的联合（逻辑“或”），
         保持比赛返回 obs 的固定格式。
    
    2. 奖励函数优化：
       根据动作更新环境状态，并返回 (observation, reward, done, info)。
        修改后的奖励逻辑：
          1. 每个 unit 单独计算 unit_reward。
          2. 若移动动作导致超出地图或目标 tile 为 Asteroid，则判定为无效，unit_reward -0.2。
          3. Sap 动作：
             - 检查 unit 局部 obs 中 relic_nodes_mask 是否存在 relic；
             - 如果存在，统计 unit 8 邻域内敌方单位数，若数目>=2，则 sap 奖励 = +1.0×敌方单位数，否则扣 -2.0；
             - 若无 relic 可见，则同样扣 -2.0。
          4. 非 sap 动作：
             - 成功移动后，检查该 unit 是否位于任一 relic 配置内的潜力点：
                  * 若首次访问该潜力点，unit_reward +2.0，并标记 visited；
                  * 如果该潜力点尚未兑现 team point，则增加 self.score 1，同时 unit_reward +3.0 并标记为 team_points_space；
                  * 如果已在 team_points_space 上，则每回合奖励 +3.0；
             - 若 unit 位于能量节点（energy == Global.MAX_ENERGY_PER_TILE），unit_reward +0.2；
             - 若 unit 位于 Nebula（tile_type==1），unit_reward -0.2；
             - 如果 unit 移动后与敌方 unit 重合，且对方能量低于己方，则对每个满足条件的敌方 unit 奖励 +1.0。
          5. 全局探索奖励：所有己方单位联合视野中新发现 tile，每个奖励 +0.1。
          6. 每一step结束，奖励 point*0.5的奖励 + 规则*0.5的奖励
    
    3. 敌方单位策略说明：
       - 敌方单位在出生后不主动行动，其位置仅由环境每 20 步整体滚动（右移 1 格）改变，
         属于被动对手。这样设计主要用于初期调试，后续可引入更主动的对抗策略。
    """
    
    def __init__(self):
        super(PPOGameEnv, self).__init__()
        
        # 修改动作空间：每个单位独立决策（动作取值范围为 0~5）
        self.action_space = spaces.MultiDiscrete([len(ActionType)] * MAX_UNITS)
        
        # 观察空间保持不变
        self.observation_space = spaces.Dict({
            "units_position": spaces.Box(
                low=0,
                high=SPACE_SIZE - 1,
                shape=(NUM_TEAMS, MAX_UNITS, 2),
                dtype=np.int32
            ),
            "units_energy": spaces.Box(
                low=0,
                high=400,  # 单位能量上限 400
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

        # 全图状态：地图瓦片、遗迹标记、能量地图
        self.tile_map = None     # -1未知、0空地、1星云、2小行星
        self.relic_map = None    # relic 存在标记，1 表示存在
        self.energy_map = None   # 每个 tile 的能量值
        
        # 单位状态：己方和敌方单位列表，每个单位以字典表示 {"x": int, "y": int, "energy": int}
        self.team_units = []    # 己方
        self.enemy_units = []   # 敌方
        
        # 出生点：己方出生于左上角，敌方出生于右下角
        self.team_spawn = (0, 0)
        self.enemy_spawn = (SPACE_SIZE - 1, SPACE_SIZE - 1)
        
        # 探索记录：全图布尔数组，记录己方联合视野中已见过的 tile（全局只记录一次）
        self.visited = None
        
        # 团队得分（己方得分）
        self.score = 0
        
        # 模拟环境的部分参数（env_cfg）
        self.env_cfg = {
            "map_width": SPACE_SIZE,
            "map_height": SPACE_SIZE,
            "max_steps_in_match": Global.MAX_STEPS_IN_MATCH,
            "unit_move_cost": Global.UNIT_MOVE_COST,
            "unit_sap_cost": Global.UNIT_SAP_COST if hasattr(Global, "UNIT_SAP_COST") else 30,
            "unit_sap_range": Global.UNIT_SAP_RANGE,
        }
        
        # 新增：用于 relic 配置相关奖励
        self.relic_configurations = []   # list of (center_x, center_y, mask(5x5 bool))
        self.potential_visited = None      # 全图记录，shape (SPACE_SIZE, SPACE_SIZE)
        self.team_points_space = None      # 全图记录，哪些格子已经贡献过 team point

        self._init_state()

    def _init_state(self):
        """初始化全图状态、单位和记录"""
        num_tiles = SPACE_SIZE * SPACE_SIZE
        
        # 初始化 tile_map：随机部分设为 Nebula (1) 或 Asteroid (2)
        self.tile_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_nebula = int(num_tiles * 0.1)
        num_asteroid = int(num_tiles * 0.1)
        indices = np.random.choice(num_tiles, num_nebula + num_asteroid, replace=False)
        flat_tiles = self.tile_map.flatten()
        flat_tiles[indices[:num_nebula]] = 1
        flat_tiles[indices[num_nebula:]] = 2
        self.tile_map = flat_tiles.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # 初始化 relic_map：随机选取 3 个位置设置为 1（表示存在 relic）
        self.relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        relic_indices = np.random.choice(num_tiles, 3, replace=False)
        flat_relic = self.relic_map.flatten()
        flat_relic[relic_indices] = 1
        self.relic_map = flat_relic.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # 初始化 energy_map：随机生成 2 个能量节点，值设为 MAX_ENERGY_PER_TILE，其余为 0
        self.energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_energy_nodes = 2
        indices_energy = np.random.choice(num_tiles, num_energy_nodes, replace=False)
        flat_energy = self.energy_map.flatten()
        for idx in indices_energy:
            flat_energy[idx] = Global.MAX_ENERGY_PER_TILE
        self.energy_map = flat_energy.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # 初始化己方单位：初始生成 1 个单位，出生于 team_spawn
        self.team_units = []
        spawn_x, spawn_y = self.team_spawn
        self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})
        
        # 初始化敌方单位：初始生成 1 个单位，出生于 enemy_spawn
        self.enemy_units = []
        spawn_x_e, spawn_y_e = self.enemy_spawn
        self.enemy_units.append({"x": spawn_x_e, "y": spawn_y_e, "energy": 100})
        
        # 初始化探索记录：全图大小，取各己方单位联合视野后标记已见区域
        self.visited = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        union_mask = self.get_global_sensor_mask()
        self.visited = union_mask.copy()
        
        # 初始化 team score
        self.score = 0
        
        # 新增：初始化 relic 配置，及潜力点记录
        self.relic_configurations = []
        relic_coords = np.argwhere(self.relic_map == 1)
        for (y, x) in relic_coords:
            # 生成一个 5x5 mask，随机选择 5 个格子为 True（训练的时候选10个吧，避免奖励太过于稀疏）
            mask = np.zeros((5,5), dtype=bool)
            indices = np.random.choice(25, 8, replace=False)
            mask_flat = mask.flatten()
            mask_flat[indices] = True
            mask = mask_flat.reshape((5,5))
            self.relic_configurations.append((x, y, mask))
        self.potential_visited = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        self.team_points_space = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        
        self.current_step = 0

    def compute_unit_vision(self, unit):
        """
        根据传入 unit 的位置计算其独立的 sensor mask，
        计算范围为单位传感器范围（切比雪夫距离），并对 Nebula tile 减少贡献。
        取消环绕，只有在地图内的 tile 才计算。
        返回布尔矩阵 shape (SPACE_SIZE, SPACE_SIZE)。
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
        返回己方所有单位 sensor mask 的联合（逻辑 OR）。
        """
        mask = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        for unit in self.team_units:
            mask |= self.compute_unit_vision(unit)
        return mask

    def get_unit_obs(self, unit):
        """
        根据传入 unit 的独立 sensor mask 构造局部观察字典，
        格式与比赛返回固定 JSON 格式相同。
        仅使用该 unit 自己能看到的区域进行过滤。
        """
        sensor_mask = self.compute_unit_vision(unit)
        map_tile_type = np.where(sensor_mask, self.tile_map, -1)
        map_energy = np.where(sensor_mask, self.energy_map, -1)
        map_features = {"tile_type": map_tile_type, "energy": map_energy}
        sensor_mask_int = sensor_mask.astype(np.int8)
        
        # 构造单位信息，分别对己方与敌方单位过滤（使用该 unit 的 sensor mask）
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
        
        # 构造 relic_nodes 信息：仅显示在 sensor_mask 内的 relic 坐标
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
        返回平铺后的全局观测字典，确保所有键与 observation_space 完全一致。
        """
        sensor_mask = self.get_global_sensor_mask()
        sensor_mask_int = sensor_mask.astype(np.int8)
        
        map_features_tile_type = np.where(sensor_mask, self.tile_map, -1)
        map_features_energy = np.where(sensor_mask, self.energy_map, -1)
        
        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)

        # 己方单位
        for i, unit in enumerate(self.team_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = np.array([ux, uy])
                units_energy[0, i] = unit["energy"]
                units_mask[0, i] = 1
        # 敌方单位
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
        重置环境状态，并返回初始的平铺观测数据。
        """
        self._init_state()
        return self.get_obs()

    def _spawn_unit(self, team):
        """生成新单位：己方或敌方，初始能量 100，出生于各自出生点"""
        if team == 0:
            spawn_x, spawn_y = self.team_spawn
            self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})
        elif team == 1:
            spawn_x, spawn_y = self.enemy_spawn
            self.enemy_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})

    def step(self, actions):
       
# """ 
# Updates the environment state based on the given actions and returns (observation, reward, done, info). Modified reward logic:

# Each unit independently calculates unit_reward.
# If a movement action results in moving out of the map bounds or onto an Asteroid tile, it is considered invalid, and unit_reward -= 0.2.
# Sap action:
# Check if the unit's local observation (relic_nodes_mask) contains a relic.
# If a relic is present, count the number of enemy units in the unit’s 8-neighbor area. If the count is >= 2, the sap reward is +1.0 × enemy count, otherwise, a penalty of -2.0 is applied.
# If no relic is visible, a penalty of -2.0 is also applied.
# Non-sap actions:
# After a successful movement, check whether the unit is located at a potential point in any relic configuration:
# If it is visiting this potential point for the first time, unit_reward += 2.0, and mark it as visited.
# If the potential point has not yet contributed to the team score, increase self.score by 1, grant unit_reward += 3.0, and mark it as team_points_space.
# If it is already in team_points_space, grant +3.0 reward per turn.
# If the unit is on an energy node (energy == Global.MAX_ENERGY_PER_TILE), unit_reward += 0.2.
# If the unit is on a Nebula tile (tile_type == 1), unit_reward -= 0.2.
# If the unit moves onto a tile occupied by an enemy unit with lower energy, reward +1.0 for each such enemy unit.
# Global exploration reward: For every newly discovered tile in the combined vision of all friendly units, reward +0.1 per tile.
# End-of-step reward: The final reward per step is calculated as (point * 0.3) + (rule-based reward * 0.7).
# Additional mechanics:
# A new unit is spawned every 3 steps.
# Every 20 steps, the entire map and enemy unit positions shift (with boundary checks applied to enemy units). """
        
        prev_score = self.score
        
        self.current_step += 1
        total_reward = 0.0

        # 处理每个己方单位
        for idx, unit in enumerate(self.team_units):
            unit_reward = 0.0
            act = actions[idx]
            action_enum = ActionType(act)
            # print(f"Unit {idx} action: {action_enum}",file=sys.stderr)
            
            # 获取该 unit 的局部 obs
            unit_obs = self.get_unit_obs(unit)
            
            # 如果动作为 sap
            if action_enum == ActionType.sap:
                # 检查局部 obs 中是否有 relic 可见
                if np.any(unit_obs["obs"]["relic_nodes_mask"] == 1):
                    # 统计 unit 周围 8 邻域内敌方单位数
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
                        unit_reward += 1.0 * enemy_count
                    else:
                        unit_reward -= 1.0
                else:
                    unit_reward -= 1.0
                # Sap 动作不改变位置
            else:
                # 计算移动方向
                if action_enum in [ActionType.up, ActionType.right, ActionType.down, ActionType.left]:
                    dx, dy = action_enum.to_direction()
                else:
                    dx, dy = (0, 0)
                new_x = unit["x"] + dx
                new_y = unit["y"] + dy
                # 检查边界和障碍
                if not (0 <= new_x < SPACE_SIZE and 0 <= new_y < SPACE_SIZE):
                    new_x, new_y = unit["x"], unit["y"]
                    unit_reward -= 0.2  # 超出边界
                elif self.tile_map[new_y, new_x] == 2:
                    new_x, new_y = unit["x"], unit["y"]
                    unit_reward -= 0.2  # 遇到 Asteroid
                else:
                    # 移动成功
                    unit["x"], unit["y"] = new_x, new_y
                
                # 重新获取移动后的局部 obs
                unit_obs = self.get_unit_obs(unit)

                # Check relic configuration Bonus: Traverse all relic configurations and determine whether the unit is in the configuration (consider boundaries when calculating)
                for (rx, ry, mask) in self.relic_configurations:
                    # relic 配置区域范围：中心 (rx, ry) ±2
                    # 如果 unit 在 [rx-2, rx+2] 和 [ry-2, ry+2] 范围内
                    if rx - 2 <= unit["x"] <= rx + 2 and ry - 2 <= unit["y"] <= ry + 2:
                        # 计算在配置 mask 中的索引
                        ix = unit["x"] - rx + 2
                        iy = unit["y"] - ry + 2
                        # 检查索引是否在 mask 范围内（考虑边界）
                        if 0 <= ix < 5 and 0 <= iy < 5:
                            # 如果该潜力点未被访问，则奖励 +2.0
                            if not mask[iy, ix]:
                                if not self.potential_visited[unit["y"], unit["x"]]:
                                    unit_reward += 1.5
                                    self.potential_visited[unit["y"], unit["x"]] = True
                            # 如果潜力点是真的points_space，则奖励 +3.0
                            else:
                                # 如果该点尚未产生 team point，则增加 team point并奖励 +3.0
                                if not self.team_points_space[unit["y"], unit["x"]]:
                                    self.score += 1
                                    unit_reward += 3.0
                                    self.team_points_space[unit["y"], unit["x"]] = True
                                else:
                                    # 已在 team_points_space 上，每回合奖励 +3.0
                                    self.score += 1
                                    unit_reward += 3.0
                # 能量节点奖励
                if unit_obs["obs"]["map_features"]["energy"][unit["y"], unit["x"]] == Global.MAX_ENERGY_PER_TILE:
                    unit_reward += 0.2
                # Nebula 惩罚
                if unit_obs["obs"]["map_features"]["tile_type"][unit["y"], unit["x"]] == 1:
                    unit_reward -= 0.2
                # 攻击行为：若与敌方单位重合且对方能量低于己方，则对每个敌人奖励 +1.0
                for enemy in self.enemy_units:
                    if enemy["x"] == unit["x"] and enemy["y"] == unit["y"]:
                        if enemy["energy"] < unit["energy"]:
                            unit_reward += 1.0
            total_reward += unit_reward
            # print("################################",file=sys.stderr)
            # print("step:",self.current_step)
            # print("")
            # print(total_reward,file=sys.stderr)

        # 全局探索奖励：利用所有己方单位联合视野中新发现的 tile
        union_mask = self.get_global_sensor_mask()
        new_tiles = union_mask & (~self.visited)
        num_new = np.sum(new_tiles)
        if num_new > 0:
            total_reward += 0.2 * num_new
        self.visited[new_tiles] = True

        # 每 3 步生成新单位（若未达到 MAX_UNITS）
        if self.current_step % 3 == 0:
            if len(self.team_units) < MAX_UNITS:
                self._spawn_unit(team=0)
            if len(self.enemy_units) < MAX_UNITS:
                self._spawn_unit(team=1)

            # Every 20 steps, scroll the entire map, ruins and energy map, as well as enemy unit positions (move right 1 square, border check)        if self.current_step % 20 == 0:
            # 这里采用 np.roll 保持地图内部数据不变，但对于敌方单位，我们检查边界
            self.tile_map = np.roll(self.tile_map, shift=1, axis=1)
            self.relic_map = np.roll(self.relic_map, shift=1, axis=1)
            self.energy_map = np.roll(self.energy_map, shift=1, axis=1)
            for enemy in self.enemy_units:
                new_ex = enemy["x"] + 1
                if new_ex >= SPACE_SIZE:
                    new_ex = enemy["x"]  # 保持不变
                enemy["x"] = new_ex

        # 在 step 结束时计算 self.score 的增加量
        score_increase = self.score - prev_score
    
        # 将总奖励合并：total_reward * 0.5 + score_increase * 0.5
        final_reward = total_reward * 0.5 + score_increase * 0.5
        # final_reward =score_increase
        
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
