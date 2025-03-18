import os
import sys
import numpy as np
from stable_baselines3 import PPO


def transform_obs(comp_obs, env_cfg=None):
    """
    将比赛引擎返回的 JSON 观测转换为模型训练时使用的平铺观测格式。

    比赛环境的观测格式（comp_obs）结构如下：
      {
        "obs": {
            "units": {"position": Array(T, N, 2), "energy": Array(T, N, 1)},
            "units_mask": Array(T, N),
            "sensor_mask": Array(W, H),
            "map_features": {"energy": Array(W, H), "tile_type": Array(W, H)},
            "relic_nodes_mask": Array(R),
            "relic_nodes": Array(R, 2),
            "team_points": Array(T),
            "team_wins": Array(T),
            "steps": int,
            "match_steps": int
        },
        "remainingOverageTime": int,
        "player": str,
        "info": {"env_cfg": dict}
      }

    我们需要构造如下平铺字典（与 PPOGameEnv.get_obs() 返回的格式一致）：
      {
        "units_position": (T, N, 2),
        "units_energy": (T, N, 1),
        "units_mask": (T, N),
        "sensor_mask": (W, H),
        "map_features_tile_type": (W, H),
        "map_features_energy": (W, H),
        "relic_nodes_mask": (R,),
        "relic_nodes": (R, 2),
        "team_points": (T,),
        "team_wins": (T,),
        "steps": (1,),
        "match_steps": (1,),
        "remainingOverageTime": (1,),
        "env_cfg_map_width": (1,),
        "env_cfg_map_height": (1,),
        "env_cfg_max_steps_in_match": (1,),
        "env_cfg_unit_move_cost": (1,),
        "env_cfg_unit_sap_cost": (1,),
        "env_cfg_unit_sap_range": (1,)
      }
    """
    # 如果存在 "obs" 键，则取其内部数据，否则直接使用 comp_obs
    if "obs" in comp_obs:
        base_obs = comp_obs["obs"]
    else:
        base_obs = comp_obs

    flat_obs = {}

    # 处理 units 数据
    if "units" in base_obs:
        flat_obs["units_position"] = np.array(base_obs["units"]["position"], dtype=np.int32)
        flat_obs["units_energy"] = np.array(base_obs["units"]["energy"], dtype=np.int32)
        # 如果 units_energy 的 shape 为 (NUM_TEAMS, MAX_UNITS) 则扩展一个维度
        if flat_obs["units_energy"].ndim == 2:
            flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)
    else:
        flat_obs["units_position"] = np.array(base_obs["units_position"], dtype=np.int32)
        flat_obs["units_energy"] = np.array(base_obs["units_energy"], dtype=np.int32)
        if flat_obs["units_energy"].ndim == 2:
            flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)

    # 处理 units_mask
    if "units_mask" in base_obs:
        flat_obs["units_mask"] = np.array(base_obs["units_mask"], dtype=np.int8)
    else:
        flat_obs["units_mask"] = np.zeros(flat_obs["units_position"].shape[:2], dtype=np.int8)

    # 处理 sensor_mask：若返回的是 3D 数组，则取逻辑 or 得到全局 mask
    sensor_mask_arr = np.array(base_obs["sensor_mask"], dtype=np.int8)
    if sensor_mask_arr.ndim == 3:
        sensor_mask = np.any(sensor_mask_arr, axis=0).astype(np.int8)
    else:
        sensor_mask = sensor_mask_arr
    flat_obs["sensor_mask"] = sensor_mask

    # 处理 map_features（tile_type 与 energy）
    if "map_features" in base_obs:
        mf = base_obs["map_features"]
        flat_obs["map_features_tile_type"] = np.array(mf["tile_type"], dtype=np.int8)
        flat_obs["map_features_energy"] = np.array(mf["energy"], dtype=np.int8)
    else:
        flat_obs["map_features_tile_type"] = np.array(base_obs["map_features_tile_type"], dtype=np.int8)
        flat_obs["map_features_energy"] = np.array(base_obs["map_features_energy"], dtype=np.int8)

    # 处理 relic 节点信息
    if "relic_nodes_mask" in base_obs:
        flat_obs["relic_nodes_mask"] = np.array(base_obs["relic_nodes_mask"], dtype=np.int8)
    else:
        max_relic = env_cfg.get("max_relic_nodes", 6) if env_cfg is not None else 6
        flat_obs["relic_nodes_mask"] = np.zeros((max_relic,), dtype=np.int8)
    if "relic_nodes" in base_obs:
        flat_obs["relic_nodes"] = np.array(base_obs["relic_nodes"], dtype=np.int32)
    else:
        max_relic = env_cfg.get("max_relic_nodes", 6) if env_cfg is not None else 6
        flat_obs["relic_nodes"] = np.full((max_relic, 2), -1, dtype=np.int32)

    # 处理团队得分与胜局
    if "team_points" in base_obs:
        flat_obs["team_points"] = np.array(base_obs["team_points"], dtype=np.int32)
    else:
        flat_obs["team_points"] = np.zeros(2, dtype=np.int32)
    if "team_wins" in base_obs:
        flat_obs["team_wins"] = np.array(base_obs["team_wins"], dtype=np.int32)
    else:
        flat_obs["team_wins"] = np.zeros(2, dtype=np.int32)

    # 处理步数信息
    if "steps" in base_obs:
        flat_obs["steps"] = np.array([base_obs["steps"]], dtype=np.int32)
    else:
        flat_obs["steps"] = np.array([0], dtype=np.int32)
    if "match_steps" in base_obs:
        flat_obs["match_steps"] = np.array([base_obs["match_steps"]], dtype=np.int32)
    else:
        flat_obs["match_steps"] = np.array([0], dtype=np.int32)

    # 注意：不在此处处理 remainingOverageTime，
    # 将在 Agent.act 中利用传入的参数添加

    # 补全环境配置信息
    if env_cfg is not None:
        flat_obs["env_cfg_map_width"] = np.array([env_cfg["map_width"]], dtype=np.int32)
        flat_obs["env_cfg_map_height"] = np.array([env_cfg["map_height"]], dtype=np.int32)
        flat_obs["env_cfg_max_steps_in_match"] = np.array([env_cfg["max_steps_in_match"]], dtype=np.int32)
        flat_obs["env_cfg_unit_move_cost"] = np.array([env_cfg["unit_move_cost"]], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_cost"] = np.array([env_cfg["unit_sap_cost"]], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_range"] = np.array([env_cfg["unit_sap_range"]], dtype=np.int32)
    else:
        flat_obs["env_cfg_map_width"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_map_height"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_max_steps_in_match"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_move_cost"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_cost"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_range"] = np.array([0], dtype=np.int32)

    return flat_obs


class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        # 如果 env_cfg 中没有 "max_units"，则补上默认值 16
        if "max_units" not in self.env_cfg:
            self.env_cfg["max_units"] = 16

        # 加载训练好的 PPO 模型（请确保模型文件路径正确）
        dirpath = os.getcwd()
        model_path = "model/ppo_game_env_model.zip"#os.path.join(dirpath, "model/ppo_game_env_model.zip")
        self.model = PPO.load(model_path)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        根据比赛观测与当前步数决定各单位动作。
        输出为形状 (max_units, 3) 的 numpy 数组，每行格式为 [动作类型, delta_x, delta_y]，
        其中非汲取动作时 delta_x 和 delta_y 固定为 0。
        """
        import sys
        # # 当 step 为 11 时打印调试信息
        # if step == 11:
        #     print("DEBUG: Agent.act() 调用参数：", file=sys.stderr)
        #     print("DEBUG: self.player =", self.player, file=sys.stderr)
        #     print("DEBUG: step =", step, file=sys.stderr)
        #     # 打印 obs 的 key 列表，可以查看观测数据的大致结构
        #     print("DEBUG: obs keys =", list(obs.keys()), file=sys.stderr)
        #     print("=============================================================", file=sys.stderr)
        #     print("DEBUG: ob =", obs, file=sys.stderr)
        #     print("DEBUG: remainingOverageTime =", remainingOverageTime, file=sys.stderr)
        #     print("#############################################################", file=sys.stderr)

        flat_obs = transform_obs(obs, self.env_cfg)
        # 如果当前 agent 为 player_1，则交换单位信息（确保和训练时候一致，己方视角永远在第一位置）
        if self.player == "player_1":
            flat_obs["units_position"] = flat_obs["units_position"][::-1]
            flat_obs["units_energy"] = flat_obs["units_energy"][::-1]
            flat_obs["units_mask"] = flat_obs["units_mask"][::-1]

        # 手动添加 remainingOverageTime（取自传入参数）
        flat_obs["remainingOverageTime"] = np.array([remainingOverageTime], dtype=np.int32)

        # if step == 11:
        #     print("------------------------------------------------------------", file=sys.stderr)
        #     print("DEBUG: flat_obs =", flat_obs, file=sys.stderr)
        #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", file=sys.stderr)
        # 使用模型预测动作（deterministic 模式）
        action, _ = self.model.predict(flat_obs, deterministic=True)
        # 确保 action 为 numpy 数组，并显式设置为 np.int32 类型
        action = np.array(action, dtype=np.int32)

        max_units = self.env_cfg["max_units"]
        actions = np.zeros((max_units, 3), dtype=np.int32)
        for i, a in enumerate(action):
            actions[i, 0] = int(a)
            actions[i, 1] = 0  # 若为 sap 动作，可在此扩展目标偏移
            actions[i, 2] = 0
        return actions