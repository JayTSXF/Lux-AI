agent.py:

import os
import sys
import numpy as np
from stable_baselines3 import PPO

def transform_obs(comp_obs, env_cfg=None):
    """
    Transform the JSON observation returned by the competition engine into a flattened observation format
    for model training.
    
    The observation format from the competition environment (comp_obs) is structured as follows:
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
    
    We need to construct the following flattened dictionary (which matches the format returned by PPOGameEnv.get_obs()):
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
    # If the "obs" key exists, use its content; otherwise, use comp_obs directly
    if "obs" in comp_obs:
        base_obs = comp_obs["obs"]
    else:
        base_obs = comp_obs

    flat_obs = {}

    # Process the "units" data
    if "units" in base_obs:
        flat_obs["units_position"] = np.array(base_obs["units"]["position"], dtype=np.int32)
        flat_obs["units_energy"] = np.array(base_obs["units"]["energy"], dtype=np.int32)
        # If units_energy has shape (NUM_TEAMS, MAX_UNITS), expand one dimension
        if flat_obs["units_energy"].ndim == 2:
            flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)
    else:
        flat_obs["units_position"] = np.array(base_obs["units_position"], dtype=np.int32)
        flat_obs["units_energy"] = np.array(base_obs["units_energy"], dtype=np.int32)
        if flat_obs["units_energy"].ndim == 2:
            flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)
    
    # Process units_mask
    if "units_mask" in base_obs:
        flat_obs["units_mask"] = np.array(base_obs["units_mask"], dtype=np.int8)
    else:
        flat_obs["units_mask"] = np.zeros(flat_obs["units_position"].shape[:2], dtype=np.int8)
    
    # Process sensor_mask: if the returned sensor_mask is a 3D array, take the logical OR to obtain a global mask
    sensor_mask_arr = np.array(base_obs["sensor_mask"], dtype=np.int8)
    if sensor_mask_arr.ndim == 3:
        sensor_mask = np.any(sensor_mask_arr, axis=0).astype(np.int8)
    else:
        sensor_mask = sensor_mask_arr
    flat_obs["sensor_mask"] = sensor_mask

    # Process map_features (tile_type and energy)
    if "map_features" in base_obs:
        mf = base_obs["map_features"]
        flat_obs["map_features_tile_type"] = np.array(mf["tile_type"], dtype=np.int8)
        flat_obs["map_features_energy"] = np.array(mf["energy"], dtype=np.int8)
    else:
        flat_obs["map_features_tile_type"] = np.array(base_obs["map_features_tile_type"], dtype=np.int8)
        flat_obs["map_features_energy"] = np.array(base_obs["map_features_energy"], dtype=np.int8)

    # Process relic node information
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

    # Process team points and wins
    if "team_points" in base_obs:
        flat_obs["team_points"] = np.array(base_obs["team_points"], dtype=np.int32)
    else:
        flat_obs["team_points"] = np.zeros(2, dtype=np.int32)
    if "team_wins" in base_obs:
        flat_obs["team_wins"] = np.array(base_obs["team_wins"], dtype=np.int32)
    else:
        flat_obs["team_wins"] = np.zeros(2, dtype=np.int32)

    # Process step information
    if "steps" in base_obs:
        flat_obs["steps"] = np.array([base_obs["steps"]], dtype=np.int32)
    else:
        flat_obs["steps"] = np.array([0], dtype=np.int32)
    if "match_steps" in base_obs:
        flat_obs["match_steps"] = np.array([base_obs["match_steps"]], dtype=np.int32)
    else:
        flat_obs["match_steps"] = np.array([0], dtype=np.int32)

    # Note: Do not process remainingOverageTime here; it will be added in Agent.act using the provided parameter

    # Complete the environment configuration information
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

        # If "max_units" is not in env_cfg, add a default value of 16
        if "max_units" not in self.env_cfg:
            self.env_cfg["max_units"] = 16

        # Load the trained PPO model (please ensure the model file path is correct)
        model_path = os.path.join(os.path.dirname(__file__), "/model/ppo_game_env_model.zip")
        self.model = PPO.load(model_path)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Determine the actions for each unit based on the competition observation and the current step.
        The output is a numpy array of shape (max_units, 3), where each row is formatted as [action type, delta_x, delta_y].
        For non-sap actions, delta_x and delta_y are fixed at 0.
        """
        import sys
        # # If step equals 11, print debug information:
        # if step == 11:
        #     print("DEBUG: Agent.act() called parameters:", file=sys.stderr)
        #     print("DEBUG: self.player =", self.player, file=sys.stderr)
        #     print("DEBUG: step =", step, file=sys.stderr)
        #     # Print the list of keys in obs to see the general structure of the observation data
        #     print("DEBUG: obs keys =", list(obs.keys()), file=sys.stderr)
        #     print("=============================================================", file=sys.stderr)
        #     print("DEBUG: ob =", obs, file=sys.stderr)
        #     print("DEBUG: remainingOverageTime =", remainingOverageTime, file=sys.stderr)
        #     print("#############################################################", file=sys.stderr)
        
        flat_obs = transform_obs(obs, self.env_cfg)
        # If the current agent is player_1, swap unit information (to ensure consistency with training, allied perspective is always in the first position)
        if self.player == "player_1":
            flat_obs["units_position"] = flat_obs["units_position"][::-1]
            flat_obs["units_energy"] = flat_obs["units_energy"][::-1]
            flat_obs["units_mask"] = flat_obs["units_mask"][::-1]
            
        # Manually add remainingOverageTime (taken from the passed parameter)
        flat_obs["remainingOverageTime"] = np.array([remainingOverageTime], dtype=np.int32)

        # if step equals 11:
        #     print("------------------------------------------------------------", file=sys.stderr)
        #     print("DEBUG: flat_obs =", flat_obs, file=sys.stderr)
        #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", file=sys.stderr)
        # Use the model to predict actions (deterministic mode)
        action, _ = self.model.predict(flat_obs, deterministic=True)
        # Ensure action is a numpy array and explicitly set its type to np.int32
        action = np.array(action, dtype=np.int32)

        max_units = self.env_cfg["max_units"]
        actions = np.zeros((max_units, 3), dtype=np.int32)
        for i, a in enumerate(action):
            actions[i, 0] = int(a)
            actions[i, 1] = 0  # For sap actions, target offset can be extended here
            actions[i, 2] = 0
        return actions
