import os
import json
import numpy as np
import tensorflow as tf

# 1) Helper: flatten an observation dict into a 1D state vector
def flatten_observation(raw_obs):
    # 1) If the raw observation is a JSON string, parse it
    if isinstance(raw_obs, str):
        obs = json.loads(raw_obs)
    else:
        obs = raw_obs

    # 2) Unwrap nested "obs" key if present (and parse if it's also a string)
    if isinstance(obs, dict) and "obs" in obs:
        inner = obs["obs"]
        obs = json.loads(inner) if isinstance(inner, str) else inner

    # Now extract and flatten all the fields
    units_pos    = np.array(obs["units"]["position"],     np.float32).flatten()
    units_energy = np.array(obs["units"]["energy"],       np.float32).flatten()
    units_mask   = np.array(obs["units_mask"],            np.float32).flatten()
    sensor_mask  = np.array(obs["sensor_mask"],           np.float32).flatten()
    map_energy   = np.array(obs["map_features"]["energy"],np.float32).flatten()
    map_type     = np.array(obs["map_features"]["tile_type"],np.float32).flatten()
    relic_mask   = np.array(obs["relic_nodes_mask"],      np.float32).flatten()
    relic_pos    = np.array(obs["relic_nodes"],           np.float32).flatten()
    team_points  = np.array(obs["team_points"],           np.float32).flatten()

    # Concatenate into one 1D state vector
    return np.concatenate([
        units_pos, units_energy, units_mask,
        sensor_mask, map_energy, map_type,
        relic_mask, relic_pos, team_points
    ], axis=0)


# 2) Helper: flatten an (N×3) action into a 1D vector
def flatten_action(action_arr):
    return np.array(action_arr, dtype=np.float32).flatten()

# 3) Load all replay.json files and build X (states) and y (actions)
replay_dir = os.getcwd() + "\\replay"  # ← your directory of replay.json files

state_list = []
action_list = []

for fname in os.listdir(replay_dir):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(replay_dir, fname)
    data = json.load(open(path))

    # support both keys "steps" and "observations"
    steps = data.get("steps") or data.get("observations") or []
    for step in steps[1:]:  # skip the initial setup step
        # if each step is a list [player0, player1], take index 0
        player0 = step[0] if isinstance(step, list) else step

        # pull raw observation from either "obs" or "observation"
        raw_obs = player0.get("obs", player0.get("observation"))
        action  = player0.get("action")
        if raw_obs is None or action is None:
            continue

        # flatten and collect
        sv = flatten_observation(raw_obs)
        av = flatten_action(action)
        state_list.append(sv)
        action_list.append(av)

# guard against empty dataset
if not state_list:
    raise RuntimeError(
        f"No samples collected from '{replay_dir}'.\n"
        "Check that the directory path is correct and your JSON uses keys "
        "'steps', 'obs' or 'observation', and 'action'."
    )

# stack into arrays
X = np.vstack(state_list)
y = np.vstack(action_list)

print("Collected", X.shape[0], "samples → state dim", X.shape[1], "action dim", y.shape[1])

# 4) Define & compile the BC model
input_dim  = X.shape[1]
output_dim = y.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(output_dim)  # linear outputs
])
model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(1e-3)
)
model.summary()

# 5) Train & save
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.1, shuffle=True)
model.save("model/ppo_game_env_model-immitation.keras")  # creates a folder bc_model/ with the SavedModel
print(" Behavior cloning model saved to bc_model/")
