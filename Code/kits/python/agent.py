import numpy as np
import tensorflow as tf

class Agent:
    def __init__(self):
        # Load the trained behavior cloning model
        model_path = "model/ppo_game_env_model-immitation.keras"  # path to the saved model (adjust if different)
        self.model = tf.keras.models.load_model(model_path)
        # Determine the expected state vector length from the model input shape
        self.state_dim = self.model.input_shape[-1]
        # Determine the number of units (N) from the model output shape (since output is 3*N)
        self.action_dim = self.model.output_shape[-1]
        # Typically, action_dim is 3 * N. Compute N:
        if self.action_dim % 3 != 0:
            raise ValueError("Model output dimension is not divisible by 3. Check the model or environment config.")
        self.max_units = self.action_dim // 3

    def preprocess_observation(self, obs_dict):
        """
        Convert the raw observation dict into a flat numpy array of length self.state_dim.
        This should mirror the preprocessing used in training.
        """
        if "obs" in obs_dict:
            obs = obs_dict["obs"]
        else:
            obs = obs_dict
        # Flatten observation components (same as flatten_observation in training script)
        units_pos = np.array(obs["units"]["position"], dtype=np.float32).flatten()
        units_energy = np.array(obs["units"]["energy"], dtype=np.float32).flatten()
        units_mask = np.array(obs["units_mask"], dtype=np.float32).flatten()
        sensor_mask = np.array(obs["sensor_mask"], dtype=np.float32).flatten()
        map_energy = np.array(obs["map_features"]["energy"], dtype=np.float32).flatten()
        map_type = np.array(obs["map_features"]["tile_type"], dtype=np.float32).flatten()
        relic_mask = np.array(obs["relic_nodes_mask"], dtype=np.float32).flatten()
        relic_pos = np.array(obs["relic_nodes"], dtype=np.float32).flatten()
        team_points = np.array(obs["team_points"], dtype=np.float32).flatten()
        # Concatenate all parts
        state_vector = np.concatenate([
            units_pos, units_energy, units_mask,
            sensor_mask, map_energy, map_type,
            relic_mask, relic_pos, team_points
        ], axis=0)
        # If needed, pad or trim state_vector to match expected state_dim (in case of slight dimension mismatches)
        if state_vector.shape[0] < self.state_dim:
            # Pad with zeros if missing features (e.g., smaller map than max expected)
            pad_length = self.state_dim - state_vector.shape[0]
            state_vector = np.pad(state_vector, (0, pad_length))
        elif state_vector.shape[0] > self.state_dim:
            # Trim if there's an excess (e.g., larger map - this shouldn't happen if model was trained on same size)
            state_vector = state_vector[:self.state_dim]
        return state_vector

    def act(self, step, obs, remainingOverageTime=0):
        """
        Given the current observation, use the imitation model to predict an action.
        """
        # Preprocess observation into state vector
        state_vector = self.preprocess_observation(obs)
        # Ensure the state_vector is the correct shape for the model (1, state_dim)
        state_vector = state_vector.reshape(1, -1)
        # Use the model to predict the action vector
        pred_action_vector = self.model.predict(state_vector, verbose=0)[0]  # shape: (action_dim,)
        # Reshape the flat action vector into (N, 3) format
        N = self.max_units
        pred_action_matrix = pred_action_vector.reshape((N, 3))
        # Convert predicted actions to integers
        action_output = []
        for i in range(N):
            act_type = int(np.rint(pred_action_matrix[i, 0]))
            # Clamp the action type to valid range [0,5]
            act_type = max(0, min(5, act_type))
            dx = int(np.rint(pred_action_matrix[i, 1]))
            dy = int(np.rint(pred_action_matrix[i, 2]))
            # (Optionally clamp dx, dy if necessary based on game rules, e.g., sap range)
            action_output.append([act_type, dx, dy])
        # Return the action array (as list of lists)
        return action_output
