from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from ppo_game_env import PPOGameEnv

# Create an instance of the environment

def make_env():
    return PPOGameEnv()


if __name__ == '__main__':
# Initialize the PPO model with a multilayer perceptron policy
# model = PPO("MultiInputPolicy", env, learning_rate=0.0005, ent_coef=0.1, vf_coef=0.3, verbose=1)

    env = SubprocVecEnv([lambda: make_env() for _ in range (8)])
    env = VecMonitor(env)
    model = PPO("MultiInputPolicy", env, 
                n_steps=2048,
                batch_size = 128,
                learning_rate=5e-4, 
                verbose=1)
    # model_1 = PPO("MultiInputPolicy", env, learning_rate=0.0005, verbose=1)

    # total_timesteps may need to be adjusted
    # model.learn(total_timesteps=960000)
    model.learn(total_timesteps=640000)

    # Save the trained model
    model.save("/model/ppo_game_env_model")

# Test: Load the model and run a simulation
# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()