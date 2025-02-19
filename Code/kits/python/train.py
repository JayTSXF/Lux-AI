from stable_baselines3 import PPO
from ppo_game_env import PPOGameEnv

env = PPOGameEnv()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

model.save("model/ppo_game_env_model")

# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()