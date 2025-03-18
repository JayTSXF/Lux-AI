from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO
# from ppo_game_env import PPOGameEnv
# import os
# env = PPOGameEnv()
# dir = os.getcwd()
# model = RecurrentPPO("MlpLstmPolicy", env, verbose=1,policy_kwargs=dict(lstm_hidden_size=256),
#                       learning_rate=3e-4,
#                       n_steps=512, batch_size=128,
#                       gamma=0.99, gae_lambda=0.95,
#                       ent_coef=0.01, clip_range=0.2, tensorboard_log = dir)
#
# model.learn(total_timesteps=10000)
#
# model.save("model/ppo_game_env_model")


from stable_baselines3 import PPO
from ppo_game_env import PPOGameEnv

# 创建环境实例
env = PPOGameEnv()

# 使用多层感知机策略初始化 PPO 模型
# model = PPO("MultiInputPolicy", env,learning_rate=0.0005,ent_coef=0.1,vf_coef = 0.3, verbose=1)
model = PPO("MultiInputPolicy", env,learning_rate=0.0005, verbose=1)
# model_1 = PPO("MultiInputPolicy", env,learning_rate=0.0005, verbose=1)



# total_timesteps may need to adjust
# model.learn(total_timesteps=960000)
model.learn(total_timesteps=6000)

# 保存训练好的模型
model.save("model/ppo_game_env_model")

# 测试：加载模型并进行一次模拟
# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
