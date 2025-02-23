import pyglet
window = pyglet.window.Window(visible=False)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the custom wrapper for Duckietown
from gym_duckietown.simulator import Simulator
from utils.custom import DuckietownGymnasiumWrapper
from stable_baselines3.common.vec_env import VecTransposeImage
from utils.env import make_env
from gym_duckietown.wrappers import *

# Prepare the environment
env = make_env() 
env = VecTransposeImage(DummyVecEnv([make_env]))

# Create PPO model
model = PPO(
    policy="CnnPolicy",
    env=env,
    policy_kwargs={"normalize_images": False},
    verbose=1,
    tensorboard_log="./ppo_duckietown_tensorboard/",
    device="mps"
)

# Train the agent
TIMESTEPS = 1
model.learn(
    total_timesteps=TIMESTEPS,
)

# # Save the trained model
# model.save("ppo_duckietown_model")

# # Evaluate the trained policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} ± {std_reward}")

# # Load the trained model
# model = PPO.load("ppo_duckietown_model")

# # Test the trained agent
# obs = env.reset()
# for _ in range(TIMESTEPS):
#     action, _states = model.predict(obs)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()

# env.close()