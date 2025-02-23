import pyglet
window = pyglet.window.Window(visible=False)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Import the custom wrapper for Duckietown
from gym_duckietown.simulator import Simulator
from utils.custom import DuckietownGymnasiumWrapper
from stable_baselines3.common.vec_env import VecTransposeImage
from utils.env import make_env
from gym_duckietown.wrappers import *

env = make_env() 

# # Test step BEFORE DummyVecEnv
# action = env.action_space.sample()
# action = np.array(action, dtype=np.float32)
# print("Action shape before step:", action.shape)
# obs, reward, terminated, truncated, info = env.step(action)

# # Apply DummyVecEnv (after verifying base env works)
# env = VecTransposeImage(DummyVecEnv([make_env]))

# # Sample action
# action = np.array(env.action_space.sample(), dtype=np.float32).reshape(1, -1)

# # Print action shape to verify
# print("Action shape before step:", action.shape)  # Should print (1,2)

# # Fix unpacking issue
# step_result = env.step(action)

# if len(step_result) == 4:
#     obs, reward, done, info = step_result
#     truncated = False  # Add missing value
# else:
#     obs, reward, done, truncated, info = step_result

# print("Final step output:", obs.shape, reward, done, truncated, info)

# Apply DummyVecEnv (after verifying base env works)
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