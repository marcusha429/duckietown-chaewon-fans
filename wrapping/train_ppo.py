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
from utils.dt import MotionBlurWrapper, NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper
from gym_duckietown.wrappers import *

# Create the environment and apply wrappers
def make_env():
    simulator_kwargs = {
        "seed": 123,
        "map_name": "loop_empty",
        "max_steps": 10,
        "domain_rand": 0,
        "camera_width": 640,
        "camera_height": 480,
        "accept_start_angle_deg": 4,
        "full_transparency": True,
    }
    # Create Duckietown environment
    env = Simulator(**simulator_kwargs)
    print("Initialized environment")

    # Apply Duckietown wrappers
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Duckietown wrappers")

    # Apply Gymnasium wrapper
    env = DuckietownGymnasiumWrapper(env)
    print("Applied Gymnasium wrapper")

    return env

# Vectorize environment/s
env = DummyVecEnv([make_env])

# Print spaces
print(f"Observation space:\t: {env.observation_space}")
print(f"Action Space:\t\t {env.action_space}")

# Create PPO model
model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs={"normalize_images": False},
    verbose=1,
    tensorboard_log="./ppo_duckietown_tensorboard/"
)

# Print spaces
print(f"Observation space:\t: {env.observation_space}")
print(f"Action Space:\t\t {env.action_space}")


# # Train the agent
# TIMESTEPS = 1
# model.learn(
#     total_timesteps=TIMESTEPS,
# )

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