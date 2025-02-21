# Running headless 
# Run in terminal 
# Xvfb :1 -screen 0 1024x768x24 &
# export DISPLAY=:1
# python ducky_project.py
# To support headless
# import os
# os.environ["DISPLAY"] = ":1"

import gym
import gym_duckietown
from gym_duckietown.simulator import Simulator
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import numpy as np
from datetime import datetime
from tensorboard_video_recorder import TensorboardVideoRecorder
from stable_baselines3.common.vec_env import VecTransposeImage


# Custom Duckietown environment wrapper to reward staying on road
class DuckietownEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DuckietownEnvWrapper, self).__init__(env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        if info['Simulator']['collision']:
            reward -= 5  # Reduce penalty severity
        else:
            reward += 2 - 2 * abs(info['Simulator']['lane_position'])  # Increase lane centering rewardfrom center lane
        
        return obs, reward, done, info

# Create the environment
def make_env():
    env = gym.make("Duckietown-loop_empty-v0")
    return env

def make_env_old():
    env = Simulator(
        seed=123,
        map_name="loop_empty",
        max_steps=5000,
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4,
        full_transparency=True,
        distortion=True,
    )
    env = DuckietownEnvWrapper(env)
    env = Monitor(env)  # Monitor training progress
    return env

# Vectorized environment setup
env = make_vec_env(lambda: make_env(), n_envs=1)
env = VecTransposeImage(env)  # Ensures correct input shape

env.render(mode="rgb_array") # disable pop up to reduce rendering... 

# Experiement set up - logging ree
UCNETID = "choiji2"
experiment_name = "ppo_ducky_town_vector_normalized"
experiment_logdir = f"ppo_duckietown_log/{experiment_name}_{UCNETID}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
# Create the RL model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO("CnnPolicy", env, verbose=1, learning_rate=3e-4, n_steps=1024, gamma=0.99, gae_lambda=0.95, ent_coef=0.01)

# Train the model
print("Training...")
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_duckietown")
print("Model saved!")

# Load the model
model = PPO.load("ppo_duckietown")

def run_trained_model():
    env = make_env()
    obs, info = env.reset(return_info=True)
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()

if __name__ == "__main__":
    run_trained_model()
