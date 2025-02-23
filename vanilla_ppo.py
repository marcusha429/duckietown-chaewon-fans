#!/usr/bin/env python3
import pyglet
# Initialize the pyglet window (non-visible, headless mode)
window = pyglet.window.Window(visible=False)
pyglet.options['shadow_window'] = False


import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from gym_duckietown.simulator import Simulator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wrapper import DuckietownWrapper


# Initialize the native Duckietown simulator directly (not via gym)
simulator = Simulator(
    domain_rand=False,  # Set to False for a deterministic simulation
    seed=123,  # Seed for reproducibility
    map_name="loop_empty",  # Define the map to use
    max_steps=500001,  # Define max steps
    camera_width=640,  # Camera width
    camera_height=480,  # Camera height
)

# Create a custom wrapped environment
simulator = DuckietownWrapper(simulator)

# Wrap the simulator for PPO training with a Monitor wrapper to log rewards
simulator = Monitor(simulator)

# Use the stable_baselines3 PPO algorithm for training
model = PPO("MlpPolicy", simulator, verbose=1, tensorboard_log="./ppo_duckietown_tensorboard/")

# Train for a specific number of timesteps
model.learn(total_timesteps=20000)

# After training, you can test the model
print("Training complete. Now testing...")

# Reset the environment
simulator.reset()

# Run the model to see how it behaves
for _ in range(1000):
    action, _states = model.predict(simulator.observation)
    observation, reward, done, _ = simulator.step(action)
    simulator.render()

    # Reset the simulator if it reaches the end of episode
    if done:
        simulator.reset()

simulator.close()  # Properly close the simulator when done