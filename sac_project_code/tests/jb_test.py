#!/usr/bin/env python3

import numpy as np
import gym_duckietown
from gym_duckietown.simulator import Simulator

env = Simulator(
        seed=123, # random seed
        map_name="loop_empty",
        max_steps=500001, # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4, # start close to straight
        full_transparency=True,
        distortion=True,
    )   

while True:
    action = np.array([0.1, 0.1], dtype=np.float64)  # Explicitly define as a NumPy array
    print(f"Action before step: {action}, Type: {type(action)}, Shape: {action.shape}")  # Debugging print

    try:
        observation, reward, done, misc = env.step(action)
        env.render()
        print(f"Observation shape: {np.shape(observation)}")
    except Exception as e:
        print(f"Error in step execution: {e}")
        break  # Stop the loop if an error occurs

    if done:
        env.reset()

