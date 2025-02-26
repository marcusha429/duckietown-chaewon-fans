import pyglet
window = pyglet.window.Window(visible=False)

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from gym_duckietown.simulator import Simulator

# Import make_envs from your project
from utils.env import make_envs

def test_dummy_vec_env():
    print("\n===== TESTING ENVIRONMENT BEFORE VECTORIZATION =====")
    
    # Create a single environment (before vectorization)
    env = make_envs()
    
    # Test step BEFORE DummyVecEnv
    action = env.action_space.sample()
    action = np.array(action, dtype=np.float32)
    
    print("Action shape before step:", action.shape)  # Expected: (2,)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("Step output BEFORE DummyVecEnv:")
    print(f"Obs shape: {obs.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

    print("\n===== TESTING ENVIRONMENT AFTER VECTORIZATION =====")

    # Apply DummyVecEnv (after verifying base env works)
    env = VecTransposeImage(DummyVecEnv([make_envs]))

    # Sample action for DummyVecEnv
    action = np.array(env.action_space.sample(), dtype=np.float32).reshape(1, -1)

    # Print action shape to verify
    print("Action shape before step (DummyVecEnv):", action.shape)  # Expected: (1,2)

    # Test step AFTER DummyVecEnv
    step_result = env.step(action)

    # Fix unpacking issue
    if len(step_result) == 4:
        obs, reward, done, info = step_result
        truncated = False  # Add missing value
    else:
        obs, reward, done, truncated, info = step_result

    print("Step output AFTER DummyVecEnv:")
    print(f"Obs shape: {obs.shape}, Reward: {reward}, Terminated: {done}, Truncated: {truncated}, Info: {info}")

    print("\n✅ TEST COMPLETED SUCCESSFULLY!\n")

if __name__ == "__main__":
    test_dummy_vec_env()