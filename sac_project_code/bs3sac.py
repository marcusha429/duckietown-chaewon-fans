import gym
import gym_duckietown
from gym_duckietown.simulator import Simulator
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import numpy as np
from datetime import datetime
import torch
from gym import spaces

class DuckietownEnvWrapper(gym.Env):
    def __init__(self):
        # Create the simulator
        self.env = Simulator(
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
        
        # Define action and observation spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Custom reward: Encourage forward movement, discourage off-track
        if info['Simulator']['collision']:  # If collision happens, large penalty
            reward -= 10
        else:
            reward += 1 - abs(info['Simulator']['lane_position'])  # Penalize deviation from center lane
        
        return obs, reward, done, info
    
    def reset(self):
        return self.env.reset()
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()

def main():
    # Create and wrap the environment
    env = DuckietownEnvWrapper()
    
    # Experiment setup - logging
    experiment_name = "sac_ducky_town"
    experiment_logdir = f"sac_duckietown_log/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Create the SAC model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAC(
        "CnnPolicy", 
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=experiment_logdir,
        device=device
    )

    # Train the model
    print("Training...")
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("sac_duckietown")
    print("Model saved!")

if __name__ == "__main__":
    main()