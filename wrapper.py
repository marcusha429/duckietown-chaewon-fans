import gym
from gym import spaces
import numpy as np
from gym_duckietown.simulator import Simulator

class DuckietownWrapper(gym.Wrapper):
    def __init__(self, simulator: Simulator):
        super(DuckietownWrapper, simulator).__init__()
        
        self.simulator = simulator
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # Control space: [steering, throttle]
        self.observation_space = spaces.Box(low=0, high=255, shape=(simulator.camera_height, simulator.camera_width, 3), dtype=np.uint8)  # RGB image from camera
        
    def reset(self):
        """Reset the simulator and return the first observation."""
        self.simulator.reset()
        return self.simulator.observation
    
    def step(self, action):
        """Take a step in the simulation based on the given action."""
        # action: [steering, throttle]
        steering, throttle = action
        action_dict = {
            'steering': steering,
            'throttle': throttle,
        }
        
        # Apply action to simulator
        observation, reward, done, info = self.simulator.step(action_dict)
        
        # Return observation, reward, done, info
        return observation, reward, done, info

    def render(self, mode='human'):
        """Render the current state of the simulator."""
        self.simulator.render()
        
    def close(self):
        """Close the simulator."""
        self.simulator.close()