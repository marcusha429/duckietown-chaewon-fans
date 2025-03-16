import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces


class FrameStack(gym.Wrapper):
    """
    Stack n frames together as observation.
    
    This customized frame stacking wrapper is designed to work with Gymnasium 1.1.1
    and outputs channel-first observations for Stable Baselines3 compatibility.
    
    :param env: The environment to wrap
    :param num_stack: Number of frames to stack
    """
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        
        # Get shape info from original observation space
        obs_shape = env.observation_space.shape
        
        # For image observations (assuming shape is (H, W, C))
        # Stack along the channel dimension to get (H, W, C*num_stack)
        # and then convert to channel-first for SB3: (C*num_stack, H, W)
        self.height, self.width, self.channels = obs_shape
        
        # Create new observation space with stacked channels (channel-first for SB3)
        stacked_shape = (self.channels * num_stack, self.height, self.width)
        
        low = np.zeros(stacked_shape, dtype=env.observation_space.dtype)
        high = np.ones(stacked_shape, dtype=env.observation_space.dtype) * 255  # Assuming uint8 images
        
        self.observation_space = spaces.Box(
            low=low, high=high, 
            dtype=env.observation_space.dtype
        )
        
    def reset(self, **kwargs):
        """Reset the environment and return stacked observations."""
        obs, info = self.env.reset(**kwargs)
        
        # Clear our frame buffer and add initial observation multiple times
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
            
        return self._get_observation(), info
    
    def step(self, action):
        """Take action and return stacked observations."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Return the stacked observations in channel-first format for SB3.
        Takes frames (H, W, C) and returns (C*num_stack, H, W)
        """
        # Make a copy of all frames
        frames = list(self.frames)
        
        # Stack all frames along the channel dimension
        # First reshape each frame to isolate the channels
        reshaped_frames = []
        for frame in frames:
            # Extract channels from each frame
            channels = [frame[:, :, i] for i in range(self.channels)]
            reshaped_frames.extend(channels)
        
        # Stack all channel planes
        observation = np.stack(reshaped_frames, axis=0)
        
        return observation