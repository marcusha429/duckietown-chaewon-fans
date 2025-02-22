import gym
import numpy as np
from gym import spaces
import gym_duckietown

class DuckietownEnvWrapper(gym.Env):
    def __init__(self):
        print("Initializing DuckietownEnvWrapper...")
        
        # Create the base environment
        self.env = gym.make("Duckietown-udem1-v0")
        
        # Define action space (velocity, steering)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space for [x, z, sin(theta), cos(theta), velocity]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 5, dtype=np.float32),
            high=np.array([np.inf] * 5, dtype=np.float32),
            dtype=np.float32
        )
        
        self.max_steps = 500  # Maximum steps per episode
        self._episode_steps = 0
        self._np_random = None
        
    def seed(self, seed=None):
        """Set the seed for this env's random number generator(s)."""
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
            return self.env.seed(seed)
        return []
        
    def step(self, action):
        """Execute one time step within the environment."""
        try:
            # Store previous position for distance calculation
            prev_pos = self.env.cur_pos.copy()
            
            # Ensure action is numpy array and properly shaped
            action = np.array(action, dtype=np.float32).flatten()
            if action.shape != (2,):
                raise ValueError(f"Invalid action shape: {action.shape}, expected (2,)")
            
            # Clip action values
            action = np.clip(action, -1.0, 1.0)
            
            # Convert to wheel commands
            vel, angle = action
            left_wheel = float(vel - angle)  # Left wheel velocity
            right_wheel = float(vel + angle)  # Right wheel velocity
            
            # Clip wheel velocities
            left_wheel = np.clip(left_wheel, -1.0, 1.0)
            right_wheel = np.clip(right_wheel, -1.0, 1.0)
            wheel_commands = np.array([left_wheel, right_wheel], dtype=np.float32)
            
            # Step the environment
            obs, _, done, info = self.env.step(wheel_commands)
            
            # Get lane position information
            lane_pos = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
            
            # Calculate reward components
            # 1. Speed and direction reward
            speed_reward = 1.0 * vel * lane_pos.dot_dir
            
            # 2. Lane position reward
            lane_reward = -10 * np.abs(lane_pos.dist)
            
            # 3. Collision penalty
            collision = info.get('collision', False)
            col_penalty = -1 if collision else 0
            collision_reward = 40 * col_penalty
            
            # 4. Distance traveled reward
            delta_pos = self.env.cur_pos - prev_pos
            distance_traveled = np.linalg.norm(delta_pos)
            distance_reward = distance_traveled * 50
            
            # Combine all rewards
            reward = speed_reward + lane_reward + collision_reward + distance_reward
            
            # Extract state information
            cur_pos = self.env.cur_pos
            x = float(cur_pos[0])
            z = float(cur_pos[2])
            angle = float(self.env.cur_angle)
            
            # Construct state
            state = np.array([
                x,
                z,
                np.sin(angle),
                np.cos(angle),
                vel  # Current velocity
            ], dtype=np.float32)
            
            # Update step counter
            self._episode_steps += 1
            
            # Check for episode termination
            truncated = self._episode_steps >= self.max_steps
            terminated = collision or done
            
            # Add reward components to info
            info.update({
                'speed_reward': speed_reward,
                'lane_reward': lane_reward,
                'collision_reward': collision_reward,
                'distance_reward': distance_reward,
                'total_reward': reward,
                'lane_position': lane_pos.dist,
                'lane_angle': lane_pos.angle_rad,
                'distance_traveled': distance_traveled
            })
            
            return state, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error during step: {str(e)}")
            print(f"Raw action was: {action}")
            raise
            
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self._episode_steps = 0
        
        if seed is not None:
            self.seed(seed)
        
        # Reset environment
        obs = self.env.reset()
        
        # Get initial state
        state = np.array([
            float(self.env.cur_pos[0]),
            float(self.env.cur_pos[2]),
            np.sin(float(self.env.cur_angle)),
            np.cos(float(self.env.cur_angle)),
            0.0  # Initial velocity
        ], dtype=np.float32)
        
        return state, {}
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Clean up resources."""
        self.env.close()