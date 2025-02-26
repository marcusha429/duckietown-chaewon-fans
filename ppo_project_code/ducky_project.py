# Running headless 
# Run in terminal 
# Xvfb :1 -screen 0 1024x768x24 &
# export DISPLAY=:1
# python ducky_project.py
import gym
import gym_duckietown
from gym_duckietown.simulator import Simulator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from datetime import datetime
import gc
import os
import torch

# Simple memory tracking function
def log_memory_usage(point_name):
    print(f"Checkpoint: {point_name}")

# Custom Duckietown environment wrapper with improved rewards
class ImprovedDuckietownEnv(gym.Env):
    def __init__(self, env_id="Duckietown-loop_empty-v0"):
        # Create the environment with reduced resolution
        self.env = gym.make(
            env_id,
            max_steps=5000,
            domain_rand=0,
            camera_width=320,  # Reduced resolution
            camera_height=240,
            accept_start_angle_deg=4,
            full_transparency=True,
            distortion=True
        )
        
        # Define action and observation space
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Initialize variables for metrics
        self.reward_sum = 0
        self.episode_length = 0
        
        # Initial position values
        self.prev_pos = None
        self.cur_pos = None
        
        # Track previous distance to center line
        self.prev_dist_to_road_center = None
        
        # Track if the agent went off-track in the previous step
        self.prev_off_track = False
        self.off_track_counter = 0
    
    def reset(self):
        # Reset metrics
        self.reward_sum = 0
        self.episode_length = 0
        self.prev_pos = None
        self.prev_dist_to_road_center = None
        self.prev_off_track = False
        self.off_track_counter = 0
        
        obs = self.env.reset()
        
        # Store initial position
        try:
            self.cur_pos = self._get_position()
        except:
            self.cur_pos = None
        
        return obs
    
    def _get_position(self):
        """Safely extract position from environment"""
        if hasattr(self.env, 'cur_pos'):
            return self.env.cur_pos
        elif hasattr(self.env.unwrapped, 'cur_pos'):
            return self.env.unwrapped.cur_pos
        return None
    
    def _get_lane_pos(self):
        """Safely extract lane position information"""
        try:
            # Try to get detailed lane position
            if hasattr(self.env, 'get_lane_pos2'):
                return self.env.get_lane_pos2(self._get_position(), self.env.unwrapped.cur_angle)
            elif hasattr(self.env.unwrapped, 'get_lane_pos2'):
                return self.env.unwrapped.get_lane_pos2(self._get_position(), self.env.unwrapped.cur_angle)
            # Fall back to simpler lane position
            elif hasattr(self.env, 'get_lane_pos'):
                return self.env.get_lane_pos()
            elif hasattr(self.env.unwrapped, 'get_lane_pos'):
                return self.env.unwrapped.get_lane_pos()
        except:
            pass
        return None
    
    def _detect_collision(self):
        """Safely check for collision"""
        try:
            if hasattr(self.env, 'check_collision'):
                return self.env.check_collision(self._get_position())
            elif hasattr(self.env.unwrapped, 'check_collision'):
                return self.env.unwrapped.check_collision(self._get_position())
        except:
            pass
        return False
    
    def _is_off_track(self):
        """Check if the robot is off track"""
        try:
            position = self._get_position()
            if position is None:
                return False
                
            # Try to use drivable tile information
            if hasattr(self.env, 'is_tile_drivable'):
                return not self.env.is_tile_drivable(position)
            elif hasattr(self.env.unwrapped, 'is_tile_drivable'):
                return not self.env.unwrapped.is_tile_drivable(position)
                
            # Fallback - check if far from centerline
            lane_pos = self._get_lane_pos()
            if lane_pos is not None:
                # If distance to center is too large, consider off track
                if isinstance(lane_pos, tuple):
                    dist = lane_pos[0]  # First element is usually distance
                else:
                    dist = lane_pos
                return abs(dist) > 0.2
        except:
            pass
        return False
    
    def step(self, action):
        # Store previous position
        self.prev_pos = self.cur_pos
        
        # Get lane position before taking action
        lane_pos_before = self._get_lane_pos()
        if lane_pos_before is not None:
            if isinstance(lane_pos_before, tuple):
                self.prev_dist_to_road_center = abs(lane_pos_before[0])
            else:
                self.prev_dist_to_road_center = abs(lane_pos_before)
        
        # Take a step
        obs, reward, done, info = self.env.step(action)
        
        # Get current position
        self.cur_pos = self._get_position()
        
        # Calculate custom reward
        custom_reward = 0
        
        # Detect collision
        collision = self._detect_collision()
        
        # Check if off-track
        off_track = self._is_off_track()
        
        # Get current lane position
        lane_pos = self._get_lane_pos()
        dist_to_road_center = None
        if lane_pos is not None:
            if isinstance(lane_pos, tuple):
                dist_to_road_center = abs(lane_pos[0])
            else:
                dist_to_road_center = abs(lane_pos)
        
        # Heavily penalize collisions
        if collision:
            custom_reward -= 10
            done = True
        
        # Penalize going off-track and encourage on-track driving
        if off_track:
            self.off_track_counter += 1
            # Increasing penalty the longer it stays off track
            custom_reward -= (1 + 0.1 * self.off_track_counter)
            
            # If too long off track, end episode
            if self.off_track_counter > 20:
                done = True
        else:
            self.off_track_counter = 0
            
            # Reward for staying on track and close to center
            if dist_to_road_center is not None:
                # Transform distance to reward (closer to center = higher reward)
                center_reward = 1.0 - dist_to_road_center * 5.0  # Scale factor makes it more sensitive
                center_reward = max(0.1, center_reward)  # Ensure minimum reward while on track
                custom_reward += center_reward
                
                # Reward improvement in staying close to center
                if self.prev_dist_to_road_center is not None:
                    if dist_to_road_center < self.prev_dist_to_road_center:
                        custom_reward += 0.5  # Reward getting closer to center
        
        # Add forward progress reward
        if self.prev_pos is not None and self.cur_pos is not None:
            # Calculate forward distance (in the direction of the track)
            # This is a simplification - ideally use curve direction
            forward_dist = np.linalg.norm(self.cur_pos - self.prev_pos)
            custom_reward += forward_dist * 10.0  # Scale to make movement significant
        
        # Penalize high angular velocities (to avoid wobbling)
        # Assuming action[1] is the steering command
        custom_reward -= abs(action[1]) * 0.5
        
        # Reward for maintaining moderate speed (assuming action[0] is throttle)
        speed_reward = action[0] * 0.5  # Linear reward for speed
        custom_reward += speed_reward
        
        # Update the reward
        reward = custom_reward
        
        # Update metrics
        self.reward_sum += reward
        self.episode_length += 1
        
        # Add metrics to info
        info['episode'] = {
            'r': self.reward_sum,
            'l': self.episode_length
        }
        info['off_track'] = off_track
        info['collision'] = collision
        info['dist_to_center'] = dist_to_road_center
        
        # Update previous states
        self.prev_off_track = off_track
        self.prev_dist_to_road_center = dist_to_road_center
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()

# Create vectorized environment
def make_env(rank):
    def _init():
        env = ImprovedDuckietownEnv()
        return env
    return _init

log_memory_usage("Before environment creation")

# Use fewer environments to save memory
n_envs = 1
env = DummyVecEnv([make_env(i) for i in range(n_envs)])

log_memory_usage("After environment creation")

# Experiment setup - logging
UCNETID = "choiji2"
experiment_name = "ppo_ducky_town_improved"
experiment_logdir = f"ppo_duckietown_log/{experiment_name}_{UCNETID}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Create directory for model checkpoints
os.makedirs(f"{experiment_logdir}/checkpoints", exist_ok=True)

# Create the RL model with optimized hyperparameters for Duckietown
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=experiment_logdir,
    device=device,
    batch_size=32,       # Smaller batch size
    n_steps=128,         # Fewer steps per update
    learning_rate=3e-4,  # Standard learning rate
    ent_coef=0.01,       # Slightly increase exploration
    clip_range=0.2,      # Standard PPO clipping
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # GAE parameter
    n_epochs=10,         # More optimization epochs
    vf_coef=0.5          # Value function coefficient
)

log_memory_usage("Before training")

# Train the model in smaller chunks
print("Training...")
total_timesteps = 500000  # Need more training steps for complex behavior
chunk_size = 10000        # Smaller chunks to avoid memory buildup

for i in range(0, total_timesteps, chunk_size):
    current_chunk = min(chunk_size, total_timesteps - i)
    print(f"Training chunk {i//chunk_size + 1}/{total_timesteps//chunk_size}: {current_chunk} steps")
    
    model.learn(total_timesteps=current_chunk, reset_num_timesteps=False)
    
    # Save checkpoint after each chunk
    model.save(f"{experiment_logdir}/checkpoints/ppo_duckietown_step_{i+current_chunk}")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log_memory_usage(f"After training chunk {i//chunk_size + 1}")

# Save the final model
model.save("ppo_duckietown_final")
print("Model saved!")

# Function to run the trained model
def run_trained_model(model_path="ppo_duckietown_final"):
    # Create an environment for evaluation that renders
    env = ImprovedDuckietownEnv()
    model = PPO.load(model_path)
    
    obs = env.reset()
    total_reward = 0
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        print(f"Reward: {reward:.2f}, Total: {total_reward:.2f}, Off-track: {info.get('off_track', False)}")
        
        if done:
            print(f"Episode finished with total reward: {total_reward}")
            total_reward = 0
            obs = env.reset()

if __name__ == "__main__":
    # Train first, then evaluate
    run_trained_model()