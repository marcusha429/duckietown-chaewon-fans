# import pyglet
# window = pyglet.window.Window(visible=False)

import gym as old_gym  # Original gym used by Duckietown
import gymnasium as gym  # Gymnasium API
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium import spaces
from gym_duckietown.simulator import Simulator


def convert_space(space):
    """
    Convert an old gym space to a gymnasium space.
    """
    # No need to convert a new gymnasium space
    if type(space).__module__.startswith("gymnasium"):
        return space

    if isinstance(space, old_gym.spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, old_gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, old_gym.spaces.Dict):
        new_spaces = {
            key: convert_space(subspace) for key, subspace in space.spaces.items()
        }
        return gym.spaces.Dict(new_spaces)
    else:
        raise NotImplementedError(
            f"Conversion for space type {type(space)} is not implemented"
        )


class DuckietownGymnasiumWrapper(gym.Env):
    """
    A Gymnasium-compatible wrapper for the Duckietown simulator.

    This wrapper adapts the original Gym-based Simulator to the Gymnasium API,
    converting spaces and modifying step/reset methods. It also implements
    DDPG-style rewards to improve agent performance.
    """

    metadata = {"render_modes": ["human", "rgb_array", "top_down", "free_cam"]}

    def __init__(self, env):
        """
        Initialize the wrapper with an existing Duckietown simulator instance.

        Args:
            env: An instance of gym_duckietown.simulator.Simulator
        """
        super().__init__()
        self.env = env
        # Set the render mode of the environment (important to recording videos of agent using RecordVideo wrapper)
        self.render_mode = env.render_mode

        # Convert spaces from old Gym to Gymnasium
        self.action_space = convert_space(self.env.action_space)
        self.observation_space = convert_space(self.env.observation_space)
        
        # Always use DDPG reward by default
        self.use_ddpg_reward = True

    def _compute_ddpg_reward(self, observation, action, info):

        try:
            # Get the unwrapped environment to access basic attributes
            unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        
            # Basic reward components
        
            # 1. Baseline reward for being alive
            baseline_reward = 0.1
        
            # 2. Reward for forward velocity (encourage movement)
            velocity = getattr(unwrapped, 'speed', 0)
            velocity_reward = velocity * 1.5
        
            # 3. Penalty for excessive steering (encourage smooth driving)
            steering = action[0]  # Assuming first element is steering
            steering_penalty = -abs(steering) * 0.3
        
            # 4. Collision penalty
            collision_penalty = -10.0 if getattr(unwrapped, 'collision', False) else 0.0
            
            # Calculate final reward
            reward = baseline_reward + velocity_reward + steering_penalty + collision_penalty
        
            # Clip reward to reasonable range
            reward = np.clip(reward, -1.0, 1.0)
        
            return reward
        
        except Exception as e:
            # If reward calculation fails, fall back to original reward
            print(f"Warning: DDPG reward calculation failed: {e}")
            return None
        
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        Adapts the original Gym API (observation, reward, done, info) to the
        Gymnasium API: (observation, reward, terminated, truncated, info).
        """
        step_result = self.env.step(action)

        # If the environment returns only 4 values, add `truncated=False`
        if len(step_result) == 4:
            observation, reward, done, info = step_result
            truncated = False  # Gymnasium requires a `truncated` value
        else:
            raise ValueError(f"Unexpected step return value length: {len(step_result)}")

        terminated = done  # Rename `done` to `terminated` for Gymnasium compatibility
        
        # Calculate DDPG-style reward if enabled
        if self.use_ddpg_reward:
            ddpg_reward = self._compute_ddpg_reward(observation, action, info)
            if ddpg_reward is not None:
                reward = ddpg_reward
        
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Returns:
            A tuple (observation, info)
        """
        # Attempt to bypass intermediate wrappers by calling unwrapped.reset()
        try:
            observation = self.env.unwrapped.reset()
        except AttributeError:
            # Fallback if unwrapped isn't available
            observation = self.env.reset()

        return observation, {}

    def render(self):
        """
        Render the environment using the specified render mode.
        """
        return self.env.render(mode="rgb_array")  # Render headless by default
        # return self.env.render(mode="human") # Render visual simulation

    def close(self):
        """
        Close the environment and perform any necessary cleanup.
        """
        self.env.close()

    @property
    def unwrapped(self):
        """
        Return the base (unwrapped) simulator.
        """
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
