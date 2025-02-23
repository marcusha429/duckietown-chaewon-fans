import pyglet
window = pyglet.window.Window(visible=False)

import gym as old_gym  # Original gym used by Duckietown
import gymnasium as gym  # Gymnasium API
import numpy as np
from gymnasium.spaces import Box, Discrete

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
        new_spaces = {key: convert_space(subspace) for key, subspace in space.spaces.items()}
        return gym.spaces.Dict(new_spaces)
    else:
        raise NotImplementedError(f"Conversion for space type {type(space)} is not implemented")

class DuckietownGymnasiumWrapper(gym.Env):
    """
    A Gymnasium-compatible wrapper for the Duckietown simulator.
    
    This wrapper adapts the original Gym-based Simulator to the Gymnasium API,
    converting spaces and modifying step/reset methods. It also renames the
    `compute_reward` method (if present) to avoid false detection as a GoalEnv.
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

        # METHOD 1: REMOVE THE compute reward method of the underlying simulator class entirely
        # # # Rename compute_reward to avoid confusion with goal-conditioned envs
        # # if hasattr(self.env, 'compute_reward'):
        # #     self.env.calculate_reward = self.env.compute_reward
        # #     del self.env.compute_reward

        # # Rename compute_reward to avoid confusion with goal-conditioned envs
        # if hasattr(self.env.__class__, "compute_reward"):
        #     self.env.__class__.calculate_reward = self.env.__class__.compute_reward
        #     delattr(self.env.__class__, "compute_reward")  # Remove from class, not instance

        # # Method 2:
        # # Rename compute_reward to avoid confusion with goal-conditioned envs
        # if hasattr(self.env, "compute_reward"):
        #     self._original_compute_reward = self.env.compute_reward  # Keep a reference
        #     setattr(self.env, "calculate_reward", self._original_compute_reward)  # Store it under a new name

        #     # Override compute_reward to prevent `check_env()` from treating it as goal-conditioned
        #     self.env.compute_reward = lambda *args, **kwargs: print("compute_reward() is overridden and unused in this context.")
        
        # Convert spaces from old Gym to Gymnasium
        self.action_space = convert_space(self.env.action_space)
        self.observation_space = convert_space(self.env.observation_space)
    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        
        Adapts the original Gym API (observation, reward, done, info) to the
        Gymnasium API: (observation, reward, terminated, truncated, info).
        """
        observation, reward, done, info = self.env.step(action)
        terminated = done    # Use the original 'done' as 'terminated'
        truncated = False    # No explicit truncation logic provided
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
            # Fallback if unwrapped isn’t available
            observation = self.env.reset()
        
        return observation, {}
    
    def render(self):
        """
        Render the environment using the specified render mode.
        """
        if self.render_mode is not None:
            return self.env.render(mode=self.render_mode)
        return self.env.render()
    
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
        return self.env

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from gym_duckietown.simulator import Simulator

    # First create the base Duckietown simulator
    env = Simulator(
        seed=123,
        map_name="loop_empty"
    )

    # Now wrap it with our Gymnasium wrapper
    env = DuckietownGymnasiumWrapper(env=env)

    # Check if the wrapped environment meets Gymnasium specifications
    check_env(env, warn=False, skip_render_check=True)

    # Optional: Test basic functionality
    obs = env.reset()
    action = np.array([0.1, 0.1], dtype=np.float64)
    obs, reward, terminated, truncated, info = env.step(action)


class TransposeImage(old_gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Original shape: (H, W, C)
        obs_shape = self.observation_space.shape
        # New shape: (C, H, W)
        new_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        # Transpose the low and high bounds as well
        self.observation_space = Box(
            low=self.observation_space.low.transpose(2, 0, 1),
            high=self.observation_space.high.transpose(2, 0, 1),
            shape=new_shape,
            dtype=np.float32
        )
        
    def observation(self, observation):
        # Transpose observation from (H, W, C) to (C, H, W)
        return observation.transpose(2, 0, 1)
