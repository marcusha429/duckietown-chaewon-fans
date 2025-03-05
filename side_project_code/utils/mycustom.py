# utils/custom.py
# import pyglet
# window = pyglet.window.Window(visible=False)

import gym as old_gym  # Original gym used by Duckietown
import gymnasium as gym  # Gymnasium API
import numpy as np
from gymnasium.spaces import Box, Discrete

from gym_duckietown.simulator import Simulator


def convert_space(space):
    """
    Convert an old gym space to a gymnasium space.
    """
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


def compute_reward(observation, action, info):
    center_weight = 1.0
    forward_weight = 40.0 
    angular_penalty_weight = 1.0
    progress_weight = 30.0
    corner_penalty_weight = 5.0

    centering_reward = center_weight * (1.0 / (1.0 + abs(info.get("distance_from_center", 0.0))))

    forward_reward = forward_weight * info.get("forward_velocity", 0.0)
    
    angular_velocity = abs(info.get("angular_velocity", 0.0))
    angular_penalty = -angular_penalty_weight * angular_velocity

    corner_penalty = 0
    pos = info.get("position", [0, 0, 0])
    angle = info.get("angle", 0)
    distance_from_center = info.get("distance_from_center", 0)

    if distance_from_center > 0.15 and angular_velocity > 0.3:
        corner_penalty = -corner_penalty_weight

    progress_reward = progress_weight * info.get("lap_progress", 0.0)
    lane_bonus = 0.1 if info.get("on_lane", True) else -2.0

    termination_penalty = -10.0 if info.get("terminated", False) else 0.0

    total_reward = (
        centering_reward 
        + forward_reward 
        + angular_penalty 
        + corner_penalty
        + progress_reward
        + lane_bonus
        + termination_penalty
    )
    
    return total_reward


class DuckietownGymnasiumWrapper(gym.Env):
    """
    A Gymnasium-compatible wrapper for the Duckietown simulator.

    This wrapper adapts the original Gym-based Simulator to the Gymnasium API,
    converting spaces and modifying step/reset methods.
    """
    metadata = {"render_modes": ["human", "rgb_array", "top_down", "free_cam"]}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.render_mode = env.render_mode

        # Convert spaces from old Gym to Gymnasium.
        self.action_space = convert_space(self.env.action_space)
        self.observation_space = convert_space(self.env.observation_space)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics and override the reward with the custom reward.
        """
        step_result = self.env.step(action)

        # If the original environment returns only 4 values, add a default for truncated.
        if len(step_result) == 4:
            observation, original_reward, done, info = step_result
            truncated = False
        else:
            raise ValueError(f"Unexpected step return value length: {len(step_result)}")

        # Compute the custom reward.
        custom_reward = compute_reward(observation, action, info)
        terminated = done  # Gymnasium compatibility: rename done to terminated.
        return observation, custom_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state.
        """
        try:
            observation = self.env.unwrapped.reset()
        except AttributeError:
            observation = self.env.reset()
        return observation, {}

    def render(self):
        """
        Render the environment in rgb_array mode.
        """
        return self.env.render(mode="rgb_array")

    def close(self):
        """
        Close the environment and perform cleanup.
        """
        self.env.close()

    @property
    def unwrapped(self):
        """
        Return the base (unwrapped) simulator.
        """
        return self.env