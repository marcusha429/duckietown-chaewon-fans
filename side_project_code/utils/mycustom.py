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
    metadata = {"render_modes": ["human", "rgb_array", "top_down", "free_cam"]}

    def __init__(self, env):
        super().__init__()
        self.env = env
        # Set the render mode of the environment (important to recording videos of agent using RecordVideo wrapper)
        self.render_mode = env.render_mode

        # Convert spaces from old Gym to Gymnasium
        self.action_space = convert_space(self.env.action_space)
        self.observation_space = convert_space(self.env.observation_space)
        
        self.use_dt_reward = True
        self.prev_pos = None

    def _compute_dt_reward(self, observation, action, info):

        unwrapped = self.env.unwrapped
        
        cur_pos = getattr(unwrapped, 'cur_pos', None)
        cur_angle = getattr(unwrapped, 'cur_angle', None)
        
        if cur_pos is None or cur_angle is None:
            return 0  # Fallback

        my_reward = -1000
        prev_pos = self.prev_pos
        self.prev_pos = cur_pos.copy() if cur_pos is not None else None

        if prev_pos is None:
            return 0  # First step

        try:
            # Get lane position
            lane_pos = unwrapped.get_lane_pos2(cur_pos, cur_angle)
        except NotInLane:
            return my_reward

        # Calculate progress
        curve_point, _ = unwrapped.closest_curve_point(cur_pos, cur_angle)
        prev_curve_point, _ = unwrapped.closest_curve_point(prev_pos, cur_angle)
        if curve_point is None or prev_curve_point is None:
            return 0

        dist = np.linalg.norm(curve_point - prev_curve_point)
        
        # Reward components
        lane_center_dist_reward = np.interp(abs(lane_pos.dist), (0, 0.05), (1, 0))
        lane_center_angle_reward = np.interp(abs(lane_pos.angle_deg), (0, 180), (1, -1))
        
        # Final reward calculation
        reward = 100 * dist + lane_center_dist_reward + lane_center_angle_reward
        return reward
        
    def step(self, action):
        step_result = self.env.step(action)

        # If the environment returns only 4 values, add `truncated=False`
        if len(step_result) == 4:
            observation, reward, done, info = step_result
            truncated = False  # Gymnasium requires a `truncated` value
        else:
            raise ValueError(f"Unexpected step return value length: {len(step_result)}")

        terminated = done  # Rename `done` to `terminated` for Gymnasium compatibility
        
        # Calculate DDPG-style reward if enabled
        if self.use_dt_reward:
            dt_reward = self._compute_dt_reward(observation, action, info)
            if dt_reward is not None:
                reward = dt_reward
        
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
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