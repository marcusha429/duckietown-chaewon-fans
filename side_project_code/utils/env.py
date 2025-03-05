from gym_duckietown.simulator import Simulator
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from gym_duckietown.wrappers import *
from .custom import DuckietownGymnasiumWrapper


# Create a raw environment
def make_raw_env(simulator_kwargs):
    # Default parameters for the environment
    default_kwargs = {
        "map_name": "loop_empty",
        "full_transparency": True,
    }

    # If the user provided any kwargs, merge them with the defaults.
    if simulator_kwargs is not None:
        default_kwargs.update(simulator_kwargs)

    # Create Duckietown environment using the merged parameters.
    env = Simulator(**default_kwargs)

    # Set the render mode of the raw environment
    env.unwrapped.render_mode = "rgb_array"
    env.render_mode = "rgb_array"

    print("Initialized environment")
    return env


# Create the environment and apply wrappers
def make_gym_env(simulator_kwargs) -> VecEnv:
    # Make a raw environment
    env = make_raw_env(simulator_kwargs)

    # Apply Duckietown wrappers
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    # env = ImgWrapper(env)
    # env = ActionWrapper(env)
    # env = DtRewardWrapper(env)
    # print("Initialized Duckietown wrappers")

    # Apply Gymnasium wrapper
    env = DuckietownGymnasiumWrapper(env)
    print("Applied Gymnasium wrapper")

    return env


def make_envs(n_envs: int = 8, simulator_kwargs={}, seed: int = 47):
    # Generate environments with seeds starting at seed argument
    env_fns = [
        lambda i=i: make_gym_env({**simulator_kwargs, "seed": seed + i})
        for i in range(n_envs)
    ]

    # Vectorize and parallelize environments
    env = DummyVecEnv(env_fns)
    print(f"Created {n_envs} environments with unique seeds starting from {seed}.")
    return env
