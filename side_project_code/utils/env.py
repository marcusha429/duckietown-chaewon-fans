from typing import Dict
from gym_duckietown.simulator import Simulator
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

# from gym_duckietown.wrappers import *
from .custom import DuckietownGymnasiumWrapper


# Create a raw environment
def make_raw_env(simulator_kwargs):
    # Default parameters for the environment
    default_kwargs = {
        "map_name": "loop_empty",
        "max_steps": 5000,
        "domain_rand": 0,
        "accept_start_angle_deg": 4,
        "full_transparency": True,
    }

    # If the user provided any kwargs, merge them with the defaults.
    if simulator_kwargs is not None:
        default_kwargs.update(simulator_kwargs)

    # Create Duckietown environment using the merged parameters.
    env = Simulator(**default_kwargs)
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


def make_envs(n_envs: int = 4, simulator_kwargs={}, seed: int = 47):
    # Vectorize and parallelize environment
    env = make_vec_env(
        env_id=lambda: make_gym_env(simulator_kwargs), n_envs=n_envs, seed=seed
    )
    print(f"Vectorized and parallelized {n_envs} environments.")
    return env
