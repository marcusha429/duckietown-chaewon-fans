# utils/env.py
from gym_duckietown.simulator import Simulator
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from gym_duckietown.wrappers import *
from .mycustom import DuckietownGymnasiumWrapper


def make_raw_env(simulator_kwargs):
    default_kwargs = {
        "map_name": "loop_empty",
        "max_steps": 250,
        "domain_rand": 0,
        "accept_start_angle_deg": 4,
        "seed": 47,
        "full_transparency": True,
    }
    if simulator_kwargs is not None:
        default_kwargs.update(simulator_kwargs)

    env = Simulator(**default_kwargs)
    env.unwrapped.render_mode = "rgb_array"
    env.render_mode = "rgb_array"

    print("Initialized environment")
    return env


def make_gym_env(simulator_kwargs) -> VecEnv:
    env = make_raw_env(simulator_kwargs)
    env = DuckietownGymnasiumWrapper(env)
    print("Applied Gymnasium wrapper")
    return env


def make_envs(n_envs: int = 4, simulator_kwargs={}, seed: int = 47):
    env = make_vec_env(
        env_id=lambda: make_gym_env(simulator_kwargs), n_envs=n_envs, seed=seed
    )
    print(f"Vectorized and parallelized {n_envs} environments.")
    return env