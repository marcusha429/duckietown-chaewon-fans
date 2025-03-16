# utils/env.py
from gym_duckietown.simulator import Simulator
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecMonitor

from gym_duckietown.wrappers import *
from .custom import DuckietownGymnasiumWrapper
from .framestack import FrameStack


def make_raw_env(simulator_kwargs):
    default_kwargs = {
        "map_name": "small_loop",
        "max_steps": 250,
        "accept_start_angle_deg": 4,
        "full_transparency": True,
        "domain_rand": False,
    }
    if simulator_kwargs is not None:
        default_kwargs.update(simulator_kwargs)
    default_kwargs.setdefault("camera_width", 84)
    default_kwargs.setdefault("camera_height", 84)

    env = Simulator(**default_kwargs)
    env.unwrapped.render_mode = "rgb_array"
    env.render_mode = "rgb_array"

    print("Initialized environment")
    return env


def make_gym_env(simulator_kwargs) -> VecEnv:
    env = make_raw_env(simulator_kwargs)
    env = DuckietownGymnasiumWrapper(env)
    env = FrameStack(env, 3)
    print("Applied Gymnasium wrapper")
    return env


def make_envs(n_envs: int = 4, simulator_kwargs={}, seed: int = 47):
    env_fns = [
        lambda i=i: make_gym_env({**simulator_kwargs, "seed": seed + i})
        for i in range(n_envs)
    ]
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)
    

    print(f"Created {n_envs} environments with unique seeds starting from {seed}.")
    print(f"Observation space after stacking: {env.observation_space}")
    return env