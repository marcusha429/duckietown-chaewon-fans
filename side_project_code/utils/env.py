from gym_duckietown.simulator import Simulator
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor

from gym_duckietown.wrappers import *
from .custom import DuckietownGymnasiumWrapper
from .myframestack import FrameStack


# Create a raw environment
def make_raw_env(simulator_kwargs):
    # Default parameters for the environment
    default_kwargs = {
        "map_name": "small_loop",
        "full_transparency": True,
        "max_steps": 250,  # adjust max step to 250
        "domain_rand": False,  # turn off domain randomization
    }

    # If the user provided any kwargs, merge them with the defaults.
    if simulator_kwargs is not None:
        default_kwargs.update(simulator_kwargs)

    default_kwargs.setdefault("camera_width", 84)  # adjust camera resolution
    default_kwargs.setdefault("camera_height", 84)  # adjust camera resolution

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
    env = FrameStack(env, 3)
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

    # Wrap with VecMonitor to log episode rewards and lengths
    env = VecMonitor(env)

    # from stable_baselines3.common.vec_env import VecFrameStack
    # env = VecFrameStack(env, n_stack=3) #add 3 stack frame

    print(f"Created {n_envs} environments with unique seeds starting from {seed}.")
    return env
