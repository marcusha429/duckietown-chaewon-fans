
from gym_duckietown.simulator import Simulator
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
# from gym_duckietown.wrappers import *
from .custom import DuckietownGymnasiumWrapper

RANDOM_SEED = 47

# Create a raw environment
def make_raw_env():
    simulator_kwargs = {
        "seed": RANDOM_SEED,
        "map_name": "loop_empty",
        "max_steps": 10,
        "domain_rand": 0,
        "camera_width": 640,
        "camera_height": 480,
        "accept_start_angle_deg": 4,
        "full_transparency": True,
    }
    # Create Duckietown environment
    env = Simulator(**simulator_kwargs)
    print("Initialized environment")
    return env


# Create the environment and apply wrappers
def make_gym_env() -> VecEnv:
    # Make a raw environment
    env = make_raw_env()

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


def make_envs(n_envs: int = 4):
    # Vectorize and parallelize environment
    env = make_vec_env(
        env_id=lambda: make_gym_env(),
        n_envs=n_envs,
        seed=RANDOM_SEED
    )
    print(f"Vectorized and parallelized {n_envs} environments.")
    return env