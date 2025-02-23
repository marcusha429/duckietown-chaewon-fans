
from gym_duckietown.simulator import Simulator
from .custom import DuckietownGymnasiumWrapper

# Create the environment and apply wrappers
def make_env():
    simulator_kwargs = {
        "seed": 123,
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

    # Apply Duckietown wrappers
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    # env = ImgWrapper(env)
    # env = ActionWrapper(env)
    # env = DtRewardWrapper(env)
    print("Initialized Duckietown wrappers")

    # Apply Gymnasium wrapper
    env = DuckietownGymnasiumWrapper(env)
    print("Applied Gymnasium wrapper")

    return env