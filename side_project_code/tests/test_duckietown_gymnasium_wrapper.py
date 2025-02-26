import pyglet
window = pyglet.window.Window(visible=False)

from stable_baselines3.common.env_checker import check_env
from gym_duckietown.simulator import Simulator
from utils.custom import DuckietownGymnasiumWrapper
import numpy as np

def test_duckietown_gymnasium_wrapper():
        
    # First create the base Duckietown simulator
    env = Simulator(
        seed=0,
        map_name="loop_empty"
    )

    # Now wrap it with our Gymnasium wrapper
    env = DuckietownGymnasiumWrapper(env=env)

    # Sanity check basic agent and environment functionality
    env.reset()
    action = np.array([0.1, 0.1], dtype=np.float64)
    env.step(action)
    
    # Check if the wrapped environment meets Gymnasium specifications
    check_env(env, warn=False, skip_render_check=True)


if __name__ == "__main__":
    test_duckietown_gymnasium_wrapper()