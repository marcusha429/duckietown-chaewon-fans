# import pyglet
# window = pyglet.window.Window(visible=False)

import random
import numpy as np
from gym_duckietown.simulator import Simulator

# List of maps
maps = [
    "4way",
    "loop_dyn_duckiebots",
    "loop_empty",
    "loop_obstacles",
    "loop_pedestrians",
    "regress_4way_adam",
    "regress_4way_drivable",
    "small_loop",
    "small_loop_cw",
    "straight_road",
    "udem1",
    "zigzag_dists",
]

for map in maps:
    # Create the environment
    env = Simulator(
        seed=3,
        map_name=map,
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        full_transparency=True,
        accept_start_angle_deg=20,
    )

    # Determine which mode to render at random
    render_human = random.choice([True, False])

    # Render the environment
    for i in range(100):
        # Spin around in place
        action = np.array(object=[0, 0.2], dtype=np.float32)

        # Perform an step in the environment
        observation, reward, done, misc = env.step(action)

        # Render simulation
        if render_human:
            env.render(mode="human")
        else:
            env.render(mode="top_down")

        if done:
            env.reset()

    env.window.close()
