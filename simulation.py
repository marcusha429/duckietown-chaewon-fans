import pyglet
window = pyglet.window.Window(visible=True)

import numpy as np
from gym_duckietown.simulator import Simulator

# Create the environment
env = Simulator(
    seed=123, # random seed
    map_name="loop_empty",
    max_steps=500001, # we don't want the gym to reset itself
    domain_rand=0,
    camera_width=640,
    camera_height=480,
    accept_start_angle_deg=4, # start close to straight
    full_transparency=True,
    distortion=True,
)   

# Render the environment
while True:
    # Consistently move forward
    action = np.array(object=[0.1, 0.1], dtype=np.float32)

    observation, reward, done, misc = env.step(action)

    env.render(mode="human") # To render graphics
    # env.render(mode="rgb_array") # To get an image back and train headless
    # env.render(mode="top_down") # To render top-down view graphics
    # env.render(mode="free_cam") # Same as human to some extent

    if done:
        env.reset()
    
    # Run pyglet app to ensure rendering works
    # pyglet.app.run()