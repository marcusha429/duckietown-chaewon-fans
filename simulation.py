import pyglet
# Create a small, hidden window to initialize the GL context. This must be at the top of the code to avoid issues with other imports...
window = pyglet.window.Window(visible=False)

pyglet.options['shadow_window'] = False
# pyglet.options['headless'] = True

import numpy as np

import gym_duckietown
from gym_duckietown.simulator import Simulator
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
while True:
    action = np.array(object=[0.1, 0.1], dtype=np.float64)

    observation, reward, done, misc = env.step(action)

    # (method) def step(action: ndarray) -> tuple[ndarray, float, bool, dict]
    # Run one timestep of the environment's dynamics.

    # When end of episode is reached, you are responsible for calling reset to reset this environment's state. Accepts an action and returns either a tuple (observation, reward, terminated, truncated, info).

    # Args
    # action : ActType
    # an action provided by the agent

    # Returns
    # observation : object
    # this will be an element of the environment's observation_space. This may, for instance, be a numpy array containing the positions and velocities of certain objects. reward (float): The amount of reward returned as a result of taking the action. terminated (bool): whether a terminal state (as defined under the MDP of the task) is reached. In this case further step() calls could return undefined results. truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied. Typically a timelimit, but could also be used to indicate agent physically going out of bounds. Can be used to end the episode prematurely before a terminal state is reached. info (dictionary): info contains auxiliary diagnostic information (helpful for debugging, learning, and logging). This might, for instance, contain: metrics that describe the agent's performance state, variables that are hidden from observations, or individual reward terms that are combined to produce the total reward. It also can contain information that distinguishes truncation and termination, however this is deprecated in favour of returning two booleans, and will be removed in a future version.

    env.render(mode="human") # To render graphics
    # env.render(mode="rgb_array") # To get an image back and train headless
    # env.render(mode="top_down") # To render top-down view graphics
    # env.render(mode="free_cam") # Same as human to some extent

    # def render(self, mode: str = "human", close: bool = False, segment: bool = False):
    # """
    # Render the environment for human viewing

    # mode: "human", "top_down", "free_cam", "rgb_array"

    # """
    # assert mode in ["human", "top_down", "free_cam", "rgb_array"]

    if done:
        env.reset()
