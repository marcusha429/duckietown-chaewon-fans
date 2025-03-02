# import pyglet
# window = pyglet.window.Window(visible=False)

from utils.env import make_gym_env

# Initialize environment
env = make_gym_env(simulator_kwargs={})

# Test rendering for a single step
obs, _ = env.reset(seed=0)
for _ in range(10):  # Run for 10 steps to see if rendering works
    action = env.action_space.sample()  # Sample random action
    obs, reward, terminated, truncated, info = env.step(action)
    img = env.render()  # Get the rendered image
    if img is not None:
        print("Rendering works!")
    if terminated or truncated:
        break
env.close()
