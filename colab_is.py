import pyglet
window = pyglet.window.Window(visible=False)
pyglet.options['shadow_window'] = False


map_name = "Duckietown-small_loop-v0"

import gym, gym_duckietown

env = gym.make(map_name)

"""
env = DuckietownEnv(
    map_name=map_name,
    draw_curve="store_true",
    draw_bbox="store_true",
    domain_rand="store_true",
    accept_start_angle_deg=4, 
)"""


env.reset()
prev_screen = env.render(mode='rgb_array')
# plt.imshow(prev_screen)

for i in range(10):
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  screen = env.render(mode='rgb_array')

  if done:
    break


print(env.action_space)

print(env.observation_space)