# import pyglet
# window = pyglet.window.Window(visible=False)

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from utils.env import make_gym_env

num_eval_episodes = 4

env = make_gym_env({})

env = RecordVideo(
    env,
    video_folder="test_videos",
    episode_trigger=lambda x: True,
    video_length=200,
    name_prefix="evaluation",
)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
env.close()

print(f"Episode time taken: {env.time_queue}")
print(f"Episode total rewards: {env.return_queue}")
print(f"Episode lengths: {env.length_queue}")
