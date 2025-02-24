import pyglet
window = pyglet.window.Window(visible=False)

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from utils.env import make_env
# from gym_duckietown.wrappers import *

RANDOM_SEED = 47

import torch
torch.set_num_threads(10)

# Prepare the environment
# env = make_env() 
# env = VecTransposeImage(DummyVecEnv([make_env]))

# Vectorized environment setup
env = make_vec_env(
    env_id=lambda: make_env(),
    n_envs=4,
    seed=RANDOM_SEED
)

# Create PPO model
model = PPO(
    policy="CnnPolicy",
    env=env,
    policy_kwargs={"normalize_images": False},
    verbose=1,
    tensorboard_log="./ppo_tensorboard_log/",
    seed=RANDOM_SEED,
    device="mps"
)

# Train the agent
model.learn(
    total_timesteps=50,
    tb_log_name="var_2"
)

# Save the trained model
model.save(
    "ppo_duckietown_model_2")

# # Evaluate the trained policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} ± {std_reward}")

# # Load the trained model
# model = PPO.load("ppo_duckietown_model")

# # Test the trained agent
# obs = env.reset()
# for _ in range(2000):
#     action, _states = model.predict(obs)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render(mode="human")
#     if done:
#         obs = env.reset()

# env.close()