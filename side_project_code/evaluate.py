# import pyglet
# window = pyglet.window.Window(visible=False)

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from utils.env import make_raw_env

env = make_raw_env()

# Load the trained model
print("Loading the model...")
model = PPO.load("ppo_duckietown_model_100.zip")

# Evaluate the trained policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
# print(f"Mean reward: {mean_reward} ± {std_reward}")

# Test and simulate the trained agent
obs = env.reset()  # For VecEnv, reset returns just the observations
for _ in range(100):
    action, _states = model.predict(obs)
    print("action: ", action)
    observation, reward, done, misc = env.step(action)
    env.render(mode="human")
    if done:
        env.reset()

env.close()
