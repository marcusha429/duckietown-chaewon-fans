#!/usr/bin/env python3

import argparse
import yaml
import pyglet

# Hide the pyglet window
window = pyglet.window.Window(visible=False)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from utils.env import make_envs

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.env = self.create_env()
        self.model = self.create_model()

    def create_env(self):
        """
        Create and return the vectorized environment/s.
        """
        n_envs = self.config.get("n_envs", 4)
        env = make_envs(n_envs=n_envs)
        return env

    def create_model(self):
        """
        Create the PPO model using CnnPolicy with parameters from the config.
        """
        random_seed = self.config.get("RANDOM_SEED", 47)
        tensorboard_log = self.config.get("tensorboard_log", "./ppo_tensorboard_log/")
        device = self.config.get("device", "cpu")

        model = PPO(
            policy="CnnPolicy",
            env=self.env,
            policy_kwargs={"normalize_images": True},
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=random_seed,
            device=device
        )
        return model

    def train(self):
        """
        Train the model using parameters from the config.
        Uses a custom callback to periodically save the full model,
        overwriting the previous saved file.
        """
        total_timesteps = self.config.get("timesteps", 100)
        model_name = self.config.get("model_name", "ppo_duckietown_model")
        tb_log_name = model_name
        save_freq = self.config.get("save_freq", 1000)

        # Save a checkpoint periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path="./model_artifacts/",
            name_prefix=model_name,
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
    
        # Begin training.
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=[checkpoint_callback],
            reset_num_timesteps=False
        )

        # After training, save the final model (this will override any previous save).
        self.model.save(f"./model_artifacts/{model_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent on Duckietown")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()