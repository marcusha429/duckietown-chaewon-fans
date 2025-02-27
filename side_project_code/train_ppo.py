# Hide pyglet window
import pyglet

window = pyglet.window.Window(visible=False)

import argparse
import yaml
import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack
from utils.env import make_envs
from utils.callbacks import VideoRecordingCallback


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.seed = self.config.get("seed", 47)
        self.simulator_kwargs = self.config.get("simulator_params", {})
        self.simulator_kwargs.setdefault("seed", self.seed)
        self.model_params = self.config.get("model_params", {})
        self.model_params.setdefault("seed", self.seed)

        self.env = self.create_env()
        self.model = self.load_or_create_model()

    def create_env(self):
        """
        Create and return the vectorized environment.
        Uses the global n_envs parameter and passes any Duckietown-specific parameters as simulator_kwargs.
        """
        n_envs = self.config.get("n_envs", 4)
        print(f"Printing the simulator kwargs: {self.simulator_kwargs}")

        env = make_envs(
            n_envs=n_envs,
            seed=self.seed,
            simulator_kwargs=self.simulator_kwargs,
        )
        print(
            f"{n_envs} environments created with seed {self.seed} and simulator arguments: {self.simulator_kwargs}"
        )
        return env

    def create_model(self):
        """
        Create the RL model (PPO or SAC) using the fixed parameters plus any additional model-specific parameters.
        Global parameters like seed are parsed from the config
        """
        # Choose the RL algorithm
        rl_algorithm = self.config.get("rl_algorithm", "PPO").upper()
        model_class = PPO if rl_algorithm == "PPO" else SAC

        print(
            f"Creating and instantiating a {rl_algorithm} model with seed {self.seed} and model parameters: {self.model_params}"
        )
        return model_class(
            policy="CnnPolicy",
            env=self.env,
            policy_kwargs={"normalize_images": True},
            tensorboard_log="logs",
            **self.model_params,
        )

    def load_or_create_model(self):
        """
        Load the model from the checkpoint if it exists, otherwise create a new model.
        """
        model_name = self.config.get("model_name", "duckietown_model")
        model_path = f"./model_artifacts/{model_name}"
        rl_algorithm = self.config.get("rl_algorithm", "PPO").upper()  # Default to PPO

        if os.path.exists(model_path + ".zip"):
            print(f"Loading {rl_algorithm} model from {model_path}")
            model_class = PPO if rl_algorithm == "PPO" else SAC
            model = model_class.load(model_path, env=self.env)
        else:
            model = self.create_model()
        return model

    def train(self):
        """
        Train the model using training parameters from the configuration.
        The training_parameters section should include at least total_timesteps (and can include other kwargs).
        """
        model_name = self.config.get("model_name", "duckietown_model")

        # Get training-specific parameters.
        total_timesteps = self.config.get("total_timesteps", 32)
        print(f"Training for {total_timesteps} timesteps")

        # Set up the checkpoint callback.
        checkpoint_callback = CheckpointCallback(
            save_freq=4096,
            save_path="./model_artifacts/",
            name_prefix=model_name,
        )

        # Create the custom callback for recording videos every 1024 steps
        video_callback = VideoRecordingCallback(
            video_folder="videos", step_interval=1024
        )

        # Begin training
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=model_name,
            # callback=[checkpoint_callback, video_callback],
            callback=[checkpoint_callback],
            reset_num_timesteps=False,
        )

        # Save the final model.
        self.model.save(f"./model_artifacts/{model_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on Duckietown")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
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
