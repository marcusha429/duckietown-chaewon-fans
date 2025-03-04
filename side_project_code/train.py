# import pyglet
# window = pyglet.window.Window(visible=False)

import argparse
import yaml
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.env import make_envs
from utils.callbacks import VideoRecordingCallback


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.seed = self.config.get("seed", 47)

        # Set up simulator and model parameters with defaults
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
        n_envs = self.config.get("n_envs", 1)
        print(f"Duckietown Simulator arguments: {self.simulator_kwargs}")

        env = make_envs(
            n_envs=n_envs,
            seed=self.seed,
            simulator_kwargs=self.simulator_kwargs,
        )
        print(
            f"Created {n_envs} environments with seed {self.seed} and simulator arguments: {self.simulator_kwargs}"
        )
        return env

    def create_model(self):
        """
        Create the RL model (PPO or SAC) using the fixed parameters plus any additional model-specific parameters.
        Global parameters like seed are parsed from the config.
        """
        # Choose the RL algorithm
        rl_algorithm = self.config.get("rl_algorithm", "PPO").upper()
        model_class = PPO if rl_algorithm == "PPO" else SAC

        print(
            f"Creating {rl_algorithm} model with seed {self.seed} and model arguments: {self.model_params}"
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
        model_path = f"./model_artifacts/{model_name}/{model_name}"
        rl_algorithm = self.config.get("rl_algorithm", "PPO").upper()

        if os.path.exists(model_path + ".zip"):
            print(f'Loading existing "{rl_algorithm}" model from {model_path}')
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
        print(f"Model name: {model_name}")
        total_timesteps = self.config.get("total_timesteps", 32)
        print(f"Training for {total_timesteps} timesteps")

        # Dynamic checkpoint save frequency (k)
        n_envs = self.config.get("n_envs", 1)
        total_agent_steps = (
            total_timesteps // n_envs
        )  # total agent steps (not parallelized)
        k = total_agent_steps // 20  # Default to 20 total checkpoints

        # Use k as the default checkpoint_save_freq
        checkpoint_save_freq = self.config.get("checkpoint_save_freq", k)
        video_save_freq = self.config.get("video_save_freq", k)

        # Printing the frequencies
        print(f"Checkpoint saving frequency: {checkpoint_save_freq}")
        print(f"Video saving frequency: {video_save_freq}")

        # Adjusting frequencies if n_envs > 1
        if n_envs > 1:
            adjusted_checkpoint_freq = checkpoint_save_freq // n_envs
            adjusted_video_freq = video_save_freq // n_envs

            # Warning message for the current behavior
            print(
                f"\033[1m\033[31mWARNING: You are using {n_envs} n_envs. This means your checkpoint and video saves will happen every "
                f"{n_envs * checkpoint_save_freq} steps (note the multiplication with n_envs). Be careful about this behavior! \033[0m"
            )

            # Recommendation message for adjustment
            print(
                f"\033[1m\033[32mNOTE: For more accurate saving of agent steps, consider adjusting the save frequencies to {adjusted_checkpoint_freq} for checkpoints, "
                f"and {adjusted_video_freq} for videos. This ensures saving occurs every real agent step. (Ignore if adjusted already) \n\033[0m"
            )

        # Set up checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=f"./model_artifacts/{model_name}",
            name_prefix=model_name,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # Create the custom callback for periodically recording videos
        video_callback = VideoRecordingCallback(
            simulator_kwargs=self.simulator_kwargs,
            video_folder=f"videos/{model_name}",
            video_length=200,
            save_freq=video_save_freq,
        )

        print("Beginning training...")
        # Begin training
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=model_name,
            callback=[checkpoint_callback, video_callback],
            reset_num_timesteps=False,
        )
        print("Training complete")

        # Save the final model.
        self.model.save(f"./model_artifacts/{model_name}/{model_name}")
        print(
            f"Saved final model artifact to ./model_artifacts/{model_name}/{model_name}"
        )


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