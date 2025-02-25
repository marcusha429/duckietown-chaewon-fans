import argparse
import yaml
import pyglet
import os

# Hide pyglet window
window = pyglet.window.Window(visible=False)

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from utils.env import make_envs

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.env = self.create_env()
        self.model = self.load_or_create_model()

    def create_env(self):
        """
        Create and return the vectorized environment.
        Uses the global n_envs parameter and passes any Duckietown-specific parameters as simulator_kwargs.
        """
        n_envs = self.config.get("n_envs", 4)
        seed = self.config.get("seed", 47)
        simulator_kwargs = self.config.get("simulator_kwargs", {})
        env = make_envs(
            n_envs=n_envs,
            seed=seed,
            simulator_kwargs=simulator_kwargs,
        )
        return env

    def create_model(self):
        """
        Create the RL model (PPO or SAC) using the fixed parameters plus any additional model-specific parameters.
        Global parameters like seed are parsed from the config
        """
        # Global parameters
        seed = self.config.get("seed", 47)
        tensorboard_log = "logs"
        
        # Choose the RL algorithm
        rl_algorithm = self.config.get("rl_algorithm", "PPO").upper()
        if rl_algorithm == "PPO":
            model_class = PPO
        elif rl_algorithm == "SAC":
            model_class = SAC
        else:
            raise ValueError(f"Unknown rl_algorithm: {rl_algorithm}")
        
        # Model-specific parameters are provided in a dedicated section
        model_params = self.config.get("model_parameters", {})

        # Instantiate and return the model
        return model_class(
            policy="CnnPolicy",
            env=self.env,
            policy_kwargs={
                "normalize_images": True
            },
            tensorboard_log=tensorboard_log,
            seed=seed,
            **model_params
        )

    def load_or_create_model(self):
        """
        Load the model from the checkpoint if it exists, otherwise create a new model.
        """
        model_name = self.config.get("model_name", "duckietown_model")
        model_path = f"./model_artifacts/{model_name}"
        rl_algorithm = self.config.get("rl_algorithm", "PPO").upper() # Default to PPO

        if rl_algorithm == "PPO":
            model_class = PPO
        elif rl_algorithm == "SAC":
            model_class = SAC
        else:
            raise ValueError(f"rl_algorithm must be PPO or SAC. Received  {rl_algorithm}")
        
        if os.path.exists(model_path + ".zip"):
            print(f"Loading model from {model_path}")
            model = model_class.load(model_path, env=self.env)
        else:
            print("No model found. Creating a new model.")
            model = self.create_model()
        
        return model

    def train(self):
        """
        Train the model using training parameters from the configuration.
        The training_parameters section should include at least total_timesteps (and can include other kwargs).
        """
        model_name = self.config.get("model_name", "duckietown_model")
        tb_log_name = model_name

        # Get training-specific parameters.
        total_timesteps = self.config.get("total_timesteps", 100)

        # Set up the checkpoint callback.
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path="./model_artifacts/",
            name_prefix=model_name,
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # Begin training. Extra training parameters (if any) are passed as kwargs.
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=[checkpoint_callback],
            reset_num_timesteps=False
        )

        # Save the final model.
        self.model.save(f"./model_artifacts/{model_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on Duckietown")
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