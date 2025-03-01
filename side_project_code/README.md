# Welcome to Reinforcement Learning in DuckieTown
You've arrived at the ultimate reinvention of duck simulation! This project is born out of a desire to supercharge the classic Duckietown environment by wrapping it in the modern, well-maintained Gymnasium interface. Gone are the days of wrestling with the outdated OpenAI Gym; our custom wrapper lets you leverage cutting-edge reinforcement learning techniques with stable-baselines3, ensuring a smoother, more powerful training experience that’s built for today’s RL community.

At its core, Duckietown RL is a modular training framework designed for rapid experimentation. With easy-to-tweak YAML configuration files and seamless design (inspired by PyTorch Lightning), you can effortlessly customize models, adjust parameters, and dive into new algorithms without the usual compatibility hassles. Whether you’re a seasoned researcher or just starting your RL journey, this open source codebase promises a fun, flexible, and innovative way to explore the exciting world of Duckietown – making every experiment a quacktastic adventure!


![A duckiebot spinning in the duckietown environment](docs/gifs/spinning_duckiebot.gif)
---
This project provides a flexible framework for training reinforcement learning models in the Duckietown simulation environment. With configurable YAML files, a dedicated training script, evaluation utilities, and custom gymnasium wrappers, users can easily experiment with various models and training setups.

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Utilities](#utilities)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

To use this environment, it is important to use conda with Python version 3.10. Follow these steps:

    1. Create a new conda environment with Python 3.10:
    
           conda create --name duckietown_env python=3.10

    2. Activate the environment:

           conda activate duckietown_env

    3. Install the dependencies:

           pip install -r requirements.txt

## Configuration

The configurations for training are stored in YAML files within the **configs** folder. These files allow you to tweak:
- The model type and parameters
- Simulator environment settings for Duckietown
- Total number of time steps
- Seed values and model naming
- Number of environments (n_envs) and other parameters

Users can easily adjust these settings declaratively and then pass the desired configuration file to the trainer.

## Training

The main training logic is implemented in **train.py**. This script:
- Parses the provided configuration file (e.g., `configs/fast_dev_run.yaml` or `configs/model1.yaml`)
- Instantiates a new model or loads one from the model artifacts
- Initializes the Duckietown environment
- Begins training for the specified number of time steps

During training, the framework:
- Periodically saves model checkpoints to the model artifacts directory
- Optionally records video snippets of the agent in MP4 format
- Logs agent performance to TensorBoard in the **logs** directory

To run the training, execute one of the following commands:

    python3 train.py --config configs/fast_dev_run.yaml
    python3 train.py --config configs/model1.yaml

You can monitor the training progress by running:

    tensorboard --logdir=./logs

## Evaluation

The **evaluate_model.py** script contains the logic for evaluating a trained model. It allows you to:
- Hardcode or specify the name of the saved model (typically in a ZIP file)
- Optionally render the environment during evaluation to observe the agent’s performance

This is useful for quickly verifying how well the trained model performs in the Duckietown simulation.

## Tests

The **tests** directory includes various scripts that:
- Help verify whether your environment is set up correctly
- Check that the custom wrappers and utility functions work as expected

These tests serve as a guide to ensure that each component of the framework is operating correctly.

## Utilities

The **utils** folder contains essential helper functions and modules. Key components include:
- Custom gymnasium environment logic that enhances compatibility with stable baselines algorithms
- Callbacks for training, such as a video recording callback (found in **utils/callbacks.py**), which periodically saves training videos
- Additional utilities for managing environment configurations and custom logic specific to Duckietown

## Usage

- **Training:** Run the training script with the desired configuration file.
- **Evaluation:** Use the evaluation script to test a saved model.
- **Monitoring:** Launch TensorBoard to monitor training progress with the command:

      tensorboard --logdir=./logs

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please submit a pull request or open an issue. Your input is greatly appreciated.

Happy training and evaluating your Duckietown models!