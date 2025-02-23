# Duckietown Gymnasium: Next-Gen Reinforcement Learning Playground 🦆🚀

Welcome to the **Duckietown Gymnasium** – a modular, clean, and cutting-edge training setup designed for reinforcement learning in Duckietown! Whether you're experimenting with legacy algorithms or leveraging the latest stable baselines, this platform is built to help you iterate rapidly and push the boundaries of your RL experiments. Get ready for a playful dive into the world of duck-powered learning! 😎

---

## Getting Started

### Environment Setup 🐍
Start by creating a dedicated Conda environment to keep your dependencies organized and conflict-free. For example, you can run:

    conda create -n duckietown_env python=3.10
    conda activate duckietown_env

Next, install all required Python packages using pip. For a faster installation experience, we recommend using the Chinese University mirror:

    pip install --upgrade pip
    pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/ --use-feature=fast-deps

*Pro tip: Our setup uses optimized dependency resolution ("fast deps") to get you up and running in no time!*

---

## Project Overview

### What’s Inside?
This project is built as a gymnasium for Duckietown, merging the best of legacy support with the flexibility of newer, state-of-the-art reinforcement learning frameworks. Here’s what you can expect:

- **Modern & Modular Architecture:** Designed to let you plug in new ideas with minimal hassle. The system is highly modular, so you can add or modify components rapidly.
- **Dual-Mode Compatibility:** Supports both older interfaces and a modernized version of the gym, allowing seamless integration of stable baselines algorithms and other advanced RL methods.
- **Cross-Platform Functionality:** The simulation engine is built with compatibility in mind, having been tested on both macOS and Linux systems.

---

## Deep Dive: How It Works

### Environment Module
The core of this project is its custom-built environment, which simulates Duckietown with high fidelity. Key features include:

- **Advanced Rendering:** The simulation leverages a robust rendering engine (using pyglet) that supports multiple view modes. Whether you need a human-friendly display or raw image arrays for headless training, the environment adapts to your needs.
- **Customizable Parameters:** From setting random seeds and choosing maps to adjusting camera dimensions and visual effects (like transparency and distortion), every aspect of the simulation is fine-tunable.
- **Modularity at Its Best:** The environment is wrapped in custom modules that abstract away the complexities, making it straightforward to integrate with various reinforcement learning libraries.

### Agent Training Module
The training side of our project is designed to work seamlessly with algorithms like PPO from Stable Baselines 3. Here’s what you should know:

- **Custom Wrappers:** We provide custom wrappers that extend the base environment, adding functionalities like image normalization and tailored preprocessing to optimize learning.
- **Flexible Pipeline:** The setup uses vectorized environments and image transposition wrappers to ensure compatibility with convolutional neural network policies. This means you can train your agents efficiently while experimenting with different architectures.
- **Easy Experimentation:** Adjusting training parameters—whether it’s the policy type, the number of timesteps, or tensorboard logging—can be done with minimal changes to the code. This flexibility is perfect for rapidly testing new ideas and iterating on your experiments.

### Simulation & Rendering Documentation
Even though this README doesn’t include the raw simulation code, here’s what you can expect in the underlying implementation:

- **Simulation Engine:** A robust simulator initializes the Duckietown environment with customizable settings such as random seeds, map selection, step limits, and more.
- **Rendering Options:** Multiple rendering modes are supported, including human-readable graphics and various analytical views (like top-down and free camera perspectives). This allows you to visualize the environment from different angles and debug your RL agent’s behavior effectively.
- **Dynamic Control:** The simulation continuously processes actions, updates the environment state, and renders the results, all within a loop that resets the simulation upon reaching terminal conditions. This design ensures continuous and stable training sessions.

---

## Why You’ll Love It ❤️

- **Speed & Flexibility:** Rapidly prototype new ideas without the overhead of outdated architectures.
- **Plug-and-Play Design:** Easily extend or modify components with a clear, modular codebase.
- **Tested Across Platforms:** Enjoy a smooth experience on both macOS and Linux, removing the “it works on my machine” excuse once and for all.
- **Enhanced Visual Feedback:** With multiple rendering modes, you get an insightful look into your simulation, making debugging and improvements a breeze.

---

## Contributing & Future Enhancements

We’re all about community and collaboration! If you’re passionate about reinforcement learning or have ideas to enhance Duckietown Gymnasium, your contributions are more than welcome. Here are some ways you can help:

- **Feature Enhancements:** Add new simulation features, support for additional algorithms, or improve the user interface.
- **Bug Fixes:** Help us track down and squash any pesky bugs.
- **Documentation:** Contribute to making the documentation even clearer so that new users can hit the ground running.
- **Community Engagement:** Share your experiences, provide feedback, and join discussions on future improvements.

---

## Final Notes

Dive in, experiment, and remember: **Reinforcement Learning is all about iterating until your duck learns to fly!** Embrace the playfulness and creativity of Duckietown Gymnasium as you push the envelope of what’s possible in AI-driven simulations. 🚀

Happy coding, and may your RL adventures be as fun and unpredictable as a flock of ducks on a windy day! 😄🦆