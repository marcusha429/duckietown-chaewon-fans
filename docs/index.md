# Project Overview
We’re bringing reinforcement learning to DuckieTown🦆 by training a duckiebot to navigate an ever-changing environment filled with stationary and moving obstacles🚧. Our purpose is to showcase how advanced RL algorithms—Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC)—thrive in continuous control tasks requiring pinpoint precision🎯. Our goals include applying theoretical RL in real-world scenarios, pushing the boundaries of safe and efficient deployment, and documenting each step of our progress. Key features include dynamic obstacle avoidance, robust policy learning, and the potential for real-life domain transfer via Sim2Real⚡️. Follow our journey here where our team, **LaneQuakers**, practically applies control and RL to robotics!

# Source Code Repository
[Link to Source Code Repository](https://github.com/marcusha429/duckietown-chaewon-fans/tree/main)

`ppo_project_code` directory: Holds all the main scripts and configurations for training and evaluating a reinforcement learning agent in the **Duckietown-loop_empty** environment. Inside, you’ll find example training files, logs, and job submission scripts demonstrating how to run **Proximal Policy Optimization (PPO)** in a simple DuckieTown loop map—both headlessly or with rendering. It's a self-contained starting point for exploring RL in DuckieTown using PPO🚗!

`sac_project_code` directory: Holds the scripts and configurations for training and evaluating a duckiebot agent using **Soft Actor-Critic (SAC)** in the **Duckietown-loop_empty** map🛣. It includes the main training script **`bs3sac.py`**, environment setup in **`env/duckietown_env.py`**, and core SAC components like the agent (**`sac/agent.py`**), neural network architectures (**`sac/networks.py`**), and replay buffer (**`sac/replay_buffer.py`**). Additionally, **`manual_control.py`** allows for manual robot control, and **`requirements.txt`** lists all the necessary Python dependencies. This project provides a structured framework to experiment with SAC for RL in Duckietown 🚗💡.


# Images and Screenshots
![Duckiebot spinning around itself in a basic looping map](gifs/spinning.gif)
![Simple navigation demo in Duckietown's environment in a map with stationary and moving obstacles](gifs/navigation_demo.gif)

# Relevant Online Resources
### Libraries:
- PyTorch
- OpenAI Gym
- OpenAI gymnasium
- Duckietown Gym
- NumPy
- Matplotlib

### GitHub Repositories:
- [Soft Actor-Critic Implementation](https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py)
- [Duckietown Gym](https://github.com/duckietown/gym-duckietown)

### Paper/Website:
- [Duckietown Model-Based RL Library](https://www.alihkw.com/duckietown-mbrl-lib/)
- [Duckietown Simulator (Running Headless)](https://docs.duckietown.com/ente/devmanual-software/intermediate/simulation/index.html#simulator-running-headless)

### Additional Resources:
- [Duckietown Documentation](https://docs.duckietown.com/)
- [Stable-Baselines3 GitHub](https://github.com/DLR-RM/stable-baselines3)
- [Gym API Documentation](https://www.gymlibrary.dev/)
- TensorBoard for logging
- StackOverflow discussions for Mac virtualization troubleshooting
- Private cloud-based HPC3 nad SLURM job scheduler for training

# Next Steps and Future Improvements
The next steps involve using a Linux laptop for training and testing the models, making sure to have display support. One key focus will be optimizing the hyperparameters of the PPO algorithm to get better performance from the model. To make the setup easier, we can use pre-built reward functions for Duckietown. In the early stages of training, it’s important to run some sanity checks with the simulation to make sure everything is on track. Adding model checkpointing callbacks will help save progress along the way. Looking ahead, we’ll need to build a solid machine learning pipeline that covers training, evaluation, and deployment. Finally, fine-tuning hyperparameters like the learning rate and batch size will be essential for improving the agent’s performance.

We also hope to explore more maps and investigate whether learning in one map can transfer to another. It will be interesting to see if the trained agent is overfitting to a specific environment or if it would perform well in the same environment but with a different random seed. Another goal is to plot the rewards as a function of hyperparameters, especially the batch size and learning rate, since PPO is sensitive to these factors. We’ll also guide our initial training with the simulation to help us better understand the agent’s performance and behavior.