---
layout: default
title: Final Report
---

## Video

Embedded video of our project can be found below:

<iframe width="560" height="315" src="https://www.youtube.com/embed/PZG3GnKm9Tc" frameborder="0" allowfullscreen></iframe>

The video includes a brief problem description using images, screenshots, or screen captures, along with examples of a simple policy (e.g., random) and our best-performing approach. The video is under three minutes, recorded in at least 720p resolution, and contains comprehensible speech where applicable.

## Project Summary

Our project focuses on developing an intelligent self-navigation system for DuckieBot within the DuckieTown simulation environment. The DuckieBot is trained using reinforcement learning (RL) techniques to recognize lanes, follow them, stop at traffic signals, and avoid obstacles such as walls, trees, and buses. We have successfully developed a simulation-based environment and are working towards deploying the trained model on a physical DuckieBot.

### Key Objectives:
- Train a DuckieBot using reinforcement learning algorithms to navigate the town autonomously.
- Compare **Soft Actor-Critic (SAC)** and **Proximal Policy Optimization (PPO)** for performance evaluation.
- Deploy the trained model onto an actual DuckieBot for real-world testing.

With recent progress, we successfully ran the DuckieBot in an empty loop environment, demonstrating basic movement capabilities. The next steps involve refining policies for better performance in complex environments.

## Approaches

### SAC Approach
- **State Space:**
  - `(x, z)`: 2D position of the DuckieBot.
  - `sin(θ), cos(θ)`: Orientation of the DuckieBot.
  - `velocity`: Speed of movement.
- **Action Space:**
  - Velocity control: `[-1, 1]` (reverse to forward motion).
  - Steering control: `[-1, 1]` (right to left turn).
- **Reward System:**
  - Small rewards for forward movement.
  - Penalties for deviation from center.
  - Large penalties for collisions.

### PPO Approach
- **Observation:** Uses a CNN-based policy to process image-based inputs.
- **Key Hyperparameters:**
  - Learning rate: `3e-4`
  - Steps per rollout: `1024`
  - Discount factor (γ): `0.99`
  - GAE Lambda: `0.95`
  - Entropy coefficient: `0.01`
- **Training Details:**
  - Conducted over `100,000` timesteps using `make_vec_env` and `VecTransposeImage` for proper input preprocessing.

## Evaluation

### Quantitative Evaluation
Metrics used for evaluation:
- **Mean Episode Reward (`ep_rew_mean`)**: Measures policy effectiveness. Currently, SAC struggles with lane adherence, leading to lower rewards.
- **Mean Episode Length (`ep_len_mean`)**: Shorter episodes indicate frequent collisions.
- **Frames per Second (`fps`)**: Low FPS (<10) affects CNN-based policies relying on visual data.

**Performance Charts:**
![SAC 500](image/sac-500.png)
![SAC 2000](image/sac-2000.png)
![SAC 3000](image/sac-3000.png)
![PPO 1](image/ppo1.png)
![PPO 2](image/ppo2.png)
![PPO 3](image/ppo3.png)

Planned improvements:
- Increase training timesteps to **500,000 - 1M**.
- Optimize hyperparameters.
- Explore **Stable-Baselines3's SAC implementation**.

### Qualitative Evaluation
- **Visual Inspection**: Reviewing recorded runs to assess lane-following behavior.
- **TensorBoard Logs**: Used for tracking policy learning progress.
- **Challenges Identified**:
  - Frame rate issues in high-performance computing (HPC) environments.
  - Poor training results for both PPO and SAC, requiring further refinement.

## References

### Libraries Used
- PyTorch
- OpenAI Gym
- Duckietown Gym
- NumPy
- Matplotlib

### GitHub Repositories
- [Pytorch SAC Implementation](https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py)
- [Duckietown Gym](https://github.com/duckietown/gym-duckietown)

### Papers and Documentation
- [Duckietown MBRL Documentation](https://www.alihkw.com/duckietown-mbrl-lib/)
- [Duckietown Simulation Guide](https://docs.duckietown.com/ente/devmanual-software/intermediate/simulation/index.html#simulator-running-headless)

## AI Tool Usage
We used AI tools in the following ways:
- **Reinforcement Learning Algorithms**: PPO and SAC were implemented using AI/ML frameworks such as PyTorch and Stable-Baselines3.
- **Visualization & Debugging**: TensorBoard and Matplotlib for logging and model performance evaluation.
- **Code Assistance**: Used GPT-based tools for debugging and optimizing hyperparameters.
- **Report Writing**: AI-assisted summarization and structure refinement.

Further refinements will involve tuning hyperparameters and deploying the final trained model onto the DuckieBot for real-world testing.

