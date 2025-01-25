---
layout: default
title: Proposal
---

# DuckieTown Project Proposal

## Summary of the Project

We decided to work on the **DuckieTown project**. The goal of our project is to develop an intelligence system for the Duckie-Bot, enabling it to self-navigate in the “Town” using reinforcement learning and machine learning. 

We will focus on developing a simulation of DuckieTown and deploying a machine-learning-based solution to train the Duckie-Bot to:
- Detect and follow lanes
- Recognize signals
- Stop and/or avoid collisions with objects such as walls, trees, and buses
- Respond quickly to different factors

Depending on the situation, we will either purchase or build a Duckie-Bot with a camera for sensing data. The Duckie-Bot should be capable of driving (accelerating), stopping, and turning.

This project is exciting and new to us. We believe it will greatly enhance our knowledge and skills. We are committed to coordinating with team members to execute this project in the best possible way.

### Goals and Outcomes
Our primary goal for this project is to gain hands-on, in-depth experience with RL by developing a Duckietown navigation model. We aim to experience the entire lifecycle of a ML project—from theoretical understanding through practical implementation, training, and potential deployment—seeing firsthand how algorithms work in complex environments. This project course is an exciting opportunity for our team to get hands-on experience. We are ambitiously committed to building an intelligent navigation agent from first principles, using theory as our guide.

---

## Methodology and Approach

In our duckie-town navigation project, we've selected the **Soft Actor-Critic (SAC)** method as our primary reinforcement learning approach. This choice stems from several key characteristics that make SAC particularly great for our autonomous duck.

The most appealing advantage of SAC is its exceptional performance in continuous state spaces. Unlike discrete action methods, SAC can handle the dynamic environment of lane navigation with high flexibility. Our miniature self-driving scenario requires an agent that can make smooth, adaptive decisions across a spectrum of potential actions, and SAC excels precisely in this domain.

Entropy regularization is a critical feature that sets SAC apart. This mechanism allows our duckie-bot to maintain exploratory behavior while learning, preventing the model from becoming overly jerky or shallow. SAC ensures that the agent can generalize effectively to unseen environmental conditions, and avoid overcommitting to untested actions.

Compared to other policy gradient methods like Proximal Policy Optimization (PPO), SAC offers some nice properties: It inherently reduces the computational overhead of hyperparameter tuning by dynamically learning the exploration-exploitation trade-off. The alpha parameter, which controls exploration intensity, is automatically adjusted during training, streamlining the learning process. In our project, we aim to be conscious of our compute.

Interpretability was another key consideration in our model selection. We tend to think of reinforcement learning algorithms as "black boxes”. To understanding the decision-making process of duckie-bot, we will employ a variety of interpretable AI methods. For instance, by analyzing the entropy terms, we can gain meaningful understanding of the agent's thought process, enabling us to look into our model and refine its navigation strategies.

If our initial SAC model doesn't perform well, we'll jump-start its learning by pre-training on existing Duckietown navigation data. This gives our model a head start by learning from previous driving experiences.

Soft Actor Critic’s robustness to environmental stochasticity makes it ideal for our dynamic lane navigation scenario. Whether dealing with unexpected obstacles, slight lane variations, or changing environmental conditions, SAC can adapt and maintain consistent performance without requiring extensive retraining.

In sum, SAC represents an optimal mix of **exploration**, **adaptation**, and **interpretable learning** – precisely what our ambitious duckie-bot navigation project demands.

## Evaluation Plan

Our evaluation plan consists of two phases:

### 1. Simulation Evaluation
- Use the DuckieTown dataset for training and testing.
- Metrics include:
  - Lane detection accuracy
  - Collision avoidance
  - Response accuracy to traffic signals

### 2. Real-World Duckie-Bot Evaluation
- Purchase or build a Duckie-Bot (~$150 from an online store).
- Metrics include:
  - Accuracy in lane-following
  - Obstacle avoidance (e.g., buses, walls, houses)
  - Response to traffic lights


Quantitative metrics will serve as our primary benchmarks for performance. We will track key indicators including success rate (percentage of episodes reaching destination), mean lane deviation (average distance from lane center), collision frequency, and action smoothness. These metrics will be computed across multiple experimental trials to ensure robustness. Additionally, we will plot the convergence / training curves of the models to understand the speed and smoothness at which the agent learns.
We will implement targeted interpretability techniques to decode our RL model's decision-making. These will include generating attention maps to visualize critical visual decision triggers, discretizing continuous actions into understandable navigation behaviors, and tracking entropy values to reveal the model's exploration-exploitation dynamics. It should be stressed that our primary concern for our project is getting a deep understanding of how things turned out the way they did.

## Plan to Meet Course Instructor

We will meet with the course instructor at least once a week but ideally twice a week to touch base about the progress of our project.  We will talk about topics such as tips and tricks for training the models,  encoding rewards into the agent, and being smart about how we develop our model in the given time-frame. We would benefit from meeting the instructor / TA in-person should that opportunity arise.
---

## AI Usage

We intend to use the following tools and frameworks:
- **OpenAI Gym**: For reinforcement learning simulations and environments
- **PyTorch**: For implementing and training neural networks
- **TensorFlow**: For experimenting with alternative architectures
- DuckieTown-specific libraries and tools for seamless integration with the simulator and Duckie-Bot.

We'll use LLMs strategically as theoretical explainers and critical review partners. Our approach prioritizes AI as a sophisticated sounding board for testing research ideas, explaining complex concepts, and generating occasional code snippets—always with explicit citation and intent to learn new topics.