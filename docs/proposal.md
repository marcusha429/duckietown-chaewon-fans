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

---

## Machine Learning Algorithms

For this project, we anticipate using the following methods:
- **Deep Q-Networks (DQN)**
- **Deep Deterministic Policy Gradient (DDPG)**
- **Proximal Policy Optimization (PPO)**
- **Deep Convolutional Neural Networks (DCN)**

These algorithms will focus on:
- Reinforcement learning with model-free, off-policy approaches for control tasks
- Leveraging DCNs for perception tasks

---

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

---

## Plan to Meet Course Instructor

We plan to meet the course instructor at the following milestones:
- **Week 3**: Discuss and get suggestions for the project plan
- **Week 5**: Mandatory check-in
- **Week 9**: Mandatory final review

---

## AI Usage

We intend to use the following tools and frameworks:
- **OpenAI Gym**: For reinforcement learning simulations and environments
- **PyTorch**: For implementing and training neural networks
- **TensorFlow**: For experimenting with alternative architectures
- DuckieTown-specific libraries and tools for seamless integration with the simulator and Duckie-Bot