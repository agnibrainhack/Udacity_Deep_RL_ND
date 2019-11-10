[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, agent navigates (and collect bananas!) in a large,square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Create a new virtual python environment in your system and then create a new directory.

2. Copy the entire folder from [this link](https://github.com/agnibrainhack/deep-reinforcement-learning/tree/master/python) to obtain the necessary list of packages and environments and place it within the newly created folder.

3. Also place the ` Solution.ipynb, model.py and dqn_agent.py ` files in this newly created folder. Run the first cell of the notebook to succesfully buid the environment and the packages involved in the process.

4. All cells in the notebook are accompanied by two python files
   ` model.py ` is the core architecture of the neural network and ` dqn_agent.py ` is the agent creation and replay buffer establishment code.