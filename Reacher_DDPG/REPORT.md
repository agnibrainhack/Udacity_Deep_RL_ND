# Project 2 : Reacher Project (Continuous Control)

## Project's goal

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, **the goal of the agent is to maintain its position at the target location for as many time steps as possible.**


## Environment details

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). The project environment provided by Udacity is similar to the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment on the Unity ML-Agents GitHub page.

> The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API. 

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

- Set-up: Double-jointed arm which can move to target locations.
- Goal: The agents must move it's hand to the goal location, and keep it there.
- Agents: The Unity environment contains 10 agent linked to a single Brain.
  - The provided Udacity agent versions are Single Agent or 20-Agents
- Agent Reward Function (independent):
  - +0.1 Each step agent's hand is in goal location.
- Brains: One Brain with the following observation/action space.
  - Vector Observation space: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  - Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
  - Visual Observations: None.
- Reset Parameters: Two, corresponding to goal size, and goal movement speed.
- Benchmark Mean Reward: 30

**In my implementation I have chosen to solve the environment using the off-policy DDPG algorithm.** 

# Learning Algorithm
- **DDPG**: The DDPG - Deep deterministic policy gradients - is a model free, off-policy actor-critic algorithm that uses deep neural networks to learn policies in high-dimensional, continuous action spaces. The actor network takes state as input and returns the action whereas the critic network takes state and action as input and returns the value. The critic in this case is a DQN with local and fixed target networks and replay buffer (memory). Both, actor and critic use two neural networks: local and fixed. The local networks are trained by sampling experiences from replay buffer and minimising the loss function.

Solutions are both based on Udacity's sample solution DDPG-pendulum. I took advantage of all online resources to come up with better optimization search. 

### Hyperparameters for the Algorithm

  | Hyperparameter                      | Value |
  | ----------------------------------- | ----- |
  | Replay buffer size                  | 1e6   |
  | Batch size                          | 64  |
  | Gamma                               | 0.99  |
  | Tau                                 | 1e-3  |
  | Learning rate of actor              | 5e-4  |
  | Learning rate of critic             | 1e-3  |
  | Actor_fc1_unit                      | 256   |
  | Actor_fc2_unit                      | action_size   |
  | Critic_fcs1_unit                    | 256   |
  | Critic_fc2_unit                     | 256 + action_size  |
  | Critic_fc3_unit                     | 128  |
  | Add_ou_noise                        | True  |
  | Mu_ou                               | 0     |
  | Theta_ou                            | 0.15  |
  | Sigma_ou                            | 0.2   |
  | Number of episodes                  | 300  |
  | Max number of timesteps per episode | 500  |


## Results

 ![results](plot.jpg)

## Ideas for improvement
I felt that hyperparameter optimization in this archiecture is very very critical and a challenging task, bayesian optimization can be used. A cyclic learning rate can be used in order to periodically incerease and decrease learning rate as there the network gets repeatedly stuck in local minimas for prolonged intervals. 