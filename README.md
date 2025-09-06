# Tetris Reinforcement Learning

Using PyTorch and Pygame, I am creating a Deep Reinforcement Learning Network to learn how to play Tetris. Unlike a classic DQN, which takes in a state and returns the action with the highest Q-Value, I've created the **Afterstate Model**, which parses through all the actions and returns the action which returns the best next state.

## Network Architecture and Motivation

Before, I tried using a classic DQN will be implemented based on [Minh et al.'s](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) paper. However, through multiple efforts and interations of this model, the AI simply wouldn't learn. Through further research, I found that **predicting the best action based on the current tetris grid state was not feasible**. I needed the network to *lookahead*, to observe all actions in the action space to see which yields the best board state going forward. For this, I created the **Afterstate Model**.

The **Afterstate Model** takes in the current board state and outputs the action that gives the best next state. Training looks a little like this:
- **Act**: Based on the current model parameters, the model predicts (what it thinks is) the best action based on the current board state. It does this by trying every action internally to see which action gives the best next board state according to the model.
- **Step**: The AI performs that action and puts new board state and observations into the replay buffer, along with rewards, the next block after queue, and the termination state.
- **Train**: During model training, the model takes a batch of these replay transitions. For each transition in the batch, the target model attempts to find what it thinks is the **next best action**, and applies the **Bellman's Equation** to get the expected return value of the action. It then compares that with the current values of the model, and updates the weights accordingly as a normal DQN would.

Treat this neural network as a mix of a **DQN** paired with an afterstate value function. I am still estimating the best action to choose based on the current state, using a **replay buffer** to train the model, simulating exploration with an epsilon greedy policy, and training a target network for stable updates. The **Afterstate Value Function** is used to determine how favorable each action is.

The loss function I will be using is **Mean Squared Error Loss (MSE Loss)**, and the optimizer I will use is **Adam** optimizer. A list of other hyperparameters that I used are in `src/hyperparameters.yaml`.

Other sources/papers that have helped with the development of this project:
- https://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
- https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

## To-Do List

This is a tentative list of what I want to achieve with this project.

- Train and adjust hyperparameters; use TensorBoard to more effectively diagnose the pitfalls of my network
- Look into Optuna for further hyperparameter tuning.
- Implement the ability to run multiple instances of Tetris at once -- may be useful in training & hyperparameter tuning

## How to run

If you want to run the training process with the DQN, first edit any hyperparameters in the `hyperparams.yaml` file, then run the command:
```bash
python run.py
```

If you want to play the tetris environment yourself, run the command:
```bash
python tetris_env.py
```
