# Tetris Reinforcement Learning

Using pytorch and pygame, I am creating a Deep Q-Learning Network (DQN) to learn how to play Tetris. The tetris game will
be built from scratch using pygame, and the DQN will be created through pytorch and Q-table practices. I will also be
implementing a regular Q-Learning network to compare the two performances on this classic game of tetris.

## Network Architecture and Motivation

DQN will be implemented based on [Minh et al.'s](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) paper. 

It utilizes both a **policy network** and a **target network**. The **policy network** acts as the "Q-Table," being used to select actions during training, evaluation, and testing. It updates frequently to factor in new experiences after each episode. The **target network** is known as the *training stabilizer* as it carries a delayed copy of the policy network to help compute target Q-values in the loss function for training. It updates every *C* episodes, introducing another hyperparameter for the user to test.

Each model will have 2 convolutional layers, one to motivate the detect the current environment state and the other to predict the motion of the falling piece. Then it will converge onto four linear layers to output an action.

The loss function I will be using is **Mean Squared Error Loss (MSE Loss)**, and the optimizer I will use is **Adam** optimizer.

Training will also rely on use of the **Bellman's Equation**, which tells us the value of transitioning into a certain state.

## To-Do List

This is a tentative list of what I want to achieve with this project.

- Complete the Tetris Environment (make sure it is fully functional)
- Setup the reinforcement learning environment; assign rewards and penalties and a way for the network to interact with the game
- Train and adjust hyperparameters

## How to run

If you want to run the training process with the DQN, first edit any hyperparameters in the `hyperparams.yaml` file, then run the command:
```bash
python run.py
```

If you want to play the tetris environment yourself, run the command:
```bash
python tetris_env.py
```