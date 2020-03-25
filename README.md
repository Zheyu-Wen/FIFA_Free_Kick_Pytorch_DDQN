# FIFA_Free_Kick_Pytorch_DDQN
#Introduction:

This repository is for final project of EECS 545 WN.
It is a pytorch environment Reinforcement Learning algorithm application in FIFA soccer game--free kick skill game.

Mainly used package:

numpy, torch, pytesseract(need tesseract installed), keyboard.

Acknowledgement:
https://towardsdatascience.com/using-deep-q-learning-in-fifa-18-to-perfect-the-art-of-free-kicks-f2e4e979ee66.

Thanks for his idea and github code. I start from this code and apply more advanced method. After I finished this project, I also tried to program for the whole game in FIFA, people who are interested in that can see the code in my another repository https://github.com/CoderNoMercy/FIFA_PLAY_WHOLE_GAME

# Algorithm Introduction

Thanks for this github's detailed illustration https://github.com/fg91/Deep-Q-Learning (in his jupyter notebook). My method is basically the same with him but a different application and a different neural network architecture. 

The method I used is DDQN, which needs the environment to return reward, observation of the game and according this judging what action to choose in action space. There is an important standard called Q-value that illustrates how good the action is. DDQN, the method I chose has a basic difference with DQN. Instead of estimating the  Q-values in the next state with only the target network, we use the main network to estimate which action is the best and then ask the target network how high the  Q-value is for that action. 

![image](https://github.com/CoderNoMercy/CoderNoMercy.github.io/blob/master/images/DDQN%20equation.png)

Other details like reply memory：We first set a maximum memory and then to observe state, do prediction, get reward, get next state. Remember these in our memory. If our memory is full, we will replace the oldest one. In training step, it draws a random minibatch from the replay memory to perform a gradient descent step.

Also, trade-off between exploration and exploitation is important. If you just do the action conressponding to the highest score from Q-value, you may stay in a sub-optimized result and never get out of that. So we need some exploration like doing some random choice with a probability of epsilion and ignore the action from Q-value prediction. This helps to lower the loss in our algorithm.

# How to use
First you should have a FIFA18 game in your windows computer. Since our program has some function that can only be used in windows system. You should do some work to change the code if you want to apply this repo to your own game on other system.

You also have to change your key-action in game. Use leftarrow to control left, rightarrow control right and space control shot.

For quick begin, you can just type

``` python
python main.py
```
After the screen shows "please type alt_l and continue", you should type as instructed after you load game and begin the FIFA free kick game.

# What's the result

There should be a plot showing the rate of winning during training. You may find that the AI can play free kick much better than not training.

The training will continue for at least 8 hours or so.

# File instruction

ENV_Creature.py is the most important part in this project since it establish an env for the game under condition that we don't have access to the game code.

Keyboard_mapping_and_screen_shot.py is doing a mapping from keyboard to system, which lets AI to play for you through keyboard

Network_model.py build network for training. In DDQN, we need two network, one is target nework and another is main network. They share the same structure. Main network gives appropriate action and target network returns Q-value.

test.py is for testing after training.

train.py is for training. It is default set in main.py

