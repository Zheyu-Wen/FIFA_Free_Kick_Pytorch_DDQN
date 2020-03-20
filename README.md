# FIFA_Free_Kick_Pytorch_DDQN
This is pytorch environment Reinforcement Learning algorithm application in FIFA soccer game--free kick skill game.
Mainly used package:
numpy, torch, pytesseract(need tesseract installed), keyboard.

Acknowledgement: https://towardsdatascience.com/using-deep-q-learning-in-fifa-18-to-perfect-the-art-of-free-kicks-f2e4e979ee66.

Thanks for his idea and github code. I start from this code and apply more advanced method.
Maybe in the future, I will also explore to train AI playing true soccer game in FIFA.

Below is the algorithm's detail:
Thanks for this github's detailed illustration https://github.com/fg91/Deep-Q-Learning (in his jupyter notebook). My method is basically the same with him but a different application and a different neural network architecture. 

First is the most important variable called Q-value, which illustrates how good the action is. DDQN, the method I chose has a basic difference with DQN. Instead of estimating the  Q-values in the next state with only the target network, we use the main network to estimate which action is the best and then ask the target network how high the  Q-value is for that action. This way, the main network will still prefer the action with the small positive Q-value but because of the noisy estimation, the target network will predict a small positive or small negative  Q-value for that action and on average, the predicted Q-values will be closer to 0. Below is the DDQN equation.

![image](https://github.com/CoderNoMercy/CoderNoMercy.github.io/blob/master/images/DDQN%20equation.png)

Other details like reply memory：We first set a maximum memory and then to observe state, do prediction, get reward, get next state. To remember these in our memory. If our memory is full, we will replace the oldest one. In training step, it draws a random minibatch from the replay memory to perform a gradient descent step.

Also, trade-off between exploration and exploitation is important. If you just do the action conressponding to the highest score from Q-value, you may stay in a sub-optimized result and never get out of that. So we need some exploration like doing some random choice with a probability of epsilion and ignore the action from Q-value prediction. This helps to lower the loss in our algorithm.


