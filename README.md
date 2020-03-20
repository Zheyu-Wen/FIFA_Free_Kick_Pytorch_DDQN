# FIFA_Free_Kick_Pytorch_DDQN
This is pytorch environment Reinforcement Learning algorithm application in FIFA soccer game--free kick skill game.
Mainly used package:
numpy, torch, pytesseract(need tesseract installed), keyboard.

Acknowledgement: https://towardsdatascience.com/using-deep-q-learning-in-fifa-18-to-perfect-the-art-of-free-kicks-f2e4e979ee66.

Thanks for his idea and github code. I start from this code and apply more advanced method.
Maybe in the future, I will also explore to train AI playing true soccer game in FIFA.

Below is the algorithm's detail:
Thanks for this github's detailed illustration https://github.com/fg91/Deep-Q-Learning. My method is basically the same with him but a different application and a different neural network architecture.
DDQN, the method I chose has a basic difference with DQN. Instead of estimating the  Q-values in the next state with only the target network, we use the main network to estimate which action is the best and then ask the target network how high the  Q-value is for that action. This way, the main network will still prefer the action with the small positive Q-value but because of the noisy estimation, the target network will predict a small positive or small negative  Q-value for that action and on average, the predicted Q-values will be closer to 0.

$a=x+b$
