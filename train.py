import numpy as np
import time
from pynput import keyboard
import os
from Keyboard_mapping_and_screen_shot import *
from ENV_Create import *
from torchvision.transforms import transforms
from torch import optim
import torch
import torch.nn.functional as F
# parameters
epsilon = .2  # exploration
num_actions = 4  # [ shoot_low, shoot_high, left_arrow, right_arrow]
max_memory = 10000  # Maximum number of experiences we are storing
batch_size = 4  # Number of experiences we use for training per batch



def save_model(model):
    if not os.path.exists("model_epoch1000_zheyu"):
        os.makedirs("model_epoch1000_zheyu")
    # serialize weights to HDF5
    torch.save(model.state_dict(), "model_epoch1000_zheyu/model.pth")

def creterion(pred, target):
    return 2 * F.smooth_l1_loss(pred, target)


def train(game, model, target_net, epochs, verbose=1):
    # Train
    # Reseting the win counter
    win_cnt = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []
    # Epochs is the number of games we play
    flag = 0
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    total_count = 0
    win_cnt = 0
    for e in range(epochs):
        loss_total = 0.
        epsilon = 4 / ((e + 1) ** (1 / 2))
        # Resetting the game
        game.reset()
        game_over = False
        # get tensorflow running first to acquire cudnn handle
        input_t = game.observe()
        if e == 0:
            paused = True
            flag = 1
            print('Training is paused. Press alt.l once game is loaded and is ready to be played.')
        else:
            paused = False
        counting = 0
        while not game_over:
            if not paused:
                # The learner is acting on the last observed game screen
                # input_t is a vector containing representing the game screen
                input_tm1 = input_t

                """
                We want to avoid that the learner settles on a local minimum.
                Imagine you are eating in an exotic restaurant. After some experimentation you find 
                that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian 
                food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
                It's simple: Sometimes, they just don't let you choose but order something random from the menu.
                Maybe you'll like it.
                The chance that your friends order for you is epsilon
                """
                if np.random.rand() <= epsilon:
                    # Eat something random from the menu
                    action = int(np.random.randint(0, num_actions, size=1))
                    print('random action')
                else:
                    # Choose yourself
                    # q contains the expected rewards for the actions
                    input_tm1 = torch.Tensor(input_tm1).float()
                    q = model(input_tm1)
                    # We pick the action with the highest expected reward
                    action = np.argmax(q.data.cpu().numpy())
                counting += 1
                # apply action, get rewards and new state
                input_t, reward = game.act(action)
                total_count += 1
                # If we managed to catch the fruit we add 1 to our win counter
                if reward == 1:
                    win_cnt += 1

                """
                The experiences < s, a, r, s’ > we make during gameplay are our training data.
                Here we first save the last experience, and then load a batch of experiences to train our model
                """

                # store experience
                game.remember([input_tm1, action, reward, input_t])

                # Load batch of experiences
                inputs, targets = game.get_batch(model, target_net, batch_size=batch_size, game_over=game_over)

                # train model on experiences
                inputs = torch.from_numpy(inputs).float()
                targets = torch.from_numpy(targets).float()
                pred = model(inputs)
                loss = creterion(pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(loss)
                loss_total += loss

            # menu control
            keys = []
            if flag == 1:
                with keyboard.Events() as events:
                    print("choose to pause or continue")
                    for event in events:
                        if event.key == keyboard.Key.alt_l:
                            print("continue action")
                            keys.append("C")
                            break
                        if event.key == keyboard.Key.ctrl_l:
                            keys.append("P")
                            print("stop train, wait for ready")
                            break
                    if 'C' in keys:
                        print("continue")
                        paused = False
                        time.sleep(1)
                flag = 0
            if counting==5:
                game_over = True
                target_net.load_state_dict(model.state_dict())
        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {}".format(e, epochs, loss_total, win_cnt))
        save_model(model)
        win_hist.append(win_cnt/total_count)
    return win_hist
