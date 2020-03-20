import time
import torchvision
import torch
import numpy as np
from pynput import keyboard
# parameters
# epsilon = .2  # exploration
num_actions = 4  # [ shoot_low, shoot_high, left_arrow, right_arrow]
max_memory = 500  # Maximum number of experiences we are storing
hidden_size = 100  # Size of the hidden layers
batch_size = 1  # Number of experiences we use for training per batch
grid_size = 10  # Size of the playing field
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(game, model, n_games, verbose=1):
    # Train
    # Reseting the win counter
    win_cnt = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []
    flag = 0
    model = model.to(device)
    # Epochs is the number of games we play
    for e in range(n_games):
        # Resetting the game
        game.reset()
        game_over = False
        # get tensorflow running first to acquire cudnn handle
        input_t = game.observe()
        if e == 0:
            paused = True
            flag = 1
            print('Training is paused. Press p once game is loaded and is ready to be played.')
        else:
            paused = False
        counting = 0
        win_cnt = 0
        total_count = 0
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

                # Choose yourself
                # q contains the expected rewards for the actions
                print("begin predict q value")
                input_tm1 = torch.Tensor(input_tm1).float()
                with torch.no_grad():
                    q = model(input_tm1)
                # We pick the action with the highest expected reward
                print('q values=' + str(q[0]))
                action = np.argmax(q[0])
                counting += 1
                # apply action, get rewards and new state
                input_t, reward, game_over = game.act(action)
                total_count += 1
                # If we managed to catch the fruit we add 1 to our win counter
                if reward == 1:
                    win_cnt += 1

                """
                The experiences < s, a, r, s’ > we make during gameplay are our training data.
                Here we first save the last experience, and then load a batch of experiences to train our model
                """

            keys = []
            if flag == 1:
                with keyboard.Events() as events:
                    print("choose to pause or continue")
                    for event in events:
                        if event.key == keyboard.Key.alt_l:
                            keys.append("C")
                            print("continue")
                            break
                if 'C' in keys:
                    print('unpausing!')
                    paused = False
                    time.sleep(1)
                flag = 0
            if counting == 5:
                game_over = True
        if verbose > 0:
            print("Game {:03d}/{:03d} | Win count {}".format(e, n_games, win_cnt))
        win_hist.append(win_cnt/total_count)
    return win_hist
