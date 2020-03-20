import pytesseract as pt
from Network_model import *
from PIL import Image
from Keyboard_mapping_and_screen_shot import *
import numpy as np
import time
import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
class myDataloader(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]

class FIFA(object):
    """
    This class acts as the intermediate "API" to the actual game. Double quotes API because we are not touching the
    game's actual code. It interacts with the game simply using screen-grab (input) and keypress simulation (output)
    using some clever python libraries.
    """

    def __init__(self, max_memory=100000, discount=.9):
        self.feature_map = FeatureMap()
        self.reward = 0
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def _get_reward(self, action):

        reward_screen = grab_screen(region=(1675, 135, 80, 40),save=1)
        try:
            reward_image = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
            custom_config = r'--oem 3 --psm 6 outputbase digits'
            ocr_result = pt.image_to_string(reward_image, config=custom_config)
            ingame_reward = float(ocr_result)
            print('current reward: ' + str(self.reward))
            print('observed reward: ' + str(ingame_reward))
            if ingame_reward - self.reward > 200:
                self.reward = ingame_reward
                ingame_reward = 1
            elif self.reward - ingame_reward == 200:
                # ball hits the Goal bar or post
                ingame_reward = -1
            else:
                # if ball hasn't been shot yet or the game restart
                self.reward = ingame_reward
                ingame_reward = 0
            print('q-learning reward: ' + str(ingame_reward))
        except:
            ingame_reward = -1
            print('exception q-learning reward: ' + str(ingame_reward))
        return ingame_reward

    def observe(self):
        print('\n\nobserve')
        # get current state s from screen using screen-grab
        screen = grab_screen(region=(420, 0, 1080, 1080))
        # if drill over, restart drill and take screenshot again
        restart_button = grab_screen((552, 815, 200, 100))
        buttom_image = Image.fromarray(restart_button.astype('uint8'), 'RGB')
        restart_text = pt.image_to_string(buttom_image)
        if "RETRY DRILL" in restart_text:
            # press enter key
            print('pressing enter, reset reward')
            self.reward = 0
            PressKey(leftarrow)
            time.sleep(0.4)
            ReleaseKey(leftarrow)
            PressKey(enter)
            time.sleep(0.4)
            ReleaseKey(enter)
            time.sleep(2)

        # process through FeatureMap to get the feature map from the raw image
        state = self.feature_map.FeatureExtract(screen)
        return state

    def act(self, action):
        display_action = ['shoot_low', 'shoot_high', 'left_arrow', 'right_arrow']
        print('action: ' + str(display_action[action]))
        keys_to_press = [[spacebar], [spacebar], [leftarrow], [rightarrow]]
        # need to keep all keys pressed for some time before releasing them otherwise fifa considers them as accidental
        # key presses.
        for key in keys_to_press[action]:
            PressKey(key)
        time.sleep(0.05) if action == 0 else time.sleep(0.2)
        for key in keys_to_press[action]:
            ReleaseKey(key)
        # wait until some time after taking action
        if action in [0, 1]:
            time.sleep(5)
        else:
            time.sleep(1)

        reward = self._get_reward(action)
        return self.observe(), reward

    def reset(self):
        self.reward = 0
        return 0

    def remember(self, states):
        # Save a state to memory
        self.memory.append(states)
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, target_net, batch_size=10, game_over=False):

        # How many experiences do we have?
        len_memory = len(self.memory)

        num_actions = 4

        # Dimensions of the game field
        env_dim = self.memory[0][0].shape[1]

        # We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        targets = np.zeros((inputs.shape[0], num_actions))
        model.eval()
        target_net.eval()
        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx]


            # add the state s to the input
            inputs[i:i + 1] = state_t

            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            # state_t = self.transform_numpy(state_t)
            # state_t = Variable(torch.from_numpy(state_t))
            state_t = torch.Tensor(state_t).float()
            with torch.no_grad():
                targets[i] = model(state_t)


            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            state_tp1 = torch.Tensor(state_tp1).float()
            with torch.no_grad():
                action_tp1 = np.argmax(model(state_tp1).numpy())
                Q_sa = target_net(state_tp1).numpy()[0][action_tp1]
            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

