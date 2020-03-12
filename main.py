import numpy as np
import pytesseract as pt
from keras.models import model_from_json
from matplotlib import pyplot as plt
import torch
from ENV_Create import *
from train import train
from test import test
from Network_model import *


def moving_average_diff(a, n=100):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def load_model():

    loaded_model = Feature2Act([1,4096])
    # load weights into new model
    loaded_model.load_state_dict(torch.load("model_epoch1000_zheyu/model.pth "))

    return loaded_model


model = Feature2Act([1, 4096])
target_net = Feature2Act([1, 4096])
target_net.load_state_dict(model.state_dict())
# model = load_model()
# model.summary()

# necessary evil
pt.pytesseract.tesseract_cmd = 'D:/Program Files/Tesseract-OCR/tesseract'
#pt.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.1.1/bin/tesseract' #used for mac checking sanity.

max_memory = 1e4
game = FIFA(max_memory=max_memory)
print("game object created")

epoch = 1000  # Number of games played in training, I found the model needs about 4,000 games till it plays well

train_mode = 1

if train_mode == 1:
    # Train the model
    hist = train(game, model, target_net, epoch, verbose=1)
    print("Training done")
else:
    # Test the model
    hist = test(game, model, epoch, verbose=1)

print(hist)
np.savetxt('win_history.txt', hist)
plt.plot(moving_average_diff(hist))
plt.ylabel('Average of victories per game')
plt.show()
