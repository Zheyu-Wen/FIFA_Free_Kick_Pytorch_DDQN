# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

import ctypes
import time
import pyautogui
import cv2
import numpy as np
import pytesseract as pt
from PIL import Image

SendInput = ctypes.windll.user32.SendInput

W = 0x11
Q = 0x10
F = 0x21
spacebar = 0x39
leftarrow = 0xcb
rightarrow = 0xcd
uparrow = 0xc8
downarrow = 0xd0
enter = 0x1c
U = 0x16
J = 0x24
H = 0x23
L = 0x26
E = 0x12

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))



def grab_screen(region,save=0):
    if save==1:
        img = pyautogui.screenshot('screen_shot.png',region=region)
    else:
        img = pyautogui.screenshot(region=region)
    img = np.array(img)
    # left, top, x2, y2 = region
    # width = x2 - left
    # height = y2 - top
    # print(img.shape)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

if __name__ == '__main__':
    pt.pytesseract.tesseract_cmd = 'D:/Program Files/Tesseract-OCR/tesseract'
    reward_screen = grab_screen(region=(1675, 135, 80, 40),save=1)
    reward_image = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    ocr_result = pt.image_to_string(reward_image, config=custom_config)
    # ingame_reward = int(''.join(c for c in ocr_result if c.isdigit()))
    ingame_reward = float(ocr_result)
    print(ingame_reward)
    # i = Image.fromarray(a.astype('uint8'), 'RGB')
    # restart_text = pt.image_to_string(i)
    # if restart_text == "RETRY DRILL":
    #     PressKey(leftarrow)
    #     time.sleep(0.4)
    #     ReleaseKey(leftarrow)
    #     PressKey(enter)
    #     time.sleep(0.4)
    #     ReleaseKey(enter)
    #     time.sleep(2)