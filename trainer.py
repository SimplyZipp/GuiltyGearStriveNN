import math
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

#here is the code to ddetect HP values and RISC values
from pymem import *
from pymem.process import *
import time

#Guilty gear preferred memory space 0x7FF7B92B0000
#base offset for health 2 0x4EC5F38
#offsets for health p2 are [0x130, 0x188, 0x6D8, 0x680, 0x1170]


pm = Pymem('GGST-Win64-Shipping.exe')

#from stack overflow https://stackoverflow.com/questions/74348681/overflow-while-using-readwritememory
#this follows the memory addresses the current one. Used to find the memory location of HP and RISC
def GetPtrAddr(base, offsets):
    addr = pm.read_longlong(base)
    for i in offsets:
        if i != offsets[-1]:
            addr = pm.read_longlong(addr + i)
    return addr + offsets[-1]

#good for testing to see if the code works
#the below offsets and base addresses may not work on different system. Make sure it doesn't break the code if it can't detect a value (i.e. if you're in the main menu)
'''while 1:
    #these are the values at the current time t, unless it works differently on different systems, don't worry about changing
    p2HP = pm.read_int(GetPtrAddr(pm.base_address + 0x4EC5F38, offsets=[0x130, 0x188, 0x6D8, 0x680, 0x1170])) 
    p1HP = pm.read_int(GetPtrAddr(pm.base_address + 0x4EC5F38, offsets = [0x130, 0x8, 0x2B0, 0x680, 0x1170]))
    p1RISC = pm.read_int(GetPtrAddr(pm.base_address + 0x4EC5F38, offsets = [0x130, 0x188, 0x2A0, 0x6D8, 0xC754]))
    p2RISC = pm.read_int(GetPtrAddr(pm.base_address + 0x4EC5F38, offsets = [0x130, 0x188, 0xC754]))
    print("healph P1:" + str(p1HP))
    print("health P2: " + str(p2HP))
    print("RISC P1: " + str(p1RISC))
    print("RISC P2: " + str(p2RISC))
    time.sleep(2)'''



class Trainer:
    def __init__(self, lr):
        self.lr = lr

    def train(self, model, logs):
        """
        Train the model using the game logs.
        The object logs is not created yet, but it should be an object containing information about how the game
        went, including states, actions, rewards, entropy, log_probs, etc.
        :param model:
        :param logs:
        :return:
        """
        pass

    def calc_A2C_loss(self):
        """
        Calculate the total loss for A2C. Not sure what it needs yet
        :return:
        """
        pass

    def get_discounted_rewards(self):
        """
        Calculate the future discounted rewards from the current state
        :return:
        """
        pass
