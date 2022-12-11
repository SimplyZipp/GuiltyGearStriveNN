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
#the below offsets and base addresses may not work on a different system. Make sure it doesn't break the code if it can't detect a value (i.e. if you're in the main menu)
def getcurrentstats():
    #these are the values at the current time t, unless it works differently on different systems, don't worry about changing
    p2HP = pm.read_int(GetPtrAddr(pm.base_address + 0x4EC5F38, offsets=[0x130, 0x188, 0x6D8, 0x680, 0x1170])) 
    p1HP = pm.read_int(GetPtrAddr(pm.base_address + 0x4EC5F38, offsets = [0x130, 0x8, 0x2B0, 0x680, 0x1170]))
    p1RISC = pm.read_int(GetPtrAddr(pm.base_address + 0x4EC5F38, offsets = [0x130, 0x188, 0x2A0, 0x6D8, 0xC754]))
    p2RISC = pm.read_int(GetPtrAddr(pm.base_address + 0x4EC5F38, offsets = [0x130, 0x188, 0xC754]))
    #print("healph P1:" + str(p1HP))
    #print("health P2: " + str(p2HP))
    #print("RISC P1: " + str(p1RISC))
    #print("RISC P2: " + str(p2RISC))
    return p1HP, p2HP, p1RISC, p2RISC

def getReward(prev_p1HP, prev_p2HP, prev_p1RISC, prev_p2RISC, p1HP, p2HP, p1RISC, p2RISC, position):
    if(position = "Player 1"): #change to whatever indicator we use to say whether the AI is player 1 or two
        mult = 1
    else:
        mult = -1
    Reward = 0
    #current reward function, can adjust values and add more cases to help improve the behavior of the actor
    Reward -= 20 * (p1HP == 0) * mult 
    Reward += 20 * (p2HP == 0) * mult #+ 20 if you win a round, -20 if you lose
    
    Reward += (prev_p1HP - p1HP)/ 42 * mult 
    Reward -= (prev_p2HP - p2hp) / 42 * mult #+1 point for every 10% damage dealt, max HP is 420
    
    Reward -= max(p1RISC - prev_p1RISC, 0) / 12800 * mult #+1 point per bar of enemy RISC filled. Max RISC value is 12800.  
    Reward += max(p2RISC - prev_p2RISC, 0) /12800 * mult #if RISC goes below zero, then the enemy is being combo'd and not part of this case
    
    Reward -= 2 * (p1RISC == 12800) * mult 
    Reward += 2 * (p2RISC == 12800) * mult #+2 if enemies RISC is full.if the RISC bar is full, really bad things can happen
    
    Reward -= (p1RISC > prev_p1RISC) * (prev_p1RISC == 0)/4 
    Reward += (p2RISC > prev_p2RISC) * (prev_p2RISC == 0)/4 #+.25 for winning neutral or okizeme
    
    Reward += (prev_p1RISC - p1RISC) * (prev_p1HP == p1HP) * (abs(prev_p1RISC - p1RISC) < 1500)) /12800 #gets some reward back for allowing RISC to lower without taking damage
    Reward -= (prev_p2RISC - p2RISC) * (prev_p2HP == p2HP) * (abs(prev_p2RISC - p2RISC) < 1500)) /12800 #if the RISC drops too much, then you likely just got back up after a combo, so no reward should be given
    
    #the expected reward should realistically stay between negative 20 and 20
    return Reward
    



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
