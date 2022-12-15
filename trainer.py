import math
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from memory import Memory

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
    p2HP = pm.read_int(GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x138, 0x8, 0x6E0, 0x688, 0x1178]))
    p1HP = pm.read_int(GetPtrAddr(pm.base_address + 0x0505E6B8, offsets = [0x138, 0x0, 0x6E0, 0x688, 0x1178]))
    p1RISC = pm.read_int(GetPtrAddr(pm.base_address + 0x0505E6B8, offsets = [0x130, 0x8, 0x688, 0xC764]))
    p2RISC = pm.read_int(GetPtrAddr(pm.base_address + 0x0505E6B8, offsets = [0x138, 0x8, 0x2A8, 0xC764]))
    #print("health P1:" + str(p1HP))
    #print("health P2: " + str(p2HP))
    #print("RISC P1: " + str(p1RISC))
    #print("RISC P2: " + str(p2RISC))
    return p1HP, p2HP, p1RISC, p2RISC

def getReward(prev_p1HP, prev_p2HP, prev_p1RISC, prev_p2RISC, p1HP, p2HP, p1RISC, p2RISC, position):
    if(position == "Player 1"): #change to whatever indicator we use to say whether the AI is player 1 or two
        mult = 1
    else:
        mult = -1
    Reward = 0
    #current reward function, can adjust values and add more cases to help improve the behavior of the actor
    Reward -= 20 * (p1HP == 0) * mult
    Reward += 20 * (p2HP == 0) * mult #+ 20 if you win a round, -20 if you lose

    Reward += (prev_p1HP - p1HP)/ 42 * mult
    Reward -= (prev_p2HP - p2HP) / 42 * mult #+1 point for every 10% damage dealt, max HP is 420

    Reward -= max(p1RISC - prev_p1RISC, 0) / 12800 * mult #+1 point per bar of enemy RISC filled. Max RISC value is 12800.
    Reward += max(p2RISC - prev_p2RISC, 0) /12800 * mult #if RISC goes below zero, then the enemy is being combo'd and not part of this case

    Reward -= 2 * (p1RISC == 12800) * mult
    Reward += 2 * (p2RISC == 12800) * mult #+2 if enemies RISC is full.if the RISC bar is full, really bad things can happen

    Reward -= (p1RISC > prev_p1RISC) * (prev_p1RISC == 0)/4
    Reward += (p2RISC > prev_p2RISC) * (prev_p2RISC == 0)/4 #+.25 for winning neutral or okizeme

    Reward += ( (prev_p1RISC - p1RISC) * (prev_p1HP == p1HP) * (abs(prev_p1RISC - p1RISC) < 1500)) /12800 #gets some reward back for allowing RISC to lower without taking damage
    Reward -= ( (prev_p2RISC - p2RISC) * (prev_p2HP == p2HP) * (abs(prev_p2RISC - p2RISC) < 1500)) /12800 #if the RISC drops too much, then you likely just got back up after a combo, so no reward should be given

    #the expected reward should realistically stay between negative 20 and 20
    return Reward




class Trainer:
    def __init__(self, model, lr=1e-3, gamma=0.99, critic_coeff=0.5, entropy_coeff=0.001, save_every=10, device='cpu', dtype=torch.float32):
        self.lr = lr

        self.gamma = gamma
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff

        self.save_every = save_every
        self.iterations = 0

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        self.device = device
        self.dtype = dtype

    def train(self, mem):
        """
        Train the model using the game logs.
        The object logs is not created yet, but it should be an object containing information about how the game
        went, including states, actions, rewards, entropy, log_probs, etc.
        :param mem:
        :return:
        """

        self.optimizer.zero_grad()

        loss = self.calc_a2c_loss(mem)
        loss.backward()

        self.optimizer.step()

        # Reset memory for next game
        mem.reset()

        # Save model once in a while
        if self.iterations % self.save_every == 0:
            # TODO: Save model
            self.iterations = 0
        self.iterations += 1

    def calc_a2c_loss(self, mem):
        """
        Calculate the total loss for A2C. Does not use GAE.
        :return: Scalar loss
        """
        entropy = mem.entropy

        advantage = self.get_discounted_rewards(mem) - torch.tensor(mem.values,
                                                                    device=self.device, dtype=self.dtype)
        critic_loss = torch.pow(advantage, 2).mean()

        # Detach advantage from the computation graph. It only acts as a scalar multiplier
        actor_loss = (-advantage.detach() * torch.tensor(mem.log_probs, device=self.device, dtype=self.dtype)).mean()

        loss = actor_loss + critic_loss * self.critic_coeff + entropy * self.entropy_coeff
        return loss

    def get_discounted_rewards(self, mem):
        """
        Calculate the future discounted rewards for all states
        :return: 1D Tensor
        """
        rewards = torch.zeros(len(mem))
        val = 0
        for i in reversed(range(len(mem))):
            _, _, value, reward = mem[i]
            val = self.gamma*val + reward
            rewards[i] = val
        return rewards
