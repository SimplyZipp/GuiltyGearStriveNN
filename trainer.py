import math
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from memory import Memory

class Trainer:
    def __init__(self, lr=1e-3, gamma=0.99, save_every=10):
        self.lr = lr
        self.gamma = gamma

        self.iterations = 0
        self.save_every = save_every

    def train(self, model, memory):
        """
        Train the model using the game logs.
        The object logs is not created yet, but it should be an object containing information about how the game
        went, including states, actions, rewards, entropy, log_probs, etc.
        :param model:
        :param logs:
        :return:
        """

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        optimizer.zero_grad()

        loss = self.calc_a2c_loss()
        loss.backward()

        optimizer.step()

        memory.reset()  # Reset memory for the next run

        # Save the model once in a while
        self.iterations += 1
        if self.iterations % self.save_every == 0:
            # TODO: Save model
            self.iterations = 0

    def calc_a2c_loss(self, memory):
        """
        Calculate the total loss for A2C. Not sure what it needs yet
        :return:
        """
        # TODO: Implement loss

        pass

    def get_discounted_rewards(self, memory):
        """
        Calculate the future discounted rewards for all states
        :return: 1D Tensor
        """
        rewards = torch.zeros(len(memory))

        val = 0
        for i in reversed(range(len(memory))):
            _, _, value, reward = memory[i]
            val = self.gamma*val + reward
            rewards[i] = val

        return rewards
