import math
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

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