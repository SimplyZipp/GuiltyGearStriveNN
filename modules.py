import math
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions import Categorical


def init(module, weight_init, bias_init):
    """
    Initialize a module using the given initializers
    :param module: the module to initialize
    :param weight_init: method to initialize the weights. From torch.nn.init
    :param bias_init: method to initialize the bias. From torch.nn.init
    :return: Initialized module
    """
    weight_init(module.weight.data)
    bias_init(module.bias.data)
    return module


def get_feature_size(input_size, channel_in):
    """

    :param input_size: Input image size
    :param channel_in:
    :return: Feature size
    """
    with torch.no_grad:  # Just in case
        x = torch.zeros((channel_in, *input_size))
        fe = FeatureExtractor(channel_in)
        x = fe(x)
        return x.shape


class ConvBlock(nn.Module):
    """
    Convolutional module for feature encoding
    """
    def __init__(self, channel_in, device='cpu', dtype=torch.float32):
        """
        :param channel_in: The number of channels (or stack frames) for the input
        """
        super().__init__()

        init_ = lambda x: x

        self.conv128 = nn.Sequential(
            init_(nn.Conv2d(channel_in, 128, 3, padding=1, device=device, dtype=dtype)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(128, 128, 3, stride=2, padding=1, device=device, dtype=dtype)),
            nn.LeakyReLU()
        )
        self.conv64 = nn.Sequential(
            init_(nn.Conv2d(128, 64, 3, padding=1, device=device, dtype=dtype)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=2, padding=1, device=device, dtype=dtype)),
            nn.LeakyReLU()
        )
        self.conv32 = nn.Sequential(
            init_(nn.Conv2d(64, 32, 3, padding=1, device=device, dtype=dtype)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1, device=device, dtype=dtype)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv128(x)
        x = self.conv64(x)
        x = self.conv32(x)

        return torch.flatten(x)


class FeatureExtractor(nn.Module):
    """
    Module for feature encoding
    """
    def __init__(self, channel_in, device='cpu', dtype=torch.float32):
        super().__init__()

        self.conv = ConvBlock(channel_in, device=device, dtype=dtype)

    def forward(self, x):
        """

        :param x: Tensor of shape (C_in, W, H)
        :return: Tensor of shape (32 * W_out * H_out)
        """
        x = self.conv(x)
        return x


class Actor(nn.Module):
    """
    Actor module for Actor-Critic
    """
    def __init__(self, in_size, nd_actions, hidden_size=128, device='cpu', dtype=torch.float32):
        """
        :param in_size: Input size of state representation
        :param nd_actions: Tuple (D, N) of N actions and D dimensions
        :param hidden_size: Size of the lstm hidden state (default: 128)
        """
        super().__init__()

        self.nd_actions = nd_actions

        self.h0 = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.c0 = torch.zeros(hidden_size, dtype=dtype, device=device)

        # self.h0.requires_grad=True  # Learn the initial state
        # self.c0.requires_grad=True  # Learn the initial state

        self.LSTM = nn.LSTMCell(in_size, hidden_size, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, nd_actions[1], device=device, dtype=dtype)

    def forward(self, x):
        """

        :param x: State representation tensor of shape (, in_size)
        :return: Action values tensor of shape (D, N). The actions (i, N)
                 are conditioned on the actions (i-1, N)
        """
        # hh = self.h0
        # cc = self.c0
        action_probs = torch.zeros(self.nd_actions, device=self.h0.device, dtype=self.h0.dtype)

        for i in range(self.nd_actions[0]):
            self.h0, self.c0 = self.LSTM(x, self.h0, self.c0)
            action_probs[i] = self.linear(self.h0)

        return action_probs


class Critic(nn.Module):
    def __init__(self, in_size, device='cpu', dtype=torch.float32):
        super().__init__()

        # Can be modified
        self.model = nn.Sequential(
            nn.Linear(in_size, 1, device=device, dtype=dtype)
        )

    def forward(self, x):
        return self.model(x)


class A2C(nn.Module):
    def __init__(self, image_size, channel_in, nd_actions, actor_hidden_size=128, device='cpu', dtype=torch.float32):
        super().__init__()

        feature_size = get_feature_size(image_size, channel_in)

        self.encoder = FeatureExtractor(channel_in, device=device, dtype=dtype)
        self.actor = Actor(feature_size, nd_actions, actor_hidden_size, device=device, dtype=dtype)
        self.critic = Critic(feature_size, device=device, dtype=dtype)

    def forward(self, state):
        features = self.encoder(state)

        policy = self.actor(features)
        value = self.critic(features)

        print(f"A2C Value output shape: {value.shape}")

        return policy, value

    def get_action(self, state):
        """

        :param state: Input state
        :return: (action, log_prob, entropy, state value)
        """
        policy, value = self(state)

        action_probs = F.softmax(policy, dim=1)
        dist = Categorical(action_probs)

        action = dist.sample()

        # Normal entropy. Try implementing Smoothed Entropy from https://arxiv.org/pdf/1806.00589.pdf
        entropy = dist.entropy()

        return action, dist.log_prob(action), entropy, value

