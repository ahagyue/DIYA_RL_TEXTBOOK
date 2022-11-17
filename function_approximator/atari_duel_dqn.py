'''

    AUTHOR       : Kang Mingyu (Github : ahagyue)
    DATE         : 2022.09.23
    AFFILIATION  : Seoul National University
    AIM          : duel deep Q network for Atari game
    REFERENCE    : Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

'''

import torch
import torch.nn as nn

import random

from function_approximator.duel_net import DuelNet
from function_approximator.q_interface import Qvalue

class AtariDuelDQN(Qvalue):
    def __init__(self, frame_size=84, action_number=6):
        """
        TODO week 2: implement duel dqn
        """
    
    def forward(self, x):
        """
        TODO week 2
        """