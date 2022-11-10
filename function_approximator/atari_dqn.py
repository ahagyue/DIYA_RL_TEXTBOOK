'''

    AUTHOR       : Kang Mingyu (Github : ahagyue)
    DATE         : 2022.09.23
    AFFILIATION  : Seoul National University
    AIM          : deep Q network for Atari game
    REFERENCE    : Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).

'''

import torch
import torch.nn as nn

import random

from function_approximator.q_interface import Qvalue

class AtariDQN(Qvalue):
    def __init__(self, frame_size=84, action_number=6):
        super(Qvalue, self).__init__()
        
        """
        TODO week 1: implement q value function approximator (deep q network)
        """
    
    def forward(self, x):
        """
        TODO week 1
        """
    
    def action(self, x, epsilon) -> int:
        if random.random() < epsilon:
            act = random.randint(0, self.action_number - 1)
        else:
            act = self.forward(x).argmax().item()
        return act