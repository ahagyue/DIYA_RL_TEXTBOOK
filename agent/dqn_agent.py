'''

    AUTHOR       : Kang Mingyu (Github : ahagyue)
    DATE         : 2022.09.23
    AFFILIATION  : Seoul National University
    AIM          : DQN agent class for Atari game
    REFERENCE    : Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).

'''

import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

from common.plot import Plots
from common.common import plot_learning_curve as plot
from utils.replay.replayer_interface import ReplayInterface
from function_approximator.q_interface import Qvalue

from IPython.display import clear_output
from typing import Callable, Dict, Tuple

class DqnAgent:
    def __init__(self, 
        env: gym.Env, q_val: Qvalue,
        replay_buffer: ReplayInterface, epsilon: Callable[[int], float],
        args: Dict):
        '''
        env:            environment
        q_val:          Q value function approximator class

        replay_buffer:  replay que which implements ReplayInterface
        epsilon:        used in epsilon-greedy algorithm

        args:
                        USE_GPU
                        GPU_NUM
                        device
                        
                        frame_num
                        learning_rate
                        discount_factor
                        update_duration
        '''

        self.env = env
        self.target_q_val = q_val().eval().to(args["device"])
        self.curr_q_val = q_val().to(args["device"])

        self.replay_buffer = replay_buffer
        self.epsilon = epsilon

        self.args = args

        self.optimizer = torch.optim.Adam(self.curr_q_val.parameters(), lr = self.args["learning_rate"])
        self.save_path = args["model_path"] + args["model_name"] + ".pt"

    
    # copy parameter of curr_q_val to target_q_val
    def copy_model_parameter(self):
        self.target_q_val.load_state_dict(self.curr_q_val.state_dict())

    # return action computed by curr_q_val
    def behavior_policy(self, state, eps: float) -> int:
        state = torch.from_numpy(state).unsqueeze(0).type(torch.FloatTensor).to(self.args["device"])
        return self.curr_q_val.action(state, eps)
    
    def get_replay(self, batch: int) -> Tuple:
        """
        TODO week 1: get observation, action, and reward from replay buffer
                    ( + get the fact that this action was last action of episode)
        """
    
    def compute_loss(self):
        """
        TODO week 1: get replay with self.get_replay(batch) method and compute loss
        """

    def training(self, verbose: bool = True):

        loss_list = []
        reward_sum_list = []
        reward_sum = 0

        observation = self.env.reset()
        for i in range(self.args["frame_num"]):
            if i % self.args["update_duration"] == 0:
                self.copy_model_parameter()

            """
            TODO week 1: implement get action with epsilon-greedy behaviour policy
                         and push it  to replay buffer
            """
            
            # reset environment
            if done:
                reward_sum_list.append(reward_sum)
                reward_sum = 0
                observation = self.env.reset()
            
            
           """
           TODO week 1: implement training code
           """

            if  verbose and (i+1) % 10000 == 0:
                clear_output(wait=True)
                learning_curve = Plots(fig=plt.figure(figsize=(12, 6)), subplot_num=2, position=(1, 2), suptitle="Learning Curve")
                plot(learning_curve, reward_sum_list, loss_list)
                
                torch.save({
                    'iteration': i,
                    'model_state_dict': self.curr_q_val.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss_list,
                    'reward': reward_sum_list
                    }, self.save_path)
    
    def get_action(self, obs):
        return self.curr_q_val.action(obs, 0)

