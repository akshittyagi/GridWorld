import os
from enum import Enum

from util import ActionSpace, GetStateNumber

class MDP():
    
    def __init__(self, board, prob_succ, prob_left_veer, prob_right_veer, prob_fail):
        self.prob_succ = prob_succ
        self.prob_left_veer = prob_left_veer
        self.prob_right_veer = prob_right_veer
        self.prob_fail = prob_fail
        self.board = board

    def getInitialState(self):
        return (0, 0)
    
    def isTerminalState(self, state):
        return GetStateNumber(state[0], state[1]) == 23

    def getActionFromPolicy(self, state):
        pass

    def TransitionFunction(self, state, action):
        pass

    def RewardFunction(self, s_t, a_t, s_t_1):
        pass 

    def runEpisode(self):
        state = self.getInitialState()
        while(not self.isTerminalState(state)):
            a_t = self.getActionFromPolicy(state)
            s_t_1 = self.TransitionFunction(state, a_t)
            r_t = self.RewardFunction(s_t, a_t, s_t_1)