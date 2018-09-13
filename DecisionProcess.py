import os
import random
from enum import Enum

from util import ActionSpace, GetStateNumber

class MDP():
    
    def __init__(self, board, prob_succ, prob_left_veer, prob_right_veer, prob_fail):
        self.prob_succ = prob_succ
        self.prob_left_veer = prob_left_veer
        self.prob_right_veer = prob_right_veer
        self.prob_fail = prob_fail
        self.board = board
        self.dimensions = board.dimensions
        self.actionSpace = {1:"up", 2:"down", 3:"left", 4:"right"}

    def getInitialState(self):
        return (0, 0)
    
    def isTerminalState(self, state):
        return GetStateNumber(state[0], state[1], self.dimensions) == 23

    def getActionFromPolicy(self, state, policy=0):
        
        if policy is 0:
            #Random Action Policy
            actionNumber = random.randint(1,4)
            action = self.actionSpace[actionNumber]
            return action

    def isValid(self, state):
        if (state[0] < self.dimensions and state[0] >= 0) and (state[1] < self.dimensions and state[1] >= 0):
            return True
        return False

    def TransitionFunction(self, state, action):
        tempState = state
        if action == "up":
            tempState[1] -= 1
        elif action == "down":
            tempState[1] += 1
        elif action == "right":
            tempState[0] += 1
        elif action == "left":
            tempState[0] -= 1
        else:
            print "Invalid Action"
            return state
        
        if self.isValid(tempState):
            return tempState
        else:
            return state

    def printBoard(self, state):
        pass

    def RewardFunction(self, s_t, a_t, s_t_1):
        pass 

    def runEpisode(self):
        state = self.getInitialState()
        while(not self.isTerminalState(state)):
            self.printBoard(state)
            a_t = self.getActionFromPolicy(state)
            s_t_1 = self.TransitionFunction(state, a_t)
            r_t = self.RewardFunction(s_t, a_t, s_t_1)
            state = s_t_1