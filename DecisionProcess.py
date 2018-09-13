import os
import random
from enum import Enum

from util import ActionSpace, GetStateNumber, Cells
from Board import Board

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
        #TODO: Add other options for policies
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
        #TODO: Add probabilities for veering and failure
        tempState = [state[0],state[1]]
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
            print "Action chosen: ", action
            return tempState
        else:
            print "Transitioning to Invalid state, choosing to STAY"
            return state

    def printBoard(self, state):
        print "----------------"
        for i in range(self.dimensions):
            currentRow = ""
            for j in range(self.dimensions):
                if state[0] == i and state[1] == j:
                    currentRow += " 1 "
                elif self.board.grid[i][j] == Cells.Regular:
                    currentRow += " * "
                elif self.board.grid[i][j] == Cells.Obstacle:
                    currentRow += " X "
                elif self.board.grid[i][j] == Cells.Water:
                    currentRow += " O "
                elif self.board.grid[i][j] == Cells.End:
                    currentRow += " [] "
            print currentRow
        print "----------------"

    def RewardFunction(self, s_t, a_t, s_t_1):
        pass 

    def runEpisode(self):
        s_t = self.getInitialState()
        while(not self.isTerminalState(s_t)):
            self.printBoard(s_t)
            a_t = self.getActionFromPolicy(s_t)
            s_t_1 = self.TransitionFunction(s_t, a_t)
            r_t = self.RewardFunction(s_t, a_t, s_t_1)
            s_t = s_t_1
        self.printBoard(s_t)
    
    def learnPolicy(self):
        #TODO: Add policy learning
        pass

if __name__ == "__main__":
    board = Board(5)
    mdp = MDP(board, 0.8, 0.05, 0.05, 0.1)
    mdp.runEpisode()