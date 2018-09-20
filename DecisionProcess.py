import os
import sys
import random
import pickle as pkl
import csv
from enum import Enum

import numpy as np
from util import ActionSpace, GetStateNumber, Cells
from Board import Board

class MDP():
    
    def __init__(self, board, prob_succ, prob_left_veer, prob_right_veer, prob_fail, gamma):
        self.prob_succ = prob_succ
        self.prob_left_veer = prob_left_veer
        self.prob_right_veer = prob_right_veer
        self.prob_fail = prob_fail
        self.board = board
        self.dimensions = board.dimensions
        self.actionSpace = {1:"up", 2:"right", 3:"down", 4:"left", 5:"stay"}
        self.gamma = gamma

    def getInitialState(self):
        return (0, 0)
    
    def isTerminalState(self, state):
        return GetStateNumber(state[0], state[1], self.dimensions) == 23

    def getActionFromPolicy(self, state, policy='uniform'):
        #TODO: Add other options for policies
        if policy is 'uniform':
            '''
            Random Action Policy
            '''
            actionNumber = random.randint(1,4)
            return actionNumber
        elif policy is 'optimal1':
            '''
            Hand fashioned optimal Policy
            '''
            s_t = GetStateNumber(state[0], state[1], self.dimensions)
            rightSet = [1,2,3,4,6,7,8,9,17,19,20,21,22]
            downSet = [5,10,13,14,15,16,18]
            upSet = [11,12]
            if s_t in rightSet:
                #Go right
                return 2
            elif s_t in downSet:
                #Go down
                return 3
            elif s_t in upSet:
                #Go up
                return 1
            elif self.isValid(state):
                print "State, Action mapping missing"
                return 5
            else:
                print "Returning random Action"
                return random.randint(1,4)
        elif policy is 'optimal2':
            '''
            Hand fashioned optimal Policy
            '''
            s_t = GetStateNumber(state[0], state[1], self.dimensions)
            rightSet = [1,2,3,4,6,7,8,9,17,19,20,21,22]
            downSet = [5,10,13,14,18]
            upSet = [15,16,11,12]
            if s_t in rightSet:
                #Go right
                return 2
            elif s_t in downSet:
                #Go down
                return 3
            elif s_t in upSet:
                #Go up
                return 1
            elif self.isValid(state):
                print "State, Action mapping missing"
                return 5
            else:
                print "Returning random Action"
                return random.randint(1,4)
        elif policy is 'goRight':
            return 2
        
    def isValid(self, state):
        if (state[0] < self.dimensions and state[0] >= 0) and (state[1] < self.dimensions and state[1] >= 0) and not(state[0] == 2 and state[1] == 2) and not(state[0] == 3 and state[1] == 2):
            return True
        return False

    def affectWithProbability(self, action, effect):
        if effect == "same":
            return self.actionSpace[action]
        elif effect == "veer left":
            if action == 1:
                return self.actionSpace[4]
            else:
                return self.actionSpace[action - 1]
        elif effect == "veer right":
            if action == 4:
                return self.actionSpace[1]
            else:
                return self.actionSpace[action + 1]
        elif effect == "stay":
            return self.actionSpace[5]

    def rollTheDice(self):
        # TODO: Change probability handler to floats in (0,1)
        proba = random.randint(1,100)
        effect = ""
        if proba <= 80:
            effect = "same"
        elif proba >= 81 and proba <= 85:
            effect = "veer left"
        elif proba >= 86 and proba <= 90:
            effect = "veer right"
        elif proba >= 91:
            effect = "stay"
        return effect, proba

    def TransitionFunction(self, state, action):
        # print "Coming with ACTION: ", self.actionSpace[action], " and at STATE: ", state[0], state[1], " STATE NUMBER ", GetStateNumber(state[0], state[1], self.dimensions)
        effect, proba = self.rollTheDice()
        # print "Effect and Probability val: ", effect, proba
        action = self.affectWithProbability(action, effect)
        tempState = [state[0],state[1]]
        if action == "up":
            tempState[0] -= 1
        elif action == "down":
            tempState[0] += 1
        elif action == "right":
            tempState[1] += 1
        elif action == "left":
            tempState[1] -= 1
        elif action == "stay":
            action = 'stay'
            # print "Choosing to stay because of failure"
        else:
            # print "Invalid Action"
            return state
        
        if self.isValid(tempState):
            # print "Action chosen: ", action
            return tempState
        else:
            # print action, " Transitioning to Invalid state, choosing to STAY"
            return state

    def printBoard(self, state, stateCounter=-1, reward=-1):
        # print "\nSTATE: ", stateCounter
        # print "----------------"
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
        # if reward != -1:
        #     print "REWARD incurred: ", reward
        # print "----------------\n"

    def RewardFunction(self, s_t, a_t, s_t_1, time_step=0):
        '''
        Reward for taking action a_t in state s_t to get into state s_t_1
        '''
        reward = 0
        before = GetStateNumber(s_t[0], s_t[1], self.dimensions)
        after = GetStateNumber(s_t_1[0], s_t_1[1], self.dimensions)
        if before == after and before == 21:
            reward -= 10*(self.gamma**time_step)
        elif self.isTerminalState(s_t_1):
            reward += 10*(self.gamma**time_step)
        return reward

    def runEpisode(self, policy='uniform',simulation_statistics=[]):
        s_t = self.getInitialState()
        incurredReward = 0
        stateCounter = 0
        history = False
        while(not self.isTerminalState(s_t)):
            # self.printBoard(s_t, stateCounter, incurredReward)
            a_t = self.getActionFromPolicy(s_t, policy=policy)
            s_t_1 = self.TransitionFunction(s_t, a_t)
            r_t = self.RewardFunction(s_t, a_t, s_t_1, stateCounter)
            if stateCounter == 8 and GetStateNumber(s_t[0], s_t[1], self.dimensions) == 18:
                simulation_statistics[0] += 1
                history = True
            if stateCounter == 19 and GetStateNumber(s_t[0], s_t[1], self.dimensions) == 21:
                simulation_statistics[1] += 1
                if history == True:
                    simulation_statistics[2] += 1
                    history = False
            s_t = s_t_1
            incurredReward += r_t
            stateCounter += 1
        #     print "Reward: ", incurredReward
        # self.printBoard(s_t, stateCounter, incurredReward)
        print "Total Reward: ", incurredReward
        return incurredReward, simulation_statistics

    def dumpData(self, data, policy):
        pkl.dump(data, open(str(len(data))+"_Episodes_"+policy+".pkl", 'w'))
        with open(str(len(data))+"_Episodes_"+policy+".csv", 'w') as data_file:
            csv_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for idx, val in enumerate(data):
                csv_writer.writerow([str(idx+1), str(val)])
        print "Saving dump to: ", str(len(data))+"_Episodes_"+"_Discount_"+str(self.gamma)+"_"+policy+".csv"
    
    def learnPolicy(self, num_episodes=100, policy="uniform",plain_text_save=True):
        print "USING POLICY: ", policy
        data = []
        simulation_statistics = [0]*3
        total_reward = 0
        for episode in range(num_episodes):
            print "At episode: ", episode
            reward, simulation_statistics = self.runEpisode(policy,simulation_statistics=simulation_statistics)
            total_reward += reward
            data.append(reward)
        if plain_text_save:
            file_writer = open(str(len(data))+"_Episodes_"+policy+".txt", 'w')
        file_str = ""
        file_str += "\n----------------------------"
        file_str += "\nAverage Reward= " +  str(total_reward*1.0/num_episodes)
        file_str += "\nMax Reward= " + str(max(data))
        file_str += "\nMin Reward= " + str(min(data))
        stddev = np.array(data) - (total_reward*1.0/num_episodes)
        stddev = stddev**2
        file_str += "\nStddev Reward= " + str(np.sqrt(np.sum(stddev))/len(stddev))
        file_str += "\nPr(S_8=18)= " + str(1.0*simulation_statistics[0] / num_episodes)
        file_str += "\nPr(S_19=21)= " + str(1.0*simulation_statistics[1] / num_episodes)
        if simulation_statistics[0]!=0:
            file_str += "\nPr(S_19=21|S_8=18)= " + str(1.0*simulation_statistics[2] / simulation_statistics[0]) 
        file_str += "\n----------------------------"
        print file_str
        if plain_text_save:
            file_writer.write(file_str)
            print "Saving plain text stats to: ", str(len(data))+"_Episodes_"+"_Discount_"+str(self.gamma)+"_"+policy+".txt"
        self.dumpData(data, policy)
    
        
if __name__ == "__main__":
    board = Board(5)
    mdp = MDP(board, 0.8, 0.05, 0.05, 0.1, 0.9)
    mdp.learnPolicy(num_episodes=10000, policy='optimal1')