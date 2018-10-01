import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum

import numpy as np
import util
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
        theta = policy
        s_t = GetStateNumber(state[0], state[1], self.dimensions)
        return np.argmax(theta[s_t-1]) + 1
            
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

    def runEpisode(self, policy='uniform'):
        s_t = self.getInitialState()
        incurredReward = 0
        while(not self.isTerminalState(s_t)):
            # self.printBoard(s_t, stateCounter, incurredReward)
            a_t = self.getActionFromPolicy(s_t, policy=policy)
            s_t_1 = self.TransitionFunction(s_t, a_t)
            r_t = self.RewardFunction(s_t, a_t, s_t_1, stateCounter)
            s_t = s_t_1
            incurredReward += r_t
            stateCounter += 1
        # self.printBoard(s_t, stateCounter, incurredReward)
        return incurredReward, simulation_statistics

    def dumpData(self, data, policy):
        pkl.dump(data, open(str(len(data))+"_Episodes_"+policy+".pkl", 'w'))
        with open(str(len(data))+"_Episodes_"+policy+".csv", 'w') as data_file:
            csv_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for idx, val in enumerate(data):
                csv_writer.writerow([str(idx+1), str(val)])
        print "Saving dump to: ", str(len(data))+"_Episodes_"+"_Discount_"+str(self.gamma)+"_"+policy+".csv"
    
    def evaluate(self, theta_k, num_episodes):
        reward = 0
        for episode in range(num_episodes):
            curr_reward, _ = self.runEpisode(policy=theta_k)
            reward += curr_reward
            if episode % num_episodes/10 == 0:
                print "At episode: ", episode
                print "Reward: ", reward
        print "Av Reward: ", reward*1.0/num_episodes
        return reward*1.0/num_episodes

    def learn_policy_bbo(self, init_population, best_ke, num_episodes, epsilon, num_iter):
        assert init_population >= best_ke
        assert num_episodes > 1
        assert epsilon < 5e-4

        curr_iter = 0
        reshape_param = (GetStateNumber(4,3,self.dimensions), len(self.actionSpace)-1)
        theta, sigma = util.get_init(state_space=reshape_param[0],action_space=reshape_param[1])
        while (curr_iter < num_iter):
            values = []
            for k in range(init_population):
                print "At : ",k
                theta_k = util.sample('gaussian', theta, sigma, reshape_param)
                j_k = self.evaluate(theta_k, num_episodes)
                values.append((theta_k, j_k))
            sorted(values, key=lambda x: x[1], reverse=True)
            theta, sigma = util.generate_new_distribution('gaussian', theta, values, best_ke, epsilon)
            curr_iter += 1

        return self.evaluate(theta, num_episodes)

if __name__ == "__main__":
    board = Board(5)
    mdp = MDP(board, 0.8, 0.05, 0.05, 0.1, 0.9)
    mdp.learn_policy_bbo(5, 3, 10, 1e-4, 2)