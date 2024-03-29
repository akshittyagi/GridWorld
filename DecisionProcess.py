import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum
import multiprocessing
from multiprocessing import Pool
import time

import numpy as np
import util
from util import ActionSpace, GetStateNumber, Cells
from Board import Board

class MDP(object):
    
    def __init__(self, board=Board(5), prob_succ=0.8, prob_left_veer=0.05, prob_right_veer=0.05, prob_fail= 0.1,gamma=0.9, debug=False):
        self.prob_succ = prob_succ
        self.prob_left_veer = prob_left_veer
        self.prob_right_veer = prob_right_veer
        self.prob_fail = prob_fail
        self.board = board
        self.dimensions = board.dimensions
        self.actionSpace = {1:"up", 2:"right", 3:"down", 4:"left", 5:"stay"}
        self.gamma = gamma
        self.debug = debug
        self.data = []
        self.max_av_reward = -2**31

    def init_q_function(self):
        self.q_vals = []
        for state in range(GetStateNumber(4, 4, self.dimensions) + 1):
            actions = np.random.uniform(0, 100, 3)
            self.q_vals.append(actions)
        return self.q_vals
    
    def init_e_function(self):
        self.e_vals = []
        for state in range(GetStateNumber(4, 4, self.dimensions) + 1):
            actions = [0,0,0]
            self.e_vals.append(actions)
        tmp = self.e_vals
        return tmp
    
    def getActionId(self, a_t):
        return a_t - 2

    def initValueFunction(self):
        self.states = [0]*(GetStateNumber(4, 4, self.dimensions) + 1)
        return self.states
    
    def getStateId(self, state):
        return GetStateNumber(state[0], state[1], self.dimensions)

    def getInitialState(self):
        return (0, 0)
    
    def isTerminalState(self, state):
        return GetStateNumber(state[0], state[1], self.dimensions) == 23

    def getActionFromPolicy(self, state, policy='uniform'):
        if isinstance(policy, str) and policy == 'uniform':
            return random.randint(1,4)
        else:
            theta = policy
            s_t = GetStateNumber(state[0], state[1], self.dimensions)
            currRow = theta[s_t-1]
            random_number = 1.0*random.randint(0,99)/100
            action_array = sorted(zip(np.arange(len(currRow)), currRow), key=lambda x: x[1], reverse=True)
            prev_proba = 0
            for action, probability in action_array:
                prev_proba += probability
                if random_number <= prev_proba:
                    if self.debug:
                        print "Action Array: ", action_array
                        print "Rand number: ",random_number
                        print "Action selected: ", action + 2    
                    return action + 2
            
            if self.debug:
                print "!!!!!!!!! NOT RETURNING ANYTHING !!!!!!!!!"
                print "Action Array: ", action_array
                print "Rand number: ",random_number
                print "Action selected: ", "NOTHING"
            
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
        if self.debug:
            print "Coming with ACTION: ", self.actionSpace[action], " and at STATE: ", state[0], state[1], " STATE NUMBER ", GetStateNumber(state[0], state[1], self.dimensions)
        effect, proba = self.rollTheDice()
        if self.debug:
            print "Effect and Probability val: ", effect, proba
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
            if self.debug:
                print "Choosing to stay because of failure"
        else:
            if self.debug:
                print "Invalid Action"
            return state
        
        if self.isValid(tempState):
            if self.debug:
                print "Action chosen: ", action
            return tempState
        else:
            if self.debug:
                print action, " Transitioning to Invalid state, choosing to STAY"
            return state

    def printBoard(self, state, stateCounter=-1, reward=-1):
        if self.debug:
            print "\nSTATE: ", stateCounter
        if self.debug:
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
        if self.debug:
            if reward != -1:
                print "REWARD incurred: ", reward
            print "----------------\n"

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

    def dumpData(self, data, policy):
        pkl.dump(data, open(str(len(data))+"_Episodes_"+policy+".pkl", 'w'))
        with open(str(len(data))+"_Episodes_"+policy+".csv", 'w') as data_file:
            csv_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for idx, val in enumerate(data):
                csv_writer.writerow([str(idx+1), str(val)])
        print "Saving dump to: ", str(len(data))+"_Episodes_"+"_Discount_"+str(self.gamma)+"_"+policy+".csv"

    def runEpisode(self, policy='uniform'):
        s_t = self.getInitialState()
        incurredReward = 0
        stateCounter = 0
        while(not self.isTerminalState(s_t) and stateCounter<1500):
            if self.debug:
                self.printBoard(s_t, stateCounter, incurredReward)
            a_t = self.getActionFromPolicy(s_t, policy=policy)
            s_t_1 = self.TransitionFunction(s_t, a_t)
            r_t = self.RewardFunction(s_t, a_t, s_t_1, stateCounter)
            s_t = s_t_1
            incurredReward += r_t
            stateCounter += 1
            if stateCounter % 5000 == 0 and self.debug:
                print "At state: ", stateCounter-1
                print "Reward in current episode: ", incurredReward 
        if self.debug:
            self.printBoard(s_t, stateCounter, incurredReward)
        return incurredReward

    def evaluate(self, theta_k, num_episodes):
        reward = 0
        for episode in range(num_episodes):
            curr_reward  = self.runEpisode(policy=theta_k)
            reward += curr_reward
            if episode % num_episodes/10 == 0 and self.debug:
                print "At episode: ", episode
                print "Reward: ", reward
        if self.debug:
            print "Av Reward: ", reward*1.0/num_episodes
        return reward*1.0/num_episodes

    def iterable(self, array):
        for elem in array:
            yield elem
    
    # Only actions being considered are 1:up, 2:right, 3:down
    def learn_policy_bbo_multiprocessing(self, init_population, best_ke, num_episodes, epsilon, num_iter, steps_per_trial=15, variance=100):
        assert init_population >= best_ke
        assert num_episodes > 1
        curr_iter = 0
        reshape_param = (GetStateNumber(4,3,self.dimensions), len(self.actionSpace)-3)
        data = []
        theta_max = []
        max_av_reward = -2**31
        while (curr_iter < num_iter):
            theta, sigma = util.get_init(state_space=reshape_param[0],action_space=reshape_param[1], sigma=variance)
            for i in range(steps_per_trial):
                values = []
                print "-----------------------------"
                print "At ITER: ", curr_iter
                print "AT step: ", i
                theta_sampled= util.sample('gaussian', theta, sigma, reshape_param, init_population)
                theta_sampled = variance*theta_sampled
                softmax_theta = np.exp(theta_sampled)
                tic = time.time()
                pool = Pool(multiprocessing.cpu_count())
                mp_obj = multiprocessing_obj(num_episodes)
                values = pool.map(mp_obj, self.iterable(softmax_theta))
                data.append(np.array(values)[:,1].tolist())
                pool.close()
                pool.join()
                toc = time.time()
                values = sorted(values, key=lambda x: x[1], reverse=True)
                print "Max reward: ", values[0][1]
                if max_av_reward < values[0][1]:
                    max_av_reward = values[0][1]
                    print "MAX REWARD UPDATED"
                    theta_max = values[0][0]
                theta, sigma = util.generate_new_distribution('gaussian', theta, values, best_ke, epsilon)
                print "-----------------------------"
            curr_iter += 1
        print "Saving data"
        pkl.dump(data, open("FILE.pkl", 'w'))
        pkl.dump(theta_max, open("THETA.pkl", 'w'))
        

    #TODO: Figure out a parallelizing strategy for FCHC
    def learn_policy_fchc_multiprocessing(self, num_iter, steps_per_trial, sigma, num_episodes):
        reshape_param = (GetStateNumber(4,3,self.dimensions), len(self.actionSpace)-1)
        curr_iter = 0
        while curr_iter < num_iter:
            theta, _ = util.get_init(state_space=reshape_param[0], action_space=reshape_param[1], sigma=sigma)
            j = self.evaluate(theta, num_episodes)
            for i in range(steps_per_trial):
                theta_sampled = util.sample(distribution='gaussian', theta=theta, sigma=sigma, reshape_param=reshape_param)
                softmax_theta = np.exp(theta_sampled)
                softmax_theta /= np.sum(softmax_theta, axis=1)[:,None]
                j_n = self.evaluate(theta_sampled, num_episodes)
                if j_n > j:
                    theta = theta_sampled
                    j = j_n
   
    def learn_policy_fchc(self, num_iter, sigma, num_episodes):
        reshape_param = (GetStateNumber(4,3,self.dimensions), len(self.actionSpace)-3)
        curr_iter = 0
        data = []
        theta_max = []
        global_max = -2**31
        theta = util.get_init(state_space=reshape_param[0], action_space=reshape_param[1], sigma=sigma, condition=True)
        softmax_theta = np.exp(theta)
        softmax_theta = softmax_theta/np.sum(softmax_theta, axis=1)[:,None]
        j = self.evaluate(softmax_theta, num_episodes)
                
        while curr_iter < num_iter:
            print "-----------------------------"
            print "At ITER: ", curr_iter
            theta_sampled = util.sample(distribution='gaussian', theta=theta, sigma=sigma, reshape_param=reshape_param)
            softmax_theta = np.exp(theta_sampled)
            softmax_theta = softmax_theta/np.sum(softmax_theta, axis=1)[:,None]
            j_n = self.evaluate(softmax_theta, num_episodes)
            data.append(j_n)
            if j_n > j:
                theta = theta_sampled
                j = j_n
                print "MAX REWARD: ", j, " AT iter: ", curr_iter
            if j_n > global_max:
                global_max = j_n
                theta_max = theta
                print "GLOBAL MAX UPDATED: ", global_max, " AT iter: ", curr_iter
            print "-----------------------------"
            curr_iter += 1
        print "Saving Data"
        pkl.dump(data, open("fchcFILE.pkl", 'w'))
        pkl.dump(theta_max, open("fchcTHETA.pkl", 'w'))

    def learn_policy_bbo(self, init_population, best_ke, num_episodes, epsilon, num_iter, steps_per_trial=15, sigma=100):
        assert init_population >= best_ke
        assert num_episodes > 1
        
        max_av_reward = -2**31
        theta_max = []
        curr_iter = 0
        reshape_param = (GetStateNumber(4,3,self.dimensions), len(self.actionSpace)-1)
        data = []
        while (curr_iter < num_iter):
            theta, sigma = util.get_init(state_space=reshape_param[0],action_space=reshape_param[1], sigma=sigma)
            for i in range(steps_per_trial):
                values = []
                print "-----------------------------"
                print "At ITER: ", curr_iter
                print "AT step: ", i
                theta_sampled= util.sample('gaussian', theta, sigma, reshape_param, init_population)
                tic = time.time()
                for k in range(init_population):
                    theta_k = softmax_theta[k]
                    theta_k = theta_k/np.sum(theta_k, axis=1)[:,None]
                    j_k = self.evaluate(theta_k, num_episodes)
                    data.append(j_k)
                    if j_k > max_av_reward:
                        max_av_reward = j_k
                        theta_max = theta_k
                        print "MAX REWARD: ", max_av_reward, " AT step, iter: ", i, curr_iter
                    values.append((theta_k.reshape(reshape_param[0]*reshape_param[1], 1), j_k))  
                toc = time.time()
                print(toc-tic)
                values = sorted(values, key=lambda x: x[1], reverse=True)
                theta, sigma = util.generate_new_distribution('gaussian', theta, values, best_ke, epsilon)
                print "-----------------------------"
            curr_iter += 1
        print "Saving Data"
        pkl.dump(data, open("FILE.pkl", 'w'))
        pkl.dump(theta_max, open("THETA.pkl", 'w'))

class multiprocessing_obj(MDP):
    def __init__(self, num_episodes):
        MDP.__init__(self)
        self.num_episodes = num_episodes
        self.data = []
        self.max_av_reward = -2**31
        self.theta_max = []
    def __call__(self, theta):
        theta = theta/np.sum(theta, axis=1)[:,None]
        j = self.evaluate(theta, self.num_episodes)
        self.data.append(j)
        if self.max_av_reward < j:
            self.max_av_reward = j
            if self.debug:
                print "Max reward: ", self.max_av_reward
        return theta.reshape(theta.shape[0]*theta.shape[1], 1), j

if __name__ == "__main__":
    board = Board(5)
    mdp = MDP(board, 0.8, 0.05, 0.05, 0.1, 0.9, False)
    # mdp.learn_policy_bbo(init_population=500, best_ke=20, num_episodes=10, epsilon=1e-4, num_iter=500, sigma=100)
    # mdp.learn_policy_bbo_multiprocessing(init_population=100, best_ke=10, num_episodes=10, epsilon=1e-2, num_iter=20, variance=10)
    # mdp.learn_policy_fchc(num_iter=500*15*100,sigma=10,num_episodes=10)
    # mpd.evaluate_policy_TD(num_training_episodes=100, num_eval_episodes=100, policy='uniform')
    