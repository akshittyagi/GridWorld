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
from matplotlib import pyplot as plt
from DecisionProcess import MDP
from Board import Board

class TD(object):
    '''TD Eval and Learning'''
    def __init__(self, mdp, num_training_episodes=100, num_eval_episodes=100, alpha=0.01, policy='uniform'):
        '''Initialising from another MDP'''
        self.mdp = mdp
        self.num_train = num_training_episodes
        self.num_eval = num_eval_episodes
        self.policy = policy
        self.alpha = alpha
        self.gamma = self.mdp.gamma
        self.value_function = self.mdp.initValueFunction()

    def estimate_value_function(self):
        '''Estimate the value for num_train episodes'''
        s_t = self.mdp.getInitialState()
        while not self.mdp.isTerminalState(s_t):
            a_t = self.mdp.getActionFromPolicy(s_t, policy=self.policy)
            s_t_1 = self.mdp.TransitionFunction(s_t, a_t)
            r_t = self.mdp.RewardFunction(s_t, a_t, s_t_1)
            s = self.mdp.getStateId(s_t)
            s_ = self.mdp.getStateId(s_t_1)
            self.value_function[s] += self.alpha*1.0*(r_t + self.gamma*1.0*(self.value_function[s_]) - self.value_function[s])
            s_t = s_t_1
  
    def evaluate_error(self, plot=False):
        '''Eval error for an episode'''
        s_t = self.mdp.getInitialState()
        error = 0
        time_step = 0
        if plot:
            X = []
            y = []
        while(not self.mdp.isTerminalState(s_t)):
            a_t = self.mdp.getActionFromPolicy(s_t, policy=self.policy)
            s_t_1 = self.mdp.TransitionFunction(s_t, a_t)
            r_t = self.mdp.RewardFunction(s_t, a_t, s_t_1)
            s = self.mdp.getStateId(s_t)
            s_ = self.mdp.getStateId(s_t_1)
            curr_error_2 = (r_t + self.gamma*1.0*(self.value_function[s_]) - self.value_function[s])**2
            error = error + curr_error_2
            s_t = s_t_1
            if plot:
                X.append(time_step)
                y.append(curr_error_2)
            time_step += 1
        if plot:
            plt.plot(X, y)
            plt.show()
        return (1.0*error)/time_step

    def update_weights(self):
        '''Update weights for a series of episodes'''
        self.value_function = self.mdp.initValueFunction()
        for episode in range(self.num_train):
            # print "UPDATING WEIGHTS FOR EPISODE: ", episode + 1
            self.estimate_value_function()

    def evaluate_policy(self, alpha):
        '''Evaluate the policy for a series of episodes'''
        error = 0.0
        X = []
        y = []
        for episode in range(self.num_eval):
            # print "EVALUATING TD ERROR FOR EPISODE: ", episode + 1
            if episode % 55 == 0:
                curr_error = self.evaluate_error(plot=False)
                error = error + curr_error
                X.append(episode + 1)
                y.append(curr_error)
            else:
                curr_error = self.evaluate_error(plot=False)
                error = error + curr_error
                X.append(episode + 1)
                y.append(curr_error)
        plt.plot(X, y)
        plt.savefig(str(alpha) + "_fig_GW.png")
        plt.clf()
        plt.cla()
        plt.close()
        error = error*1.0/self.num_eval
        return error

    def create_plots_for_alphas(self, alphas):
        '''Create plots for log scaled alphas'''
        X, y = [], []
        for alpha in alphas:
            print "At alpha: ", alpha
            self.alpha = alpha
            X.append(alpha)
            self.update_weights()
            y.append(self.evaluate_policy(alpha))
        X = np.log(np.array(X))/np.log(10)
        plt.plot(X, y)
        plt.show()

class Sarsa(TD):
    '''Sarsa docstring'''
    def __init__(self, mdp, epsilon, alpha, train_episodes):
        super(Sarsa, self).__init__(mdp, alpha=alpha)
        self.episodes = train_episodes
        self.epsilon = epsilon
        self.q_values = self.mdp.init_q_function()
        self.gamma = self.mdp.gamma

    def epsilon_greedy_action_selection(self, state):
        s = self.mdp.getStateId(state)
        q_values = np.array(self.q_values[s])
        arg_max = np.argmax(q_values)
        proba = [self.epsilon/len(q_values)]*len(q_values)
        proba[arg_max] += 1 - self.epsilon
        random_number = 1.0*random.randint(0,99)/100
        action_array = sorted(zip(np.arange(len(proba)), proba), key=lambda x: x[1], reverse=True)
        prev_proba = 0
        for action, probability in action_array:
            prev_proba += probability
            if random_number <= prev_proba:
                return action + 1

    def learn(self):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        for episode in range(self.episodes):
            print "------------------------------"
            print "AT EPISODE: ", episode + 1
            s_t = self.mdp.getInitialState()
            a_t = self.epsilon_greedy_action_selection(s_t)
            mse = 0
            time_step = 0
            while not self.mdp.isTerminalState(s_t) and time_step <= 1000:
                s_t_1 = self.mdp.TransitionFunction(s_t, a_t)
                r_t = self.mdp.RewardFunction(s_t, a_t, s_t_1)
                a_t_1 = self.epsilon_greedy_action_selection(s_t_1)
                s, s_ = map(self.mdp.getStateId, [s_t, s_t_1])
                a, a_ = map(self.mdp.getActionId, [a_t, a_t_1])
                q_td_error = r_t + self.gamma*(self.q_values[s_][a_]) - self.q_values[s][a]
                self.q_values[s][a] += self.alpha*(q_td_error)
                s_t = s_t_1
                a_t = a_t_1
                if time_step % 100 == 0:
                    print "SQ TD ERROR: ", q_td_error**2
                X.append(global_time_step)
                y.append(q_td_error**2)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(mse)
            print "AV MSE: ", mse
            print "------------------------------"
        
        plt.plot(X, y)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(X_ep, y_ep)
        plt.show()
        
if __name__ == "__main__":
    board = Board(5)
    mdp = MDP(board, 0.8, 0.05, 0.05, 0.1, 0.9, False)
    td = TD(mdp, 100, 100)
    # td.create_plots_for_alphas([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
    sarsa = Sarsa(mdp, epsilon=0.1, alpha=1e-4, train_episodes=100)
    sarsa.learn()