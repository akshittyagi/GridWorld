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

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / temperature
        s = self.mdp.getStateId(state)
        q_values = self.q_values[s]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            action = int(argmax) + 2
            return action
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            action = coin_toss + 2
            return action

    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.getInitialState()
            a_t = self.epsilon_greedy_action_selection(s_t)
            mse = 0
            time_step = 0
            temperature = 1
            g = 0
            while not self.mdp.isTerminalState(s_t) and time_step <= 20000:
                alpha = alpha / temperature
                s_t_1 = self.mdp.TransitionFunction(s_t, a_t)
                r_t = self.mdp.RewardFunction(s_t, a_t, s_t_1)
                a_t_1 = self.epsilon_greedy_action_selection(s_t_1, temperature=temperature)
                s, s_ = map(self.mdp.getStateId, [s_t, s_t_1])
                a, a_ = map(self.mdp.getActionId, [a_t, a_t_1])
                q_td_error = r_t + self.gamma*(self.q_values[s_][a_]) - self.q_values[s][a]
                self.q_values[s][a] += alpha*(q_td_error)
                s_t = s_t_1
                a_t = a_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)

class Qlearning(TD):
    '''QLearning docstring'''
    def __init__(self, mdp, epsilon, alpha, train_episodes):
        super(Qlearning, self).__init__(mdp, alpha=alpha)
        self.episodes = train_episodes
        self.epsilon = epsilon
        self.q_values = self.mdp.init_q_function()
        self.gamma = self.mdp.gamma

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / temperature
        s = self.mdp.getStateId(state)
        q_values = self.q_values[s]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            action = int(argmax) + 2
            return action
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            action = coin_toss + 2
            return action

    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.getInitialState()
            mse = 0
            time_step = 0
            temperature = 1.0
            g = 0
            while not self.mdp.isTerminalState(s_t) and time_step <= 20000:
                alpha = alpha / temperature
                a_t = self.epsilon_greedy_action_selection(s_t, temperature=temperature)
                s_t_1 = self.mdp.TransitionFunction(s_t, a_t)
                r_t = self.mdp.RewardFunction(s_t, a_t, s_t_1)
                s, s_ = map(self.mdp.getStateId, [s_t, s_t_1])
                a = self.mdp.getActionId(a_t)
                q_td_error = r_t + self.gamma*(np.amax(self.q_values[s_])) - self.q_values[s][a]
                self.q_values[s][a] += alpha*(q_td_error)
                s_t = s_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)


def get_hyperparams(range_of_param, interval, multiplicative=True):
    start, end = range_of_param
    ret = []
    while(start <= end):
        ret.append(start)
        if multiplicative:
            start *= interval
        else:
            start += interval
    return ret

if __name__ == "__main__":
    board = Board(5)
    mdp = MDP(board, 0.8, 0.05, 0.05, 0.1, 0.9, False)
    td = TD(mdp, 100, 100)
    num_trials = 100
    num_training_episodes = 100
    hyperparam_search = False
    switch_sarsa = False
    X = np.arange(num_training_episodes)
    Y = []

    if switch_sarsa:
        print "------------" 
        print "SARSA" 
        print "------------"
    else: 
        print "------------"
        print "Q-LEARNING"
        print "------------"

    if hyperparam_search:
        '''HyperParameter Search'''
        alphas = get_hyperparams(range_of_param=[1e-3, 1e-1], interval=10, multiplicative=True)
        epsilons = get_hyperparams(range_of_param=[1e-2, 1e-1], interval=0.02, multiplicative=False)
        reduction_factors = get_hyperparams(range_of_param=[3,10], interval=1, multiplicative=False)
        G = -2**31
        params = []
        for alpha in alphas:
            for epsilon in epsilons:
                for reduction_factor in reduction_factors:
                    print "RETURN for alpha", str(alpha), " epsilon ", str(epsilon), " reductionFactor ", str(reduction_factor), " : "
                    if switch_sarsa:
                        sarsa = Sarsa(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
                        _, y, g = sarsa.learn(reduction_factor=reduction_factor)
                    else:
                        qlearn = Qlearning(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
                        _, y, g = qlearn.learn(reduction_factor=reduction_factor)
                    print g
                    if G < g:
                        G = g
                        params = [alpha, epsilon, reduction_factor]
                        print "BEST PARAMS: "
                        print params
   
    if not hyperparam_search:
        #alpha, epsilon, reduction_factor: alpha = alpha/(temp**red_fac)
        params = [6e-1, 9e-1, 6]

    for trial in range(num_trials):
        print "AT TRIAL: ", trial + 1
        if switch_sarsa:
            sarsa = Sarsa(mdp, epsilon=params[1], alpha=params[0], train_episodes=num_training_episodes)
            _, y, _ = sarsa.learn(reduction_factor=params[2])
        else:
            qlearn = Qlearning(mdp, epsilon=params[1], alpha=params[0], train_episodes=num_training_episodes)
            _, y, _ = qlearn.learn(reduction_factor=params[2], plot=False, debug=False)
        Y.append(y)
    Y = np.array(Y)
    Y_mean = np.sum(Y, axis=0)
    Y_mean = Y_mean/num_trials
    Y_diff = np.repeat(Y_mean.reshape(1, num_training_episodes), num_trials, axis=0)    
    Y_diff = Y - Y_diff
    Y_diff = Y_diff ** 2
    Y_diff = np.sum(Y_diff, axis=0) / num_trials
    Y_diff = np.sqrt(Y_diff)
    plt.errorbar(X, Y_mean, yerr=Y_diff, fmt='o')
    
    if switch_sarsa:
        print "------------" 
        print "SARSA" 
        print "------------"
    else: 
        print "------------"
        print "Q-LEARNING"
        print "------------"

    plt.show()
