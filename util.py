from enum import Enum

import numpy as np

class Cells(Enum):
    Regular = 0
    Obstacle = 1
    Water = 10
    End = 100

class ActionSpace(Enum):
    AU = 1
    AD = 2
    AL = 3
    AR = 4

def GetStateNumber(x, y, size):
    if x<0 or y<0 or x>=size or y>=size:
        return -1
    if x<2 or x==2 and y<2:
        return size*x + y + 1
    elif (x==2 and y>2) or (x==3 and y<2):
        return size*x + y
    elif (x==3 and y>2) or x==4:
        return size*x + y - 1
    
def sample(distribution, theta, sigma, reshape_param):
    if distribution == 'gaussian':
        distribution = np.random.multivariate_normal
    theta_k = distribution(theta, sigma, 1).T

    return theta_k.reshape(reshape_param[0], reshape_param[1])

def get_init(state_space, action_space):
    theta = np.random.rand(state_space, action_space)
    theta = theta.reshape(state_space*action_space, 1)
    shape = theta.shape[0]
    sigma_diag = 1
    sigma = (sigma_diag*sigma_diag)*np.identity(shape)
    return theta, sigma

def generate_new_distribution(distribution, mean_vector, values, best_k, epsilon):
    if distribution == 'gaussian':
        distribution = np.random.multivariate_normal
    values = values[:best_k]
    sum_vector = np.array([element[0] for element in values])
    sum_vector -= mean_vector
    average_vector = np.sum(sum_vector, axis=0)/best_k
    sigma = (epsilon*np.identity(values[0].shape[0]))
    for vector in sum_vector:
        sigma += vector.T.dot(vector)
    sigma /= (epsilon + best_k)

    return average_vector, sigma
