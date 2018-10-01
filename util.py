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
    
def sample(distribution, theta, sigma):
    pass

def get_init(state_space, action_space):
    theta = np.random.rand(state_space, action_space)
    theta = theta.reshape(state_space*action_space, 1)
    shape = theta.shape[0]
    sigma_diag = 1
    sigma = (sigma_diag*sigma_diag)*np.identity(shape)
    return theta, sigma

def generate_new_distribution(distribution, values, best_k, epsilon):
    pass