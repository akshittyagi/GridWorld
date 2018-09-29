from enum import Enum

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

def get_init():
    pass

def get_new_values(values, best_k, epsilon):
    pass