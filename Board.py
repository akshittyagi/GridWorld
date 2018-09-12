import os
from enum import Enum

import numpy as np
from util import Cells

class Board():

    def __init__(self, size):
        self.dimensions = size
        self.grid = np.array([[Cells.Regular]*size]*size)
        #obstacles
        self.grid[2,2] = Cells.Obstacle
        self.grid[3,2] = Cells.Obstacle
        #water state
        self.grid[4,2] = Cells.Water
        #end state
        self.grid[4,4] = Cells.End