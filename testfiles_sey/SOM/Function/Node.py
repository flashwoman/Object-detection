from random import *


class Node :

    def __init__(self, FV_size=10, PV_size=10, Y=0, X=0):
        self.FV_size = FV_size
        self.PV_size = PV_size
        self.FV = [0.0] * FV_size
        self.PV = [0.0] * PV_size
        self.X = X
        self.Y = Y

        for i in range(FV_size):
            self.FV[i] = random() # random number from 0 to 1

        for i in range(PV_size):
            self.PV[i] = random() # random number from 0 to 1