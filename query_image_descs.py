import numpy as np

class QueryImageDescs():

    descs = []
    xy_coords = []
    name = ''

    def __init__(self, name):
        self.name = name
        self.descs = np.empty((0, 128))
        self.xy_coords = np.empty((0,2))

    def add_desc(self, desc):
        self.descs = np.concatenate((self.descs, desc), axis = 0)

    def add_xy_coord(self, xy):
        self.xy_coords = np.concatenate((self.xy_coords, xy), axis = 0)