import numpy as np

class Point3D():

    descs = []
    mean_desc = []
    id = 0;

    def __init__(self, id):
        self.id = id
        self.descs = np.empty((0, 128))
        self.mean_desc = np.empty((0, 128))

    def add_desc(self, desc):
        self.descs = np.concatenate((self.descs, desc), axis=0)

    def avg_descs(self):
        self.mean_desc = np.mean(self.descs, axis=0)