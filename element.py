import numpy as np

class TimeMoment:
    def __init__(self,ndof):
        self.u = np.zeros(ndof)


class Element:
    def __init__(self,ndof,nt):
       self.M    = np.zeros((ndof,ndof))
       self.C    = np.zeros((ndof, ndof))
       self.K    = np.zeros((ndof, ndof))
       self.s    = np.zeros(2)
       self.f    = np.zeros(2)
       self.x    = np.zeros(2)
       self.time = []
       for i in range(nt):
           self.time.append(TimeMoment(ndof))
