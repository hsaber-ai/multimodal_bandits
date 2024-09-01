import numpy as np
from math import log,sqrt
from BanditTools import *
 
class FTL:
    """follow the leader (a.k.a. greedy strategy)"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
    
    def chooseArmToPlay(self):
        if (min(self.nbDraws)==0):
            return randmax(-self.nbDraws)
        else:
            return randmax(self.cumRewards/self.nbDraws)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

    def name(self):
        return "FTL"


class UniformExploration:
    """a strategy that uniformly explores arms"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
    
    def chooseArmToPlay(self):
        return np.random.randint(0,self.nbArms)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

    def name(self):
        return "Uniform"
    

class gaussianKLUCB:
    """ KL-UBC for gaussian means"""
    def __init__(self,nbArms, variance):
        self.nbArms = nbArms
        self.var = variance
        self.clear()

    def clear(self):
        self.TotalnbDraws = 0
        self.nbDraws = [0 for a in range(self.nbArms)]
        self.cumRewards = [0 for a in range(self.nbArms)]
        self.means = [0 for a in range(self.nbArms)]
        self.ahatstar = randmax(self.means)
        self.maxMean = self.means[self.ahatstar]
        self.indexes = [0 for a in range(self.nbArms)]
        
    def f(self):
        if self.TotalnbDraws < 1:
            return 0
        else:
            return np.log(self.TotalnbDraws)
        
    def chooseArmToPlay(self):
        return randmax(self.indexes)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.TotalnbDraws = self.TotalnbDraws + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.maxMean = max(self.means)
        self.ahatstar = randmax(self.means)

        for a in range(self.nbArms):
            if self.nbDraws[a]>0:
                self.indexes[a] = self.means[a]+np.sqrt(2*self.var*self.f()/self.nbDraws[a])
            else:
                self.indexes[a] = np.inf
    def name(self):
        return "KLUCB"
    







class IMED:
    """Indexed Minimum Empirical Divergence"""
    def __init__(self,nbArms, kullback, xi=0):
        self.nbArms = nbArms  # N
        self.kl = kullback  # \kl
        self.xi = 0       # \xi
        self.clear()
    
    def f(self, t):    # \f_\xi(\cdot)
        if t > exp(1) :
            return log(t) + self.xi * log(log(t))
        else :
            return log(max(1,t))
    
        
    def clear(self):
        self.TotalnbDraws = 0        # t
        self.nbDraws = np.zeros(self.nbArms) # (N_a)_{a\in\cA}
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)   # (\muhat_a(t))_{a\in\cA}
        self.maxMean = max(self.means)  # \muhat^\star(t)
        self.noStructureInformativeArm = randmax(self.means) # \adot_t
        self.noStructureIndexes = [] # (I_a(t))_{a\in\cA}
        self.minNoStructureIndexes = 0  #\min_{a\in\cA}I_a(t)
        
        
    def chooseArmToPlay(self): # Algorithm
        if self.TotalnbDraws < self.nbArms :
            return self.TotalnbDraws
        else:
            return self.noStructureInformativeArm
                
                            
    def receiveReward(self,arm,reward): #Update
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.TotalnbDraws = self.TotalnbDraws + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.maxMean = max(self.means)
        
        if self.TotalnbDraws >= self.nbArms :
            
            self.noStructureIndexes = [self.nbDraws[a]*self.kl(self.means[a],self.maxMean) + self.f(self.nbDraws[a])  for a in range(self.nbArms)]
        
            self.noStructureInformativeArm = randmin(self.noStructureIndexes)
        
            self.minNoStructureIndexes = self.noStructureIndexes[self.noStructureInformativeArm]
  
    def name(self):
        return "IMED"






