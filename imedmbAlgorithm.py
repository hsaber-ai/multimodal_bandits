## Modules
import numpy as np



# Bandit specific functions and classes 
from StochasticBandit import *
from BanditTools import * 


## Algorithms

class IMEDMB:
    """Indexed Minimum Empirical Divergence for Multimodal Bandits"""
    def __init__(self,nbArms, kullback, xi = 0, M = 1, Nlen = 1, neigh=[], isexp =0):
        self.nbArms = nbArms  # N
        self.kl = kullback  # \kl
        self.xi = xi      # \xi
        self.M = M        #\M
        self.Nlen = Nlen
        self.neigh = neigh
        self.isexp = isexp
        self.clear()
    
    def f(self, t):    # \f_\xi(\cdot)
        if t > exp(1) :
            return log(t) + self.xi * log(max(1,log(t)))
        else :
            return log(max(1,t))
    
    def g(self,t):
        return log(max(1,t)) #sqrt(t)
    
    def phi(self,t):
        return log(max(1,t))
    
    def psi(self,t):
        if self.isexp ==0:
            return np.inf # max(t,np.exp(t**1.1))
        else:
            return np.exp(t)
    
    def klp(self, x,y):
        if x < y:
            return self.kl(x,y)
        else:
            return 0
        
    def clear(self):
        self.arms = [a for a in range(self.nbArms)] 
        self.noStructureInformativeArmSequence = self.arms
        self.TotalnbDraws = 0        # t
        self.nbDraws = np.zeros(self.nbArms) # (N_a)_{a\in\cA}
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)   # (\muhat_a(t))_{a\in\cA}
        self.maxMean = max(self.means)  # \muhat^\star(t)
        self.ahatstar = randmax(self.means)
        self.imedIndexes = [0 for a in range(self.nbArms)]
        self.imedArm = randmin(self.imedIndexes)
        self.firstIndexes = [np.infty for a in range(self.nbArms)]
        self.firstArm = randmin(self.firstIndexes)
        self.secondArm = self.ahatstar
        self.Ap = []
        self.Am=[]
        self.neighbours = [[] for a in range(self.nbArms)]
        self.test = True
        
      
        if self.neigh==[]:
            for a in range(self.nbArms):
                V = []
                for e in [-self.Nlen, self.Nlen]:
                    if (a+e>-1) and (a+e<self.nbArms):
                        V = V + [a+e]
                self.neighbours[a] = V
        else:
            self.neighbours=self.neigh
        
        
    def chooseArmToPlay(self): # Algorithm
        if self.TotalnbDraws < self.nbArms:
            return randmin(self.nbDraws)
        if len(self.Ap) < self.M:
            return self.imedArm
        else:
            return self.firstArm
             
                            
    def receiveReward(self,arm,reward): #Update
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.TotalnbDraws = self.TotalnbDraws + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.maxMean = max(self.means)
        self.ahatstar = randmax(self.means)

        
        self.imedIndexes = [self.nbDraws[a]*min(self.klp(self.means[a], self.maxMean),self.phi(self.nbDraws[a]))+self.f(self.nbDraws[a]) for a in range(self.nbArms)]
        self.imedArm = randmin(self.imedIndexes)

        self.Ap = []
        for a in range(self.nbArms):
            aInAp = self.nbDraws[a] > 0
            for b in self.neighbours[a]:
                aInAp = aInAp and (self.means[b]<=self.means[a])
            if aInAp:
                self.Ap = self.Ap + [a]

        self.Am = self.Ap
      
        
        if len(self.Am) >= self.M:
            if self.M>1:
                self.AmMeans = [self.means[a]  for a in self.Am if a != self.ahatstar]
                sort = sorted(zip(self.AmMeans, [a for a in self.Am if a!= self.ahatstar]), reverse=True)[:(self.M-1)]
                self.Am = [self.ahatstar]+[sort[m][1] for m in range(self.M-1)]
            else:
                self.Am = [self.ahatstar]
        
        self.firstIndexes = [self.psi(self.imedIndexes[a]) for a in range(self.nbArms)]
        for ap in self.Am:
            self.firstIndexes[ap] = self.imedIndexes[ap]
            for a in self.neighbours[ap]:
                self.firstIndexes[a] = self.imedIndexes[a]
        
        self.firstArm = randmin(self.firstIndexes)

    def name(self):
        if self.isexp==0:
            return "IMED-MB (M="+str(self.M)+")"
        else:
            return "IMED-MB (M="+str(self.M)+", exp)"
    

