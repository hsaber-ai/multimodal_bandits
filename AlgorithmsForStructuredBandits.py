## Modules

import numpy as np
import scipy


from scipy.stats import rankdata

# Bandit specific functions and classes 
from StochasticBandit import *
from BanditTools import * 


## Algorithms


class IMEDUB:
    """Indexed Minimum Empirical Divergence for Unimodal Bandit"""
    def __init__(self,nbArms, kullback, xi = 0):
        self.nbArms = nbArms  # N
        self.kl = kullback  # \kl
        self.xi = xi      # \xi
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
        self.structureInformativeArm = randmax(self.means)  # \aol_t
        self.noStructureInformativeArm = randmax(self.means)# \adot_t
        self.structureIndexes = [] # (\Iol_a(t))_{a\in\cA}
        self.noStructureIndexes = [] # (I_a(t))_{a\in\cA}
        self.minStructureIndexes = 0 #\min_{a\in\cA}\Iol_a(t)
        self.minNoStructureIndexes = 0  #\min_{a\in\cA}I_a(t)
        
        
    def chooseArmToPlay(self): # Algorithm
        if False:#self.TotalnbDraws < self.nbArms :
            return self.TotalnbDraws
        else:
          return self.structureInformativeArm
                
                            
    def receiveReward(self,arm,reward): #Update
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.TotalnbDraws = self.TotalnbDraws + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.maxMean = max(self.means)
        
        if self.TotalnbDraws >= self.nbArms :
            
            self.noStructureIndexes = [self.nbDraws[a]*self.kl(self.means[a],self.maxMean) + self.f(self.nbDraws[a])  if self.nbDraws[a] > 0 else -np.infty for a in range(self.nbArms)]
        
            self.noStructureInformativeArm = randmin(self.noStructureIndexes)
        
            self.minNoStructureIndexes = self.noStructureIndexes[self.noStructureInformativeArm]
        
        
            self.structureIndexes = [ self.noStructureIndexes [a] if (self.means[a]==self.maxMean) or ((a>0) and (self.means[a-1]==self.maxMean)) or ((a<self.nbArms-1) and (self.means[a+1]==self.maxMean)) else np.infty  for a in range(self.nbArms) ] 
        
            self.structureInformativeArm = randmin(self.structureIndexes)
        
            self.minStructureIndexes = self.structureIndexes[self.structureInformativeArm]
        

    def name(self):
        return "IMED-UB"
          
        
class CKL:
    """CKL"""
    def __init__(self,nbArms,kullback, k=0.01):
        self.nbArms = nbArms
        self.kl = kullback
        self.k=k
        self.clear()
    
    def klp(self, x,y):
        if x <=y:
            return self.kl(x,y)
        else:
            return 0

    
    def clear(self):
        self.TotalnbDraws = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.bestarm = randmax(self.means)
        self.maxMean = 0
        self.indexes = np.zeros(self.nbArms)
        self.L = np.zeros(self.nbArms)
        self.lesspulledArm = 0
        self.leader =0
        self.informative = 0
        self.leaderIndex =0

    def f(self,t):
        return np.log(max(1,t))+(3*self.nbArms+1)*np.log(max(1,np.log(max(1,t))))
    
    def u_function(self, arm, a, b, f, e = 10**(-4) ):
        mu_arm = self.means[arm]
        if mu_arm >= 3000:
            return 3000
        else:
            while b - a > e :
                m = (b+a)/2
                naklm = 0
                nakla = 0
                for aa in range(self.nbArms):
                   nakla = nakla+self.nbDraws[aa]+self.klp(self.means[aa], mu_arm+a - self.k*abs(aa-arm))
                for aa in range(self.nbArms):
                    naklm = naklm+self.nbDraws[aa]+self.klp(self.means[aa], mu_arm+m - self.k*abs(aa-arm))
                
                if ((nakla-f)*(naklm-f)<=0):
                    return self.u_function(arm, a, m, f )
                else:
                    return self.u_function(arm, m, b, f )
            return mu_arm + a
        
    
    def chooseArmToPlay(self):
        if self.TotalnbDraws < self.nbArms :
            return self.TotalnbDraws
        else:
            if self.nbDraws[self.lesspulledArm]<np.log(max(1,np.log(max(1,self.TotalnbDraws)))):
                return self.lesspulledArm
            else:
                if self.informative==self.leader:
                    return self.leader
                else:
                    return randmin([self.nbDraws[a] if self.indexes[a] > self.leaderIndex else np.inf for a in range(self.nbArms)])

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.TotalnbDraws = self.TotalnbDraws + 1
        if self.TotalnbDraws%10000==0:
            print("time step:", self.TotalnbDraws )
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.bestarm = randmax(self.means)
        self.maxMean = self.means[self.bestarm]
        self.L[self.bestarm ] = self.L[self.bestarm] + 1

        for a in range(self.nbArms):
            if self.nbDraws[a]==0:
                self.indexes[a] = 1
            else:
                self.indexes[a] = self.u_function(a, 0, 3000-self.means[a], self.f(self.TotalnbDraws))


        self.lesspulledArm = randmin(self.nbDraws)
        self.leader = randmax(self.L)
        self.informative = randmax(self.indexes)
        self.leaderIndex = self.indexes[self.leader]
    def name(self):
        return "CKLUCB"        
        


class OSUB:
    """Optimal Sampling for Unimodal Bandit"""
    def __init__(self, nbRows, nbColums, kullback, f):
        self.nbR = nbRows
        self.nbC = nbColums
        self.nbArms = self.nbR*self.nbC  # N
        self.kl = kullback
        self.f = f
        self.clear()
    
    
    def clear(self):
        self.TotalnbDraws = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.bestarm = randmax(self.means)
        self.maxMean = 0
        self.indexes = np.zeros(self.nbArms)
        self.L = np.zeros(self.nbArms)
        self.neighbours = [[] for a in range(self.nbArms)]
        for a in range(self.nbArms):
            r = a//self.nbC
            c = a - r*self.nbC
            V = []
            for eR in [-1, 0, 1]:
                for eC in [-1, 0, 1]:
                    if (eR!=0) or (eC!=0):
                        if (r+eR>-1) and (r+eR<self.nbR) and (c+eC>-1) and (c+eC<self.nbC):
                            V = V + [(r+eR)*self.nbC + c+eC]
            
            self.neighbours[a] = V
    
    def u_function(self, arm, a, b, f, e = 10**(-5) ):
        mu_arm = self.means[arm]
        if mu_arm == 1:
            return 1
        else:
            while b - a > e :
                N_arm = self.nbDraws[arm]
                m = (b+a)/2
                if ((N_arm*self.kl(mu_arm,mu_arm+a)-f)*(N_arm*self.kl(mu_arm,mu_arm+m)-f)<=0):
                    return self.u_function(arm, a, m, f )
                else:
                    return self.u_function(arm, m, b, f )
            return mu_arm + a
        
    
    def chooseArmToPlay(self):
        if self.TotalnbDraws < self.nbArms :
            return self.TotalnbDraws
        else:
            if self.L[self.bestarm]%9 == 1 :
                return self.bestarm
            else:
                return randmax(self.indexes)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.TotalnbDraws = self.TotalnbDraws + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.bestarm = randmax(self.means)
        self.maxMean = self.means[self.bestarm]
        self.L[self.bestarm ] = self.L[self.bestarm] + 1
        if self.L[self.bestarm]%9 != 1 :
            self.indexes = []
            for a in range(self.nbArms):
                if (a != self.bestarm) and (not (a in self.neighbours[self.bestarm])):
                    self.indexes = self.indexes + [-np.inf]
                elif self.nbDraws[a]==0:
                    self.indexes = self.indexes + [1]
                else:
                    self.indexes = self.indexes + [self.u_function(a, 0, 1-self.means[a], self.f(self.L[self.bestarm]))]

    def name(self):
        return "OSUB"



class UTS:
    """Unimodal Thomson Sampling"""
    def __init__(self,nbRows, nbColums):
        self.nbR = nbRows
        self.nbC = nbColums
        self.nbArms = nbRows*nbColums  # N
        self.clear()
    
    
    def clear(self):
        self.TotalnbDraws = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.bestarm = randmax(self.means)
        self.maxMean = 0
        self.indexes = np.zeros(self.nbArms)
        self.L = np.zeros(self.nbArms)
        self.neighbours = [[] for a in range(self.nbArms)]
        for a in range(self.nbArms):
            r = a//self.nbC
            c = a - r*self.nbC
            V = []
            for eR in [-1, 0, 1]:
                for eC in [-1, 0, 1]:
                    if (eR!=0) or (eC!=0):
                        if (r+eR>-1) and (r+eR<self.nbR) and (c+eC>-1) and (c+eC<self.nbC):
                            V = V + [(r+eR)*self.nbC + c+eC]
            
            self.neighbours[a] = V
    
    
        
    
    def chooseArmToPlay(self):
        if False: #self.TotalnbDraws < self.nbArms :
            return self.TotalnbDraws
        else:
            if self.L[self.bestarm]%2 == 0 :
                return self.bestarm
            else:
                return randmax(self.indexes)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.TotalnbDraws = self.TotalnbDraws + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.bestarm = randmax(self.means)
        self.maxMean = self.means[self.bestarm]
        self.L[self.bestarm ] = self.L[self.bestarm] + 1
        if self.L[self.bestarm]%9 != 0 :
            self.indexes = []
            for a in range(self.nbArms):
                if (a != self.bestarm) and (not (a in self.neighbours[self.bestarm])):
                    self.indexes = self.indexes + [-np.inf]
                elif self.nbDraws[a]==0:
                    self.indexes = self.indexes + [1]
                else:
                    self.indexes = self.indexes + [np.random.beta(max(self.cumRewards[a],0) + 1, max(self.nbDraws[a] - self.cumRewards[a],0) + 1)]

    def name(self):
        return "UTS"

class OSSB:
    """Optimal Sampling for Structured Bandits"""
    def __init__(self,nbArms, Theta, kullback, discretization=False, discretizationConstant=0.05, sigmaParameter="means",e=0, g=0):
        self.nbArms = nbArms
        self.Theta = Theta
        self.kl = kullback
        self.ThetaSize = len(self.Theta)
        self.initialValidTheta = self.initialValidThetaFunction()
        self.e = e
        self.g = g
        self.discretization = discretization
        self.dConstant = discretizationConstant
        self.sigmaParameter = sigmaParameter
        self.clear()
    
    def initialValidThetaFunction(self):
        return [[ i for i in range(self.ThetaSize) if min(self.Theta[i][a])>=0 ] for a in range(self.nbArms)]
    
    def clear(self):
        self.TotalnbDraws = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.bestarm = randmax(self.means)
        self.maxMean = self.means[self.bestarm]
        self.s = 0
        self.Xul = randmin(self.nbDraws)
        self.Xol = self.Xul
        self.lbTest = True
        self.ubTest = True
        self.linearProgramming = None
        self.c = [0 for a in range(self.nbArms)]
        self.optPbDelta = []
        self.ArmsTimesTheta = []
        self.optPbKL = []
        self.optPbLowerBounds = []
        self.ThetaStar = []
        self.validTheta = []
        self.ArmsTimesTheta = []
        

    def sigmaFunction(self, l, m ):
        output = [0]*l
    
        for i,x in enumerate(l - rankdata(m,method='ordinal')):
            output[x] = i
        return output
        
        
        
    def ThetaStarFunction(self):
        if self.ThetaSize == 1:
            return self.Theta
        else:
            a = self.sigma[self.counter]
            ThetaS = [ theta for theta in self.ThetaStar if min([theta[a][b] - (self.means[a] - self.means[b]) for b in range(self.nbArms) ])>=-self.discretization*self.dConstant]
            if ThetaS:
                return ThetaS
            else:
                if self.counter == 0 :
                    self.counter = self.nbArms
                    return []
                else:
                    self.counter = self.nbArms
                    return self.ThetaStar
        

                
        
    
    def validThetaFunction(self):
        if len(self.ThetaStar)==1:
            return [[self.Theta.index(self.ThetaStar[0])] for a in range(self.nbArms)]
        else:
            return [  [ i for i in self.initialValidTheta[a] if (self.Theta[i] in self.ThetaStar) ] for a in range(self.nbArms) ]
            #[  [ i for i in self.initialValidTheta[a] if (min([ [ min([ self.Theta[i][a][b] <= self.Theta[i][astar][b]  for b in range(self.nbArms) if  self.Theta[i][a][b] <= (self.maxMean - self.means[b])   ]) ] for astar in self.Astar ])) and min([ min([ min([theta[b][c] <= self.Theta[i][b][c] for c in range(self.nbArms)]) for theta in self.ThetaStar]) for b in range(self.nbArms) if self.Theta[i][a][b] <= ( self.maxMean - self.means[b])])  ] for a in range(self.nbArms) ]
    
        
    def ArmsTimesThetaFunction(self):
        ArmsTimesTheta = []
        for a in range(self.nbArms):
            for i in self.validTheta[a]:
                ArmsTimesTheta = ArmsTimesTheta + [[a,self.Theta[i]]]
        
        return ArmsTimesTheta
    
    def chooseArmToPlay(self):
        if self.lbTest :
            return self.bestarm
        else:
            self.s = self.s + 1 
            if self.ubTest :
                return self.Xul
            else:
                return self.Xol

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm] + reward
        self.TotalnbDraws = self.TotalnbDraws + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.bestarm = randmax(self.means)
        self.maxMean = self.means[self.bestarm]

        self.Xul = randmin(self.nbDraws)
        self.ubTest = self.nbDraws[self.Xul] <= self.e*self.s

        self.ThetaStar = self.Theta
        if self.sigmaParameter == "means":
            self.sigma = self.sigmaFunction(l=self.nbArms, m=self.means)
        else:
            self.sigma = self.sigmaFunction(l=self.nbArms, m=self.nbArms)

        self.counter = -1

        while self.ThetaStar and (self.counter < self.nbArms-1):
            self.counter = self.counter + 1
            self.ThetaStar = self.ThetaStarFunction()
        
        if not self.ThetaStar:
                self.ThetaStar = [self.Theta[np.random.randint(self.ThetaSize)]]
                
        if self.ThetaStar:
            self.optPbDelta = [self.maxMean - self.means[a] for a in range(self.nbArms)]
            self.optPbDelta = np.array(self.optPbDelta)
            
            self.validTheta = self.validThetaFunction()
            self.ArmsTimesTheta = self.ArmsTimesThetaFunction()
            
            self.optPbKL = [ [ self.kl( self.means[b], max( self.means[b], self.maxMean - aTheta[1][aTheta[0]][b]) ) for b in range(self.nbArms) ]  for aTheta in self.ArmsTimesTheta]
            
            self.optPbKL = (-1)*np.array(self.optPbKL)
            
            self.optPbLowerBounds = [  1*(self.maxMean - self.means[aTheta[0]]>0)*(self.Theta.index(aTheta[1]) in self.validTheta[aTheta[0]]) for aTheta in self.ArmsTimesTheta]
            
            self.optPbLowerBounds = (-1)*np.array(self.optPbLowerBounds)
            
            self.linearProgramming = scipy.optimize.linprog(self.optPbDelta, A_ub=self.optPbKL, b_ub=self.optPbLowerBounds , method='simplex')
            
            self.c = self.linearProgramming.x

            if np.isnan(self.c).any(): # the simplex method fails to find a solution to the current optimization problem
                self.lbTest = False
                self.Xol = self.Xul
            
            else :
                self.Xol = randmin([ self.nbDraws[a]/self.c[a] if self.c[a] > 0 else np.infty for a in range(self.nbArms)])
                self.lbTest = min([self.nbDraws[a] >= (1+self.g)*self.c[a]*log(self.TotalnbDraws) for a in range(self.nbArms)])
        
        else:
            self.lbTest = False
            self.Xol = self.Xul

        
    def name(self):
        return "OSSB"
    


        



class linearUCB:
    """Upper Confidence Bounds for Linear Bandit"""
    def __init__(self,nbArms,learnerReg, dicretization):
        self.nbArms = nbArms
        self.learnerReg = learnerReg
        self.x = dicretization
        self.pulls =0
        self.clear()

    def clear(self):
        self.pulls = 0
        self.learnerReg.clear()
        self.indexes = np.zeros(self.nbArms)
    
    def chooseArmToPlay(self):
        if (self.pulls//10**4)>0 and self.pulls%(10**4)==0:
            print('Linear:', self.pulls)
        if self.pulls < self.nbArms:
            return self.pulls
        return randmax(self.indexes)

    def receiveReward(self,arm,reward):
        self.pulls =self.pulls +1
        self.learnerReg.addTrainingInput(self.x(arm), reward)
        self.indexes = [self.learnerReg.confidenceSet(self.x(a),delta=self.learnerReg.delta)[1] for a in range(self.nbArms)]
    
    def name(self):
        return "LinearUCB"


