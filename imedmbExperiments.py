## Modules
#from google.colab import files

import sys
path = 'plots/'
sys.path.append(path)

import datetime


#from mpl_toolkits import mplot3d

import numpy as np
import scipy

from matplotlib import pyplot as plt
import time
from scipy.optimize import linprog
from scipy.stats import rankdata

# Bandit specific functions and classes
import Arms as arm
from StochasticBandit import *
from BanditTools import *
import BanditBaselines as alg
from AlgorithmsForStructuredBandits import*
from imedmbAlgorithm import*


import Regression_Tools
import Expert_Regression as algLn

def standardComplexity(means):
    bestArm = randmax(means)
    meanMax = means[bestArm]
    complexity = 0
    for a in range(nbArms):
        if means[a]<meanMax:
            complexity = complexity + (meanMax - means[a])/klGauss(means[a],meanMax)
    return complexity

def multimodalComplexity(means, Ap, neighbours):
    bestArm = randmax(means)
    meanMax = means[bestArm]
    complexity = 0
    for a in Ap:
        if means[a]<meanMax:
            complexity = complexity + (meanMax - means[a])/klGauss(means[a],meanMax)
        for b in neighbours[a]:
            complexity = complexity + (meanMax - means[b])/klGauss(means[b],meanMax)
    return complexity

def LipschitzComplexity(means, k):
    nbArms = len(means)
    bestArm = randmax(means)
    meanMax = means[bestArm]

    c = [meanMax - means[a] for a in range(nbArms)]

    A =[]
    b=[]
    for a in range(nbArms):
        if a!=bestArm:
            A = A + [[-np.float32(means[b] < meanMax - k*abs(a - b) )*klGauss(means[b],meanMax - k*abs(a - b)) for b in range(nbArms)] ]
            b = b + [-1]

    res = linprog(c, A_ub=A, b_ub=b,bounds=(0, None))
    return res.fun



## Multimodals Means
nbArms = 50

var= round(1/klGauss(0,sqrt(2)),2)



## Experiment parameters
N_exp = 10
timeHorizon = 10**4
tsave = np.arange(1, timeHorizon, timeHorizon//100)




def LipschitzGaussianBandit(nbArms):
    """define a Lipschitz Gaussian MAB """
    means= [0 for a in range(nbArms)]
    means[0]=(1-2*np.random.randint(2))*np.random.random()*0.1
    for a in range(1,nbArms):
        means[a] = means[a-1]+(1-2*np.random.randint(2))*np.random.random()*0.01
    return GaussianBandit(means)

mcomplexity = 0
lcomplexity = None

while lcomplexity is None:
    bandit = LipschitzGaussianBandit(nbArms)

    multimodalMeans = [m for m in bandit.means]



    neighbours = [[] for a in range(nbArms)]
    for a in range(nbArms):
        V = []
        #for e in [-6,-1,1,6]:
        for e in [-1,1]:
            if (a+e>-1) and (a+e<nbArms):
                V = V + [a+e]
        neighbours[a] = V


    Ap = []
    for a in range(nbArms):
        aInAp = True
        for b in neighbours[a]:
            aInAp = aInAp and (bandit.means[b]<=bandit.means[a])
        if aInAp:
            Ap = Ap + [a]

    print("Number of modes:",len(Ap))
    mcomplexity = multimodalComplexity(multimodalMeans, Ap, neighbours)
    lcomplexity = LipschitzComplexity(multimodalMeans, 0.01)



## Gaussian Bandits
Gbandit = GaussianBandit(multimodalMeans)






## Gaussian Algorithms

klucbGlearner = alg.gaussianKLUCB(nbArms, var)
klucbGlearner.clear()

imedmbexpGlearner = IMEDMB(Gbandit.nbArms, klGauss, M = len(Ap), neigh=neighbours, isexp=1)
imedmbexpGlearner.clear()

imedmbGlearner = IMEDMB(Gbandit.nbArms, klGauss,  M = len(Ap), neigh=neighbours)# , M = int(0.5+0.25*nbArms)+1, neigh=neighbours)
imedmbGlearner.clear()

cklucbGlearner = CKL(Gbandit.nbArms, klGauss, 0.01)
cklucbGlearner.clear()



## Gaussian Regrets
klucbGregret = OneBanditOneLearnerMultipleRuns("Gaussian", Gbandit, klucbGlearner, timeHorizon, N_exp, tsave, progression=1)
imedmbGregret = OneBanditOneLearnerMultipleRuns("Gaussian", Gbandit, imedmbGlearner, timeHorizon, N_exp, tsave, progression=1) 
imedmbexpGregret = OneBanditOneLearnerMultipleRuns("Gaussian", Gbandit, imedmbexpGlearner, timeHorizon, N_exp, tsave, progression=1) 
cklucbGregret = OneBanditOneLearnerMultipleRuns("Gaussian", Gbandit, cklucbGlearner, timeHorizon, N_exp, tsave, progression=1) 

## Save
Regrets = np.array([klucbGregret, imedmbGregret, imedmbexpGregret, cklucbGregret])

#save
ID = str(datetime.datetime.now()).replace(":","").replace(" ","")
#save
np.save(path+ID+" - Lipschitz - GRegrets.npy", Regrets)






## Mean Plots
plt.clf()

plt.plot([a for a in range(nbArms)], Gbandit.means, color="k",  linestyle='solid')#, label=imedmbGlearner.name())
plt.plot(randmax(multimodalMeans), max(multimodalMeans), color='r', marker='*', label='global maximum' )
label=True
for a in Ap:
    if label:
        plt.plot(a, multimodalMeans[a], color='purple', marker='*', label= str(len(Ap)-1) + ' strict local maximums' )
    else:
        plt.plot(a, multimodalMeans[a], color='purple', marker='*')
    label=False
plt.plot(randmax(multimodalMeans), max(multimodalMeans), color='r', marker='*')
plt.xlabel("Arm")
plt.ylabel('Mean')
#plt.title("Multimodal Lipschitz distributions")
plt.legend(fontsize = 6)



plt.savefig(path+ID+" - Lipschitz - MultimodalMeans.png")

#imedmbColors = [(2*lmlist-k)/(2*lmlist) for k in range(lmlist)]
## Gaussian plots
plt.clf()
# KLUCB regret
plt.plot(tsave, np.mean(klucbGregret, 0), color="k", linewidth=1.5, label=klucbGlearner.name())
# KLUCB quantiles
plt.plot(tsave, np.quantile(klucbGregret, 0.9, 0), tsave, np.quantile(klucbGregret,0.1,0), markersize = 3, linestyle=':',linewidth=0.7, color="k")
# IMED-MB-exp regret
plt.plot(tsave, np.mean(imedmbexpGregret, 0), marker='v', markevery=10, linewidth=1.5, color='darkblue', label=imedmbexpGlearner.name())
# IMED-MB-exp quantiles
plt.plot(tsave, np.quantile(imedmbexpGregret, 0.9, 0), tsave, np.quantile(imedmbexpGregret,0.1,0), marker='v', markevery=10, markersize = 3, linestyle=':',linewidth=0.7, color='darkblue')
# IMED-MB regret
plt.plot(tsave, np.mean(imedmbGregret, 0), marker='v', markevery=10, linewidth=1.5, color='b', label=imedmbGlearner.name())
# IMED-MB quantiles
plt.plot(tsave, np.quantile(imedmbGregret, 0.9, 0), tsave, np.quantile(imedmbGregret,0.1,0), marker='v', markevery=10, markersize = 3, linestyle=':',linewidth=0.7, color='b')
# CKLUCB regret
plt.plot(tsave, np.mean(cklucbGregret, 0), marker='v', markevery=10, linewidth=1.5, color='red', label=cklucbGlearner.name())
# CKLUCB quantiles
plt.plot(tsave, np.quantile(cklucbGregret, 0.9, 0), tsave, np.quantile(cklucbGregret,0.1,0), marker='v', markevery=10, markersize = 3, linestyle=':',linewidth=0.7, color='red')
# Axis labels
plt.xticks([1, timeHorizon],[1, timeHorizon])
plt.xlabel("Time step")
plt.ylabel("Regret")
plt.legend(fontsize = 6)

plt.savefig(path+ID+" - Lipschitz - Regrets.png")

plt.show()
