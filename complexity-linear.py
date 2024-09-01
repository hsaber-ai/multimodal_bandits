## Modules
import sys
path = 'plots/'
sys.path.append(path)

import datetime


import numpy as np

from matplotlib import pyplot as plt

from BanditTools import *

## Multimodals Means

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


nbArms = 500

nbModals = 5

runs= 1000
lin = [] 
multi = []
stand = []
ratio = []
ratiosm = []
ratiosl = []
lenmax = []
for r in range(runs):
    if r%10==0:
        print(r)
    
    m = nbModals

    d = 2*m-1

    def phi(x, m):
        return np.array([np.cos(2*np.pi*a*x) for a in range(m)] + [np.sin(2*np.pi*a*x) for a in range(1,m)])

    def x(arm,N = nbArms):
        return arm/N
    
    simplexTheta = np.random.exponential(1,d+1)
    simplexTheta = simplexTheta/sum(simplexTheta)
    theta = [(1-2*np.random.randint(2))*simplexTheta[a] for a in range(d)]

    means = [np.dot(phi(x(a),m), np.array(theta)) for a in range(nbArms)]
    

    neighbours = [[] for a in range(nbArms)]
    for a in range(nbArms):
        V = []
        for e in [-1,1]:
            if (a+e>-1) and (a+e<nbArms):
                V = V + [a+e]
        neighbours[a] = V


    Ap = []
    for a in range(nbArms):
        aInAp = True
        for b in neighbours[a]:
            aInAp = aInAp and (means[b]<=means[a])
        if aInAp:
            Ap = Ap + [a]
    
    mc = multimodalComplexity(means, Ap, neighbours)
    lc=1
    sc = standardComplexity(means)
    
    if lc is None:
        print('Lipschitz: None')
    else:
        lin = lin + [lc]
        multi = multi + [mc]
        ratio  = ratio + [mc/lc]
        ratiosm = ratiosm + [sc/mc]
        ratiosl = ratiosl + [sc/lc]

plt.clf()
plt.boxplot(ratiosm)
plt.title("Ratios between asymptotic optimal regrets")
plt.xticks([1], ["no str./Multimodal"])
#plt.xlabel("From a sequence of "+str(r) + " random Lipschitz means with 500 arms each")

ID = str(datetime.datetime.now()).replace(":","").replace(" ","")
plt.savefig(path +"ID = "+ID+" - Linear complexity"+ ".png")
plt.show()