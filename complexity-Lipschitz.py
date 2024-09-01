## Modules
import sys
path = 'plots/'
sys.path.append(path)

import datetime


import numpy as np

from matplotlib import pyplot as plt

from scipy.optimize import linprog

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




runs= 1000
lip = [] 
multi = []
stand = []
ratio = []
ratiosm = []
ratiosl = []
lenmax = []
for r in range(runs):
    if r%10==0:
        print(r)
    
    nbArms = 500
    variance = round(1/klGauss(0,sqrt(2)),2)

    k = sqrt(3/2)*sqrt(variance/(2*(nbArms-1)))
    means = [0 for a in range(nbArms)]
    means[0] = (1-2*np.random.random())*sqrt(2/3)*sqrt(variance/2)
    for a in range(1,nbArms):
        means[a] = means[a-1]+(1-2*np.random.randint(2))*np.random.random()*sqrt(3/2)*sqrt(variance/(2*(nbArms-1)))


    neighbours = [[] for a in range(nbArms)]
    for a in range(nbArms):
        V = []
        for e in [-5,-4,-3,-2,-1,1,2,3,4,5]:
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
    lenmax = lenmax + [len(Ap)/nbArms]
    
    lc = LipschitzComplexity(means,k)
    mc = multimodalComplexity(means, Ap, neighbours)
    sc = standardComplexity(means)
    if lc is None:
        print('Lipschitz: None')
    else:
        lip = lip + [lc]
        multi = multi + [mc]
        ratio  = ratio + [mc/lc]
        ratiosm = ratiosm + [sc/mc]
        ratiosl = ratiosl + [sc/lc]

plt.clf()
plt.boxplot(np.transpose(np.array([ratiosl, ratiosm, ratio ])))
plt.title("Ratios between asymptotic optimal regrets")
plt.xticks([1,2,3], ["no str./Lipschitz", "no str./Multimodal", "Multimodal/Lipschitz"])
#plt.xlabel("From a sequence of "+str(r) + " random Lipschitz means with 500 arms each")

ID = str(datetime.datetime.now()).replace(":","").replace(" ","")
plt.savefig(path +"ID = "+ID+" - Lipschitz complexity"+ ".png")

plt.clf()
plt.boxplot(lenmax)
plt.xticks(visible=False)
plt.title("The number of local maximums over the number of arms")
#plt.xlabel("from a sequence of "+str(r) + " random Lipschitz means with 500 arms each")

ID = str(datetime.datetime.now()).replace(":","").replace(" ","")
plt.savefig(path +"ID = "+ID+" - Lipschitz - proportion of local maximums"+ ".png")
plt.show()

#print('Lipschitz complexity:', round(LipschitzComplexity(means,k),0))
#print('Multimodal complexity:', round(multimodalComplexity(means, Ap, neighbours),0))

#plt.clf()
#plt.plot([a+1 for a in range(nbArms)], means, color="k",  linestyle='solid')#, label=imedmbGlearner.name())
#label=True
#for a in Ap:
#    if label:
#        plt.plot(a, means[a], color='purple', marker='*', label= str(len(Ap)-1) + ' strict local maximums' )
#    else:
#        plt.plot(a, means[a], color='purple', marker='*')
#    label=False
#plt.plot(randmax(means), max(means), color='r', marker='*', label='global maximum' )
#plt.xlabel("Arm")
#plt.ylabel('Mean')
#plt.title("Multimodal Lipschitz structure")
#plt.legend(fontsize = 6)
#plt.show()

