# -*- coding: utf-8 -*-
'''
Useful functions for bandit algorithms (especially KL-UCB)
'''

from math import log, sqrt, exp
import numpy as np

## A function that returns an argmax at random in case of multiple maximizers

def randmax(A):
    maxValue=max(A)
    index = [i for i in range(len(A)) if A[i]==maxValue]
    return np.random.choice(index)

def randmin(A):
    minValue=min(A)
    index = [i for i in range(len(A)) if A[i]==minValue]
    return np.random.choice(index)

## Kullback-Leibler divergence in exponential families

eps = 1e-15

def klBern(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1-eps)
    y = min(max(y, eps), 1-eps)
    return x*log(x/y) + (1-x)*log((1-x)/(1-y))


def klGauss(x, y, sig2 = 0.25):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return (x-y)*(x-y)/(2*sig2)


def klPoisson(x, y):
    """Kullback-Leibler divergence for Poison distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return y-x+x*log(x/y)


def klExp(x, y):
    """Kullback-Leibler divergence for Exponential distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return (x/y - 1 - log(x/y))


## computing the KL-UCB indices

def klucb(x, level, div, upperbound, lowerbound=-float('inf'), precision=1e-6):
    """Generic klUCB index computation using binary search:
    returns u>x such that div(x,u)=level where div is the KL divergence to be used.
    """
    l = max(x, lowerbound)
    u = upperbound
    while u-l>precision:
        m = (l+u)/2
        if div(x, m)>level:
            u = m
        else:
            l = m
    return (l+u)/2


def klucbBern(x, level, precision=1e-6):
    """returns u such that kl(x,u)=level for the Bernoulli kl-divergence."""
    upperbound = min(1.,x+sqrt(level/2))
    return klucb(x, level, klBern, upperbound, precision)


def klucbGauss(x, level, sig2=1., precision=0.):
    """returns u such that kl(x,u)=level for the Gaussian kl-divergence (can be done in closed form).
    """
    return x + sqrt(2*sig2*level)


def klucbPoisson(x, level, precision=1e-6):
    """returns u such that kl(x,u)=level for the Poisson kl-divergence."""
    upperbound = x+level+sqrt(level*level+2*x*level)
    return klucb(x, level, klPoisson, upperbound, precision)


def klucbExp(x, d, precision=1e-6):
    """returns u such that kl(x,u)=d for the exponential kl divergence."""
    if d<0.77:
        upperbound = x/(1+2./3*d-sqrt(4./9*d*d+2*d)) # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
    else:
        upperbound = x*exp(d+1)
    if d>1.61:
        lowerbound = x*exp(d)
    else:
        lowerbound = x/(1+d-sqrt(d*d+2*d))
    return klucb(x, d, klExp, upperbound, lowerbound, precision)


    
## Bandits

def complexity(bandit, kullback=klBern):
    maxMean = max(bandit.means)
    return sum([(maxMean - bandit.means[a])/kullback(bandit.means[a],maxMean) for a in range(bandit.nbArms) if bandit.means[a] != maxMean])

def unimodalComplexity(bandit, kullback=klBern):
    if bandit.bestarm == 0:
        Vbestarm = [1]
    elif bandit.bestarm == bandit.nbArms-1:
        Vbestarm = [bandit.nbArms-2]
    else:
        Vbestarm = [bandit.bestarm - 1, bandit.bestarm + 1]
        
        
    maxMean = bandit.means[bandit.bestarm]
    return sum([(maxMean - bandit.means[a])/kullback(bandit.means[a],maxMean) for a in Vbestarm])



## Run Algorithms on Bandits

def OneBanditOneLearnerOneRun(bandit, learner, timeHorizon, pulls=False, nR=1):
    """
    Run a bandit strategy (learner) on a MAB instance (bandit) for (timeHorizon) time steps
    output : sequence of arms chosen, sequence of rewards obtained
    """
    selections = []
    rewards = []
    bestarm = randmax(bandit.means)
    nC = bandit.nbArms//nR
    draws = np.array([ [0 for c in range(nC)] for r in range(nR)])
    
    learner.clear() # reset previous history
    for t in range(timeHorizon):
        arm = learner.chooseArmToPlay()
        if arm != bestarm:
            r = arm//nC
            c = arm - r*nC
            draws[r,c] = draws[r,c] + 1 
        reward = bandit.generateReward(arm)
        learner.receiveReward(arm, reward)
        # Update statistics
        selections.append(arm)
        rewards.append(reward)
    if pulls:
        return selections, rewards, draws
    else:
        return selections, rewards
    
def CumulativeRegret(bandit,selections):
    """Compute the pseudo-regret associated to a sequence of arm selections"""
    return np.cumsum(max(bandit.means)*np.ones(len(selections)) - np.array(bandit.means)[selections])


def OneBanditOneLearnerMultipleRuns(distribution, bandit, learner, timeHorizon, N_exp, tsave=[], progression = 1, pulls=False, nR=1):
    if (len(tsave) == 0):
        tsave = np.arange(timeHorizon)
    
    savedTimes = len(tsave)
    Regret = np.zeros((N_exp, savedTimes)) 
    
    nC = bandit.nbArms//nR
    Draws = np.zeros((nR,nC))
    
    for n in range(N_exp):
        if (n+1)%progression==0:
            print(learner.name(), ": Run", n+1, " (", distribution,")")
        np.random.seed()
        if pulls:
            selections, rewards, draws = OneBanditOneLearnerOneRun(bandit, learner, timeHorizon, pulls=pulls, nR = nR)
            regret_one_run = CumulativeRegret(bandit, selections)
            Regret[n, :] = np.array(regret_one_run)[tsave]
            Draws = Draws + draws
        else:
            selections, rewards = OneBanditOneLearnerOneRun(bandit, learner, timeHorizon)
            regret_one_run = CumulativeRegret(bandit, selections)
            Regret[n, :] = np.array(regret_one_run)[tsave] 
    if pulls:    
        Draws = Draws//N_exp
        return Regret, Draws
    else:
        return Regret