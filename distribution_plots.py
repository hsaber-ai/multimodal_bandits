import sys
path = 'plots/'
sys.path.append(path)

from matplotlib import pyplot as plt

import numpy as np

from BanditTools import randmax

nbArms = 18
multimodalMeans = [0, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1]

Ap = [2, 7, 14]


plt.plot([a+1 for a in range(nbArms)], multimodalMeans, color="k",  linestyle='solid')#, label=imedmbGlearner.name())

for a in range(nbArms):
    plt.plot(a+1,multimodalMeans[a], marker=".", color = 'k' )
    
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*', label='global maximum' )
label=True
for a in Ap:
    if label:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*', label= str(len(Ap)-1) + ' strict local maximums' )
    else:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*')
    label=False
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*')

plt.xlabel("Arm", fontsize = 12)
plt.ylabel('Mean', fontsize=12)
plt.xticks([a+1 for a in range(nbArms)], [a+1 for a in range(nbArms)])
plt.yticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.45],["",0, 0.1, 0.2, 0.3, 0.4, ""])
#plt.title("Multimodal Mean Distributions")
plt.legend(fontsize = 10)
plt.savefig(path+"multimodal_distributions.png")

plt.clf()



nbArms = 18
multimodalMeans = [0, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1]

Ap = [2, 7]


plt.plot([a+1 for a in range(nbArms)], multimodalMeans, color="k",  linestyle='solid')#, label=imedmbGlearner.name())

for a in range(nbArms):
    plt.plot(a+1,multimodalMeans[a], marker=".", color = 'k' )
    
plt.plot(7+1, multimodalMeans[7], color='r', marker='*', label='global maximum')
label=True
for a in Ap:
    if label:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*', label= str(len(Ap)-1) + ' strict local maximum')
    else:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*')
    label=False
plt.plot(7+1, multimodalMeans[7], color='r', marker='*')

plt.xlabel("Arm", fontsize = 12)
plt.ylabel('Mean', fontsize=12)
plt.xticks([a+1 for a in range(nbArms)], [a+1 for a in range(nbArms)])
plt.yticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.45],["",0, 0.1, 0.2, 0.3, 0.4, ""])
#plt.title("Multimodal Mean Distributions")
plt.legend(fontsize = 10)
plt.savefig(path+"misindification_distributions.png")

plt.clf()

nbArms = 18
multimodalMeans = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

Ap = [9]


plt.plot([a+1 for a in range(nbArms)], multimodalMeans, color="k",  linestyle='solid')#, label=imedmbGlearner.name())

for a in range(nbArms):
    plt.plot(a+1,multimodalMeans[a], marker=".", color = 'k' )
    
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*', label='global maximum' )
label=True
for a in Ap:
    if label:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*', label= str(len(Ap)-1) + ' strict local maximums' )
    else:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*')
    label=False
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*')

plt.xlabel("Arm", fontsize = 12)
plt.ylabel('Mean', fontsize=12)
plt.xticks([a+1 for a in range(nbArms)], [a+1 for a in range(nbArms)])
plt.yticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],["", 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, ""])
#plt.title("Multimodal Mean Distributions")
plt.legend(fontsize = 10)
plt.savefig(path+"unimodal_distributions.png")





plt.clf()
nbArms = 18
multimodalMeans = [np.cos(2*np.pi*a/nbArms)+np.sin(4*np.pi*a/nbArms) for a in range(nbArms)]

Ap = []
for a in range(nbArms):
    if a==0 and multimodalMeans[0]>multimodalMeans[1]:
        Ap.append(a)
    elif a==nbArms-1 and multimodalMeans[a-1]<multimodalMeans[a]:
        Ap.append(a)
    else:
        if -1<a-1 and multimodalMeans[a-1]<multimodalMeans[a]:
            if a+1<nbArms and multimodalMeans[a] > multimodalMeans[a+1]:
                Ap.append(a)


plt.plot([a+1 for a in range(nbArms)], multimodalMeans, color="k",  linestyle='solid')#, label=imedmbGlearner.name())

for a in range(nbArms):
    plt.plot(a+1,multimodalMeans[a], marker=".", color = 'k' )
    
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*', label='global maximum' )
label=True
for a in Ap:
    if label:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*', label= str(len(Ap)-1) + ' strict local maximums' )
    else:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*')
    label=False
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*')

plt.xlabel("Arm", fontsize = 12)
plt.ylabel('Mean', fontsize=12)
plt.xticks([a+1 for a in range(nbArms)], [a+1 for a in range(nbArms)])
#plt.yticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],["", 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, ""])
#plt.title("Multimodal Mean Distributions")
plt.legend(fontsize = 10)
plt.savefig(path+"polynomial_distributions.png")




plt.clf()
nbArms = 500

multimodalMeans = [0 for a in range(nbArms)]
multimodalMeans[0]=(1-2*np.random.randint(2))*np.random.random()*0.1
for a in range(1,nbArms):
            multimodalMeans[a] = multimodalMeans[a-1]+(1-2*np.random.randint(2))*np.random.random()*0.01

Ap = []
for a in range(nbArms):
    if a==0 and multimodalMeans[0]>multimodalMeans[1]:
        Ap.append(a)
    elif a==nbArms-1 and multimodalMeans[a-1]<multimodalMeans[a]:
        Ap.append(a)
    else:
        if -1<a-1 and multimodalMeans[a-1]<multimodalMeans[a]:
            if a+1<nbArms and multimodalMeans[a] > multimodalMeans[a+1]:
                Ap.append(a)


plt.plot([a+1 for a in range(nbArms)], multimodalMeans, color="k",  linestyle='solid')#, label=imedmbGlearner.name())

for a in range(nbArms):
    plt.plot(a+1,multimodalMeans[a], marker=".", color = 'k' )
    
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*', label='global maximum' )
label=True
for a in Ap:
    if label:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*', label= str(len(Ap)-1) + ' strict local maximums' )
    else:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*')
    label=False
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*')

plt.xlabel("Arm", fontsize = 12)
plt.ylabel('Mean', fontsize=12)
plt.xticks([a for a in [1, 100, 200, 300, 400, 500]], [a for a in [1, 100, 200, 300, 400, 500]])
#plt.yticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],["", 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, ""])
#plt.title("Multimodal Mean Distributions")
plt.legend(fontsize = 10)
plt.savefig(path+"lipschitz_distributions.png")


plt.clf()


nbArms = 18
multimodalMeans = [0, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1]

means3 = [0, 0.1, 0.4, 0.1, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1]
means9 = [0, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1]
Ap = [2, 7, 14]


plt.plot([a+1 for a in range(nbArms)], multimodalMeans, color="k",  linestyle='solid')#, label=imedmbGlearner.name())
plt.plot([a+1 for a in range(nbArms)], means3, color="brown",  linestyle='dashed', label="confusing distributions for arm 3")
plt.plot([a+1 for a in range(nbArms)], means9, color="orange",  linestyle='dashdot', label="confusing distributions for arm 9")


for a in range(nbArms):
    plt.plot(a+1,multimodalMeans[a], marker=".", color = 'k' )
    
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*', markersize=15, label='global maximum' )
label=True
for a in Ap:
    if label:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*', markersize=15, label= str(len(Ap)-1) + ' strict local maximums' )
    else:
        plt.plot(a+1, multimodalMeans[a], color='purple', marker='*', markersize=15)
    label=False
plt.plot(randmax(multimodalMeans)+1, max(multimodalMeans), color='r', marker='*', markersize=15)
plt.plot(3, max(multimodalMeans), color='brown', marker='*', markersize=15)
plt.plot(9, max(multimodalMeans), color='orange', marker='*', markersize=15)




plt.xlabel("Arm", fontsize = 12)
plt.ylabel('Mean', fontsize=12)
plt.xticks([a+1 for a in range(nbArms)], [a+1 for a in range(nbArms)])
plt.yticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.45],["",0, 0.1, 0.2, 0.3, 0.4, ""])
#plt.title("Multimodal Mean Distributions")
plt.legend(fontsize = 10)
plt.savefig(path+"lb_distributions.png")
plt.show()

plt.clf()