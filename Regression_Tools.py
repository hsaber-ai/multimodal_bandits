import numpy as np



# ----------------------------------------------------------------------------
# I. Matrix inversion and parameter estimation
# ----------------------------------------------------------------------------


def updateInverseFeatureMatrix(inverseFeatureMatrix, phin):
    k = np.dot(inverseFeatureMatrix,phin)
    b = 1 + np.dot(phin.getT(),k)
    return inverseFeatureMatrix - np.dot(k,np.dot(b.getI(),k.getT()))

def updateFeatureMatrix(featureMatrix, phin):
    return featureMatrix + np.dot(phin,phin.getT())

def updatePseudoInverseFeatureMatrix(FeatureMatrix,InverseFeatureMatrix, phi):
    # Assumes that FeatureMatrix is not updated yet.
    id=np.identity(len(phi))
    #print("id",len(phi),id)
    u = np.dot(id-  np.dot(FeatureMatrix,InverseFeatureMatrix),phi)
    v = u.getT()#np.dot(phi.getT(),id-  np.dot(InverseFeatureMatrix,FeatureMatrix))
    norm2u = np.dot(u.getT(),u)[0,0]
    #print("u",u)
    #print("v",v)
    k = np.dot(InverseFeatureMatrix,phi)
    h = k.getT()#np.dot(phi.getT(),InverseFeatureMatrix)
    b=1+np.dot(phi.getT(),k)
    # print("b",b)
    # print("k",k)
    # print("h",h)
    # print("norm2u", norm2u)
    if(norm2u <=0.001):
        if (b[0,0]<=0.001):
            norm2k = np.dot(k.getT(),k)[0,0]
            kaux = k.getT()/norm2k
            norm2h=np.dot(h,h.getT())[0,0]
            haux = h.getT()/norm2h
            return InverseFeatureMatrix - np.dot(np.dot(k,kaux),InverseFeatureMatrix) - np.dot(InverseFeatureMatrix,np.dot(haux,h)) + np.dot(np.dot(np.dot(kaux,InverseFeatureMatrix),haux),np.dot(k,h))
        else:
            barb = b
            p = -k
            q = -h
            sigma = b*b
            return  InverseFeatureMatrix  - np.dot(k,np.dot(b.getI(),h))
    else:
        uaux = u.getT()/norm2u
        if (b[0,0]<=0.001):
            norm2h=np.dot(h,h.getT())[0,0]
            haux = h.getT()/norm2h
            return InverseFeatureMatrix - np.dot(InverseFeatureMatrix,np.dot(haux,h)) - np.dot(k,uaux)
        else:
            norm2v = np.dot(v,v.getT())[0,0]
            vaux =  v.getT()/norm2v
            #if(uaux != 0 and vaux != 0):
            return InverseFeatureMatrix - np.dot(k,u.getT())/norm2u - np.dot(v.getT(),h)/norm2v + np.dot(v.getT(),np.dot(b,u.getT()))/(norm2u*norm2v)


def computePseudoInverseFeatureMatrix(FeatureMatrix):
    return np.linalg.pinv(FeatureMatrix)
#----------------------------------------------




# ----------------------------------------------------------------------------
# II. Variance estimate and bound
# ----------------------------------------------------------------------------

def noiseEvaluateRegularized(fn,yn,phix, inverseFeatureMatrix,featureMatrix, regularization, delta):
    normModelx = np.sqrt(np.dot(np.dot(phix.getT(),inverseFeatureMatrix),phix)[0,0])
    detReg=np.linalg.det(featureMatrix/np.sqrt(regularization))
    lmin = np.amin(np.linalg.eigvals(featureMatrix))
    qtt= 2*np.log( (detReg)  /delta)
    return max((np.abs(fn-yn) - regularization/np.sqrt(lmin)*normModelx),0)/( np.sqrt(qtt)*normModelx + np.sqrt(2*np.log(1./delta)))

def noiseEvaluateOrdinary(fn,yn,phix, pseudoInverseFeatureMatrix,featureMatrix, delta):
    normModelx = np.sqrt(np.dot(np.dot(phix.getT(),pseudoInverseFeatureMatrix),phix)[0,0])
    lmax = np.amax(np.linalg.eigvals(featureMatrix))
    d = len(featureMatrix[0])
    x = np.square(np.e*lmax)
    kd=2./3.*np.square(np.pi*np.log(x/np.e))*np.ceil(np.log(x)/2.)*(d + np.exp(d*np.log(12*(d+1)*np.sqrt(d)*x)))
    qtt= 4*np.log( kd  /delta)
    return np.abs(fn-yn)/( np.sqrt(qtt)*normModelx + np.sqrt(2*np.log(1./delta)))

# Bound with \hat R and no R+
# Fast update use updated computation of sigma2N
def varianceEstimateUnknownBoundRegularized(numberOfObservations,sigma2N,featureMatrix,regularization,delta):
    n = numberOfObservations
    dimension = len(featureMatrix[0])
    lambdamin = np.amin(np.linalg.eigvals(featureMatrix))
    lambdamax = np.amax(np.linalg.eigvals(featureMatrix))
    B = np.sqrt(10*dimension)
    C = (np.log(1/delta)+1)*( 1+np.log( np.square(np.pi)*np.log(max(n,2))/6)/np.log(1/delta) )
    D = 2*np.log(  np.sqrt( np.linalg.det(featureMatrix/regularization) ) / delta )
    #Lower and Upper bounds on R
    sigmaNoiseEstimateMin = max( np.sqrt(sigma2N)-B*np.sqrt(regularization*(1- regularization/lambdamax)/n),0)/(1+np.sqrt(2*C/n))
    qtt1 = max(1-np.sqrt(C/n)-np.sqrt( (C + D*(1+regularization/lambdamin)) /n),0.00001)
    qtt2 = 2*np.square(regularization)*B*np.sqrt(D)/(n*np.sqrt(lambdamin)*lambdamin)
    sigmaNoiseEstimateMax = np.square( (np.sqrt(4*np.sqrt(sigma2N)*qtt1 + qtt2)+ np.sqrt(qtt2))/(2*qtt1) )
    #print(n,B,C,D,qtt1,qtt2,lambdamin,sigmaNoiseEstimateMin,sigmaNoiseEstimateMax)
    return sigmaNoiseEstimateMin,sigmaNoiseEstimateMax

def varianceEstimateUnknownBoundOrdinary(numberOfObservations,sigma2N,featureMatrix,lambda0,delta):
    n = numberOfObservations
    dimension = len(featureMatrix[0])
    lambdamax = np.amax(np.linalg.eigvals(featureMatrix))
    x = np.e*lambdamax/lambda0#np.square(np.e*lambdamax)
    kd=2./3.*np.square(np.pi*np.log(x/np.e))*np.ceil(np.log(x)/2.)*(dimension + np.exp(dimension*np.log(12*(dimension+1)*np.sqrt(dimension)*x)))
    B = np.sqrt(10*dimension)
    C = (np.log(1/delta)+1)*( 1+np.log( np.square(np.pi)*np.log(max(n,2))/6)/np.log(1/delta) )
    D = 4*np.log(  kd/ delta )
    #Lower and Upper bounds on R
    sigmaNoiseEstimateMin = np.sqrt(sigma2N)/(1+np.sqrt(2*C/n))
    qtt1 = max(1-np.sqrt(C/n)-np.sqrt( (C + D) /n),0.00001)
    sigmaNoiseEstimateMax = np.sqrt(sigma2N)/qtt1
    #print(n,B,C,D,qtt1,qtt2,lambdamin,sigmaNoiseEstimateMin,sigmaNoiseEstimateMax)
    return sigmaNoiseEstimateMin,sigmaNoiseEstimateMax

# Bound with \hat R with R+
def varianceEstimateKnownBoundRegularized(numberOfObservations,sigma2N,Rmax,featureMatrix,regularization,delta):
    n = numberOfObservations
    dimension = len(featureMatrix[0])
    lambdamin = np.amin(np.linalg.eigvals(featureMatrix))
    lambdamax = np.amax(np.linalg.eigvals(featureMatrix))
    B = np.sqrt(10*dimension)
    C = (np.log(1/delta)+1)*( 1+np.log( np.square(np.pi)*np.log(max(n,2))/6)/np.log(1/delta) )
    D = 2*np.log(  np.sqrt( np.linalg.det(featureMatrix/regularization) ) / delta )
    #print("B,C,D,n:",B,C,D,n)
    #Lower and Upper bounds on R
    sigmaNoiseEstimateMin = max( np.sqrt(sigma2N)-B*np.sqrt(regularization*(1- regularization/lambdamax)/n) - Rmax*np.sqrt(2*C/n),0)
    qtt1 = max(1-np.sqrt(C/n)-np.sqrt( (C + D*(1+regularization/lambdamin)) /n),0.00001)
    qtt2 = 2*np.square(regularization)*B*np.sqrt(D)/(n*np.sqrt(lambdamin)*lambdamin)
    sigmaNoiseEstimateMax = min(Rmax,np.sqrt(sigma2N) + Rmax*(np.sqrt(C/n) + np.sqrt((C+2*D)/n)) + np.sqrt(2*np.sqrt(regularization*D)*Rmax*B/n ))
    #print(Rmax*(np.sqrt(C/n) + np.sqrt((C+2*D)/n)) + np.sqrt(2*np.sqrt(regularization*D)*Rmax*B/n ))
    #print(n,B,C,D,qtt1,qtt2,lambdamin,sigmaNoiseEstimateMin,sigmaNoiseEstimateMax)
    return sigmaNoiseEstimateMin,sigmaNoiseEstimateMax

def varianceEstimateKnownBoundOrdinary(numberOfObservations,sigma2N,Rmax,featureMatrix,lambda0,delta):
    n = numberOfObservations
    dimension = len(featureMatrix[0])
    lambdamax = np.amax(np.linalg.eigvals(featureMatrix))
    x = np.e*lambdamax/lambda0#np.square(np.e*lambdamax)
    kd=2./3.*np.square(np.pi*np.log(x/np.e))*np.ceil(np.log(x)/2.)*(dimension + np.exp(dimension*np.log(12*(dimension+1)*np.sqrt(dimension)*x)))
    B = np.sqrt(10*dimension)
    C = (np.log(1/delta)+1)*( 1+np.log( np.square(np.pi)*np.log(max(n,2))/6)/np.log(1/delta) )
    D = 4*np.log(  kd/ delta )
    #Lower and Upper bounds on R
    sigmaNoiseEstimateMin = max(np.sqrt(sigma2N) - Rmax*np.sqrt(2*C/n),0.)
    qtt1 = max(1-np.sqrt(C/n)-np.sqrt( (C + D) /n),0.00001)
    sigmaNoiseEstimateMax = min(Rmax,np.sqrt(sigma2N) + Rmax*(np.sqrt(C/n)+np.sqrt( (C + D) /n)))
    #print(n,B,C,D,qtt1,qtt2,lambdamin,sigmaNoiseEstimateMin,sigmaNoiseEstimateMax)
    return sigmaNoiseEstimateMin,sigmaNoiseEstimateMax


def updateIncrementalVarianceEstimate(numberObservations, quadraticCumulative, featureMatrix, parameter, sumCumulative):
    # sqrt(  1/N sum yn^2 + theta^top G_N theta /N  - 2/N theta^\top \sum_n y_n \phi_n)
     #self.quadraticCumulative= self.quadraticCumulative+yn*yn
     #self.sumCumulative = self.sumCumulative +yn*phi
    v1 = np.dot(parameter.getT(),np.dot(featureMatrix,parameter))[0,0]
    v2 = np.dot(parameter.getT(),sumCumulative)[0,0]
    return  np.sqrt(quadraticCumulative / numberObservations + v1/numberObservations - 2/numberObservations * v2)




# ----------------------------------------------------------------------------
# II. Confidence Bounds
# ----------------------------------------------------------------------------

def boundThetaRegularized(featureMatrix, regularization, noise, delta):
    B = np.sqrt(10*len(featureMatrix[0])) # Bound on ||\theta*||
    detReg=np.linalg.det(featureMatrix/np.sqrt(regularization))
    lmin = np.amin(np.linalg.eigvals(featureMatrix))
    qtt= 2*np.log( (detReg)  /delta)
    return noise*np.sqrt(qtt) + regularization/np.sqrt(lmin)*B

def boundYRegularized(phix, inverseFeatureMatrix, featureMatrix, regularization, noise, delta):
    normModelx = np.sqrt(np.dot(np.dot(phix.getT(),inverseFeatureMatrix),phix)[0,0])
    bTheta = boundThetaRegularized(featureMatrix, regularization, noise, delta)
    return bTheta*normModelx + noise*np.sqrt(2*np.log(1./delta))

def boundThetaOrdinary(featureMatrix, noise, delta):
    d = len(featureMatrix[0])
    lmax = np.amax(np.linalg.eigvals(featureMatrix))
    if(lmax>0.):
        x = np.square(np.e*lmax)
        kd=2./3.*np.square(np.pi*np.log(x/np.e))*np.ceil(np.log(x)/2.)*(d + np.exp(d*np.log(12*(d+1)*np.sqrt(d)*x)))
        qtt= 4*np.log( kd  /delta)
        return noise*np.sqrt(qtt)
    else:
        return 10000

def boundYOrdinary(phix, inverseFeatureMatrix, featureMatrix, noise, delta):
    normModelx = np.sqrt(np.dot(np.dot(phix.getT(),inverseFeatureMatrix),phix)[0,0])
    bTheta = boundThetaOrdinary(featureMatrix, noise, delta)
    return bTheta*normModelx + noise*np.sqrt(2*np.log(1./delta))

# ----------------------------------------------------------------------------
# Mean bounds from the literature
# ----------------------------------------------------------------------------
# Bound from OFUL (regularized)
# Abbasi-Yadkori, Y., Pal, D. and Szepesvari, C. (2011). Improved Algorithms for Linear Stochastic Bandits. Nips 2011, p. 1�19.

def boundTheta_OFUL(featureMatrix, regularization, noise, delta):
    B = np.sqrt(10*len(featureMatrix[0])) # Bound on ||\theta*||
    detReg=np.linalg.det(featureMatrix/np.sqrt(regularization))
    lmin = np.amin(np.linalg.eigvals(featureMatrix))
    qtt= 2*np.log( (detReg)  /delta)
    return noise*np.sqrt(qtt) + np.sqrt(regularization)*B

def boundY_OFUL(phix, inverseFeatureMatrix, featureMatrix, regularization, noise, delta):
    normModelx = np.sqrt(np.dot(np.dot(phix.getT(),inverseFeatureMatrix),phix)[0,0])
    bTheta = boundTheta_OFUL(featureMatrix, regularization, noise, delta)
    return bTheta*normModelx + noise*np.sqrt(2*np.log(1./delta))

# Bound from Linearly parameterized bandits (ordinary)
# Rusmevichientong, P. and Tsitsiklis, J. N. (2010). Linearly parameterized bandits. Mathematics of Operations Research 35 p. 395�411.
# bmax \geq  max_n ||\phi(x_n)||
def boundTheta_RTLin(featureMatrix, nbObservations, bMax, noise, delta):
    d = len(featureMatrix[0])
    B = np.sqrt(10*d) # Bound on ||\theta*||
    lmin = np.amin(np.linalg.eigvals(featureMatrix))
    lambda0=lmin
    if(lmin>0.1):
        val = 36*bMax*bMax/lambda0
        n=nbObservations
        qtt= 16*(1+np.log(1+val))*(d*np.log(val*n) + np.log(1/delta))*np.log(n)
        #print("RTLin",noise*np.sqrt(qtt))
        return noise*np.sqrt(qtt)
    else:
        return 1000

def boundY_RTLin(phix, inverseFeatureMatrix, featureMatrix, nbObservations, bMax, noise, delta):
    normModelx = np.sqrt(np.dot(np.dot(phix.getT(),inverseFeatureMatrix),phix)[0,0])
    bTheta = boundTheta_RTLin(featureMatrix, nbObservations, bMax, noise, delta)
    return bTheta*normModelx + noise*np.sqrt(2*np.log(1./delta))

# ----------------------------------------------------------------------------
