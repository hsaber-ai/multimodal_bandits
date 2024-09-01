import numpy as np
import pylab as pl
import time
import matplotlib.pyplot as plt
import copy
from Regression_Tools import *
import Signal_Generation as sg

def positive(x):
    return max(x,0)


def quadraticLoss(f,x,y):
    return np.square(f(x)-y)
def selfInformationLoss(p,x,y):
    distribution = p(x)
    return (-np.log(max(distribution(y),0.00001)))


# ----------------------------------------------------------------------------
#  I. Linear Regression class
# ----------------------------------------------------------------------------
class LeastSquaresPredictor:
    def __init__(self,model,regularization=1.,delta = 0.05, sigmaNoise=0.,knownUpperBoundOnSigmaNoise=-1.,optimisticVarianceEstimate=True):
        #sigmaNoise : constant such that log Esp exp (lambda Noise) \leq \lambda^2sigmaNoise^2/2
        #sigmaNoise = -1 : The algorithm estimates sigmaNoise
        #sigmaNoise = -2 : The algorithm performs corrective estimation of sigmaNoise
        #regularization : A penalty "regularization * ||\theta||_2^2" to the least squares objective function.
        # knownUpperBoundOnSigmaNoise : such that \geq sigmaNoise. If set to -1, then assume no bound is known (+infty)
        
        self.delta = 0.05
        self.model=model
        self.modelDimension = len(model(0.))
        self.parameterTheta = np.matrix(self.model(0.)).getT() #of dimension self.modelDimension
        # Regularized or not
        if(regularization>0):
            self.regularization=regularization
            self.isRegularized=True
        else:
            self.regularization=0.
            self.isRegularized=False

        #  Fixed, Adaptive, or Corrective noise?
        if(sigmaNoise>=0):
            self.isCorrectiveNoise=False
            self.isEstimatedNoise=False
            self.sigmaNoise= sigmaNoise
        else:
            self.isEstimatedNoise=(sigmaNoise<0 and sigmaNoise>=-1.)
            self.isCorrectiveNoise=(sigmaNoise<-1 and sigmaNoise>=-2.)
            self.sigmaNoise= 0.00001
            self.maxSigmaNoiseCorrective= self.sigmaNoise
        self.knownUpperBoundOnSigmaNoise = knownUpperBoundOnSigmaNoise
        # Estimation of R = subGaussian level.
        self.sigmaNoiseEstimate = self.sigmaNoise
        self.sigmaNoiseEstimateMin = self.sigmaNoise
        self.sigmaNoiseEstimateMax = self.sigmaNoise
        self.isVarOptimist = optimisticVarianceEstimate#Use sigmaNoiseEstimateMin (optimistic) or sigmaNoiseEstimateMax (pessimistic) in bounds?

        #For fast updates of sigmaNoiseEstimate
        self.cumulativeYnSquare = 0.
        self.cumulativeYnPhi = 0.*np.matrix(self.model(0.)).getT()

        # Prepare for rank-1 updates
        if(self.regularization>0):
            self.FeatureMatrix= (self.regularization)*np.identity(self.modelDimension)
            self.InverseFeatureMatrix= (1./self.regularization)*np.identity(self.modelDimension)
        else:
            self.FeatureMatrix= 0.*np.identity(self.modelDimension)
            self.InverseFeatureMatrix= 0.*np.identity(self.modelDimension) #This is the pseudoinverse.
        self.IsInvertible=self.isRegularized
        self.minimalEigenvalueThreshold=0.01 # (\lambda0) minimal value that must be reached in order to declare Invertibility.
        # Training history
        #self.XObservationPoints = []
        #self.YObservationValues = []
        self.numberOfObservations = 0
        #Accumulated loss over all past steps
        self.cumulativeQuadraticLoss = 0.
        self.cumulativeSelfInformationLoss = 0.
        self.cumulativeTranportationLoss = 0.
        # Loss at last step
        self.lastQuadraticLoss = 0.
        self.lastSelfInformationLoss = 0.
        self.lastTranportationLoss = 0.
        # Only for the bound from RTLin. Max of ||\phi(x_n)||:
        self.bMax = 0.
        
    def clear(self,regularization=1.,sigmaNoise=0.,knownUpperBoundOnSigmaNoise=-1.,optimisticVarianceEstimate=True):
        self.modelDimension = len(self.model(0.))
        self.parameterTheta = np.matrix(self.model(0.)).getT() #of dimension self.modelDimension
        # Regularized or not
        if(regularization>0):
            self.regularization=regularization
            self.isRegularized=True
        else:
            self.regularization=0.
            self.isRegularized=False

        #  Fixed, Adaptive, or Corrective noise?
        if(sigmaNoise>=0):
            self.isCorrectiveNoise=False
            self.isEstimatedNoise=False
            self.sigmaNoise= sigmaNoise
        else:
            self.isEstimatedNoise=(sigmaNoise<0 and sigmaNoise>=-1.)
            self.isCorrectiveNoise=(sigmaNoise<-1 and sigmaNoise>=-2.)
            self.sigmaNoise= 0.00001
            self.maxSigmaNoiseCorrective= self.sigmaNoise
        self.knownUpperBoundOnSigmaNoise = knownUpperBoundOnSigmaNoise
        # Estimation of R = subGaussian level.
        self.sigmaNoiseEstimate = self.sigmaNoise
        self.sigmaNoiseEstimateMin = self.sigmaNoise
        self.sigmaNoiseEstimateMax = self.sigmaNoise
        self.isVarOptimist = optimisticVarianceEstimate#Use sigmaNoiseEstimateMin (optimistic) or sigmaNoiseEstimateMax (pessimistic) in bounds?

        #For fast updates of sigmaNoiseEstimate
        self.cumulativeYnSquare = 0.
        self.cumulativeYnPhi = 0.*np.matrix(self.model(0.)).getT()

        # Prepare for rank-1 updates
        if(self.regularization>0):
            self.FeatureMatrix= (self.regularization)*np.identity(self.modelDimension)
            self.InverseFeatureMatrix= (1./self.regularization)*np.identity(self.modelDimension)
        else:
            self.FeatureMatrix= 0.*np.identity(self.modelDimension)
            self.InverseFeatureMatrix= 0.*np.identity(self.modelDimension) #This is the pseudoinverse.
        self.IsInvertible=self.isRegularized
        self.minimalEigenvalueThreshold=0.01 # (\lambda0) minimal value that must be reached in order to declare Invertibility.
        # Training history
        #self.XObservationPoints = []
        #self.YObservationValues = []
        self.numberOfObservations = 0
        #Accumulated loss over all past steps
        self.cumulativeQuadraticLoss = 0.
        self.cumulativeSelfInformationLoss = 0.
        self.cumulativeTranportationLoss = 0.
        # Loss at last step
        self.lastQuadraticLoss = 0.
        self.lastSelfInformationLoss = 0.
        self.lastTranportationLoss = 0.
        # Only for the bound from RTLin. Max of ||\phi(x_n)||:
        self.bMax = 0.
        

    def name(self):
        xx = str(self.model).split(" ")
        ne =xx[1]
        if(self.isRegularized):
            ne = ne + "-reg_"+str(self.regularization)
        if(self.isEstimatedNoise):
            ne = ne + "-estNoise"
            if (self.isVarOptimist):
                ne = ne + "_opti"
            else:
                ne = ne + "_pess"
        if(self.isCorrectiveNoise):
            ne = ne + "-corNoise"
        if(self.knownUpperBoundOnSigmaNoise!=-1):
            ne = ne + "-maxNoise_" + str(self.knownUpperBoundOnSigmaNoise)
        return ne

    def setKnownUpperBoundOnSigmaNoise(self,sigma):
        self.knownUpperBoundOnSigmaNoise = sigma
    def setVarOptmist(self,bool):
        self.isVarOptimist = bool

    def getmodel(self):
        return self.model


    def addTrainingInput(self,xn,yn):
        phi = np.matrix(self.model(xn)).getT()
        # Only used for RTLin confidence bound:
        self.bMax = max(self.bMax, np.sqrt(np.dot(phi.getT(),phi)))

        # Prepare for parameter estimation (incremental update of useful quantities)
        # ************************************************************************
        self.cumulativeYnPhi = self.cumulativeYnPhi +yn*phi
        self.FeatureMatrix = updateFeatureMatrix(self.FeatureMatrix,phi)
        if (self.IsInvertible):
            self.InverseFeatureMatrix = updateInverseFeatureMatrix(self.InverseFeatureMatrix,phi)
        else:
            self.InverseFeatureMatrix = computePseudoInverseFeatureMatrix(self.FeatureMatrix)
            self.IsInvertible = (np.amin(np.linalg.eigvals(self.FeatureMatrix))>self.minimalEigenvalueThreshold)
            #print(xn,np.linalg.det(self.InverseFeatureMatrix))
            if(self.IsInvertible):
                print("Becomes invertible at step ",xn)

        # Conmpute parameter estimation
        # ************************************************************************
        self.parameterTheta = np.dot(self.InverseFeatureMatrix, self.cumulativeYnPhi)
        self.numberOfObservations=self.numberOfObservations+1
        # Function estimation:
        fp = np.dot(self.parameterTheta.getT(),phi)[0,0]



        delta=self.delta
        n = self.numberOfObservations
        # Loss estimation (if required)
        # ************************************************************************
        #self.lastQuadraticLoss = np.square(yn-fp)
        #self.lastSelfInformationLoss =self.selfInformationLoss(xn,yn) # Uses predict, not predictPlausible
        #self.lastTranportationLoss = self.transportationLossEstimate(xn,yn,delta)
        #self.cumulativeQuadraticLoss = self.cumulativeQuadraticLoss+ self.lastQuadraticLoss
        #self.cumulativeSelfInformationLoss = self.cumulativeSelfInformationLoss + self.lastSelfInformationLoss
        #self.cumulativeTranportationLoss = self.cumulativeTranportationLoss + self.lastTranportationLoss
        #print("Losses:",self.lastQuadraticLoss,self.lastSelfInformationLoss,self.lastTranportationLoss,self.cumulativeQuadraticLoss,self.cumulativeSelfInformationLoss,self.cumulativeTranportationLoss)

        # Perform variance estimation or corrective estimation (if needed)
        # ************************************************************************
        if (self.isCorrectiveNoise):
            deltan = delta/(3*n*n)*6/np.square(np.pi)
            if(self.isRegularized):
                R= noiseEvaluateRegularized(fp,yn,phi, self.InverseFeatureMatrix,self.FeatureMatrix, self.regularization, deltan)
            else:
                R= noiseEvaluateOrdinary(fp,yn,phi, self.InverseFeatureMatrix,self.FeatureMatrix, deltan)
            self.maxSigmaNoiseCorrective = max(self.maxSigmaNoiseCorrective,R,0.0001)
            self.sigmaNoiseEstimateMin = self.maxSigmaNoiseCorrective
            self.sigmaNoiseEstimateMax = self.maxSigmaNoiseCorrective

        if (self.isEstimatedNoise):
            self.cumulativeYnSquare= self.cumulativeYnSquare+yn*yn
            sigma2N=  updateIncrementalVarianceEstimate(self.numberOfObservations, self.cumulativeYnSquare, self.FeatureMatrix,  self.parameterTheta , self.cumulativeYnPhi)
            #print("Sigma:",sigma2N,self.cumulativeYnSquare)
            if (self.knownUpperBoundOnSigmaNoise>=0):
                if(self.isRegularized):
                    [sigmaMin,sigmaMax] = varianceEstimateKnownBoundRegularized(self.numberOfObservations,sigma2N,self.knownUpperBoundOnSigmaNoise,self.FeatureMatrix,self.regularization,delta)
                else:
                    if(self.IsInvertible):
                        [sigmaMin,sigmaMax] = varianceEstimateKnownBoundOrdinary(self.numberOfObservations,sigma2N,self.knownUpperBoundOnSigmaNoise,self.FeatureMatrix,self.minimalEigenvalueThreshold,delta)
                    else:
                        [sigmaMin,sigmaMax] = [0,10000]
                self.sigmaNoiseEstimateMin = sigmaMin
                self.sigmaNoiseEstimateMax = sigmaMax
                if (self.isVarOptimist):
                    self.sigmaNoiseEstimate = min(sigmaMin,self.knownUpperBoundOnSigmaNoise) #optimistic
                else:
                    self.sigmaNoiseEstimate = min(sigmaMax,self.knownUpperBoundOnSigmaNoise)
            else: # No UpperBoundOnSigmaNoise is known
                if(self.isRegularized):
                    [sigmaMin,sigmaMax] = varianceEstimateUnknownBoundRegularized(self.numberOfObservations,sigma2N,self.FeatureMatrix,self.regularization,delta)
                else:
                    if(self.IsInvertible):
                        [sigmaMin,sigmaMax] = varianceEstimateUnknownBoundOrdinary(self.numberOfObservations,sigma2N,self.FeatureMatrix,self.minimalEigenvalueThreshold,delta)
                    else:
                        [sigmaMin,sigmaMax] = [0,10000]
                self.sigmaNoiseEstimateMin = sigmaMin
                self.sigmaNoiseEstimateMax = sigmaMax
                if (self.isVarOptimist):
                    self.sigmaNoiseEstimate = sigmaMin
                else:
                    self.sigmaNoiseEstimate = sigmaMax

       #print("Noise:",self.isCorrectiveNoise,self.isEstimatedNoise,self.sigmaNoise,self.sigmaNoiseEstimate,self.sigmaNoiseEstimateMin,self.sigmaNoiseEstimateMax)


    # Output: f and noise level
    # ************************************************************************
    def f(self,x):# Return the estimated value of f at point x.
        phi = np.mat(self.model(x)).getT()
        fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
        return fp

    def noise(self): # Reutrn the current estimation of the noise level.
         if (self.isCorrectiveNoise):
            no = self.maxSigmaNoiseCorrective
         else:
            no = self.sigmaNoise
         if(self.isEstimatedNoise):
            no = self.sigmaNoiseEstimate
         return max(no,0.00001)


    # Other outputs: probability of observation, confidence sets  etc.
    # ************************************************************************
    def predict(self,x):
        # return density of distribution of yp= f(xp)+\xi
        # Most basic distribution is Gaussian at point xp:
         phi = np.mat(self.model(x)).getT()
         fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
         noise = self.noise()
         return lambda y: np.exp(-np.square(y-fp)/(2*noise))/(np.sqrt(2*np.pi)*noise)
    #
    # def predictPlausible(self,x):
    #     # return density of distribution of yp= f(xp)+\xi, taking into account all uncertainty.
    #     phi = np.mat(self.model(x)).getT()
    #     fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
    #     normModelx = np.sqrt(np.dot(np.dot(phi.getT(),self.InverseFeatureMatrix),phi)[0,0])
    #     noise = self.noise()
    #     if(self.isRegularized):
    #         shift = normModelx*boundThetaRegularized(self.FeatureMatrix, self.regularization, noise, 1./3.)
    #     else:
    #         shift = normModelx*boundThetaOrdinary(self.FeatureMatrix, noise, 1./3.)
    #     a = fp-shift
    #     b = fp+shift
    #     c = noise*(normModelx+1)
    #     return lambda y: np.exp(- np.square(positive(a-y)/c)/2 -np.square(positive(y-b)/c)/2)/ (np.sqrt(2*np.pi)*c+ b-a)
    #
    # # --------------------------------
    def confidenceSet(self,x,delta):
        phi = np.mat(self.model(x)).getT()
        fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
        noise = self.noise()
        if(self.isRegularized):
            borneY = boundYRegularized(phi,self.InverseFeatureMatrix,self.FeatureMatrix,self.regularization,noise,delta/3.)
        else:
            borneY = boundYOrdinary(phi,self.InverseFeatureMatrix,self.FeatureMatrix,noise,delta/3.)
        return fp-borneY,fp+borneY
    #
    # def inModelConfidenceSet(self,x,delta):
    #     phi = np.mat(self.model(x)).getT()
    #     fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
    #     noise = self.noise()
    #     borneY  =  noise*np.sqrt(2*np.log(2./delta))
    #     return fp-borneY,fp+borneY
    #
    # # Confidence bound from OFUL, for comparaison:
    # def confidenceSet_OFUL(self,x,delta):
    #     phi = np.mat(self.model(x)).getT()
    #     fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
    #     noise = self.noise()
    #     borneY = boundY_OFUL(phi, self.InverseFeatureMatrix, self.FeatureMatrix, self.regularization, noise, delta/3.)
    #     return fp-borneY,fp+borneY
    #
    # # Confidence bound from Linearly parameterized bandits, for comparaison:
    # def confidenceSet_RTLin(self,x,delta):
    #     phi = np.mat(self.model(x)).getT()
    #     fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
    #     noise = self.noise()
    #     normModelx = np.sqrt(np.dot(np.dot(phi.getT(),self.InverseFeatureMatrix),phi)[0,0])
    #     bTheta = boundTheta_RTLin(self.FeatureMatrix,  self.numberOfObservations, self.bMax, noise, delta)
    #     borneY= bTheta*normModelx + noise*np.sqrt(2*np.log(1./delta))
    #     return fp-borneY,fp+borneY
    #
    #
    # # --------------------------------
    # #Loss estimation
    # # --------------------------------
    # def transportationPredLossEstimate(self,x,delta):
    #     [ymin,ymax] = self.confidenceSet(x,delta)
    #     mu = (ymin+ymax)/2
    #     deltasig=(self.sigmaNoiseEstimateMax-self.sigmaNoiseEstimateMin)
    #     return max(np.square(ymin-mu) + np.square(deltasig),np.square(ymax-mu)  + np.square(deltasig))
    # def selfInformationPredLossEstimate(self,x,delta):
    #     distribution = self.predictPlausible(x)
    #     [ymin,ymax] = self.confidenceSet(x,delta)
    #     return (-np.log(max(distribution(ymin),distribution(ymax),0.00001)))
    # def quadraticLossPredEstimate(self,x,delta):
    #     [ymin,ymax] = self.confidenceSet(x,delta)
    #     mu = (ymin+ymax)/2
    #     return max(np.square(ymin-mu),np.square(ymax-mu))
    #
    # def transportationLossEstimate(self,x,y,delta):
    #     [ymin,ymax] = self.confidenceSet(x,delta)
    #     deltasig=(self.sigmaNoiseEstimateMax-self.sigmaNoiseEstimateMin)
    #     return  max( np.square(ymin-y),np.square(ymax-y))+ np.square(deltasig)
    # def selfInformationLossEstimate(self,x,y,delta):
    #     distribution = self.predictPlausible(x)
    #     return (-np.log(max(distribution(y),0.00001)))
    # def quadraticLossEstimate(self,x,y,delta):
    #     #return np.square(self.f(x)-y)
    #     [ymin,ymax] = self.confidenceSet(x,delta)
    #     return max( np.square(ymin-y),np.square(ymax-y))
    #
    # def quadraticLoss(self,x,y):
    #     return np.square(self.f(x)-y)
    # def selfInformationLoss(self,x,y):
    #     distribution = self.predict(x)
    #     return (-np.log(max(distribution(y),0.00001)))
    #
    # def quadraticLossOptimisticestimate(self,x,y):
    #     phi = np.mat(self.model(x)).getT()
    #     fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
    #     noise = self.noise()
    #     normModelx = np.sqrt(np.dot(np.dot(phi.getT(),self.InverseFeatureMatrix),phi)[0,0])
    #     if(self.isRegularized):
    #         shift = normModelx*boundThetaRegularized(self.FeatureMatrix, self.regularization, noise, 1./3.)
    #     else:
    #         shift = normModelx*boundThetaOrdinary(self.FeatureMatrix, noise, 1./3.)
    #     a = fp-shift
    #     b = fp+shift
    #     c = noise*(normModelx+1)
    #
    #     R = noise*noise
    #     C = c*c
    #
    #     y0 = min(max(y,a),b)
    #     y1 = max(b,(y/R + b /C)/(1./R+1./C))
    #     y2 = min(a,(y/R + a/ C)/(1./R+1./C))
    #     # multiply by 2R:
    #     g  = lambda z: np.square(y-z) + np.square(max(a-z,0))*R/C +np.square(max(z-b,0))*R/C
    #
    #     return  min(g(y0),g(y1),g(y2))
    #
    # def lossestimate(self,x,delta): #Quadratic? Self-Information? Wasserstein?
    #     # Wasserstein with fixed noise = Quadratic.
    #     # Wasserstein with adaptive noise = ?
    #     # Wasserstein with estimated noise = should be doable.
    #     # return an upper bound estimate on the prediction loss at point x
    #     # sup{ -ln (\rho(y|x)) : \delta(y,x)\geq delta}
    #     # sup  np.square(y-fp)/(2*noise*noise) + 1/2 log(2*np.pi*noise*noise) :
    #      phi = np.mat(self.model(x)).getT()
    #      fp = np.dot(self.parameterTheta.getT(),phi)[0,0]
    #      noise = self.noise()
    #      #[cm,cM] = self.confidenceSet(x,delta)
    #      #return max(np.square(cm-fp),np.square(cM-fp))/(2*noise*noise)-0.5*np.log(2*np.pi*noise*noise)
    #      normModelx = np.sqrt(np.dot(np.dot(phi.getT(),self.InverseFeatureMatrix),phi)[0,0])
    #      if(self.isRegularized):
    #         shift = normModelx*boundThetaRegularized(self.FeatureMatrix, self.regularization, noise, 1./3.) #1/3. for self-info loss?
    #      else:
    #         shift = normModelx*boundThetaOrdinary(self.FeatureMatrix, self.regularization, 1./3.)  #1./3. for self-info loss?
    #      a = fp-shift
    #      b = fp+shift
    #      c = noise*(normModelx+1)
    #      z = c*np.sqrt(2*np.log(1/ delta)) #Probabilist
    #     #z = np.sqrt(2*c*c*np.log(max(1/ (delta*(np.sqrt(2*np.pi)*c+ b-a))),1)) # Possibilist
    #      ym = a-z
    #      yp = b+z
    #      return np.abs(max(np.square(ym-fp),np.square(yp-fp)))
    #      #return max(np.square(ym-fp),np.square(yp-fp))/(2*noise*noise)+0.5*np.log(2*np.pi*noise*noise)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
#  II. Testing an expert: sequential training, predictions and loss.
# ----------------------------------------------------------------------------

def TestExpert(expert,XObservationPoints,YObservationValues,FObservationValues,folder='.',numFigure=1):
    #Example: expert=LeastSquaresPredictor(model,regularization=1.,sigmaNoise=0.,knownUpperBoundOnSigmaNoise=-1.,optimisticVarianceEstimate=True)
    NumberOfObservations = len(XObservationPoints)
    #Values:
    fValues= []
    delta=0.05
    MinConfidenceValues= []
    MaxConfidenceValues = []
    # Losses:
    #SelfInformationLosses = []
    CumulativeQuadraticLosses = []
    CumulativeQuadraticLoss = 0
    #SelfInformationLossesPlausible= []
    for n in range(0,NumberOfObservations):
        xn = XObservationPoints[n]
        yn=  YObservationValues[n]
        fn = FObservationValues[n]

        [mn,Mn] = expert.confidenceSet(xn,delta)
        MinConfidenceValues.append(mn)
        MaxConfidenceValues.append(Mn)
        fValues.append(expert.f(xn))
        CumulativeQuadraticLoss+=quadraticLoss(expert.f,xn,yn)
        CumulativeQuadraticLosses.append(CumulativeQuadraticLoss)
        #SelfInformationLosses.append(selfInformationLoss(expert.predict,xn,yn))
        #SelfInformationLossesPlausible.append(selfInformationLoss(expert.predictPlausible,xn,yn))

        expert.addTrainingInput(xn,yn)

    # Print the expert predictions and losses
    ts = int(time.time())
    title = folder+ '\Xp-' +str(ts)+'-'+expert.name()+'-'

    colors = ['blue','red', 'orange', 'green', 'cyan', 'yellow']
    NumFigure=numFigure
    pl.figure(NumFigure)
    pl.xlabel("x", fontsize=16)
    pl.ylabel("y", fontsize=16)
    pl.plot(XObservationPoints, YObservationValues, 'black', linewidth=4, marker='o', markeredgewidth=2,
            markerfacecolor='none', markersize=8, label="Signal")
    pl.fill_between(XObservationPoints, MinConfidenceValues, MaxConfidenceValues)
    pl.plot(range(0,NumberOfObservations), fValues, 'cyan', linewidth=1, marker='', markeredgewidth=2, markerfacecolor='none', markersize=8, label="Self Information")
    pl.savefig(title+'ConfidenceSets.png')
    pl.savefig(title+'ConfidenceSets.eps')
    pl.savefig(title+'ConfidenceSets.pdf')
    NumFigure+=1
    pl.figure(NumFigure)
    pl.xlabel("x", fontsize=16)
    pl.ylabel("y", fontsize=16)
    pl.plot(range(0,NumberOfObservations), CumulativeQuadraticLosses, 'red', linewidth=4, marker='', markeredgewidth=2, markerfacecolor='none', markersize=8, label="Self Information")
    pl.savefig(title+'CumulativeQuadraticLoss.png')
    pl.savefig(title+'CumulativeQuadraticLoss.eps')
    pl.savefig(title+'CumulativeQuadraticLoss.pdf')
    return NumFigure+1

# ----------------------------------------------------------------------------
# III. Demo
# ----------------------------------------------------------------------------
def demo():
    # Construction of the pieces and of the signal:
    MaxNumberOfObservations = 500
    NumberOfPieces = 1
    MinLengthofPiece = 100
    MaxLengthofPiece = 200
    chanceThatAPiecePreviouslyAppeared = 0.2
    noiseLevel=1.

    XObservationPoints, YObservationValues,  FObservationValues, ChangePointLocations, piecewiseModels = sg.generateSignal(MaxNumberOfObservations,NumberOfPieces,MinLengthofPiece,MaxLengthofPiece,chanceThatAPiecePreviouslyAppeared,noiseLevel)
    folder = "Xps/"
    numFigure=sg.plotSignal(XObservationPoints, YObservationValues,  FObservationValues,piecewiseModels, folder)

    for m in range(len(sg.models)):
        expert= LeastSquaresPredictor(sg.models[m],regularization=1.,sigmaNoise=-1.,knownUpperBoundOnSigmaNoise=-1.,optimisticVarianceEstimate=True)
        print(expert.name())
        numFigure=TestExpert(expert,XObservationPoints,YObservationValues,FObservationValues,folder,numFigure)


#demo()