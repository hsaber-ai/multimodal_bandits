import numpy as np
import pylab as pl
import time

# ----------------------------------------------------------------------------
#  I. Models
# ----------------------------------------------------------------------------
# Definition of a few models
def truemodel0(x):
    return 1.,0
def truemodel1(x):
    return 1.,x/2.
def truemodel2(x):
    return 1.,x/2.,x*x/6.
def truemodel3(x):
    return 1.,np.log(1+x)/2.,np.sqrt(x)/6.,x/24.
def truemodel4(x):
    return 1.,x/2.,x*x/24.,np.sin(x/10.)*10.
def truemodel5(x):
    return 1.,x/2.,np.sin(x/10.)*10.,np.sin(x/20.)*10.
def cosmodel1(x):
    return 1.,np.sin(x/10.)*2.
def cosmodel2(x):
    return 1.,np.sin(x/10.)*2.,np.sin(x/20.)*3.
def cosmodel3(x):
    return 1.,np.sin(x/10.)*2.,np.sin(x/20.)*3.,np.sin(x/50.)*5.

models = {0:truemodel0,1:cosmodel1,2:cosmodel2,3:cosmodel3}

# ----------------------------------------------------------------------------
# II. Generate pieces
# ----------------------------------------------------------------------------
def generatePiecewiseModels(NumberOfPieces,MinLengthofPiece,MaxLengthofPiece,chanceThatAPiecePreviouslyAppeared,noiseLevel=1.):
    piecewiseModels = []
    startChange = 0
    for change in range(0, NumberOfPieces):
        b = np.random.binomial(1, chanceThatAPiecePreviouslyAppeared)
        if ((b==1) and (change > 0)):
            cpast = np.random.randint(0, change)
            result = [startChange, piecewiseModels[cpast][1], piecewiseModels[cpast][2], piecewiseModels[cpast][3]]
        else:
            m = np.random.randint(0, len(models))
            mo=models[m]
            y= len(mo(1.))
            theta= []
            for i in range(y):
                theta.append(np.random.randn())
            result = [startChange, mo, tuple(theta), noiseLevel * np.random.rand()]
        piecewiseModels.append(result)
        startChange = startChange + np.random.randint(MinLengthofPiece, MaxLengthofPiece)
    MaxNumberOfObservations = startChange - 1
    return  MaxNumberOfObservations,piecewiseModels
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# III. Generate the signal
# ----------------------------------------------------------------------------
def generateSignal(MaxNumberOfObservations,NumberOfPieces,MinLengthofPiece,MaxLengthofPiece,chanceThatAPiecePreviouslyAppeared,noiseLevel=1.):
# This generates
#XObservationPoints contains the x points,
#YObservationValues contains the y values,
#FObservationValues contains the  f values, such as y(x) = f(x)+\xi.
    MaxNumberOfObservations2,piecewiseModels = generatePiecewiseModels(NumberOfPieces,MinLengthofPiece,MaxLengthofPiece,chanceThatAPiecePreviouslyAppeared,noiseLevel)
    # Construction of the signal
    # Parameter: number of desired observations
    NumberOfObservations = min(MaxNumberOfObservations,MaxNumberOfObservations2)
    XObservationPoints = []
    FObservationValues = []
    YObservationValues = []
    ChangePointLocations = [p[0] for p in piecewiseModels]
    change=0
    for x in range(0,NumberOfObservations):
         XObservationPoints.append(x)
         f = 0
         if(change+1<NumberOfPieces):
            if (x>= piecewiseModels[change+1][0]):
                change=change+1
         [startChange,trueModel,trueModelParameter,trueModelStd] = piecewiseModels[change]
         f = np.dot(trueModelParameter,trueModel(x-startChange))

         FObservationValues.append(f)
         xi = trueModelStd*np.random.standard_normal()
         YObservationValues.append(f+xi)
    return XObservationPoints, YObservationValues,  FObservationValues, ChangePointLocations, piecewiseModels
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# IV. Plot the generated signal
# ----------------------------------------------------------------------------
def plotSignal(XObservationPoints, YObservationValues,  FObservationValues, piecewiseModels, folder=".",numFigure=1):
    NumFigure=numFigure
     # Print
    ts = int(time.time())
    title = folder+ '\Xp-' +str(ts)+'-TargetSignal-'
    pl.figure(NumFigure)
    pl.xlabel("Nb of observations", fontsize=16)
    pl.ylabel("Values", fontsize=16)
    pl.plot(XObservationPoints, FObservationValues, 'red', linewidth=4, marker='', markeredgewidth=2, markerfacecolor='none', markersize=8, label="Target signal")
    pl.plot(XObservationPoints, YObservationValues, 'black', linewidth=0, marker='o', markeredgewidth=2, markerfacecolor='none', markersize=8, label="Observations")
    pl.xticks([x[0] for x in piecewiseModels if x[0]<len(XObservationPoints)])
    pl.legend(loc='upper left', fontsize=12)
    pl.savefig(title + 'Observations.png')
    pl.savefig(title + 'Observations.eps')
    pl.savefig(title + 'Observations.pdf')
    return NumFigure+1
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# V. Demonstration
# ----------------------------------------------------------------------------
def demo():
    # Construction of the pieces and of the signal:
    MaxNumberOfObservations = 500
    NumberOfPieces = 1
    MinLengthofPiece = 100
    MaxLengthofPiece = 200
    chanceThatAPiecePreviouslyAppeared = 0.2
    noiseLevel=1.

    XObservationPoints, YObservationValues,  FObservationValues, ChangePointLocations, piecewiseModels = generateSignal(MaxNumberOfObservations,NumberOfPieces,MinLengthofPiece,MaxLengthofPiece,chanceThatAPiecePreviouslyAppeared,noiseLevel)
    folder = "."
    plotSignal(XObservationPoints, YObservationValues,  FObservationValues,piecewiseModels, folder)

#demo()