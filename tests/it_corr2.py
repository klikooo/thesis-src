import numpy as np
import math

size = 1000000
x = np.arange(1, size)

y = 5 * x
y = y + [np.random.normal(0, 10) for i in range(len(x))]
X = x
Y = y

split = int(size / 2)
a = [x[:split], x[split:]]
b = [y[:split], y[split:]]

###############################
# VARIABLES TO KEEP TRACK OFF #
###############################
sumX = 0.0
sumY = 0.0
totalN = 0.0
numerator = 0.0
dl = 0.0
dr = 0.0

##############################
# THE CORRELATION IN THE END #
##############################
correlation = 0

###########################
# RUN OVER THE DATA PARTS #
###########################
for z in range(len(a)):

    #################
    # LOAD THE DATA #
    #################
    x = a[z]
    y = b[z]

    # Length of current data
    n = len(x)

    # Calculate the sums
    sumX += np.sum(x)
    sumY += np.sum(y)
    totalN += len(x)

    # Calculate some means
    meanX = sumX / float(totalN)
    meanY = sumY / float(totalN)
    meanX2 = float(meanX * meanX)
    meanY2 = float(meanY * meanY)
    meanXY = float(meanX * meanY)
    nMeanXY = float(totalN * meanXY)

    # Calculate part of covariance and std
    numerator += np.sum(x * y)
    dl += np.sum(x * x)
    dr += np.sum(y * y)
    d = math.sqrt(dl - totalN*meanX2) * math.sqrt(dr - totalN*meanY2)

    # Calculate the correlation
    corr = (numerator - nMeanXY) / float(d)

    correlation = corr

print()
print("My2   corr {}".format(correlation))
print("Real Corr: {}".format(np.corrcoef(X, Y)[0, 1]))


