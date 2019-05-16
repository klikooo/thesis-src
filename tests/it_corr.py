import numpy as np
import math
from datetime import datetime

size = 1000000
x = np.arange(1, size)

y = 5 * x
y = y + [np.random.normal(0, size) for i in range(len(x))]
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

correlation = 0


t1 = datetime.now()
for z in range(len(a)):

    # Load data here
    x = a[z]
    y = b[z]

    # Length of current data
    n = len(x)

    # Calculate the sums
    for i in range(n):
        sumX += x[i]
        sumY += y[i]
        totalN += 1

    meanX = sumX / float(totalN)
    meanY = sumY / float(totalN)
    meanX2 = float(meanX * meanX)
    meanY2 = float(meanY * meanY)
    meanXY = float(meanX * meanY)
    nMeanXY = float(totalN * meanXY)

    for i in range(n):
        numerator += x[i] * y[i]
        dl += x[i] * x[i]
        dr += y[i] * y[i]

    d = math.sqrt(dl - totalN*meanX2) * math.sqrt(dr - totalN*meanY2)
    corr = (numerator - nMeanXY) / float(d)
    # print("Corr: {}".format(np.corrcoef(x, y)[0, 1]))
    # print("My Corr: {}".format(corr))
    # print()
    correlation = corr

t2 = datetime.now()
print("Time diff: {}".format(t2 - t1))

print()
print("My2   corr {}".format(correlation))
print("Real Corr: {}".format(np.corrcoef(X, Y)[0, 1]))


