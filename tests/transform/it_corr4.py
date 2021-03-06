import numpy as np
import math
np.seterr(all='warn')

# path = "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/student-datasets/Random_Delay_Large/"
path = "/media/rico/Data/TU/thesis/data/Random_Delay_Large/"
traces_path = "{}/{}/".format(path, "traces")
model_path = "{}/{}/".format(path, "Value")

traces_filename = traces_path + "/traces_{}.csv.npy"
model_filename = model_path + "model_{}.csv.npy"
key_guesses_filename = model_path + "key_guesses_{}.csv.npy"
result_filename = path + "correlation_{}"

num_traces = 20000
num_features = 2


step_size = 20000
num_steps = int(num_traces / step_size)


###############################
# VARIABLES TO KEEP TRACK OFF #
###############################
sumX = [[0.0] * num_features] * 256
sumY = [[0.0] * num_features] * 256
totalN = [[0.0] * num_features] * 256
numerator = [[0.0] * num_features] * 256
dl = [[0.0] * num_features] * 256
dr = [[0.0] * num_features] * 256

##############################
# THE CORRELATION IN THE END #
##############################
correlation = np.zeros((num_steps, 256, num_features))
print(np.shape(correlation))

###########################
# RUN OVER THE DATA PARTS #
###########################
for step_index in range(num_steps):

    #################
    # LOAD THE DATA #
    #################
    file_index = (step_index + 1) * step_size
    traces = np.array(np.load(traces_filename.format(file_index)), dtype=np.float64)
    # model_values = np.array(np.load(model_filename.format(file_index)), dtype=np.float64)
    key_guesses = np.array(np.load(key_guesses_filename.format(file_index)), dtype=np.float64)

    print("Opened {} with shape {}".format(traces_filename.format(file_index), traces.shape))

    for feature_index in range(num_features):
        # print("Starting witg index: {}".format(feature_index))
        for subkey in range(256):
            # Select the correct data
            x = traces[:, feature_index]
            y = key_guesses[:, subkey]
            # Calculate the sums
            sumX[subkey][feature_index] += np.sum(x)
            sumY[subkey][feature_index] += np.sum(y)
            totalN[subkey][feature_index] += len(x)

            # Calculate some means
            meanX = sumX[subkey][feature_index] / float(totalN[subkey][feature_index])
            meanY = sumY[subkey][feature_index] / float(totalN[subkey][feature_index])
            meanX2 = float(meanX * meanX)
            meanY2 = float(meanY * meanY)
            meanXY = float(meanX * meanY)
            nMeanXY = float(totalN[subkey][feature_index] * meanXY)

            # Calculate the something similar to covariance and std
            numerator[subkey][feature_index] += np.sum(x * y)  # sum( x_i * y_i)
            dl[subkey][feature_index] += np.sum(x * x)  # This is sum(x_i * x_i)
            dr[subkey][feature_index] += np.sum(y * y)  # This is sum(y_i * y-i)
            d = math.sqrt(dl[subkey][feature_index] - totalN[subkey][feature_index]*meanX2) * \
                math.sqrt(dr[subkey][feature_index] - totalN[subkey][feature_index]*meanY2)
            # Step above is:
            # \sqrt{dl - n * \bar{x}^2} * \sqrt{dr - n * \bar{y}^2}

            # Calculate the correlation
            corr = (numerator[subkey][feature_index] - nMeanXY) / float(d)  # (numerator - n * \bar{x} * \bar{y}) / d

            correlation[step_index][subkey][feature_index] = corr
    print("Saving result")
    np.save(result_filename.format((step_index+1) * step_size), correlation[step_index])
    print("Done saving result")

