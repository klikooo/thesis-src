#!/usr/bin/env python
# coding: utf-8

# # Generate Simulated Data: Case 4


# import python libraries
import random
from random import randint
from random import gauss
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:90% !important; }</style>"))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import util

# Define lengh for the data set
data_len = 100000
numberOfSamples = 700

# Define key for encryption
key = 23

# Position of the masked and leakage (index starts with 0)
fieldMaskedLeakage = 24
fieldMasked = 4

# Define gauss niose
noiseStart = 0.0
noiseEnd = 1.0

# Define jitter shift range
activateJitter = False

if activateJitter:
    # Value for jitter
    jitterLeft = -4
    jitterRight = 4
else:
    jitterLeft = None
    jitterRight = None

# If masked used True
maskedValue = True

# Define settings table
settingsTable = pd.DataFrame([None])
settingsTable['data_leng'] = pd.DataFrame([data_len])
settingsTable['numberOfSamples'] = numberOfSamples
settingsTable['key'] = key
settingsTable['noiseStart'] = noiseStart
settingsTable['noiseEnd'] = noiseEnd
settingsTable['jitterLeft'] = jitterLeft
settingsTable['jitterRight'] = jitterRight
settingsTable['masked'] = maskedValue
settingsTable = settingsTable.drop(0, axis=True)

# In[4]:


# AES Sbox
AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

# Generate random plaintext
plaintext = []
mask = []
for x in range(data_len):
    plaintext.append(random.randint(0, 255))
    mask.append(random.randint(0, 255))

print("plaintext:", plaintext[0:5], '\nmask', mask[0:5])

# Verify size of plaintext
len(plaintext)

# Verify size of mask
len(mask)

# Generate the label with HW
labels = []
for d in plaintext:
    labels.append(bin(AES_Sbox[d ^ key]).count("1"))
print("\nlabels:", labels[0:5])


maskedData = []
maskedNoise = []
maskIndex = 0
for d in plaintext:
    maskedData.append((bin(((AES_Sbox[d ^ key]) ^ mask[maskIndex])).count("1")) + gauss(noiseStart, noiseEnd))
    maskedNoise.append((bin(mask[maskIndex]).count("1")) + gauss(noiseStart, noiseEnd))
    maskIndex = maskIndex + 1
print("maskedData:", maskedData[0:5], "\nmaskedNoise:", maskedNoise[0:5])

# Verify size of maskedData
len(maskedData)

# Generate random traces and repalce the t=25 with the leaked data
rawTraces = []
randomJitter = []
# Loop to generate the traces
for i in range(data_len):
    rawData = []

    # loop to repalce the sample
    for z in range(numberOfSamples):

        # replace sample at 5th position with the mask and added noise
        if z == fieldMasked:
            rawData.append(maskedNoise[i])

        # Replace sample at 25th position with HW( sbox( plaintext xor key ) xor mask) + noise
        elif z == fieldMaskedLeakage:
            rawData.append(maskedData[i])

        # Else create random sample
        else:
            rawData.append(bin(randint(0, 255)).count("1") + gauss(0, 1))

    # Add jitter
    if activateJitter:
        item = deque(rawData)
        tmpJitter = random.randint(jitterLeft, jitterRight)
        item.rotate(tmpJitter)
        randomJitter.append(tmpJitter)

        # Full dataset with all traces
        rawTraces.append(item)
    else:
        rawTraces.append(rawData)

print("First generated trace: ", rawTraces[0])
print("Second generated trace:", rawTraces[1])
print("Third generated trace: ", rawTraces[2])


# Verify size of leakedData
len(rawTraces)

# ## Plot Simulated Data

# plt.figure()
# plt.plot(rawTraces[0], '-bD', markevery=[fieldMaskedLeakage, fieldMasked])
#
# plt.figure()
# plt.plot(rawTraces[1], '-yD', markevery=[fieldMaskedLeakage, fieldMasked])
#
# plt.figure()
# plt.plot(rawTraces[2], '-gD', markevery=[fieldMaskedLeakage, fieldMasked])
# plt.show()

# ### Create Dataset


# Generate file id
if not activateJitter:
    fileID = 'D' + str(int((noiseEnd * 100))) + '_0'
else:
    fileID = 'D' + str(int((noiseEnd * 100))) + '_' + str(int(jitterRight))


# Traces to csv
simulated_traces = pd.DataFrame(rawTraces)
path = "/media/rico/Data/TU/thesis/data/Simulated_Mask/"
traces = simulated_traces.to_numpy()
# simulated_traces.to_csv(path + fileID + '.csv', index=False)
np.save(path + "traces/traces_complete.csv", traces)

# Metadata to csv
simulated_matadata = pd.DataFrame(plaintext)
print(plaintext[0:5])
simulated_matadata['labels'] = labels
simulated_matadata['mask'] = mask
print(simulated_matadata)


print("Plain: {}, key: {}".format(plaintext[0], key))
unmasked = util.SBOX[plaintext[0] ^ key]
label = unmasked ^ mask[0]
print("Masked: {}, hw : {}".format(label, util.HW[label]))
print("Unmasked: {}, hw: {}".format(unmasked, util.HW[unmasked]))


key_guesses = np.zeros((data_len, 256), dtype=int)
for trace_num in range(data_len):
    for key_guess in range(256):
        key_guesses[trace_num][key_guess] = util.SBOX[plaintext[trace_num] ^ key_guess]

np.save(path + "/Value/key_guesses_ALL_transposed.csv", key_guesses)
np.save(path + "/Value/model.csv", simulated_matadata['labels'].to_numpy(dtype=int))

