import numpy as np

traces_indices = []
res = []
for i in range(5000):
    traces_indices.append([])
    res.append([])

kernel_size = 25
filename = "/media/rico/Data/TU/thesis/runs3/" \
           "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04_L2_0.0005_kaiming/train40000" \
           "/model_r0_SmallCNN_k{}".format(kernel_size) + "_m64_c128.pt.correct_indices.npy"

print(filename)

indices = np.load(filename, allow_pickle=True)

# For each trace, state whithout each filter is correctly predicted
for i in range(indices.shape[0]):
    for trace_index in indices[i]:
        traces_indices[trace_index].append(i)

count_traces_indices = []
for i in range(len(traces_indices)):
    count_traces_indices.append(len(traces_indices[i]))

print(count_traces_indices)
sorted_indices = [i[0] for i in sorted(enumerate(count_traces_indices), key=lambda x: x[1])]

for i in range(len(traces_indices)):
    res[i] = traces_indices[sorted_indices[i]]

for i in range(len(traces_indices)):
    if len(res[i]) != 0:
        print(f"i{i}: {res[i]}")

traces_map = {}
for i in range(len(traces_indices)):
    if len(res[i]) == 1:
        index = res[i][0]
        value = traces_map.get(index)
        if value is None:
            value = []
        value.append(i)
        traces_map.update({index: value})

print("Traces map single filter")
for k, v in traces_map.items():
    print(f"{k}: {v}")

import util

loader_function = util.load_data_set(util.DataSet.RANDOM_DELAY_NORMALIZED)
traces_path = "/media/rico/Data/TU/thesis/data/"
x_attack, _, _ = loader_function({'use_hw': False,
                                  'traces_path': traces_path,
                                  'raw_traces': False,
                                  'start': 40000 + 1000,
                                  'size': 5000,
                                  'domain_knowledge': True,
                                  'use_noise_data': False,
                                  'data_set': util.DataSet.RANDOM_DELAY_NORMALIZED})

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pdb
for k, v in traces_map.items():
    if len(v) > 5:
        print(f"Plotting filter {k}, with {len(v)} traces")
        samples = []
        for t_index in v:
            plt.plot(x_attack[t_index][500:1000])
            samples.append(x_attack[t_index])
        # plt.figure()

        pca = PCA(2)
        # pdb.set_trace()
        components = pca.fit_transform(samples)
        X = components[:, 0]
        Y = components[:, 1]
        fig, ax = plt.subplots(figsize=(7, 5))
        for j in range(len(X)):
            ax.scatter(X[j], Y[j])

        plt.show()

# z = traces_map[103]
# t1 = z[0]
# t2 = z[1]
# print(np.corrcoef(x_attack[t1], x_attack[t2]))
