import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

train = np.load('/media/rico/Data/TU/thesis/data/KEYS/traces/train_traces.npy')
test = np.load('/media/rico/Data/TU/thesis/data/KEYS_1B/traces/test_traces.npy')

# mean_train = np.mean(train, axis=0)
# mean_test = np.mean(test, axis=0)
# corr_mean = np.corrcoef(mean_train, mean_test)
#
# std_train = np.mean(train, axis=0)
# std_test = np.mean(test, axis=0)
# corr_std = np.corrcoef(std_train, std_test)
#
# plt.figure()
# plt.plot(mean_train, label="Train data")
# plt.plot(mean_test, label="Attack data")
#
# plt.figure()
# plt.plot(std_train, label="Train data")
# plt.plot(std_test, label="Attack data")

# print(corr_mean)
# print(corr_std)


scale = StandardScaler()
train_normalized = scale.fit_transform(train)
mean_train = scale.mean_
std_train = scale.var_

test_normalized = scale.fit_transform(test)
mean_test = scale.mean_
std_test = scale.var_


plt.figure()
plt.xlabel('Feature')
plt.ylabel('Mean')
plt.grid(True)
plt.plot(mean_train, label="Profiling traces")
plt.plot(mean_test, label="Attack traces")
plt.plot(np.abs(mean_train - mean_test), label="Difference")
plt.legend()
fig_mean = plt.gcf()

plt.figure()
plt.xlabel('Feature')
plt.ylabel('Variance')
plt.grid(True)
plt.plot(std_train, label="Profiling traces")
plt.plot(std_test, label="Attack traces")
plt.plot(np.abs(std_train - std_test), label="Difference")
plt.legend()
fig_std = plt.gcf()


fig_mean.savefig("/media/rico/Data/TU/thesis/report/img/porta/traces_mean.pdf")
fig_std.savefig("/media/rico/Data/TU/thesis/report/img/porta/traces_std.pdf")

#######################
# Normalized data #####
#######################
plt.figure()
plt.xlabel('Feature')
plt.ylabel('Mean')
plt.grid(True)
train_mean_norm = np.mean(train_normalized, axis=0)
test_mean_norm = np.mean(test_normalized, axis=0)
# plt.plot(train_mean_norm, label="Train data")
# plt.plot(test_mean_norm, label="Attack data")
plt.plot(np.abs(train_mean_norm - test_mean_norm), label="Difference")
plt.legend()
fig_mean_norm = plt.gcf()

plt.figure()
plt.xlabel('Feature')
plt.ylabel('Variance')
plt.grid(True)
train_std_norm = np.std(train_normalized, axis=0)
test_std_norm = np.std(test_normalized, axis=0)
# plt.plot(train_std_norm, label="Train data")
# plt.plot(test_std_norm, label="Attack data")
plt.plot(np.abs(train_std_norm - test_std_norm), label="Difference")
plt.legend()
fig_std_norm = plt.gcf()

fig_mean_norm.savefig("/media/rico/Data/TU/thesis/report/img/porta/traces_norm_mean.pdf")
fig_std_norm.savefig("/media/rico/Data/TU/thesis/report/img/porta/traces_norm_std.pdf")


plt.show()

