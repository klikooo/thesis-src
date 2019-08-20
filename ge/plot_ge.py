import matplotlib.pyplot as plt
import util


kernel_size = 100
run = 1
file = "/media/rico/Data/TU/thesis/runs3/" \
       "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04_L2_0.0005_kaiming/train40000" \
       "/model_r{}_SmallCNN_k{}".format(run, kernel_size) + "_m64_c128_f{}.exp"

for i in range(128):
    channel = util.load_csv(file.format(i), delimiter=",")
    plt.plot(channel)

plt.grid(True)
plt.show()

