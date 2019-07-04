import matplotlib.pyplot as plt
import util

file = "/media/rico/Data/TU/thesis/runs3/" \
       "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04_L2_0.0005_kaiming/train40000" \
       "/model_r0_SmallCNN_k50_m64_c128_f{}.exp"

for i in range(128):
    channel = util.load_csv(file.format(i), delimiter=",")
    plt.plot(channel)

plt.grid(True)
plt.show()

