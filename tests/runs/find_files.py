from os import listdir
from os.path import isfile, join
import glob


path = "/media/rico/Data/TU/thesis/runs3/Random_Delay_Normalized/subkey_2/"
l2_big_kaiming_path = path + "/ID_SF1_E75_BZ100_LR1.00E-04_L2_0.05_kaiming/train40000/"
l2_small_kaiming_path = path + "/ID_SF1_E75_BZ100_LR1.00E-04_L2_0.005_kaiming/train40000/"
l2_big_path = path + "/ID_SF1_E75_BZ100_LR1.00E-04_L2_0.05/train40000/"
l2_small_path = path + "/ID_SF1_E75_BZ100_LR1.00E-04_L2_0.005/train40000/"


def get_files(p, t=""):
    print("For {}\n{}\n".format(p, t))
    f_regex = p + "model_r*_VGGNumLayers_k*_c*_l1.exp"
    # files = listdir(path)
    # print(files)
    # print(listdir(p))
    # files = [f for f in listdir(p) if isfile(join(p, f))]
    # print(files)
    files_paths = glob.glob(f_regex)
    filenames = [x.split("/")[-1] for x in files_paths]
    print(filenames)
    for channel_size in [16, 32]:
        for kernel_size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            for run in range(5):
                f = "model_r{}_VGGNumLayers_k{}_c{}_l1.exp".format(run, kernel_size, channel_size)
                if f not in filenames:
                    print("Missing {}".format(f))


get_files(l2_big_kaiming_path, "0.05 kaiming")
get_files(l2_small_kaiming_path, "0.005 kaiming")
get_files(l2_big_path, "0.05 ")
get_files(l2_small_path, "0.005")


