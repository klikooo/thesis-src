import subprocess
import os

path = "/media/rico/Data/TU/thesis/runs3/ASCAD/subkey_2/"
filename = path + "denseFiles.txt"

with open(filename, "r") as f:
    lines = f.readlines()

    # for file in lines:
    #     f = path + file.replace("./", "").strip()
    #     print(f)
        # if os.path.islink(f):
        #     print("LINK")
        # exit()


    # exit()
    for file in lines:
        file = file.strip().replace("./", "/")
        if "SF3" in file:
            continue

        # print(file)
        for spread_factor in [3, 6, 9, 12]:
            new_file = file.replace("SF1", "SF{}".format(spread_factor))
            directory = os.path.dirname(os.path.realpath(path + new_file))
            if os.path.isfile(path + new_file):
                print("Exists {}".format(new_file))
                continue

            # Change directory
            # os.chdir(directory)
            # subprocess.run(["ln", "-s", "../../{}".format(file), "."])

            # print("{} - {} ".format(new_file, directory))
            # print("Link file: {} to {}".format(file, new_file))
            # exit()
        # print()


        # exit()

    # print(lines)


