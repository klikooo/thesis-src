from scipy.io import loadmat
from Crypto.Cipher import AES
import util
import numpy as np


num_traces = 2000000
file_size = 20000


data_set = "03_19_fpga_aes_ches11_desynch"
# data_set = "unprotected"
step_size = 1000
if data_set == "unprotected":
    step_size = 5000

path = "/media/rico/Data/TU/thesis/data/RD2/{}/matlab_format".format(data_set)
value_path = "/media/rico/Data/TU/thesis/data/Random_Delay_Large/Value/"
traces_path = "/media/rico/Data/TU/thesis/data/Random_Delay_Large/traces/"
key_path = "/media/rico/Data/TU/thesis/data/Random_Delay_Large/"
key_guesses_filename = "key_guesses_{}.csv"
traces_filename = "traces_{}.csv"
model_filename = "model_{}.csv"


# Define a leakage function
def leakage_function(ct, key_guess):
    inverse_sbox = util.SBOX_INV[ct ^ key_guess]
    return abs(ct - inverse_sbox)


def leakage_function2(ct, key_guess):
    inverse_sbox = util.SBOX_INV[ct ^ key_guess]
    return ct ^ inverse_sbox


# Convert matlab plaintext to hex
def conv_to_plain(ptext):
    hex_string = "{}{}{}{}".format(hex(ptext[3])[2:].zfill(8), hex(ptext[2])[2:].zfill(8),
                                   hex(ptext[1])[2:].zfill(8), hex(ptext[0])[2:].zfill(8))
    return bytes.fromhex(hex_string)


# Create the key guesses for a single ciphertext byte
def generate_key_guess(ct):
    key_guesses = []
    for key_guess in range(256):
        key_guesses.append(leakage_function2(ct, key_guess))
    return key_guesses


# HD(cts[j][0] , INV_SBOX[cts[j][0]^ last_r_key[0]]) = cts[j][0]^ INV_SBOX[cts[j][0]^ last_r_key[0]]
# master key 2b7e151628aed2a6abf7158809cf4f3c
# plain 61e99ea0f35fa767dd5554e7485c3261
# cipher 3a	bd	71	1d	d0	44	89	0f	0b	83	88	5f	6b	6a	f47d

last_r_key = [0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xa1, 0x3f, 0x0c, 0xc8, 0xb6, 0x63, 0x0c, 0xa6]
master_key = bytes([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c])


sub_key = 0
round_sub_key = last_r_key[sub_key]
aes = AES.new(master_key, AES.MODE_ECB)


for file_step in range(int(num_traces / file_size)):
    # Variables we want to store
    all_key_guesses = []
    model_values = []
    traces = []
    ciphertexts = []

    end_size = file_size * (file_step+1)
    start = file_size * file_step + step_size

    # Perform the conversion
    for file_index in range(start, end_size+1, step_size):
        file = "traces{}.mat".format(file_index)
        x = loadmat('{}/{}'.format(path, file))
        ptexts = x['ptexts']

        print("Opened {}".format(file_index))

        for i in range(step_size):
            # Do the encryption of the plaintext
            plain = conv_to_plain(x['ptexts'][i])
            bytes_ct = aes.encrypt(plain)

            # Add the traces
            traces.append(x['traces'][i])

            # Select a byte
            ct_byte = int(bytes_ct[sub_key])
            ciphertexts.append(ct_byte)

            # Do inv sbox and calculate HD
            leakage = leakage_function2(ct_byte, last_r_key[sub_key])

            # Store the HD for model file
            model_values.append(leakage)

            # Generate the key guesses:
            all_key_guesses.append(generate_key_guess(ct_byte))

        # Save the key guesses to a file

    # traces = np.array(traces)[:, 900:1200]

    print("Saving {}".format(key_guesses_filename.format(end_size)))
    print("s to {}/{}".format(value_path, key_guesses_filename.format(end_size)))
    print("Traces format: {}".format(np.shape(traces)))
    np.save("{}/{}".format(value_path, key_guesses_filename.format(end_size)), all_key_guesses)
    np.save("{}/{}".format(traces_path, traces_filename.format(end_size)), traces)
    np.save("{}/{}".format(value_path, model_filename.format(end_size)), model_values)

# Save the key
# util.save_np("{}/secret_key.csv".format(key_path), [last_r_key[sub_key]])
# all_key_guesses = np.array(all_key_guesses)
# traces = np.array(traces)
# print("Shape model values: {}".format(np.shape(model_values)))
# print("Shape traces: {}".format(np.shape(traces)))
#
#
# num_features = 6250
# plot_real_key = []
# # plot_real_key[4].append(1)
#
# for trace_point in range(0, num_features):
#     probabilities = [0] * 256
#     for kguess in range(0, 256):
#         kguess_vals = all_key_guesses[:, kguess]
#         traces_points = traces[:, trace_point]
#         corr = np.corrcoef(traces_points, kguess_vals)[1][0]
#         # print("For kguess {}: {}".format(kguess, corr))
#         probabilities[kguess] = abs(corr)
#     plot_real_key.append(probabilities)
#     print("Point {}, Max: {}".format(trace_point, np.argmax(probabilities)))
#
# plt.figure()
# plt.plot(plot_real_key)
# real = np.array(plot_real_key)[:, 208]
# print(np.array(plot_real_key)[:, 208])
# plt.plot(real, label="Real key", marker="+", color='gold')
# plt.legend()
# plt.show()
# 72
# 32
# 4
# 61
# 76
# 40
# 25
# 93
# 194
# 154
# 204
# 138
# 92
# 178
# 211
# 55
# 32
# 214
# 164
# 254
# 103
# 129
# 183
# 9
# 242
# 105
# 82
# 176
# 253
# 254
# 0
# 54
# 248
# 18
# 44
# 135
# 24
# 40
# 38
# 165
# 97
# 38
# 207
# 204
# 189
# 191
# 149
# 1
# 214
# 6
