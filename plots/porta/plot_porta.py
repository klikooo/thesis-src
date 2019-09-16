import os
import numpy as np
import matplotlib.pyplot as plt


list_num_traces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 200]
# list_num_traces = [5, 10, 100]


def load(train_size, epochs, hw, num_traces, batch_size=256):
    hw_string = 'HW' if hw else 'ID'
    path = '/media/rico/Data/TU/thesis/runs3/KEYS/subkey_2/'
    path = path + f'{hw_string}_SF1_E{epochs}_BZ{batch_size}_LR1.00E-04/train{train_size}/'

    file_path_acc = path + f'traces_{num_traces}_avg_accuracy.npy'
    file_path_ge = path + f'traces_{num_traces}_avg_ge.npy'
    accuracy = np.load(file_path_acc)
    ge = np.load(file_path_ge)
    return accuracy, ge


def retrieve_data(train_sizes, list_epochs):
    results_hw_acc = {}
    results_hw_ge = {}
    results_id_acc = {}
    results_id_ge = {}
    for train_size in train_sizes:
        for epochs in list_epochs:
            key = f"{train_size}_{epochs}"
            single_results_id_acc = []
            single_results_id_ge = []
            single_results_hw_acc = []
            single_results_hw_ge = []
            results_hw_acc.update({key: single_results_hw_acc})
            results_hw_ge.update({key: single_results_hw_ge})
            results_id_acc.update({key: single_results_id_acc})
            results_id_ge.update({key: single_results_id_ge})
            for num_traces in list_num_traces:
                hw_acc, hw_ge = load(train_size, epochs, True, num_traces)
                id_acc, id_ge = load(train_size, epochs, False, num_traces)
                print(id_ge)
                single_results_hw_acc.append(hw_acc)
                single_results_hw_ge.append(hw_ge)
                single_results_id_ge.append(id_ge)
                single_results_id_acc.append(id_acc)
            single_results_id_acc.reverse()
            single_results_hw_acc.reverse()
            single_results_id_ge.reverse()
            single_results_hw_ge.reverse()

    return (results_hw_ge, results_hw_acc), (results_id_ge, results_id_acc)


def create_plot_train_sizes(data, train_sizes, epochs, title=""):
    plt.figure()
    plt.xlabel('Traces')
    plt.ylabel('GE')
    plt.title(f'Epochs {epochs} {title}')
    for train_size in train_sizes:
        key = f"{train_size}_{epochs}"
        x = data[key]
        plt.plot(x, label=f"Train size {train_size}")
    plt.show()
    return plt.gcf()


def create_plot_epochs(data, list_epochs, train_size, title=""):
    plt.figure()
    plt.xlabel('Traces')
    plt.ylabel('GE')
    plt.title(f'Train size {train_size} {title}')
    for epochs in list_epochs:
        key = f"{train_size}_{epochs}"
        x = data[key]
        print(f"x = {x}")
        z = np.mean(x, axis=1)
        # print(z)
        # exit()
        plt.plot(list_num_traces, z, label=f"Epochs {epochs}")
    plt.legend()
    plt.grid(True)
    return plt.gcf()


def run():
    train_sizes = [20000]
    epochs = [75]
    hw, iv = retrieve_data(train_sizes, epochs)
    hw_acc, hw_ge = hw
    iv_acc, iv_ge = iv

    train_size = 20000
    create_plot_epochs(hw_acc, epochs, train_size=train_size, title="hw")
    # create_plot_epochs(iv_acc, epochs, train_size=train_size, title="id")
    plt.show()


if __name__ == "__main__":
    run()
