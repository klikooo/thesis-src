import os
import numpy as np
import matplotlib.pyplot as plt
import util


list_num_traces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 200]


def load(train_size, epochs, hw, num_traces, batch_size=256, data_set=util.DataSet.KEYS):
    hw_string = 'HW' if hw else 'ID'
    path = f'/media/rico/Data/TU/thesis/runs3/{str(data_set)}/subkey_2/'
    path = path + f'{hw_string}_SF1_E{epochs}_BZ{batch_size}_LR1.00E-04/train{train_size}/'

    file_path_acc = path + f'traces_{num_traces}_avg_accuracy.npy'
    file_path_ge = path + f'traces_{num_traces}_avg_ge.npy'
    accuracy = np.load(file_path_acc)
    ge = np.load(file_path_ge)
    return accuracy, ge


def retrieve_data(train_sizes, list_epochs, data_set):
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
                hw_acc, hw_ge = load(train_size, epochs, True, num_traces, data_set=data_set)
                id_acc, id_ge = load(train_size, epochs, False, num_traces, data_set=data_set)
                single_results_hw_acc.append(hw_acc)
                single_results_hw_ge.append(hw_ge)
                single_results_id_ge.append(id_ge)
                single_results_id_acc.append(id_acc)
    return (results_hw_ge, results_hw_acc), (results_id_ge, results_id_acc)


def create_plot_train_sizes(data, train_sizes, epochs, ge):
    plt.figure()
    plt.xlabel('Traces')
    if ge:
        y_label = 'Guessing entropy'
        title = "Guessing entropy"
        y_lim = [0, 140]
    else:
        y_label = "Accuracy"
        title = "Accuracy"
        y_lim = [0, 100]
    axes = plt.gca()
    axes.set_ylim(y_lim)
    plt.ylabel(y_label)
    plt.title(f'Epochs {epochs} {title}')
    for train_size in train_sizes:
        key = f"{train_size}_{epochs}"
        x = data[key]
        x = data[key]
        # print(x)
        z = np.mean(x, axis=1)
        plt.plot(list_num_traces, z, label=f"Train size {train_size}")
    plt.legend()
    plt.grid(True)
    return plt.gcf()


def create_plot_epochs(data, list_epochs, train_size, ge):
    plt.figure()
    plt.xlabel('Traces')
    if ge:
        y_label = 'Guessing entropy'
        title = "Guessing entropy"
        y_lim = [-5, 140]
    else:
        y_label = "Accuracy"
        title = "Accuracy"
        y_lim = [0, 100]
    axes = plt.gca()
    axes.set_ylim(y_lim)
    plt.ylabel(y_label)
    # plt.title(f'Train size {train_size} {title}')
    for epochs in list_epochs:
        key = f"{train_size}_{epochs}"
        x = data[key]
        # print(x)
        z = np.mean(x, axis=1)
        plt.plot(list_num_traces, z, label=f"Epochs {epochs}")
    plt.legend()
    plt.grid(True)
    return plt.gcf()


def save_fig(fig, path):
    # fig.set_size_inches(16, 16)
    fig.savefig(path, dpi=1000)


def run():
    data_set = util.DataSet.KEYS_1
    train_sizes = [1000, 2000, 5000, 10000, 20000]
    epochs = [10, 25, 50, 75]
    # train_sizes = [20000]
    # epochs = [75]
    hw, iv = retrieve_data(train_sizes, epochs, data_set)
    hw_ge, hw_acc = hw
    id_ge, id_acc = iv

    # Select the dirs
    path = f"/media/rico/Data/TU/thesis/report/img/porta/{str(data_set)}/"
    path_epochs = path + "epochs/"
    path_train_size = path + "trainsize/"

    # Make the dirs
    if not os.path.exists(path_epochs):
        os.makedirs(path_epochs)
    if not os.path.exists(path_train_size):
        os.makedirs(path_train_size)

    # Loop over epochs
    for epoch in epochs:
        fig_ge = create_plot_train_sizes(id_ge, train_sizes, epoch, ge=True)
        fig_acc = create_plot_train_sizes(id_acc, train_sizes, epoch, ge=False)
        save_fig(fig_ge, path_epochs + f"ge_id_epoch{epoch}.pdf")
        save_fig(fig_acc, path_epochs + f"acc_id_epoch{epoch}.pdf")

        # Hamming weight
        fig_ge = create_plot_train_sizes(hw_ge, train_sizes, epoch, ge=True)
        fig_acc = create_plot_train_sizes(hw_acc, train_sizes, epoch, ge=False)
        save_fig(fig_ge, path_epochs + f"ge_hw_epoch{epoch}.pdf")
        save_fig(fig_acc, path_epochs + f"acc_hw_epoch{epoch}.pdf")
        plt.close("all")

    for train_size in train_sizes:
        fig_ge = create_plot_epochs(id_ge, epochs, train_size=train_size, ge=True)
        fig_acc = create_plot_epochs(id_acc, epochs, train_size=train_size, ge=False)
        save_fig(fig_ge, path_train_size + f"ge_id_tz{train_size}.pdf")
        save_fig(fig_acc, path_train_size + f"acc_id_tz{train_size}.pdf")

        # Hamming weight
        fig_ge = create_plot_epochs(hw_ge, epochs, train_size=train_size, ge=True)
        fig_acc = create_plot_epochs(hw_acc, epochs, train_size=train_size, ge=False)
        save_fig(fig_ge, path_train_size + f"ge_hw_tz{train_size}.pdf")
        save_fig(fig_acc, path_train_size + f"acc_hw_tz{train_size}.pdf")
        # plt.show()
        plt.close("all")


if __name__ == "__main__":
    run()
