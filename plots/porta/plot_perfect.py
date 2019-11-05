from plots.porta.plot_porta import load, list_num_traces
import matplotlib.pyplot as plt
import numpy as np


def plot(data):
    plt.figure()
    plt.xlabel('Attack traces')
    y_label = 'Guessing entropy'
    # y_lim = [0, 140]
    axes = plt.gca()
    # axes.set_ylim(y_lim)
    plt.ylabel(y_label)
    # plt.title('Guessing entropy perfect model')
    mean = np.mean(data, axis=1)
    plt.plot(list_num_traces, mean)
    # plt.legend()
    plt.grid(True)
    return plt.gcf()


def plot_acc(data):
    plt.figure()
    plt.xlabel('Attack traces')
    y_label = 'Accuracy'
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.ylabel(y_label)
    mean = np.mean(data, axis=1)
    plt.plot(list_num_traces, mean)
    # plt.legend()
    plt.grid(True)
    return plt.gcf()


if __name__ == "__main__":
    ge, acc = [], []
    for num_traces in list_num_traces:
        accuracy, x_ge = load(20000, 75, False, num_traces)
        print(f"Accuracy: {accuracy} {np.mean(accuracy)}")
        ge.append(x_ge)
        acc.append(accuracy)
        print(x_ge)

    fig = plot(ge)
    path = f"/media/rico/Data/TU/thesis/report/img/porta/KEYS/"
    fig.savefig(path + "perfect_model_id.pdf")

    fig = plot_acc(acc)
    path = f"/media/rico/Data/TU/thesis/report/img/porta/KEYS/"
    fig.savefig(path + "perfect_model_id_acc.pdf")

    ge, acc = [], []
    for num_traces in list_num_traces:
        accuracy, x_ge = load(20000, 75, True, num_traces)
        print(f"Accuracy: {accuracy} {np.mean(accuracy)}")
        ge.append(x_ge)
        acc.append(accuracy)
        print(x_ge)

    fig = plot(ge)
    path = f"/media/rico/Data/TU/thesis/report/img/porta/KEYS/"
    fig.savefig(path + "perfect_model_hw.pdf")

    fig = plot_acc(acc)
    path = f"/media/rico/Data/TU/thesis/report/img/porta/KEYS/"
    fig.savefig(path + "perfect_model_hw_acc.pdf")
    plt.show()







