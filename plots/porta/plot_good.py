from decimal import Decimal

import util
import matplotlib.pyplot as plt
import numpy as np

from plots.spread.plot import get_ge


def do(train_size, experiment, data_set):
    load_args = {
        "experiment": f"{experiment}",
        "data_set": data_set,
        "subkey_index": 2,
        "unmask": True,
        "desync": 0,
        "spread_factor": 1,
        "epochs": 75,
        "batch_size": 256,
        "lr": '%.2E' % Decimal(0.0001),
        "l2_penalty": 0,
        "train_size": train_size,
        "use_hw": False,
        # "kernel_sizes": [0],
        # "num_layers": [0],
        # "channel_sizes": [0],
        "runs": range(5),
        "init_weights": "",
        # "title": "",
    }
    model_args = {

    }
    name = "DenseNet"
    return get_ge(name, model_args, load_args)


def create_plots(ge_x, ge_y, data):
    mean_y = np.mean(ge_y, axis=0)
    plt.figure()
    plt.grid(True)
    plt.xlabel('Attack traces') #, fontsize=font_size)
    plt.ylabel('Guessing Entropy') #, fontsize=font_size)
    plt.plot(ge_x[0], mean_y)
    ax = plt.axes()
    ax.yaxis.get_major_formatter().set_powerlimits((-2, 3))

    # import matplotlib.ticker as mtick
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    figure = plt.gcf()

    (lvl, ltl, lta, lva) = data
    ta = np.mean(lta, axis=0) * 100
    va = np.mean(lva, axis=0) * 100
    tl = np.mean(ltl, axis=0)
    vl = np.mean(lvl, axis=0)

    plt.figure()
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(ta, label="Training accuracy")
    plt.plot(va, label="Validation accuracy")
    plt.legend()
    figure_acc = plt.gcf()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.plot(tl, label="Training loss")
    plt.plot(vl, label="Validation loss")
    plt.legend()
    figure_loss = plt.gcf()

    return figure, figure_acc, figure_loss


if __name__ == "__main__":
    # This is a copy of using 39999 traces which have been normalized and attacked with normalized data
    ge_x, ge_y, data = do(39998, '', util.DataSet.KEYS)
    fig, fig_acc, fig_loss = create_plots(ge_x, ge_y, data)
    fig.savefig("/media/rico/Data/TU/thesis/report/img/porta/ge_good_norm.pdf")
    fig_acc.savefig("/media/rico/Data/TU/thesis/report/img/porta/acc_good_norm.pdf")
    fig_loss.savefig("/media/rico/Data/TU/thesis/report/img/porta/loss_good_norm.pdf")

    plt.show()

    # figure = plt.gcf()
    # figure.savefig(fig_save_name)




