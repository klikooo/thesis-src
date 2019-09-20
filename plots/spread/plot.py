import util
import util_classes
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
# matplotlib.rcParams.update({'font.size': 18})


# Function to load the GE of a single model
def get_ge(net_name, model_parameters, load_parameters):
    args = util.EmptySpace()
    for key, value in load_parameters.items():
        setattr(args, key, value)
    folder = "/media/rico/Data/TU/thesis/runs{}/{}".format(args.experiment, util.generate_folder_name(args))

    ge_x, ge_y = [], []
    lta, lva, ltl, lvl = [], [], [], []
    # print(load_parameters['runs'])
    # exit()
    for run in load_parameters['runs']:
        filename = '{}/model_r{}_{}'.format(
            folder,
            run,
            util_classes.get_save_name(net_name, model_parameters))
        ge_path = '{}.exp'.format(filename)

        y_r = util.load_csv(ge_path, delimiter=' ', dtype=np.float)
        x_r = range(len(y_r))
        ge_x.append(x_r)
        ge_y.append(y_r)

        ta, va, tl, vl = util.load_loss_acc(filename)
        lta.append(ta)
        lva.append(va)
        ltl.append(tl)
        lvl.append(vl)

    return ge_x, ge_y, (lta, lva, ltl, lvl)


########################################
# Load the GE results  of the networks #
########################################
def create_plot(network_settings, fig_save_name,
                x_lim,
                y_lim,
                show_loss=True, show_acc=True, font_size=18):
    plt.close('all')
    # matplotlib.rcParams.update({'font.size': font_size})
    ranks_x = []
    ranks_y = []
    rank_mean_y = []
    name_models = []
    model_params = {}
    all_loss_acc = []  # ([], [], [], [])
    plot_colors = []
    for network_name, network_setting in network_settings.items():
        def retrieve_ge(net_setting):
            print(model_params)
            ge_x, ge_y, loss_acc = get_ge(network_name, model_params, net_setting)
            mean_y = np.mean(ge_y, axis=0)
            ranks_x.append(ge_x)
            ranks_y.append(ge_y)
            rank_mean_y.append(mean_y)
            name_models.append(util_classes.get_save_name(network_name, model_params))

            (lvl, ltl, lta, lva) = loss_acc
            # loss_vali, loss_train, acc_train, acc_vali

            net_setting['ge_x'].append(ge_x[0])
            net_setting['ge_y'].append(mean_y)

            net_setting['ta'].append(np.mean(lta, axis=0))
            net_setting['va'].append(np.mean(lva, axis=0))
            net_setting['tl'].append(np.mean(ltl, axis=0))
            net_setting['vl'].append(np.mean(lvl, axis=0))
            net_setting['line_title'].append(util_classes.get_save_name(network_name, model_params))

            all_loss_acc.append(loss_acc)

        for setting in network_setting:
            for cs in setting['channel_sizes']:
                model_params.update({"channel_size": cs})
                for i in range(len(setting['num_layers'])):
                    print(cs)
                    model_params.update({"kernel_size": setting['kernel_sizes'][i]})
                    model_params.update({"num_layers": setting['num_layers'][i]})
                    plot_colors.append(setting['plot_colors'][i])
                    retrieve_ge(setting)
    ###############################################
    # Plot the runs of the same model in one plot #
    ###############################################
    for model_name, model_settings in network_settings.items():
        for model_setting in model_settings:
            # Plot GE
            plt.figure()
            plt.xlabel('Number of traces')#, fontsize=16)
            plt.ylabel('Guessing Entropy')#, fontsize=16)
            plt.grid(True)
            axes = plt.gca()
            axes.set_ylim([0, 256])
            plt.title("{} - {}".format(model_name, model_setting['title']))

            # print(model_setting)

            for i in range(len(model_setting['ge_x'])):
                plt.plot(model_setting['ge_x'][i], model_setting['ge_y'][i],
                         label="{} - {}".format(model_name, model_setting['line_title'][i]),
                         color=model_setting['plot_colors'][i])
            plt.legend()

            # Plot accuracy if asked for
            if show_acc:
                plt.figure()
                plt.title("Accuracy during training {} - {}".format(model_name, model_setting['title']))
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.grid(True)
                for i in range(len(model_setting['ge_x'])):
                    plt.plot(model_setting['ta'][i] * 100, label="Train {}".format(model_setting['line_title'][i]),
                             color='orange', marker=model_setting['plot_markers'][i])
                    plt.plot(model_setting['va'][i] * 100, label="Validation {}".format(model_setting['line_title'][i]),
                             color='green', marker=model_setting['plot_markers'][i])
                plt.legend()

            if show_loss:
                plt.figure()
                plt.title("Loss during training {} - {}".format(model_name, model_setting['title']))
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                for i in range(len(model_setting['ge_x'])):
                    plt.plot(model_setting['tl'][i], label="Train {}".format(model_setting['line_title'][i]),
                             color='orange', marker=model_setting['plot_markers'][i])
                    plt.plot(model_setting['vl'][i], label="Validation {}".format(model_setting['line_title'][i]),
                             color='green', marker=model_setting['plot_markers'][i])
                plt.legend()

        # Plot all GE in same plot

    plt.figure()
    plt.xlabel('Number of traces') #, fontsize=font_size)
    plt.ylabel('Guessing Entropy') #, fontsize=font_size)
    plt.grid(True)
    axes = plt.gca()
    # axes.set_ylim([0, 95])
    axes.set_ylim(y_lim)
    # axes.set_xlim([-1, 65])
    axes.set_xlim(x_lim)
    # plt.title("{} - {}".format(model_name, model_setting['title']))
    for model_name, model_settings in network_settings.items():
        for model_setting in model_settings:
            for i in range(len(model_setting['ge_x'])):
                plt.plot(model_setting['ge_x'][i], model_setting['ge_y'][i],
                         label=f"{model_setting['line_title2']}",
                         color=model_setting['plot_colors'][i],
                         marker=model_setting['plot_markers'][i],
                         markevery=0.1)
    plt.legend()
    # print(dir(mng))
    # exit()
    figure = plt.gcf()
    i = 1
    # figure.set_size_inches(16*i, 9*i)
    figure.savefig(fig_save_name)
