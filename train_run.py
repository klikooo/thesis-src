from train_runner import run
import os
import argparse
import torch.nn as nn


from util import BoolAction, DataSet, func_in_list
from util_classes import get_init_func, require_domain_knowledge

if __name__ == "__main__":

    traces_path = '/media/rico/Data/TU/thesis/data/'
    model_save_path = '/media/rico/Data/TU/thesis/runs/'

    # Default Parameters
    data_set = DataSet.RANDOM_DELAY
    network_names = ["KernelBigTestM"]
    use_hw = False
    runs = 1
    train_sizes = [1000]
    epochs = 75
    batch_size = 100
    lr = 0.0001
    subkey_index = 2
    checkpoints = None
    unmask = True  # Only matters for ASCAD
    raw_traces = True
    desync = 0
    validation_size = 1000
    kernel_size = 5
    channel_size = 8
    num_layers = 4
    spread_factor = 1
    loss_function = nn.CrossEntropyLoss()
    ############################

    ###################
    # Parse arguments #
    ###################
    parser = argparse.ArgumentParser('Train a nn on the ascad db')
    parser.add_argument('-a', "--raw_traces", default=raw_traces, type=bool,
                        help="Load raw traces", action=BoolAction)
    parser.add_argument('-b', "--batch_size", default=batch_size, type=int, help="Batch size")
    parser.add_argument('-c', "--num_layers", default=num_layers, type=int, help="Number of layers for some networks")
    parser.add_argument('-d', "--data_set", default=data_set, type=DataSet.from_string, choices=list(DataSet),
                        help="The data set to use")
    parser.add_argument('-e', "--epochs", default=epochs, type=int, help='Number of epochs')
    parser.add_argument('-f', "--spread_factor", default=spread_factor, type=int, help="The spread factor")
    parser.add_argument('-g', "--channel_size", default=channel_size, type=int, help="Channel size for a CNN")
    parser.add_argument('-i', "--l2_penalty", default=0, type=float, help="L2 penalty")

    parser.add_argument('-k', "--kernel_size", default=kernel_size, type=int, help="Kernel size for a CNN")
    parser.add_argument('-l', "--lr", default=lr, type=float, help="The learning rate")
    parser.add_argument('-m', "--model_save_path", default=model_save_path, type=str,
                        help="Path were the models are saved")
    parser.add_argument('-n', "--use_noise_data", default=False, action=BoolAction, type=bool,
                        help="Use noise in the data set for RD")

    parser.add_argument('-p', "--traces_path", default=traces_path, type=str, help="Path to the traces")

    parser.add_argument('-q', "--desync", default=desync, type=int, help="Desync for ASCAD db")

    parser.add_argument('-r', "--runs", default=runs, type=int, help='Number of runs')
    parser.add_argument('-s', "--subkey_index", default=subkey_index, type=int, help="The subkey index")
    parser.add_argument('-t', "--train_sizes", nargs='+', default=train_sizes, type=int, help='List of train sizes')
    parser.add_argument('-u', "--unmask", default=unmask, type=bool,
                        help="Mask the data with a the mask r[-s]", action=BoolAction)
    parser.add_argument('-v', "--validation_size", default=validation_size, type=int, help="Validation size")
    parser.add_argument('-w', '--network_names', nargs='+', help='List of networks', default=network_names)

    parser.add_argument('-y', "--use_hw", default=use_hw, type=bool, help='Use hamming weight', action=BoolAction)

    args = parser.parse_args()
    print(args)

    # TODO: move this to some util thing, or even the enum class
    def get_raw_feature_size(the_data_set):
        switcher = {DataSet.RANDOM_DELAY: 3500,
                    DataSet.DPA_V4: 3000,
                    DataSet.RANDOM_DELAY_LARGE: 6250,
                    DataSet.RANDOM_DELAY_DK: 3500}
        return switcher[the_data_set]
    # Change input shape according to the selected data set
    input_shape = 700 if args.data_set == DataSet.ASCAD else get_raw_feature_size(args.data_set) if args.raw_traces else 50

    if not os.path.isdir(args.model_save_path):
        print("Model save path ({}) does not exist.".format(args.model_save_path))
        exit(-1)

    print('Using traces path: {}'.format(args.traces_path))
    print('Using model save path: {}'.format(args.model_save_path))

    for train_size in args.train_sizes:
        for network_name in args.network_names:
            init_func = get_init_func(network_name)
            run(use_hw=args.use_hw, spread_factor=args.spread_factor, runs=args.runs,
                train_size=train_size, epochs=args.epochs, lr=args.lr,
                subkey_index=args.subkey_index, batch_size=args.batch_size,
                init=init_func,
                input_shape=input_shape,
                checkpoints=checkpoints,
                unmask=args.unmask,
                traces_path=args.traces_path,
                model_save_path=args.model_save_path,
                data_set=args.data_set,
                raw_traces=raw_traces,
                domain_knowledge=require_domain_knowledge(network_name),
                desync=args.desync,
                validation_size=args.validation_size,
                kernel_size=args.kernel_size,
                loss_function=loss_function,
                use_noise_data=args.use_noise_data,
                channel_size=args.channel_size,
                num_layers=args.num_layers,
                l2_penalty=args.l2_penalty)
