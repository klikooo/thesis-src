from models.ConvNet import ConvNet
from models.DenseNet import DenseNet
from models.DenseSpreadNet import DenseSpreadNet
from models.SpreadNet import SpreadNet
from models.CosNet import CosNet
from models.SpreadV2 import SpreadV2
from train_runner import run
import os
import argparse

from util import BoolAction, DataSet

if __name__ == "__main__":

    traces_path = '/media/rico/Data/TU/thesis/data/'
    model_save_path = '/media/rico/Data/TU/thesis/runs/'

    # Default Parameters
    data_set = DataSet.RANDOM_DELAY
    init_funcs = [ConvNet.init]
    use_hw = False
    spread_factor = 1
    runs = 2
    train_sizes = [5000]
    epochs = 80
    batch_size = 100
    lr = 0.0001
    # lr = 0.001
    subkey_index = 2
    checkpoints = None
    unmask = False  # False if subkey_index < 2 else True
    raw_traces = True
    ############################

    # Parse arguments
    parser = argparse.ArgumentParser('Train a nn on the ascad db')
    parser.add_argument('-y', "--use_hw", default=use_hw, type=bool, help='Use hamming weight', action=BoolAction)
    parser.add_argument('-r', "--runs", default=runs, type=int, help='Number of runs')
    parser.add_argument('-t', "--train_sizes", nargs='+', default=train_sizes, type=int, help='List of train sizes')
    parser.add_argument('-e', "--epochs", default=epochs, type=int, help='Number of epochs')
    parser.add_argument('-b', "--batch_size", default=batch_size, type=int, help="Batch size")
    parser.add_argument('-l', "--lr", default=lr, type=float, help="The learning rate")
    parser.add_argument('-s', "--subkey_index", default=subkey_index, type=int, help="The subkey index")
    parser.add_argument('-u', "--unmask", default=unmask, type=bool,
                        help="Mask the data with a the mask r[-s]", action=BoolAction)
    parser.add_argument('-p', "--traces_path", default=traces_path, type=str, help="Path to the traces")
    parser.add_argument('-m', "--model_save_path", default=model_save_path, type=str,
                        help="Path were the models are saved")
    parser.add_argument('-f', "--spread_factor", default=spread_factor, type=int, help="The spread factor")
    parser.add_argument('-d', "--data_set", default=data_set, type=DataSet.from_string, choices=list(DataSet),
                        help="The data set to use")
    parser.add_argument('-a', "--raw_traces", default=raw_traces, type=bool,
                        help="Load raw traces", action=BoolAction)
    args = parser.parse_args()
    print(args)

    def get_raw_input_size(the_data_set):
        switcher = {DataSet.RANDOM_DELAY: 3500,
                    DataSet.DPA_V4: -1}
        return switcher[the_data_set]
    # Change input shape according to the selected data set
    input_shape = 700 if args.data_set == DataSet.ASCAD else get_raw_input_size(args.data_set) if args.raw_traces else 50

    if not os.path.isdir(args.model_save_path):
        print("Model save path ({}) does not exist.".format(args.model_save_path))
        exit(-1)

    print('Using traces path: {}'.format(args.traces_path))
    print('Using model save path: {}'.format(args.model_save_path))

    for train_size in args.train_sizes:
        for init_func in init_funcs:
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
                raw_traces=raw_traces)
