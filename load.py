import argparse

from load.load_threaded import run_load

from util import DataSet, BoolAction


if __name__ == "__main__":
    traces_path = '/media/rico/Data/TU/thesis/data/'
    models_path = '/media/rico/Data/TU/thesis/runs/'
    # traces_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/student-datasets/'
    # models_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/rtubbing/'

    use_hw = False
    n_classes = 9 if use_hw else 256
    spread_factor = 1
    runs = [x for x in range(1)]
    train_size = 45000
    epochs = 5
    batch_size = 100
    lr = 0.0001
    sub_key_index = 2
    attack_size = 1000
    rank_step = 1
    type_network = 'HW' if use_hw else 'ID'
    unmask = True  # False if sub_key_index < 2 else True
    data_set = DataSet.ASCAD_NORMALIZED
    kernel_sizes = [15]
    channel_sizes = [32]
    num_layers = [4]
    init_weights = ""

    # network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
    network_names = ['VGGNumLayers']
    desync = 0
    num_exps = 10
    raw_traces = True
    validation_size = 1000
    use_noise_data = False
    #####################################################################################

    parser = argparse.ArgumentParser('Calculate GE for a nn')
    parser.add_argument('-a', "--attack_size", default=attack_size, type=int, help="Attack size")
    parser.add_argument('-b', "--batch_size", default=batch_size, type=int, help="Batch size")
    parser.add_argument('-c', "--num_layers", nargs='+', default=num_layers, type=int, help='List of number of layers')
    parser.add_argument('-d', "--data_set", default=data_set, type=DataSet.from_string, choices=list(DataSet),
                        help="The data set to use")
    parser.add_argument('-e', "--epochs", default=epochs, type=int, help='Number of epochs')
    parser.add_argument('-f', "--spread_factor", default=spread_factor, type=int, help="The spread factor")
    parser.add_argument('-g', "--l2_penalty", default=0, type=float, help="L2 penalty")

    parser.add_argument('-i', "--channel_sizes", nargs='+', default=channel_sizes, type=int,
                        help='List of kernel sizes')

    parser.add_argument('-k', "--kernel_sizes", nargs='+', default=kernel_sizes, type=int, help='List of kernel sizes')
    parser.add_argument('-l', "--lr", default=lr, type=float, help="The learning rate")
    parser.add_argument('-m', "--models_path", default=models_path, type=str,
                        help="Path were the models are saved")
    parser.add_argument('-n', '--network_names', nargs='+', help='List of networks', default=network_names)
    parser.add_argument('-o', "--use_noise_data", default=use_noise_data, type=bool,
                        help="Use noise data for RD", action=BoolAction)
    parser.add_argument('-p', "--traces_path", default=traces_path, type=str, help="Path to the traces")
    parser.add_argument('-q', "--desync", default=desync, type=int, help="Desync for ASCAD db")
    parser.add_argument('-r', '--runs', nargs='+', help='List of the runs', default=runs)
    parser.add_argument('-s', "--subkey_index", default=sub_key_index, type=int, help="The subkey index")
    parser.add_argument('-t', "--train_size", default=train_size, type=int, help='Train size')
    parser.add_argument('-u', "--unmask", default=unmask, type=bool,
                        help="Mask the data with a the mask r[-s]", action=BoolAction)
    parser.add_argument('-v', "--validation_size", default=validation_size, type=int, help="Validation size used")
    parser.add_argument('-w', "--raw_traces", default=raw_traces, type=bool,
                        help="Use raw traces", action=BoolAction)
    parser.add_argument('-x', "--num_exps", default=num_exps, type=int, help="Number of experiments for GE")
    parser.add_argument('-y', "--use_hw", default=use_hw, type=bool, help='Use hamming weight', action=BoolAction)
    parser.add_argument('-z', "--init_weights", default=init_weights, type=str,
                        help="Specify how the weights are initialized")

    args = parser.parse_args()
    print(args)

    run_load(args)
