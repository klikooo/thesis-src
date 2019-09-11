import argparse

from load.load_threaded import run_load

from util import DataSet, BoolAction


if __name__ == "__main__":
    traces_path = '/media/rico/Data/TU/thesis/data/'
    models_path = '/media/rico/Data/TU/thesis/runs3/'
    # traces_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/student-datasets/'
    # models_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/rtubbing/'

    use_hw = True
    n_classes = 9 if use_hw else 256
    spread_factor = 1
    runs = [x for x in range(5)]
    train_size = 45000
    epochs = 75
    batch_size = 100
    lr = 0.0001
    sub_key_index = 2
    attack_size = 10000
    rank_step = 1
    type_network = 'HW' if use_hw else 'ID'
    unmask = True
    data_set = DataSet.ASCAD_NORM
    kernel_sizes = [100]
    channel_sizes = [32]
    num_layers = [2]
    init_weights = "kaiming"

    network_names = ['VGGNumLayers']
    desync = 50
    num_exps = 1000
    raw_traces = True
    validation_size = 1000
    use_noise_data = False
    max_pool = 4
    l2_penalty = 0.0
    noise_level = 0.0
    load_predictions = True
    save_predictions = False
    #####################################################################################

    parser = argparse.ArgumentParser('Calculate GE for a nn')
    parser.add_argument('-a', "--attack_size", default=attack_size, type=int, help="Attack size")
    parser.add_argument('-b', "--batch_size", default=batch_size, type=int, help="Batch size")
    parser.add_argument('-c', "--num_layers", nargs='+', default=num_layers, type=int, help='List of number of layers')
    parser.add_argument('-d', "--data_set", default=data_set, type=DataSet.from_string, choices=list(DataSet),
                        help="The data set to use")
    parser.add_argument('-e', "--epochs", default=epochs, type=int, help='Number of epochs')
    parser.add_argument('-f', "--spread_factor", default=spread_factor, type=int, help="The spread factor")
    parser.add_argument('-g', "--l2_penalty", default=l2_penalty, type=float, help="L2 penalty")

    parser.add_argument('-i', "--channel_sizes", nargs='+', default=channel_sizes, type=int,
                        help='List of kernel sizes')
    parser.add_argument('-j', "--max_pool", default=max_pool, type=int, help="Max pooling")

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

    parser.add_argument("--noise_level", default=noise_level, type=float, help="Noise level")
    parser.add_argument("--load_predictions", default=load_predictions, type=bool, action=BoolAction,
                        help="Load predictions if existing")
    parser.add_argument("--save_predictions", default=save_predictions, type=bool, action=BoolAction,
                        help="Save predictions")
    parser.add_argument("--only_predictions", default=False, type=bool, action=BoolAction,
                        help="Save only predictions")

    args = parser.parse_args()

    run_load(args)
