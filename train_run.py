from models.DenseNet import DenseNet
from models.DenseSpreadNet import DenseSpreadNet
from models.SpreadNet import SpreadNet
from train_runner import run


def init_dense(spread_factor, input_shape, out_shape):
    return DenseSpreadNet(spread_factor=spread_factor, out_shape=out_shape, input_shape=input_shape)


def init_spread(spread_factor, input_shape, out_shape):
    return SpreadNet(spread_factor=spread_factor, out_shape=out_shape, input_shape=input_shape)


def init_mlp_best(spread_factor, input_shape, out_shape):
    return DenseNet(input_shape=input_shape, n_classes=out_shape)


if __name__ == "__main__":
    # Parameters
    init_funcs = [init_mlp_best, init_dense, init_spread]
    use_hw = True
    spread_factor = 6
    runs = 5
    train_sizes = [2000, 5000, 10000]
    epochs = 80
    batch_size = 100
    lr = 0.00001
    # lr = 0.001
    subkey_index = 2
    input_shape = 700
    checkpoints = [100]
    unmask = False if subkey_index < 2 else True

    ############################

    for train_size in train_sizes:
        for init_func in init_funcs:
            run(use_hw=use_hw, spread_factor=spread_factor, runs=runs,
                train_size=train_size, epochs=epochs, lr=lr, subkey_index=subkey_index, batch_size=batch_size,
                init=init_func,
                input_shape=input_shape,
                checkpoints=checkpoints,
                unmask=unmask)
