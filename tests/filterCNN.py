from decimal import Decimal

import util
from models.load_model import load_model
from util_classes import get_save_name

import numpy as np

base_path = '/media/rico/Data/TU/thesis/runs2/'
args = {
    "models_path": base_path,
    "data_set": util.DataSet.RANDOM_DELAY,
    "subkey_index": 2,
    "unmask": True,
    "desync": 0,
    "type_network": 'ID',
    "spread_factor": 1,
    "epochs": 120,
    "batch_size": 100,
    "train_size": 20000,
    "lr": 0.001,
    "runs": [0]
}
network_name = "ConvNetKernel"
model_params = {
    "kernel_size": 17,
    "channel_size": 8
}

THRESHOLD = 1/model_params['kernel_size']
print("Threshold:  {}".format(THRESHOLD))


folder = '{}/{}/subkey_{}/{}{}{}_SF{}_' \
         'E{}_BZ{}_LR{}/train{}/'.format(
            args['models_path'],
            str(args['data_set']),
            args['subkey_index'],
            '' if args['unmask'] else 'masked/',
            '' if args['desync'] is 0 else 'desync{}/'.format(args['desync']),
            args['type_network'],
            args['spread_factor'],
            args['epochs'],
            args['batch_size'],
            '%.2E' % Decimal(args['lr']),
            args['train_size'])

# Calculate the predictions before hand
predictions = []
for run in args['runs']:
    model_path = '{}/model_r{}_{}.pt'.format(
        folder,
        run,
        get_save_name(network_name, model_params))
    print('path={}'.format(model_path))

    model = load_model(network_name, model_path)

    variables = ["conv1", "conv2", "conv3"]
    for var in variables:
        weights = model.__getattr__(var).weight.data.cpu().numpy()
        ones = np.ones(np.shape(weights))
        zeros = np.zeros(np.shape(weights))

        plus = np.where(weights < THRESHOLD, ones, zeros)
        minus = np.where(-THRESHOLD < weights, ones, zeros)
        z = plus + minus
        res = np.where(z == 2, ones, zeros)
        # print(res)
        count = np.sum(res, axis=2)

        # ones = np.ones(np.shape(res))
        # zeros = np.zeros(np.shape(res))
        useless_filters = np.where(count > 3, 1, 0)
        print("For {} there are {} weird filters of total of {} filters".format(
            var, np.sum(useless_filters), np.size(plus)))


