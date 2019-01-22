import torch
import torch.nn.functional as F
import numpy as np

from ascad import HW, test_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(x_attack, y_attack, metadata_attack, network, sub_key_index, use_hw=True, attack_size=10000, rank_step=10
         , unmask=False):
    # Cut to the correct attack size
    x_attack = x_attack[0:attack_size]
    y_attack = y_attack[0:attack_size]

    metadata_attack = metadata_attack[0:attack_size]
    if unmask:
        y_attack = np.array([y_attack[i] ^ metadata_attack[i]['masks'][0] for i in range(attack_size)])

    # Convert values to hamming weight if asked for
    if use_hw:
        y_attack = np.array([HW[val] for val in y_attack])

    # Test the model
    with torch.no_grad():
        data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
        print('x_test size: {}'.format(data.cpu().size()))

        predictions = F.softmax(network(data).to(device), dim=-1).to(device)
        d = predictions[0].cpu().numpy()
        print('Sum predictions: {}'.format(np.sum(d)))

        # Print accuracy
        accuracy(network, x_attack, y_attack)

        # Calculate num of traces needed
        return test_model(predictions.cpu().numpy(), metadata_attack, sub_key_index,
                          use_hw=use_hw,
                          show_plot=False,
                          rank_step=rank_step,
                          unmask=unmask)


def accuracy(network, x_test, y_test):
    data = torch.from_numpy(x_test.astype(np.float32)).to(device)
    predictions = F.softmax(network(data).to(device), dim=-1).to(device)

    _, pred = predictions.max(1)
    z = pred == torch.from_numpy(y_test).to(device)
    num_correct = z.sum().item()
    print('Correct: {}'.format(num_correct))
    print('Accuracy: {}'.format(num_correct / len(y_test)))
