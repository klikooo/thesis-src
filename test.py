import torch
import torch.nn.functional as F
import numpy as np

from util import HW, test_model, device


def test(x_attack, y_attack, metadata_attack, network, sub_key_index, use_hw=True, attack_size=10000, rank_step=10,
         unmask=False,
         only_accuracy=False):
    # Cut to the correct attack size
    x_attack = x_attack[0:attack_size]
    y_attack = y_attack[0:attack_size]

    metadata_attack = metadata_attack[0:attack_size]
    if unmask:
        y_attack = np.array([y_attack[i] ^ metadata_attack[i]['masks'][15] for i in range(attack_size)])

    # Convert values to hamming weight if asked for
    if use_hw:
        y_attack = np.array([HW[val] for val in y_attack])

    # Test the model
    with torch.no_grad():
        data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
        print('x_test size: {}'.format(data.cpu().size()))

        predictions = F.softmax(network(data).to(device), dim=-1).to(device)
        # d = predictions[0].cpu().numpy()
        # print('Sum predictions: {}'.format(np.sum(d)))

        # Print accuracy
        accuracy(network, x_attack, y_attack)

        if not only_accuracy:
            # Calculate num of traces needed
            return test_model(predictions.cpu().numpy(), metadata_attack, sub_key_index,
                              use_hw=use_hw,
                              rank_step=rank_step,
                              unmask=unmask)
        else:
            return None, None


def accuracy(network, x_test, y_test):
    data = torch.from_numpy(x_test.astype(np.float32)).to(device)
    predictions = F.softmax(network(data).to(device), dim=-1).to(device)

    _, pred = predictions.max(1)
    z = pred == torch.from_numpy(y_test).to(device)
    num_correct = z.sum().item()
    print('Correct: {}'.format(num_correct))
    print('Accuracy: {}'.format(num_correct / len(y_test)))


def test_with_key_guess(x_attack, y_attack, key_guesses, network,
                        attack_size=10000, n_classes=9, real_key=108):

    # Test the model
    with torch.no_grad():
        data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
        print('x_test size: {}'.format(data.cpu().size()))

        predictions = F.softmax(network(data).to(device), dim=-1).to(device)
        # d = predictions[0].cpu().numpy()
        accuracy(network, x_attack, y_attack)

    ranks = np.zeros(attack_size)
    predictions = predictions.cpu().numpy()
    probabilities = np.zeros(n_classes)
    for trace_num in range(attack_size):
        for key_guess in range(n_classes):
            sbox_out = key_guesses[trace_num][key_guess]
            probabilities[key_guess] += predictions[trace_num][sbox_out]

        res = np.argmax(np.argsort(probabilities)[::-1] == real_key)
        ranks[trace_num] = res
    print('Key guess: {}'.format(np.argmax(probabilities)))
    print(np.sort(probabilities))
    print(probabilities[real_key])

    # sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    # real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]

    return np.array(range(attack_size)), ranks




