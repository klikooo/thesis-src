import torch
import torch.nn.functional as F
import numpy as np

from util import HW, test_model, device


def test(x_attack, y_attack, metadata_attack, network, sub_key_index, use_hw=True, attack_size=10000, rank_step=10,
         unmask=False,
         only_accuracy=False,
         plain=None):
    # Cut to the correct attack size
    x_attack = x_attack[0:attack_size]
    y_attack = y_attack[0:attack_size]

    metadata_attack = metadata_attack[0:attack_size]
    if unmask:
        y_attack = np.array([y_attack[i] ^ metadata_attack[i]['masks'][sub_key_index-2] for i in range(attack_size)])

    # Convert values to hamming weight if asked for
    if use_hw:
        y_attack = np.array([HW[val] for val in y_attack])

    # Test the model
    with torch.no_grad():
        data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
        print('x_test size: {}'.format(data.cpu().size()))

        if plain is None:
            predictions = F.softmax(network(data).to(device), dim=-1).to(device)
        else:
            plain = torch.from_numpy(plain.astype(np.float32)).to(device)
            predictions = F.softmax(network(data, plain).to(device), dim=-1).to(device)

        # Print accuracy
        accuracy(network, x_attack, y_attack, plain=plain)
        if not only_accuracy:
            # Calculate num of traces needed
            return test_model(predictions.cpu().numpy(), metadata_attack, sub_key_index,
                              use_hw=use_hw,
                              rank_step=rank_step,
                              unmask=unmask)
        else:
            return None, None


def accuracy(network, x_test, y_test, plain=None, batch_size=100):
    with torch.no_grad():
        data = torch.from_numpy(x_test.astype(np.float32)).to(device)
        if plain is not None:
            plain = torch.from_numpy(plain.astype(np.float32)).to(device)
        size = np.shape(x_test)[0]
        predi = torch.from_numpy(np.array([]).astype(np.float32)).to(device)
        for i in range(0, size, batch_size):
            d = data[i:i+batch_size]
            if plain is None:
                predictions = F.softmax(network(d).to(device), dim=-1).to(device)
            else:
                p = plain[i:i+batch_size]
                predictions = F.softmax(network(d, p).to(device), dim=-1).to(device)
            predi = torch.cat((predi, predictions), 0)

        _, pred = predi.max(1)
        z = pred.long() == torch.from_numpy(y_test.reshape(len(y_test))).long().to(device)
        # print(predi[0])
        # exit()
        num_correct = z.sum().item()
        acc = num_correct / len(y_test) * 100
        print('Correct: {}'.format(num_correct))
        print('Accuracy: {} - {}%'.format(num_correct / len(y_test), acc))
        return predi, acc


def accuracy2(network, x_test, y_test, plain=None, batch_size=100):
    with torch.no_grad():
        data = torch.from_numpy(x_test.astype(np.float32)).to(device)
        if plain is not None:
            plain = torch.from_numpy(plain.astype(np.float32)).to(device)
        size = np.shape(x_test)[0]
        predi = torch.from_numpy(np.array([]).astype(np.float32)).to(device)
        for i in range(0, size, batch_size):
            d = data[i:i+batch_size]
            if plain is None:
                predictions = F.softmax(network(d).to(device), dim=-1).to(device)
            else:
                p = plain[i:i+batch_size]
                predictions = F.softmax(network(d, p).to(device), dim=-1).to(device)
            predi = torch.cat((predi, predictions), 0)

        _, pred = predi.max(1)
        z = pred.long() == torch.from_numpy(y_test.reshape(len(y_test))).long().to(device)

        num_correct = z.sum().item()
        acc = num_correct / len(y_test)
        return predi, acc


def test_with_key_guess(x_attack, y_attack, key_guesses, network, use_hw, real_key,
                        attack_size=10000,
                        plain=None):

    # Test the model
    with torch.no_grad():
        data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
        print('x_test size: {}'.format(data.cpu().size()))
        data_plain = None

        if plain is None:
            predictions = F.softmax(network(data).to(device), dim=-1).to(device)
        else:
            data_plain = torch.from_numpy(plain.astype(np.float32)).to(device)
            predictions = F.softmax(network(data, data_plain).to(device), dim=-1).to(device)
        # d = predictions[0].cpu().numpy()
        accuracy(network, x_attack, y_attack, plain=data_plain)

    ranks = np.zeros(attack_size)
    predictions = predictions.cpu().numpy()
    probabilities = np.zeros(256)
    if not use_hw:
        for trace_num in range(attack_size):
            for key_guess in range(256):
                sbox_out = key_guesses[trace_num][key_guess]
                probabilities[key_guess] += predictions[trace_num][sbox_out]

            res = np.argmax(np.argsort(probabilities)[::-1] == real_key)
            ranks[trace_num] = res
    else:
        for trace_num in range(attack_size):
            for key_guess in range(256):
                sbox_out = key_guesses[trace_num][key_guess]
                probabilities[key_guess] += predictions[trace_num][HW[sbox_out]]
            res = np.argmax(np.argsort(probabilities)[::-1] == real_key)
            ranks[trace_num] = res

    print('Key guess: {}'.format(np.argmax(probabilities)))
    # print(np.sort(probabilities))
    # print(probabilities[real_key])

    # sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    # real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]

    return np.array(range(1, attack_size+1)), ranks


def test_with_key_guess_p(key_guesses, predictions, use_hw, real_key,
                          attack_size=10000,
                          ):
    ranks = np.zeros(attack_size)
    probabilities = np.zeros(256)
    if not use_hw:
        for trace_num in range(attack_size):
            for key_guess in range(256):
                sbox_out = key_guesses[trace_num][key_guess]
                if predictions[trace_num][sbox_out] > 0.0:
                    probabilities[key_guess] += np.log(predictions[trace_num][sbox_out])

            res = np.argmax(np.argsort(probabilities)[::-1] == real_key)
            ranks[trace_num] = res
    else:
        for trace_num in range(attack_size):
            for key_guess in range(256):
                sbox_out = key_guesses[trace_num][key_guess]
                probabilities[key_guess] += predictions[trace_num][HW[sbox_out]]
            res = np.argmax(np.argsort(probabilities)[::-1] == real_key)
            ranks[trace_num] = res

    final_guess = np.argmax(probabilities)
    return np.array(range(1, attack_size+1)), ranks, final_guess

