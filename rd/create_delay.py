import csv

import numpy as np


def create_delay(a, b, n):
    delays = np.zeros(n)
    m = np.random.randint(0, a - b)
    div = int(n / 2)
    for i in range(div):
        delays[i] = m + np.random.randint(0, b)
    for i in range(div):
        index = i + div
        delays[index] = a - m - np.random.randint(0, b)
    return delays, m


def create_sample(selected_features):
    y = np.random.randint(0, np.shape(selected_features)[0])

    selected_feature = selected_features[y]

    hot_encoded = np.random.rand(256)
    for x in selected_feature:
        hot_encoded[x] = np.random.uniform(low=1.5, high=2, size=1)

    # print('Hot encoded: {}'.format(hot_encoded))
    return hot_encoded, y


def generate_features_with_delay(features, max_features):
    num_features = np.shape(features)[0]
    d, m = create_delay(180, 30, num_features)
    x, y = create_sample(features)
    insert_at = [20, 40, 60, 80, 100, 120, 140, 180, 200, 220]
    total_delay = 0.0
    index = 0
    delay_vector = x
    for delay in d:
        add = np.random.rand(int(delay))
        # print(add)
        # exit()
        delay_vector = np.insert(delay_vector, int(total_delay + insert_at[index]), add)
        total_delay += insert_at[index] + int(delay)
        # index += 1

    num_to_add = int(max_features - len(delay_vector))
    if num_to_add < 0:
        print("ERROR {}".format(len(delay_vector)))
        exit()
    add = np.random.rand(num_to_add)
    delay_vector = np.insert(delay_vector, len(delay_vector), add)
    return delay_vector, y


if __name__ == "__main__":
    num_features = 10
    features = np.random.choice(np.arange(0, 256), (num_features, num_features))

    # max_features = 350

    with open('testX.csv', 'w') as csvFileX:
        with open('testY.csv', 'w') as csvFileY:
            writerX = csv.writer(csvFileX)
            writerY = csv.writer(csvFileY)
            for i in range(20000):
                x, y = generate_features_with_delay(features, max_features=1250)
                writerX.writerow(x)
                writerY.writerow([y])

    # x2, y2 = generate_features_with_delay(features, max_features=350)
    # print(len(x2))
    # print(len(x))


# print(y)
