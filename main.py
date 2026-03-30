import numpy as np
import os

import NN
from NN import update_weights
from dataset import Dataset, load_csv
import time

def knn_factory(k, training : Dataset, distance_func):
    def knn(x_new):
        dist_vector = distance_func(training.data, x_new)
        combo_vec = np.c_[(np.c_[training.data, training.labels]), dist_vector]
        combo_vec = combo_vec[combo_vec[:, -1].argsort()]
        closest = combo_vec[:k, :]
        return np.sum(closest[:, -2]) / closest.shape[0]

    return knn

def vector_euclidean_distance(vector1, new_x):
    minus = vector1 - new_x
    square = np.square(minus)
    sum_vec = np.sum(square, axis=1)
    sqrt_vec = np.sqrt(sum_vec)
    return sqrt_vec

def conn(configs, dataset, budget, k, size_eval, evaluation_func):
    np.random.shuffle(configs)
    eval_configs_list = [measure(dataset, x) for x in configs[:size_eval]]
    remainder_configs = configs[size_eval:].tolist()
    while budget > 0:
        part1 = [i[1] for i in eval_configs_list]
        part2 = [j[0] for j in eval_configs_list]
        eval_configs : Dataset = Dataset(np.array(part1).squeeze(), np.array(part2).squeeze())
        model = knn_factory(k, eval_configs, vector_euclidean_distance)
        predict_xs = []
        for l in range(0, len(remainder_configs)):
            predict_xs.append((l, model(np.array(remainder_configs[l]))))
        max_point = evaluation_func(predict_xs, key=lambda x: x[1])
        acquired_point = measure(dataset, remainder_configs[max_point[0]])
        eval_configs_list.append(acquired_point)
        del remainder_configs[max_point[0]]
        budget -= 1
    min_ = np.inf
    final = []
    for value in eval_configs_list:
        if evaluation_func(value[1].item(), min_) == value[1].item():
            min_ = value[1].item()
            final = value
    return final

def measure(dataset, x):
    find_x = np.all(dataset.data == x, axis=1)
    return dataset.data[find_x], dataset.labels[find_x]

def main(directory, search_function, **params):
    start = time.time()
    files = os.listdir(directory)
    for filename in files:
        data: Dataset = load_csv(filename, directory, dtype=float)
        best_complexity = search_function(data.data.copy(), data, **params)
        print(f"System: {filename}\n" +
              f"\tBest solution:\t\t{best_complexity[0].squeeze().tolist()}\n" +
              f"\tBest performance:\t{best_complexity[1].item()}")
        real_best_complexity = np.min(data.labels)
        abs_error = np.abs(best_complexity[1] - real_best_complexity)
        with open(f"./error/{filename.split('.')[0]}_error.csv", "a") as f:
            f.write(f"{params['k']},{abs_error.item()}\n")
        with open(f'./results/{filename.split(".")[0]}_results.csv', 'a') as f:
            f.write(",".join(best_complexity[0].flatten().astype(str)) + f",{best_complexity[1].item()}\n")
        print(f"Absolute error: {abs_error.item():.4f}\n")
    print(f"Total time taken: {(time.time() - start):.2f}s")

def test_train(directory):
    data: Dataset = load_csv("./storm.csv", directory, dtype=float)
    test_set = data[:np.floor(0.8*data.data.shape[1]).astype(int)]
    hidden_size = 10
    input_size = test_set.data.shape[1]
    max_row = np.max(test_set.data, axis=0, keepdims=True)

    epochs = 300
    learning_rate = 0.001

    model = NN.MLP(input_size, hidden_size)

    for epoch in range(epochs):
        # test_set.apply(np.random.shuffle)
        test_set_data = test_set.data / max_row
        output = model.forward(test_set_data)
        flat_output = output.flatten()
        loss = ((data.labels[:flat_output.shape[0]] - flat_output)**2).mean()
        model.backward(test_set_data, test_set.labels, output)
        update_weights(model, learning_rate)
        if (epoch % 10 == 0):
            print(f"loss: {loss}")

if __name__ == "__main__":
    # for count_i in range(0, 10):
    #     main("datasets", conn, budget=100-2, k=3, size_eval=2, evaluation_func=min)
    test_train("datasets")