import numpy as np
import os
from dataset import Dataset, load_csv

def knn_factory(k, training : Dataset, distance_func):
    def knn(x_new):
        closest = []
        for i in range(0, len(training.data)):
            x = training.data[i, :]
            y = training.labels[i]
            d = distance_func(x_new, x)
            if len(closest) < k:
                closest.append((x, y, d))
            else:
                for point in range(0, len(closest)):
                    if d < closest[point][2]:
                        del closest[point]
                        closest.append((x, y, d))
        sum_y = 0
        for (x, y, d) in closest:
            sum_y += y
        return sum_y / len(closest)
    return knn

def conn(configs, dataset, budget, k, size_eval, evaluation_func):
    np.random.shuffle(configs)
    eval_configs_list = [measure(dataset, x) for x in configs[:size_eval]]
    remainder_configs = configs[size_eval:].tolist()
    while budget > 0:
        part1 = [i[1] for i in eval_configs_list]
        part2 = [j[0] for j in eval_configs_list]
        eval_configs : Dataset = Dataset(np.array(part1).squeeze(), np.array(part2).squeeze())
        model = knn_factory(k, eval_configs, (lambda q, p : np.sum(np.sqrt(q**2 + p**2))))
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
    files = os.listdir(directory)
    for filename in files:
        data: Dataset = load_csv(filename, directory, dtype=float)
        best_complexity = search_function(data.data.copy(), data, **params)
        print(f"System: {filename}\n\tBest solution:\t\t{best_complexity[0].tolist()}\n\tBest performance:\t{best_complexity[1].item()}\n")

if __name__ == "__main__":
    main("datasets", conn, budget=100, k=3, size_eval=10, evaluation_func=min)