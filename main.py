# is wrong. Please refer to the reference solution
# LLM USE: for debugging only

import numpy as np
from dataset import Dataset, load_csv
from sys import float_info

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

def conn(configs, dataset, budget, size):
    print(configs)
    np.random.shuffle(configs)
    print(configs)
    eval_configs_list = [measure(dataset, x) for x in configs[:size]]
    remainder_configs = configs[size:].tolist()
    while budget > 0:
        part1 = [i[1] for i in eval_configs_list]
        part2 = [j[0] for j in eval_configs_list]
        eval_configs : Dataset = Dataset(np.array(part1).squeeze(), np.array(part2).squeeze())
        # eval_configs.apply(print)
        # print(len(eval_configs.data))
        model = knn_factory(1, eval_configs, (lambda q, p : np.sum(np.sqrt(q**2 + p**2))))
        predict_xs = []
        for l in range(0, len(remainder_configs)):
            predict_xs.append((l, model(np.array(remainder_configs[l]))))
        max_point = min(predict_xs, key=lambda x: x[1])
        acquired_point = measure(dataset, remainder_configs[max_point[0]])
        eval_configs_list.append(acquired_point)
        del remainder_configs[max_point[0]]
        budget -= 1
    min_ = np.inf
    final = []
    for value in eval_configs_list:
        if value[1].item() < min_:
            min_ = value[1].item()
            final = value
    return final

def measure(dataset, x):
    find_x = np.all(dataset.data == x, axis=1)
    return dataset.data[find_x], dataset.labels[find_x]

def random_search(data: Dataset):
    x_max = np.max(data.data, axis=0)
    x_min = np.min(data.data, axis=0)
    x = np.random.randint(x_min, x_max+1, data.data.shape[1])
    find_x = np.all(data.data == x, axis=1)
    if not np.any(find_x):
        return np.empty_like(x), float_info.max
    return x, data.labels[np.where(find_x)]

def main(filename, directory, minimise, R, search_function):
    data: Dataset = load_csv(filename, directory, dtype=float)[:20]
    num_configurable_params = data.data.shape[1]
    best_complexity = conn(data.data, data, R, 6)
    # best_complexity = (None, float_info.max)
    # for count in range(R):
    #     contender = search_function(data)
    #     print(contender)
    #     if contender[0] is None:
    #         count += 1
    #         continue
    #     if minimise:
    #         if best_complexity[-1] > contender[-1]:
    #             best_complexity = contender
    #     else:
    #         if best_complexity[-1] < contender[-1]:
    #             best_complexity = contender
    return best_complexity

if __name__ == "__main__":
    print(main("brotli.csv", "datasets", True, 10, random_search))