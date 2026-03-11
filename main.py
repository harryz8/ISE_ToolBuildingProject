# is wrong. Please refer to the reference solution
# LLM USE: for debugging only

import numpy as np
from dataset import Dataset, load_csv
from sys import float_info

def random_search(data: Dataset):
    x_max = np.max(data.data, axis=0)
    x_min = np.min(data.data, axis=0)
    x = np.random.randint(x_min, x_max+1, data.data.shape[1])
    find_x = np.all(data.data == x, axis=1)
    if not np.any(find_x):
        return np.empty_like(x), float_info.max
    return x, data.labels[np.where(find_x)]

def main(filename, directory, minimise, R, search_function):
    data: Dataset = load_csv(filename, directory, dtype=float)
    best_complexity = (None, float_info.max)
    for count in range(R):
        contender = search_function(data)
        print(contender)
        if contender[0] is None:
            count += 1
            continue
        if minimise:
            if best_complexity[-1] > contender[-1]:
                best_complexity = contender
        else:
            if best_complexity[-1] < contender[-1]:
                best_complexity = contender
    return best_complexity

if __name__ == "__main__":
    print(main("brotli.csv", "datasets", True, 100, random_search))