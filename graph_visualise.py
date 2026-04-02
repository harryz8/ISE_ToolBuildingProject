import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset, load_csv
from configuration_tuning import hyperparameters

def generate_mask(data : np.ndarray, titles, budget, hp_filter) -> np.ndarray:
    hp_copy = hyperparameters
    hp_copy['budget'] = budget
    mask = (data[:, titles.index(hp_filter[0])] == hp_copy[hp_filter[0]])
    hp_filter.pop(0)
    for hp in hp_filter:
        mask = mask & (data[:, titles.index(hp)] == hp_copy[hp])
    return mask

def plot_error():
    data: Dataset = load_csv("spear_error", "error_new", title_row=False, dtype=float)
    data_0_100 = data.data[:, 0] == 100
    data_1_100 = data.data[:, 1] == 10
    data_2_100 = data.data[:, 2] == 10
    data_3_100 = data.data[:, 3] == 300
    data_4_100 = data.data[:, 4] == 0.01
    x, y = (data.data[data_0_100 & data_4_100 & data_1_100 & data_3_100][:, 2],
            data.labels[data_0_100 & data_4_100 & data_1_100 & data_3_100])
    plt.ylabel("error")
    plt.plot(x, y, 'ro')
    b, a = np.polyfit(x, y, 1)
    plt.plot(x, b * x + a)
    plt.show()

def plot_hyperparameters(dataset : str, budget, hyperparameter, metric):
    data: Dataset = load_csv(f"{dataset}_error", "error_new", title_row=True, dtype=float)
    hp_filter = data.title.copy()
    hp_filter.remove(hyperparameter)
    hp_filter.remove("total_time")
    hp_filter.remove("absolute_error")
    print(hyperparameter)
    print(hp_filter)
    mask = generate_mask(data.data, data.title, budget, hp_filter)
    x = data.data[mask][:, data.title.index(hyperparameter)]
    y = data.data[mask][:, 5] if metric == "total_time" else data.labels[mask]
    plt.ylabel(metric)
    plt.xlabel(hyperparameter)
    plt.plot(x, y, 'ro')
    b, a = np.polyfit(x, y, 1)
    plt.plot(x, b * x + a)
    plt.show()

def plot_results(program : str):
    data: Dataset = load_csv(f"{program}_results", "results", title_row=False, dtype=float)
    y = data.labels
    x = range(0, data.data.shape[0])
    plt.plot(x, y)
    plt.ylabel("performance")
    plt.xlabel("iteration")
    plt.show()

if __name__ == '__main__':
    plot_error()
    plot_time()