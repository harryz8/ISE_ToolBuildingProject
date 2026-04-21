import os

import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset, load_csv
from configuration_tuning import hyperparameters

def generate_mask(data : np.ndarray, titles, budget, hp_filter) -> np.ndarray:
    hp_copy = hyperparameters
    hp_copy['budget'] = int(budget)
    mask = (data[:, titles.index(hp_filter[0])] == hp_copy[hp_filter[0]])
    hp_filter.pop(0)
    for hp in hp_filter:
        mask = mask & (data[:, titles.index(hp)] == hp_copy[hp])
    return mask

def plot_hyperparameters(dataset : str, budget, hyperparameter, metric):
    data: Dataset = load_csv(f"{dataset}_error", "error", title_row=True, dtype=float)
    hp_filter = data.title.copy()
    hp_filter.remove(hyperparameter)
    hp_filter.remove("total_time")
    hp_filter.remove("absolute_error")
    mask = generate_mask(data.data, data.title, budget, hp_filter)
    x = data.data[mask][:, data.title.index(hyperparameter)]
    y = data.data[mask][:, -1] if metric == "total_time" else data.labels[mask]

    categories = set(x)
    cats = []
    mean_cat = []
    for category in categories:
        cats.append(category)
        filter_cat = x == category
        mean_cat.append(np.mean(y[filter_cat]))

    plt.title(f"{dataset} {metric} when {hyperparameter} is varied")
    plt.ylabel(metric)
    plt.xlabel(hyperparameter)
    plt.plot(x, y, 'rx', alpha=0.3, markersize=5, label="Raw test run results")

    x_mean, y_mean = np.array(cats), np.array(mean_cat)

    plt.plot(x_mean, y_mean, 'bo', label=f"Mean {metric} over each {hyperparameter}")
    b, a = np.polyfit(x_mean, y_mean, 1)
    plt.plot(x_mean, b * x_mean + a, label="Linear trend line over the means")

    plt.legend()
    return plt

def plot_results(program : str):
    data: Dataset = load_csv(f"{program}_results", "results", title_row=True, dtype=float)
    y = data.labels
    x = range(0, data.data.shape[0])
    plt.plot(x, y, label="Performance")
    best_performance = np.min(y)
    best_index = np.nonzero(y == best_performance)[0][0]
    plt.plot(best_index, best_performance, marker="*", color="red", markersize=12, label="Best Point")
    plt.title(f"{program} Results")
    plt.ylabel("performance")
    plt.xlabel("iteration")
    plt.legend()
    return plt

def plot_result_histograms():
    for program_full in os.listdir("datasets"):
        program = program_full[:-4]
        data: Dataset = load_csv(f"{program}_results", "results", title_row=True, dtype=float)
        plt.hist(data.labels, bins=10)
        plt.title(f"Performance Histogram for {program}")
        plt.ylabel("frequency")
        plt.xlabel("performance")
        plt.savefig(f"graphs/{program}_histogram.png")
        plt.clf()

if __name__ == "__main__":
    plot_result_histograms()