import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset, load_csv

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

def plot_time():
    data: Dataset = load_csv("spear_error", "error_new", title_row=False, dtype=float)
    data_0_100 = data.data[:, 0] == 100
    data_1_100 = data.data[:, 1] == 10
    data_2_100 = data.data[:, 2] == 10
    data_3_100 = data.data[:, 3] == 300
    data_4_100 = data.data[:, 4] == 0.01
    x, y = (data.data[data_0_100 & data_4_100 & data_1_100 & data_3_100][:, 2],
            data.data[data_0_100 & data_4_100 & data_1_100 & data_3_100][:, 5])
    plt.ylabel("time")
    plt.plot(x, y, 'ro')
    b, a = np.polyfit(x, y, 1)
    plt.plot(x, b * x + a)
    plt.show()

def plot_results(program : str):
    data: Dataset = load_csv(f"{program}_results", "results", title_row=False, dtype=float)
    y = data.labels
    x = range(0, data.data.shape[0])
    b, a = np.polyfit(x, y, 1)
    plt.plot(x, y)
    plt.plot(x, b * x + a)
    plt.ylabel("performance")
    plt.xlabel("iteration")
    plt.show()

if __name__ == '__main__':
    plot_error()
    plot_time()