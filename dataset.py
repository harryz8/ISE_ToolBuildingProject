import numpy as np
from csv import reader
from collections.abc import Callable
from typing import Any

class Dataset:
    def __init__(self, labels, data):
        if labels.shape[0] != data.shape[0]:
            raise ValueError("The number of labels and the number of data does not match")
        self.labels = labels
        self.data = data

    def apply(self, func, **kwargs):
        whole = np.c_[self.data, self.labels]
        out = func(whole, **kwargs)
        whole = whole if out is None else out
        self.data, self.labels = whole[:, :-1], whole[:, -1]

    def __getitem__(self, index):
        return Dataset(self.labels[index], self.data[index])

    def __len__(self):
        return self.labels.shape[0]

def load_csv(filename: str, directory: str, dtype : Callable[[Any], Any] = None) -> Dataset:
    with open("./"+directory+"/"+filename, "r", encoding="utf-8") as file:
        if not dtype is None:
            whole = np.array(
                list(map(
                    lambda x: list(map(dtype, x)),
                    list(reader(file, delimiter=","))[1:]
                )))
        else:
            whole = np.array(list(reader(file, delimiter=","))[1:])
        data, labels = whole[:, :-1], whole[:, -1]
        return Dataset(labels, data)