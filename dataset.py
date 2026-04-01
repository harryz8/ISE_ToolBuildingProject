import numpy as np
from csv import reader
from collections.abc import Callable
from typing import Any

class Dataset:

    _titles = []

    def __init__(self, labels, data):
        if labels.shape[0] != data.shape[0]:
            raise ValueError("The number of labels and the number of data does not match")
        self.labels = labels
        self.data = data

    def set_titles(self, titles):
        self._titles = titles

    @property
    def title(self):
        return self._titles

    def apply(self, func, **kwargs):
        whole = np.c_[self.data, self.labels]
        out = func(whole, **kwargs)
        whole = whole if out is None else out
        self.data, self.labels = whole[:, :-1], whole[:, -1]

    def __getitem__(self, index):
        return Dataset(self.labels[index], self.data[index])

    def copy(self, index):
        return Dataset(self.labels[index].copy(), self.data[index].copy())

    def __len__(self):
        return self.labels.shape[0]

def load_csv(filename: str, directory: str, title_row = True, dtype : Callable[[Any], Any] = None) -> Dataset:
    with open("./"+directory+"/"+filename+".csv", "r", encoding="utf-8") as file:
        file_read = list(reader(file, delimiter=","))
        start_index = 1 if title_row else 0
        if not dtype is None:
            whole = np.array(
                list(map(
                    lambda x: list(map(dtype, x)),
                    file_read[start_index:]
                )))
        else:
            whole = np.array(file_read[start_index:])
        data, labels = whole[:, :-1], whole[:, -1]
        ret_dataset = Dataset(labels, data)
        if title_row:
            ret_dataset.set_titles(file_read[0])
        return ret_dataset