import tkinter as tk
from tkinter import ttk
from configuration_tuning import tc_running_flag, tune_time, hyperparameters
import threading

# inspired by https://www.askpython.com/python-modules/tkinter/tkinter-spinbox-and-progressbar-widgets
class TrackTuning(tk.Tk):
    def __init__(self, budget):
        super().__init__()
        super().title("Please wait...")
        super().protocol("WM_DELETE_WINDOW", self.close)

        max_count = (((budget - hyperparameters["size_eval"]) * hyperparameters["epochs_1batch"])
                     + hyperparameters["epochs"])

        wait_lb = tk.Label(self, text="Please wait...")
        wait_lb.pack()

        self.progress_bar = ttk.Progressbar(self, maximum=max_count, value=0)
        self.progress_bar.pack()
        self.update()

        self.close_bt = tk.Button(self, text="Close", command=self.close)

        self.progress_bar['value'] = 0
        self.update()

    def run(self, function, **kwargs):
        thread = threading.Thread(target=function, kwargs=kwargs)
        thread.start()
        if tc_running_flag:
            self.progress_bar['value'] = tune_time
            self.update()
        else:
            thread.join()


    def close(self):
        if not tc_running_flag:
            self.destroy()