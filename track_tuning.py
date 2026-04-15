import tkinter as tk
from tkinter import ttk
import configuration_tuning
import threading
import time

# inspired by https://www.askpython.com/python-modules/tkinter/tkinter-spinbox-and-progressbar-widgets
class TrackTuning(tk.Tk):
    def __init__(self, budget):
        super().__init__()
        super().title("Please wait...")
        super().protocol("WM_DELETE_WINDOW", self.close)
        super().config(width=400)

        self.max_count = (((budget - configuration_tuning.hyperparameters["size_eval"])
                      * configuration_tuning.hyperparameters["epochs_1batch"])
                     + configuration_tuning.hyperparameters["epochs"])

        wait_lb = tk.Label(self, text="Please wait...")
        wait_lb.pack()

        self.progress_bar = ttk.Progressbar(self, maximum=self.max_count, value=0)
        self.progress_bar.pack(fill="x")
        self.update()

        self.close_bt = tk.Button(self, text="Close", command=self.close)

        self.progress_bar['value'] = 0
        self.update()

    def run(self, function, **kwargs):
        thread = threading.Thread(target=function, kwargs=kwargs)
        thread.start()
        while not configuration_tuning.tc_running_flag:
            time.sleep(0.1)
        while True:
            if configuration_tuning.tc_running_flag:
                self.progress_bar['value'] = configuration_tuning.tune_time
                self.update()
            else:
                thread.join()
                break


    def close(self):
        if not configuration_tuning.tc_running_flag:
            self.destroy()