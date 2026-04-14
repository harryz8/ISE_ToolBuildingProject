import tkinter as tk
from tkinter import ttk
from configuration_tuning import tune_configuration_all_programs, tune_configuration


# inspired by https://www.askpython.com/python-modules/tkinter/tkinter-spinbox-and-progressbar-widgets
class MeasureHyperparameterPerformance(tk.Tk):
    running = False

    def __init__(self):
        super().__init__()
        super().title("Please wait...")
        super().protocol("WM_DELETE_WINDOW", self.close)

        wait_lb = tk.Label(self, text="Please wait...")
        wait_lb.pack()

        self.progress_bar = ttk.Progressbar(self, maximum=590, value=0)
        self.progress_bar.pack()
        self.update()

        self.close_bt = tk.Button(self, text="Close", command=self.close)

        self.progress_bar['value'] = 0
        self.update()

    def run(self):
        self.running = True
        for count_i in range(0, 90):
            tune_configuration_all_programs("datasets", tune_configuration, budget=100,
                                            size_eval=10 + (10 * (count_i // 10)), hidden_size=10, epochs=300,
                                            epochs_1batch=30, learning_rate=0.01, evaluation_func=min)
            self.progress_bar['value'] += 1
            self.update()
        for count_i in range(0, 100):
            tune_configuration_all_programs("datasets", tune_configuration, budget=50 + (10 * (count_i // 10)),
                                            size_eval=10, hidden_size=10, epochs=300, epochs_1batch=30,
                                            learning_rate=0.01, evaluation_func=min)
            self.progress_bar['value'] += 1
            self.update()
        for count_i in range(0, 100):
            tune_configuration_all_programs("datasets", tune_configuration, budget=100, size_eval=10,
                                            hidden_size=10 + (10 * (count_i // 10)), epochs=300, epochs_1batch=30,
                                            learning_rate=0.01, evaluation_func=min)
            self.progress_bar['value'] += 1
            self.update()
        for count_i in range(0, 100):
            tune_configuration_all_programs("datasets", tune_configuration, budget=100, size_eval=10,
                                            hidden_size=10, epochs=100 + (100 * (count_i // 20)), epochs_1batch=30,
                                            learning_rate=0.01, evaluation_func=min)
            self.progress_bar['value'] += 1
            self.update()
        for count_i in range(0, 100):
            tune_configuration_all_programs("datasets", tune_configuration, budget=100, size_eval=10,
                                            hidden_size=10, epochs=300, epochs_1batch=10 + (10 * (count_i // 10)),
                                            learning_rate=0.01, evaluation_func=min)
            self.progress_bar['value'] += 1
            self.update()
        for count_i in range(0, 100):
            tune_configuration_all_programs("datasets", tune_configuration, budget=100, size_eval=10,
                                            hidden_size=10, epochs=300, epochs_1batch=30,
                                            learning_rate=0.1 * 10 ** (-(count_i // 10) / 2), evaluation_func=min)
            self.progress_bar['value'] += 1
            self.update()
        self.running = False

    def close(self):
        if not self.running:
            self.destroy()