import tkinter as tk
from tkinter import ttk
import os
from configuration_tuning import tune_configuration_for
from graph_visualise import plot_results


class Window(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Configuration Tuning")

        datasets = [file[:-4] for file in os.listdir("datasets")]
        self.chosen_dataset = tk.StringVar()

        dataset_label = tk.Label(self, text="Choose a program to tune:")
        dataset_label.grid(column=0, row=1)
        self.dataset_combo_box = ttk.Combobox(self, values=datasets, textvariable=self.chosen_dataset)
        self.dataset_combo_box.grid(column=0, row=2)

        budget_label = tk.Label(self, text="Set the budget:")
        budget_label.grid(column=0, row=3)
        self.budget_entry = tk.Entry(self, validate="all", validatecommand=(self.register(self.invalid_budget), '%P'))
        self.budget_entry.grid(column=0, row=4)
        self.budget_entry.insert(0, "100")

        self.tuned_config_frame = tk.Text(self)
        self.tuned_config_frame.insert(tk.END, "Results:\n")

        button_frame = ButtonFrame(self)
        button_frame.grid(column=0, row=6)

    def invalid_budget(self, P):
        return str.isdigit(P) or P == ""

    def show_tuned_config(self):
        if self.chosen_dataset.get() == "":
            return
        config, time_taken = tune_configuration_for(self.chosen_dataset.get(), budget=int(self.budget_entry.get()), evaluate=True)
        self.tuned_config_frame.insert(tk.END, f"\tConfiguration: {config[0]}\n\tPerformance: {config[1]}\n")
        self.tuned_config_frame.insert(tk.END, f"\tTime elapsed: {time_taken}\n")
        self.tuned_config_frame.grid(column=0, row=5)

    def show_results_plot(self):
        if self.chosen_dataset.get() == "":
            return
        plot_results(self.chosen_dataset.get())


class ButtonFrame(tk.Frame):
    def __init__(self, master : Window):
        super().__init__(master)

        self.search_button = tk.Button(self, text="Tune configuration", width=20, command=master.show_tuned_config, state="disabled")
        self.search_button.pack(side="left")

        self.prev_results_button = tk.Button(self, text="View previous results", width=20, command=master.show_results_plot, state="disabled")
        self.prev_results_button.pack(side="left")

        master.dataset_combo_box.bind("<<ComboboxSelected>>", self.enable_conditional_buttons)

        hyperparameter_results = tk.Button(self, text="Evaluate hyperparameters", width=20)
        hyperparameter_results.pack(side="right")

    def enable_conditional_buttons(self, *event):
        self.search_button.config(state="normal")
        self.prev_results_button.config(state="normal")

if __name__ == '__main__':
    window = Window()
    window.mainloop()