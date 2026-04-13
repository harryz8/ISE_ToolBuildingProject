import tkinter as tk
from tkinter import ttk
import os
from configuration_tuning import tune_configuration_for
from graph_visualise import plot_results, plot_hyperparameters

conversion = {
    "Budget": "budget",
    "Evaluation size": "size_eval",
    "Hidden layer size": "hidden_size",
    "Epochs for one batch" : "epochs_1batch",
    "Epochs for evaluation batch": "epochs",
    "Learning rate": "learning_rate",
    "Time taken": "total_time",
    "Mean Absolute Error": "absolute_error"
}


class Window(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Configuration Tuning")

        datasets = [file[:-4] for file in os.listdir("datasets")]
        self.chosen_dataset = tk.StringVar()
        self.budget = tk.IntVar()

        dataset_label = tk.Label(self, text="Choose a program to tune:")
        dataset_label.grid(column=0, row=1)
        self.dataset_combo_box = ttk.Combobox(self, values=datasets, textvariable=self.chosen_dataset)
        self.dataset_combo_box.grid(column=0, row=2)

        budget_label = tk.Label(self, text="Set the budget:")
        budget_label.grid(column=0, row=3)
        budget_entry = tk.Spinbox(self, from_ = 0, to = float("inf"), increment=1, textvariable=self.budget, validate="all", validatecommand=(self.register(self.invalid_budget), '%P'))
        budget_entry.grid(column=0, row=4)
        self.budget.set(100)

        self.tuned_config_frame = tk.Text(self)
        self.tuned_config_frame.insert(tk.END, "Results:\n")

        self.hyperparameter_graph_select_frame = HyperparameterGraphSelect(self)
        self.back_button = tk.Button(self, command=self.hide_hyperparameter_graph_select_frame, text="↩ Back")

        self.button_frame = ButtonFrame(self)
        self.button_frame.grid(column=0, row=6)

    def invalid_budget(self, P):
        return str.isdigit(P) or P == ""

    def show_tuned_config(self):
        if self.chosen_dataset.get() == "":
            return
        config, time_taken = tune_configuration_for(self.chosen_dataset.get(), budget=int(self.budget.get()), evaluate=True)
        self.tuned_config_frame.insert(tk.END, f"\tConfiguration: {config[0]}\n\tPerformance: {config[1]}\n")
        self.tuned_config_frame.insert(tk.END, f"\tTime elapsed: {time_taken}\n\n")
        self.tuned_config_frame.grid(column=0, row=5)

    def show_results_plot(self):
        if self.chosen_dataset.get() == "":
            return
        plot_results(self.chosen_dataset.get())

    def show_hyperparameter_graph_select_frame(self):
        self.hyperparameter_graph_select_frame.grid(column=0, row=5)
        self.button_frame.grid_forget()
        self.back_button.grid(column=0, row=6)

    def hide_hyperparameter_graph_select_frame(self):
        self.hyperparameter_graph_select_frame.grid_forget()
        self.button_frame.grid(column=0, row=6)


class ButtonFrame(tk.Frame):
    def __init__(self, master : Window):
        super().__init__(master)

        self.search_button = tk.Button(self, text="Tune configuration", width=20, command=master.show_tuned_config, state="disabled")
        self.search_button.pack(side="left")

        self.prev_results_button = tk.Button(self, text="View previous results", width=20, command=master.show_results_plot, state="disabled")
        self.prev_results_button.pack(side="left")

        master.dataset_combo_box.bind("<<ComboboxSelected>>", self.enable_conditional_buttons)

        hyperparameter_results = tk.Button(self, text="Evaluate hyperparameters", width=20, command=master.show_hyperparameter_graph_select_frame)
        hyperparameter_results.pack(side="right")

    def enable_conditional_buttons(self, *event):
        self.search_button.config(state="normal")
        self.prev_results_button.config(state="normal")

class HyperparameterGraphSelect(tk.Frame):
    def __init__(self, master : Window):
        super().__init__(master)
        self.master = master

        x_axis_label = tk.Label(self, text="X axis:")
        x_axis_label.pack()
        hyperparameter_frame = tk.Frame(self)
        hyperparameter_frame.pack()
        hyperparameter_radiobuttons = []
        self.x_selected = tk.StringVar()
        for hyperparameter in ["Budget", "Evaluation size", "Hidden layer size", "Epochs for evaluation batch", "Learning rate", "Epochs for one batch"]:
            hyperparameter_radiobuttons.append(
                tk.Radiobutton(hyperparameter_frame, text=hyperparameter, variable=self.x_selected, value=conversion[hyperparameter],
                               indicatoron=False, width=20, padx=5, pady=5)
            )
            hyperparameter_radiobuttons[-1].pack(side="left")

        y_axis_label = tk.Label(self, text="Y axis:")
        y_axis_label.pack()
        metric_frame = tk.Frame(self)
        metric_frame.pack()
        metric_radiobuttons = []
        self.y_selected = tk.StringVar()
        for metric in ["Time taken", "Mean Absolute Error"]:
            metric_radiobuttons.append(
                tk.Radiobutton(metric_frame, text=metric, variable=self.y_selected, value=conversion[metric],
                               indicatoron=False, width=20, padx=5, pady=5)
            )
            metric_radiobuttons[-1].pack(side="left")
        render_button = tk.Button(self, text="Render Graph", command=self.generate_graph)
        render_button.pack()

    def generate_graph(self):
        if self.x_selected.get() == "":
            return
        if self.y_selected.get() == "":
            return
        plot_hyperparameters(self.master.chosen_dataset.get(), self.master.budget.get(), self.x_selected.get(), self.y_selected.get())


if __name__ == '__main__':
    window = Window()
    window.mainloop()