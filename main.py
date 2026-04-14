import tkinter as tk
from tkinter import ttk
import os
from configuration_tuning import tune_configuration_for
from graph_visualise import plot_results, plot_hyperparameters

settings_info = {
    "Budget": {"csv_title": "budget", "description": "The maximum number of times the real performance can be measured."},
    "Evaluation size": {"csv_title": "size_eval", "description": "This refers to the amount of the budget that is used to create labelled training data for training the initial model."},
    "Hidden layer size": {"csv_title": "hidden_size", "description": "This refers to the number of artificial neurons in the hidden layer of the neural network (the layer between the input and output layers)."},
    "Epochs for one batch": {"csv_title": "epochs_1batch", "description": "This refers to the number of times that the backwards pass is run and the weights are updated when updating the model with a new piece of labelled data."},
    "Epochs for evaluation batch": {"csv_title": "epochs", "description": "This refers to the number of times that the backwards pass is run and the weights are updated when training the initial model."},
    "Learning rate": {"csv_title": "learning_rate", "description": "This hyper-parameter controls how quickly the neural network model learns from the error of the previous pass"},
    "Time taken": {"csv_title": "total_time", "description": ""},
    "Mean Absolute Error": {"csv_title": "absolute_error", "description": ""}
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
                tk.Radiobutton(hyperparameter_frame, text=hyperparameter, variable=self.x_selected, value=settings_info[hyperparameter]['csv_title'],
                               indicatoron=False, width=20, padx=5, pady=5)
            )
            hyperparameter_radiobuttons[-1].pack(side="left")
            self.master.update()
            ToolTip(self, hyperparameter_radiobuttons[-1], settings_info[hyperparameter]['description'])
        y_axis_label = tk.Label(self, text="Y axis:")
        y_axis_label.pack()
        metric_frame = tk.Frame(self)
        metric_frame.pack()
        metric_radiobuttons = []
        self.y_selected = tk.StringVar()
        for metric in ["Time taken", "Mean Absolute Error"]:
            metric_radiobuttons.append(
                tk.Radiobutton(metric_frame, text=metric, variable=self.y_selected, value=settings_info[metric]['csv_title'],
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

#class based on https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
class ToolTip:
    def __init__(self, parent, widget, text):
        self.widget = widget
        self.text = text
        self.parent = parent
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.waiter = None
        self.toplevel_window = None

    def widget_after(self, event=None):
        self.waiter = self.widget.after(100, lambda:self.show(event))

    def show(self, event=None):
        self.toplevel_window = tk.Toplevel(self.widget)
        self.toplevel_window.wm_overrideredirect(True)
        x, y, _, _ = self.widget.bbox("insert")
        lb = tk.Label(self.toplevel_window, text=self.text, wraplength=200)
        lb.config(bg="white", padx=5, pady=5, borderwidth=1, relief="solid", font=("Arial", 8))
        lb.pack()
        if event:
            self.toplevel_window.wm_geometry(
                f"+{self.parent.master.winfo_x() + self.parent.winfo_x() + self.widget.winfo_x() + event.x + 5}" +
                f"+{self.parent.master.winfo_y() + self.parent.winfo_y() + self.widget.winfo_y() + event.y + 5}")
        else:
            self.toplevel_window.wm_geometry(
                f"+" +
                f"{self.parent.master.winfo_x() + self.parent.winfo_x() + self.widget.winfo_x() + self.widget.winfo_width()}" +
                f"+{self.parent.master.winfo_x() + self.parent.winfo_x() + self.widget.winfo_x() + self.widget.winfo_height() + 5}"
            )

    def hide(self):
        if self.toplevel_window:
            self.toplevel_window.destroy()
            self.toplevel_window = None

    def enter(self, event):
        if (self.waiter is None) and (self.toplevel_window is None):
            self.widget_after(event)

    def leave(self, event=None):
        if self.waiter:
            self.widget.after_cancel(self.waiter)
        self.waiter = None
        self.hide()

if __name__ == '__main__':
    window = Window()
    window.mainloop()