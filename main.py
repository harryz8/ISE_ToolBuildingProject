import tkinter as tk
from tkinter import ttk, messagebox
import os
from configuration_tuning import tune_configuration_for, clear_folder
from graph_visualise import plot_results, plot_hyperparameters
import measure_hyperparameter_performance

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

def clear_folder_with_feedback(folder):
    sure = messagebox.askyesno("Are you sure?", "Are you sure you want to delete all " +
                               f"{'results' if folder == 'results' else 'hyperparameter measurements'}?")
    if sure:
        clear_folder(folder)
        messagebox.showinfo("Done", "All " +
                            f"{'results' if folder == 'results' else 'hyperparameter measurements'} " +
                            "successfully deleted.")
    else:
        messagebox.showinfo("Cancelled", "Deletion action cancelled")


class Window(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Configuration Tuning")

        datasets = [file[:-4] for file in os.listdir("datasets")]
        self.chosen_dataset = tk.StringVar()
        self.budget = tk.IntVar()

        self.heading_title = tk.Label(self, text="Configuration Tuning", font=("Arial", 14, "bold"))
        self.heading_title.grid(column=0, row=0)

        dataset_label = tk.Label(self, text="Choose a program:")
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

        self.all_program_actions_frame = AllProgramActions(self)
        self.all_program_actions_frame.grid(column=0, row=7)

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
        self.all_program_actions_frame.grid_forget()
        self.heading_title.config(text="Select graph to view:")

    def hide_hyperparameter_graph_select_frame(self):
        self.hyperparameter_graph_select_frame.grid_forget()
        self.button_frame.grid(column=0, row=6)
        self.all_program_actions_frame.grid(column=0, row=7)
        self.heading_title.config(text="Configuration Tuning")


class AllProgramActions(tk.LabelFrame):
    def __init__(self, master):
        super().__init__(master)
        super().config(text="Actions over all programs:")

        self.clear_results_button = tk.Button(self, text="Clear all results", width=30,
                                              command=lambda: clear_folder_with_feedback("results"))
        self.clear_results_button.pack(side="left")

        self.save_result_graphs = tk.Button(self, text="Save all result graphs", width=30,
                                            command=lambda: print("coming soon"))
        self.save_result_graphs.pack(side="right")


class ButtonFrame(tk.Frame):
    def __init__(self, master : Window):
        super().__init__(master)

        self.search_button = tk.Button(self, text="Tune configuration", width=20,
                                       command=master.show_tuned_config, state="disabled")
        self.search_button.pack(side="left")

        self.prev_results_button = tk.Button(self, text="View previous results", width=20,
                                             command=master.show_results_plot, state="disabled")
        self.prev_results_button.pack(side="left")

        master.dataset_combo_box.bind("<<ComboboxSelected>>", self.enable_conditional_buttons)

        hyperparameter_results = tk.Button(self, text="Evaluate hyperparameters", width=20,
                                           command=master.show_hyperparameter_graph_select_frame)
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
        for hyperparameter in ["Budget", "Evaluation size", "Hidden layer size", "Epochs for evaluation batch",
                               "Learning rate", "Epochs for one batch"]:
            hyperparameter_radiobuttons.append(
                tk.Radiobutton(hyperparameter_frame, text=hyperparameter, variable=self.x_selected,
                               value=settings_info[hyperparameter]['csv_title'], indicatoron=False, width=20,
                               padx=5, pady=5)
            )
            hyperparameter_radiobuttons[-1].pack(side="left")
            self.master.update()
            ToolTip(self, hyperparameter_radiobuttons[-1], settings_info[hyperparameter]['description'], depth=1)
        y_axis_label = tk.Label(self, text="Y axis:")
        y_axis_label.pack()
        metric_frame = tk.Frame(self)
        metric_frame.pack()
        metric_radiobuttons = []
        self.y_selected = tk.StringVar()
        for metric in ["Time taken", "Mean Absolute Error"]:
            metric_radiobuttons.append(
                tk.Radiobutton(metric_frame, text=metric, variable=self.y_selected,
                               value=settings_info[metric]['csv_title'], indicatoron=False, width=20, padx=5, pady=5)
            )
            metric_radiobuttons[-1].pack(side="left")

        render_button = tk.Button(self, text="Render Graph", command=self.generate_graph)
        render_button.pack()

        separator = ttk.Separator(self, orient="horizontal")
        separator.pack(fill="x")

        all_program_actions_frame = AllProgramActionsHyperparameter(self)
        all_program_actions_frame.pack()

    def generate_graph(self):
        if self.x_selected.get() == "":
            return
        if self.y_selected.get() == "":
            return
        plot_hyperparameters(self.master.chosen_dataset.get(), self.master.budget.get(), self.x_selected.get(),
                             self.y_selected.get())


class AllProgramActionsHyperparameter(tk.LabelFrame):
    def __init__(self, master):
        super().__init__(master)
        super().config(text="Actions over all programs and hyperparameters:")

        self.save_hyperparameter_graphs = tk.Button(self, text="Save all hyperparameter\ngraphs", width=20,
                                              command=lambda: print("Coming soon"))
        self.save_hyperparameter_graphs.pack(side="left")

        self.measure_hyperparameters_button = tk.Button(self, text="Measure hyperparameters", width=20,
                                                        command=self.start_measure_hyperparameter_performance)
        self.measure_hyperparameters_button.pack(side="left")
        ToolTip(self, self.measure_hyperparameters_button, "Warning: this can take hours", depth=2)

        self.clear_hyperparameter_measurements_button = tk.Button(self, text="Clear all hyperparameter\nmeasurements",
                                                                  width=20,
                                                                  command=lambda: clear_folder_with_feedback("error"))
        self.clear_hyperparameter_measurements_button.pack(side="right")

    @staticmethod
    def start_measure_hyperparameter_performance():
        mhp_loading_window = measure_hyperparameter_performance.MeasureHyperparameterPerformance()
        mhp_loading_window.run()
        mhp_loading_window.close()


#class based on https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
class ToolTip:
    def __init__(self, parent, widget, text, depth):
        self.widget = widget
        self.text = text
        self.parent = parent
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.waiter = None
        self.toplevel_window = None
        self.depth = depth

    def widget_after(self, event=None):
        self.waiter = self.widget.after(100, lambda:self.show(event))

    def show(self, event=None):
        self.toplevel_window = tk.Toplevel(self.widget)
        self.toplevel_window.wm_overrideredirect(True)
        x, y, _, _ = self.widget.bbox("insert")
        lb = tk.Label(self.toplevel_window, text=self.text, wraplength=200)
        lb.config(bg="white", padx=5, pady=5, borderwidth=1, relief="solid", font=("Arial", 8))
        lb.pack()
        x_given_depth = 0
        y_given_depth = 0
        current_master = self.parent
        for _ in range(self.depth):
            current_master = current_master.master
            x_given_depth += current_master.winfo_x()
            y_given_depth += current_master.winfo_y()
        if event:
            self.toplevel_window.wm_geometry(
                f"+{x_given_depth + self.parent.winfo_x() + self.widget.winfo_x() + event.x + 5}" +
                f"+{y_given_depth + self.parent.winfo_y() + self.widget.winfo_y() + event.y + 5}")
        else:
            self.toplevel_window.wm_geometry(
                f"+" +
                f"{x_given_depth + self.parent.winfo_x() + self.widget.winfo_x() + self.widget.winfo_width()}" +
                f"+{y_given_depth + self.parent.winfo_x() + self.widget.winfo_x() + self.widget.winfo_height() + 5}"
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