import numpy as np
import os
import NN
from NN import update_weights
from dataset import Dataset, load_csv
import time

hyperparameters = {
    "size_eval": 10,
    "hidden_size": 10,
    "epochs": 300,
    "epochs_1batch": 30,
    "learning_rate": 0.01
}

def tune_configuration(configs, dataset, budget, size_eval, hidden_size, epochs, learning_rate, epochs_1batch, evaluation_func):
    max_row = np.max(dataset.data, axis=0, keepdims=True)
    max_label = np.max(dataset.labels)
    np.random.shuffle(configs)
    eval_configs_list = [measure(dataset, x) for x in configs[:size_eval]]
    budget -= size_eval
    remainder_configs = configs[size_eval:].tolist()

    part1 = [i[1] for i in eval_configs_list]
    part2 = [j[0] for j in eval_configs_list]
    eval_configs: Dataset = Dataset(np.array(part1).squeeze() / max_label, np.array(part2).squeeze() / max_row)
    model = train_nn(eval_configs, hidden_size, epochs, learning_rate)

    while budget > 0:
        predict_xs = []
        for l in range(0, len(remainder_configs)):
            predict_xs.append((l, model.forward(np.array(remainder_configs[l]) / max_row)))
        max_point = evaluation_func(predict_xs, key=lambda x: x[1])
        acquired_point = measure(dataset, remainder_configs[max_point[0]])
        for epoch in range(epochs_1batch):
            model.backward(np.array(remainder_configs[max_point[0]]) / max_row, acquired_point[1] / max_label, max_point[1])
            update_weights(model, learning_rate=learning_rate)
        # print(f"flash loss: {((acquired_point[1] / max_label - max_point[1])**2)}")
        eval_configs_list.append(acquired_point)
        del remainder_configs[max_point[0]]
        budget -= 1
    min_ = np.inf
    final = []
    for value in eval_configs_list:
        if evaluation_func(value[1].item(), min_) == value[1].item():
            min_ = value[1].item()
            final = value
    return final

def measure(dataset, x):
    find_x = np.all(dataset.data == x, axis=1)
    return dataset.data[find_x], dataset.labels[find_x]

def main(directory, search_function, **params):
    start = time.time()
    files = [file[:-4] for file in os.listdir(directory)]
    for filename in files:
        file_start = time.time()
        data: Dataset = load_csv(filename, directory, dtype=float)
        best_complexity = search_function(data.data.copy(), data, **params)
        print(f"System: {filename}\n" +
              f"\tBest solution:\t\t{best_complexity[0].squeeze().tolist()}\n" +
              f"\tBest performance:\t{best_complexity[1].item()}")
        real_best_complexity = np.min(data.labels)
        abs_error = np.abs(best_complexity[1] - real_best_complexity)
        if not os.path.isfile(f"./error_new_new/{filename.split('.')[0]}_error.csv"):
            with open(f"./error_new_new/{filename.split('.')[0]}_error.csv", "w") as f:
                f.write("budget,size_eval,hidden_size,epochs,epochs_1batch,learning_rate,total_time,absolute_error\n")
        with open(f"./error_new_new/{filename.split('.')[0]}_error.csv", "a") as f:
            f.write(f"{params['budget']},{params['size_eval']},{params['hidden_size']},{params['epochs']},{params['epochs_1batch']},{params['learning_rate']},{(time.time() - file_start)},{abs_error.item()}\n")
        with open(f'./results/{filename.split(".")[0]}_results.csv', 'a') as f:
            f.write(",".join(best_complexity[0].flatten().astype(str)) + f",{best_complexity[1].item()}\n")
        print(f"Absolute error: {abs_error.item():.4f}\n")
    print(f"Total time taken: {(time.time() - start):.2f}s")

def tune_configuration_for(filename, budget, evaluate=False):
    file_start = time.time()
    data: Dataset = load_csv(filename, "datasets", dtype=float)
    best_complexity = tune_configuration(data.data.copy(), data, **hyperparameters, budget=budget, evaluation_func=min)
    total_time = time.time() - file_start
    if evaluate:
        real_best_complexity = np.min(data.labels)
        abs_error = np.abs(best_complexity[1] - real_best_complexity)
        if not os.path.isfile(f"./error_new_new/{filename.split('.')[0]}_error.csv"):
            with open(f"./error_new_new/{filename.split('.')[0]}_error.csv", "w") as f:
                f.write("budget,size_eval,hidden_size,epochs,epochs_1batch,learning_rate,total_time,absolute_error\n")
        with open(f"./error_new_new/{filename.split('.')[0]}_error.csv", "a") as f:
            f.write(f"{budget},{hyperparameters['size_eval']},{hyperparameters['hidden_size']},{hyperparameters['epochs']},{hyperparameters['epochs_1batch']},{hyperparameters['learning_rate']},{total_time},{abs_error.item()}\n")
    with open(f'./results/{filename.split(".")[0]}_results.csv', 'a') as f:
        f.write(",".join(best_complexity[0].flatten().astype(str)) + f",{best_complexity[1].item()}\n")
    return best_complexity, total_time

def train_nn(test_set : Dataset, hidden_size, epochs, learning_rate):
    input_size = test_set.data.shape[1]

    model = NN.MLP(input_size, hidden_size)
    # print("Training...")

    for epoch in range(epochs):
        test_set.apply(np.random.shuffle)
        output = model.forward(test_set.data)
        flat_output = output.flatten()
        loss = ((test_set.labels[:flat_output.shape[0]] - flat_output)**2).mean()
        model.backward(test_set.data, test_set.labels, output)
        update_weights(model, learning_rate)
        # if (epoch % 100 == 0):
        #     print(f"\tloss: {loss}")

    return model

if __name__ == "__main__":
    for count_i in range(0, 90):
        main("datasets", tune_configuration, budget=100, size_eval=10+(10 * (count_i //10)), hidden_size=10, epochs=300, epochs_1batch = 30, learning_rate=0.01, evaluation_func=min)
    for count_i in range(0, 100):
        main("datasets", tune_configuration, budget=50+(10 * (count_i //10)), size_eval=10, hidden_size=10,
                 epochs=300, epochs_1batch=30, learning_rate=0.01, evaluation_func=min)
    for count_i in range(0, 100):
        main("datasets", tune_configuration, budget=100, size_eval=10, hidden_size=10+(10 * (count_i //10)), epochs=300, epochs_1batch = 30, learning_rate=0.01, evaluation_func=min)
    for count_i in range(0, 100):
        main("datasets", tune_configuration, budget=100, size_eval=10, hidden_size=10,
                 epochs=100+(100 * (count_i //20)), epochs_1batch=30, learning_rate=0.01, evaluation_func=min)
    #0.1 * 10**(-(count_i // 10)/2)
    for count_i in range(0, 100):
        main("datasets", tune_configuration, budget=100, size_eval=10, hidden_size=10, epochs=300, epochs_1batch = 10+(10 * (count_i //10)), learning_rate=0.01, evaluation_func=min)
    for count_i in range(0, 100):
        main("datasets", tune_configuration, budget=100, size_eval=10, hidden_size=10,
                 epochs=300, epochs_1batch=30, learning_rate=0.1 * 10**(-(count_i // 10)/2), evaluation_func=min)