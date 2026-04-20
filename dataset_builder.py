import os

program_name = str(input("What is the name of your program: "))
config_params = int(input("How many configurable options does your program have: "))
titles = []
params = []
total_params = 1

for param in range(config_params):
    titles.append(str(input(f"What is the name of option {param+1}: ")))
    if titles == "":
        raise Exception("Invalid entry. Please try again.")
    param_type = str(input(f"What is the type of option {param+1} (Binary/Numeric/Other): ")).lower()
    if param_type == "other":
        print("Encode your options as consecutive numbers starting from 0.")
        param_num = int(input("How many options are there?: "))
        total_params *= param_num
        params.append([x for x in range(0, param_num)])
    elif param_type == "binary":
        param_num = 2
        total_params *= param_num
        params.append([x for x in range(0, param_num)])
    elif param_type == "numeric":
        max_param = int(input(f"Enter the maximum possible value (inclusive) of option {param+1}: "))
        min_param = int(input(f"Enter the minimum possible value (inclusive) of option {param + 1}: "))
        if max_param == min_param:
            raise Exception("Max and min should be different. Please try again.")
        data_type = str(input(f"Is the numeric range continuous or discrete? (continuous/discrete): ")).lower()
        if data_type != "continuous" and data_type != "discrete":
            raise Exception("Invalid entry. Please try again.")
        step_param = int(input("What is the step size between options?: " if data_type == "discrete" else "How regularly across the range should will be measured (step size)?: "))
        if step_param <= 0:
            raise Exception("Invalid entry. Please try again.")
        option_list = [x for x in range(min_param, max_param+1, step_param)]
        params.append(option_list)
        total_params *= len(option_list)
    else:
        raise Exception("Invalid entry. Please try again.")

#generate dataset
if os.path.isfile(f"./datasets/{program_name}.csv"):
    raise Exception(f"File {program_name} already exists. Please try again.")
with open(f"./datasets/{program_name}.csv", "w") as f:
    f.write(",".join(titles)+",performance\n")
    counter = total_params
    all_para = [x for x in range(config_params)]
    while counter > 0:
        for param in range(config_params-1):
            divi = counter
            for para in all_para[param+1:]:
                divi = divi // len(params[para])
            divi = divi % len(params[param])
            f.write(str(params[param][divi])+",")
        f.write(str(params[config_params-1][counter % len(params[config_params-1])]))
        f.write(","+str(float("inf")))
        f.write("\n")
        counter -= 1