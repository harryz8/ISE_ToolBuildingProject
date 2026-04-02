import os

files = os.listdir("datasets")

for file in files:
    current = ""
    titles = ""
    with open("./datasets/" + file, "r") as f:
        titles = f.read()
    with open("./results/" + file[:-4]+"_results.csv", "r") as f:
        current = f.read()
        print(current)
    with open("./results/" + file[:-4]+"_results.csv", "w") as f:
        current_list = current.split("\n")
        f.write(f"{titles}\n"+"\n".join(current_list[1:]))