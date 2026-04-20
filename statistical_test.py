from dataset import *
from scipy.stats import ranksums
import os

print(f"Current location: {os.getcwd()}")
path_to_file_this = str(input("Please enter the path to the results from algorithm 1 (e.g. \"./results\" <- Note: no trailing slash): "))
filename_this = str(input("Please enter the name of the results file for algorithm 1 (Note: no file extension) (e.g. \"Apache_results\"): "))
path_to_file_other = str(input("Please enter the path to the results from algorithm 2 (e.g. \"./results\" <- Note: no trailing slash): "))
filename_other = str(input("Please enter the name of the results file for algorithm 2 (Note: no file extension) (e.g. \"Apache_results\"): "))
data = load_csv(filename_this, path_to_file_this, True, float)
other_data = load_csv(filename_other, path_to_file_other, True, float)
answer = ranksums(data.labels, other_data.labels, alternative="two-sided")
print(f"p_value: {answer[1]}, Algorithm 1 median: {np.median(data.labels)}, Algorithm 2 median: {np.median(other_data.labels)}")