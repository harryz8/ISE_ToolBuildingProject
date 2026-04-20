from dataset import *
import os
from scipy.stats import ranksums

for file in ["7z", "Apache", "brotli", "LLVM", "PostgreSQL", "spear", "storm", "x264"]:
    data = load_csv(file+"_results", "data", True, float)
    other_data = load_csv(file+"_search_results", "data", True, float)
    answer = ranksums(data.labels, other_data.labels, alternative="two-sided")
    print(f"file: {file}, p_value: {answer[1]}, New solution median: {np.median(data.labels)}, Random search median: {np.median(other_data.labels)}")