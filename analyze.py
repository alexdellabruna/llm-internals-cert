import json
from common import *
import numpy as np
import os
import matplotlib.pyplot as plt

total_distance_arr=[]
per_layer_distance_arr=[]

for r in range(REPETITIONS):
    with open(f"./out/out{r+1}.json","r") as f:
        data=json.load(f)
        total_distance_arr.append(data["total_distance"])
        per_layer_distance_arr.append(data["per_layer_distance"])

# plot total distance for each repetition
td = np.array(total_distance_arr)
runs = np.arange(1, td.size + 1)

plt.figure(figsize=(8, 4))
plt.plot(runs, td, marker='o', linestyle='-')
plt.xlabel('Run')
plt.ylabel('Total distance')
plt.title('Total distance per run')
plt.grid(True)
plt.tight_layout()
plt.savefig('./out/total_distance_per_run.png')
plt.show()

plt.figure(figsize=(14, 5))

for r, run_dict in enumerate(per_layer_distance_arr):
    layers = list(run_dict.keys())
    distances = list(run_dict.values())
    plt.plot(layers, distances, marker="o", alpha=0.7, label=f"run {r+1}")

plt.xticks(rotation=45, ha="right")
plt.ylabel("Distance")
plt.title("Per-layer distance for each run")
plt.legend()
plt.tight_layout()
plt.savefig('./out/per_layer_distance_per_run.png')
plt.show()

