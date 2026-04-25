import json
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from common import *
from model import *
import numpy as np

def save_per_layer_predictions(v_copy_arr, plot_name):
    # aggregate a scalar summary per layer for each sample in v_copy_arr
    per_layer_vals = {}
    for sample in v_copy_arr:
        if isinstance(sample, dict):
            items = sample.items()
        else:
            items = enumerate(sample)
        for k, v in items:
            # convert tensor/array to a single scalar (use mean absolute)
            if isinstance(v, torch.Tensor):
                val = v.detach().cpu().abs().mean().item()
            else:
                try:
                    arr = np.array(v)
                    val = float(np.abs(arr).mean())
                except Exception:
                    continue
            per_layer_vals.setdefault(str(k), []).append(val)

    if not per_layer_vals:
        print("No per-layer values to plot.")
        pass
    else:
        # ensure consistent ordering (try numeric if possible)
        names = list(per_layer_vals.keys())
        try:
            names = [n for n in sorted(names, key=lambda x: int(x))]
        except Exception:
            names = list(names)

        # compute mean per layer
        means = [np.mean(per_layer_vals[n]) for n in names]

        plt.figure(figsize=(8, 4))
        # plot each sample as a light line
        n_samples = len(next(iter(per_layer_vals.values())))
        for i in range(n_samples):
            ys = [per_layer_vals[n][i] for n in names]
            plt.plot(names, ys, color="gray", alpha=0.3, linewidth=0.8)
        # plot mean
        plt.plot(names, means, marker="o", color="red", linewidth=2, label="mean")
        plt.xlabel("Layer")
        plt.ylabel("Mean |prediction|")
        plt.title("Per-layer predictions across samples")
        plt.legend()
        plt.tight_layout()
        if DRAW_PLOT:
            plt.show()
        else:
            os.makedirs("./out", exist_ok=True)
            plt.savefig(f"./out/{plot_name}.png", dpi=150)
        plt.close()

for r in range(REPETITIONS):
    print("Repetition "+str(r+1)+" of "+str(REPETITIONS))

    total_distance=0
    total_per_layer_distance={}

    net1 = Model()
    net1.load_state_dict(torch.load("./models/model_1.pth", weights_only=True))
    net1.eval()

    net2 = Model()
    net2.load_state_dict(torch.load("./models/model_2.pth", weights_only=True))
    net2.eval()

    out=0
    v_copy_arr_1=[]
    v_copy_arr_2=[]

    # generate test random input
    test_input=[]

    for i in range(RANDOM_DATASET_SIZE):
        test_input.append(torch.randn(1,4))

    for x in test_input:
        tracking=NNTracking()
        tracking.get_all_layers(net1)
        out=net1(x)
        v_copy_arr_1.append(tracking.visualisation.copy())
    
    save_per_layer_predictions(v_copy_arr_1, f"model_1_per_layer_predictions_{r+1}")

    for x_idx,x in enumerate(test_input):
        tracking=NNTracking()
        tracking.get_all_layers(net2)
        out=net2(x)
        v_copy_arr_2.append(tracking.visualisation.copy())
        distance,per_layer_distance=tracking.calc_distance(v_copy_arr_1[x_idx])
        total_distance+=distance/(RANDOM_DATASET_SIZE*REPETITIONS)
        for k,v in per_layer_distance.items():
            if k not in total_per_layer_distance.keys():
                total_per_layer_distance[k]=0
            else:
                total_per_layer_distance[k]+=v/(RANDOM_DATASET_SIZE*REPETITIONS)

    save_per_layer_predictions(v_copy_arr_2, f"model_2_per_layer_predictions_{r+1}")

    final_res={"total_distance": total_distance, "per_layer_distance": total_per_layer_distance}

    os.makedirs("./out", exist_ok=True)

    with open(f"./out/out{r+1}.json","w") as f:
        json.dump(final_res, f)

    print("Total distance: "+str(total_distance))

    print("Per layer distance:")

    for k,v in total_per_layer_distance.items():
        print(k,v)

    if DRAW_PLOT:
        names, counts = zip(*total_per_layer_distance.items())
        plt.plot(names, counts)
        plt.show()