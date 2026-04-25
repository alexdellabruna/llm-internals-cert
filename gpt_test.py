import json
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from common import *
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

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
            os.makedirs("./gpt", exist_ok=True)
            plt.savefig(f"./gpt/{plot_name}.png", dpi=150)
        plt.close()

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model(**inputs)
logits = outputs.logits  # shape: [batch, seq_len, vocab_size]

print(outputs)
print(logits.shape)
exit(1)


out=0
v_copy_arr_1=[]

# generate test random input
test_input=[]

for i in range(RANDOM_DATASET_SIZE):
    test_input.append(torch.randn(1,4))

for x in test_input:
    tracking=NNTracking()
    tracking.get_all_layers(model)
    out=model(x)
    v_copy_arr_1.append(tracking.visualisation.copy())

print(v_copy_arr_1)

save_per_layer_predictions(v_copy_arr_1, f"model_1_per_layer_predictions_{r+1}")