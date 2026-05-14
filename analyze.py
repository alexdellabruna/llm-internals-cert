import os
import json
import shutil
from matplotlib import pyplot as plt
import numpy as np
import scipy.spatial.distance as spd
import argparse

def softmax(x):
    e_x = np.exp(x - np.max(x)) # Sottrazione del max per stabilità numerica
    return e_x / e_x.sum()

parser = argparse.ArgumentParser()
parser.add_argument("--comp-out-dir", "-c", help="Compare output directory", default=None)
args, _ = parser.parse_known_args()

if not args.comp_out_dir:
    print("Compare output directory not provided. Please provide it using the --comp-out-dir or -c argument.")
    exit(1)

comp_out_dir = args.comp_out_dir

shutil.rmtree('./analyze_out/', ignore_errors=True)

with open(os.path.join("gpt_out", "global.json"), "r") as result_file:
    res_file_raw=result_file.read()

    result_arr = json.loads(res_file_raw)

    with open(os.path.join(comp_out_dir, "global.json"), "r") as compare_file:
        compare_file_raw=compare_file.read()
        
        compare_result_arr = json.loads(compare_file_raw)

        plt.plot(range(len(compare_result_arr)), compare_result_arr)
        plt.plot(range(len(result_arr)), result_arr)
        os.makedirs("./analyze_out", exist_ok=True)
        plt.savefig(os.path.join("analyze_out/", "out.png"))
        #plt.show()
        plt.close()

        # Trasformazione dei dati
        compare_result_arr = softmax(compare_result_arr)
        result_arr = softmax(result_arr)

        js_distance=spd.jensenshannon(compare_result_arr, result_arr)

        integral=False

        if js_distance < 0.1:
            integral=True
        
        print(f"Jensen-Shannon distance: {js_distance}")
        print(f"Integral: {integral}")

        with open(os.path.join("analyze_out/", "distance.json"), "w") as distance_file:
            distance_file.write(json.dumps({"js_distance": js_distance, "integral": integral}))