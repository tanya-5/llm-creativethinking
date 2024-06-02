import json
import numpy as np

file_path = "data/SP_eval_data_for_practice.npy"
out_path = "SP-eval.json"

data = np.load(file_path, allow_pickle=True).tolist()
# print(data)

with open(out_path, 'w') as f:
    json.dump(data, f, indent=2)