import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval"
)
args = parser.parse_args()


base_dir = f"../models/{args.dataset}"
des_dir = f"../data/{args.dataset}"

for root, dirs, files in os.walk(base_dir):
    if "eval_results.json" in files:
        folder_name = os.path.basename(root)
        file_path = os.path.join(root, "eval_results.json")
        destination = os.path.join(
            des_dir, f'{folder_name.replace("-sanitized", "")}.json'
        )
        shutil.copy(file_path, destination)
        print(f"Copied eval_results.json from {root} to {destination}")
