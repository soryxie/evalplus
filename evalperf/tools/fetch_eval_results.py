"""
Fetch eval_results.json from the models directory and copy them to the data directory.
"""
import argparse
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--models-dir", type=str, default="/data/songrun-data/evalperf/model")
    parser.add_argument("--output-dir", type=str, default="./evalperf/data")
    args = parser.parse_args()

    base_dir = f"{args.models_dir}"
    des_dir = f"{args.output_dir}/{args.dataset}"

    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    for root, dirs, files in os.walk(base_dir):
        if "eval_results.json" in files:
            folder_name = os.path.basename(root)
            file_path = os.path.join(root, "eval_results.json")
            destination = os.path.join(
                des_dir, f'{folder_name.replace("-sanitized", "")}.json'
            )
            shutil.copy(file_path, destination)
            print(f"Copied eval_results.json from {root} to {destination}")
