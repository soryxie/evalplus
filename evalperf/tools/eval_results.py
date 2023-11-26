import argparse
import json
import os
from collections import namedtuple

import numpy as np

k = 4
k_means_iters = 10000000
perf_times = 10
avg_result_path = "./model_avg_data.json"


def process_model_avg_data(model_results_path: str):
    model_avg_data = {}

    files = os.listdir(model_results_path)

    for file in files:
        if os.path.isfile(os.path.join(model_results_path, file)) and ".json" in file:
            print("{} in this directory".format(file))

            res = json.load(open(model_results_path + file, "r"))["eval"]
            for task_id, task_results in res.items():
                if task_results:
                    file_number = len(task_results["0"]["perf_result"])
            print("file_number is", file_number)

            for task_id, task_results in res.items():
                total_sum = 0
                total_n, total_plus_n = 0, 0

                for file_no in range(file_number):
                    succ_n = 0
                    rtime_sum = 0
                    rtime_avg = 0.0
                    for run_time in range(perf_times):
                        if str(run_time) in task_results:
                            status = task_results[str(run_time)]["perf_result"][
                                file_no
                            ][0]
                            if status == "success":
                                succ_n += 1
                                rtime_sum += task_results[str(run_time)]["perf_result"][
                                    file_no
                                ][1]

                    if succ_n == perf_times:
                        total_sum += rtime_sum
                        total_n += perf_times

                if task_id not in model_avg_data:
                    model_avg_data[task_id] = {}
                model_avg_data[task_id][file[:-5]] = (
                    (total_sum / total_n) if total_n else 0
                )

    with open(avg_result_path, "w") as f:
        json.dump(model_avg_data, f)


def k_means_clustering(data, k, max_iterations):
    centroids = np.random.choice(data, size=k, replace=False)

    for _ in range(max_iterations):
        distances = np.abs(data[:, np.newaxis] - centroids)
        clusters = np.argmin(distances, axis=1)
        new_centroids = np.array([data[clusters == i].mean() for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return clusters, centroids


def print_task_cv():
    data = json.load(open(avg_result_path, "r"))
    selected_models = []
    clu_res = {}
    print("{}\t{}\t{}\t{}".format("task_id", "k-means-cv", "groups", "rtime"))
    for task_id, task_results in data.items():
        avgs = np.array([], dtype=float)
        for model, avg in task_results.items():
            if avg > 0:
                avgs = np.append(avgs, avg)
        if len(avgs) >= k:
            clusters, centroids = k_means_clustering(avgs, k, k_means_iters)
        else:
            centroids = avgs
        if len(centroids) > 0:
            mean = np.mean(centroids)
            std = np.std(centroids)
            cv = np.abs(std / mean)
            print("{}\t{}\t{}\t{}".format(task_id, cv, len(centroids), mean))
        else:
            print("{}\t{}".format(task_id, "No data"))

        clu_res[task_id] = centroids.tolist()
    with open("clu_res.json", "w") as f:
        json.dump(clu_res, f)


def ranking_all_models_by_perf(select_tasks_path: str):

    data = json.load(open(avg_result_path, "r"))
    clusters = json.load(open("clu_res.json", "r"))
    with open(select_tasks_path, "r") as file:
        lines = file.readlines()
    selected_task_id = [line.strip() for line in lines]

    selected_models = {}

    for task_id, task_results in data.items():
        if task_id not in selected_task_id:
            continue
        avgs = np.array([], dtype=float)
        centroids = clusters[task_id]
        centroids.append(0)
        centroids = sorted(centroids)

        for model, avg in task_results.items():
            if model not in selected_models:
                selected_models[model] = {}

            for level in range(1, len(centroids)):
                if centroids[len(centroids) - 1] < avg:
                    selected_models[model][task_id] = 5
                    break
                if centroids[level - 1] < avg and avg <= centroids[level]:
                    selected_models[model][task_id] = 5 * level / len(centroids)
                    break

    selected_task_id = sorted(selected_task_id, key=lambda x: int(x[len("Mbpp_") :]))

    print("problem\t", end="")
    for task_id in selected_task_id:
        print(f"{task_id}\t", end="")
    print("")

    for model in selected_models:
        print(f"{model}\t", end="")
        for task_id in selected_task_id:
            if task_id in selected_models[model]:
                print(f"{selected_models[model][task_id]}\t", end="")
            else:
                print("\t", end="")
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval"
    )
    args = parser.parse_args()

    model_results_path = f"../data/{args.dataset}/"
    select_tasks_path = f"./selected_{args.dataset}_tasks.txt"

    # 1. generate avg_result.json
    process_model_avg_data(model_results_path)
    # 2. generate clu_res.json and task cv table
    print_task_cv()
    # 3. generate ranking table
    ranking_all_models_by_perf(select_tasks_path)
