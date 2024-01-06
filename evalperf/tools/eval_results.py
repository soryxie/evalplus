import argparse
import json
import os
from collections import namedtuple
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

k = 4


class AllModelResults:
    def __init__(self, model_results_path: str):
        self.model_results_path = model_results_path
        self.model_results = self.gather_all_models_results()
        self.outputs = {task_id: {} for task_id in self.model_results.keys()}

    def gather_all_models_results(self) -> Dict[str, Dict[str, float]]:
        """
        Gather all models' results into one json file.
        Each model's results: {task_id: ["success", average_rtime] or ["failed", 0]}
        Gathered results: {task_id: {model_name: [task_results]}}
        """
        model_results = {}

        files = os.listdir(self.model_results_path)
        for file in files:
            file_path = os.path.join(self.model_results_path, file)
            if os.path.isfile(file_path) and ".json" in file:
                print(f"{file} in this directory")
                model_result = json.load(open(file_path, "r"))["eval"]
                for task_id, task_result in model_result.items():
                    if task_id not in model_results:
                        model_results[task_id] = {}
                    model_results[task_id][file[:-5]] = task_result

        return model_results

    def clustering(self, k: int, seed: int):
        """
        For each task, cluster the models based on their average runtime.
        """
        for task_id, model_results in self.model_results.items():
            avgs = np.array([], dtype=float)
            model_names = []
            for model_name, task_result in model_results.items():
                if task_result[0] == "success":
                    avgs = np.append(avgs, task_result[1])
                    model_names.append(model_name)
            if len(avgs) == 0:
                print(f"[Clustering FAIL] Task {task_id}: No model success")
                self.outputs[task_id]["centroids"] = []
                self.outputs[task_id]["clusters"] = dict()
                continue

            # clustering, get centroids and clusters
            kmeans = KMeans(n_clusters=min(k, len(avgs)), random_state=seed, n_init='auto')
            kmeans.fit(avgs.reshape(-1, 1))
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
            clusters = {i: [] for i in range(len(centroids))}
            for i in range(len(model_names)):
                clusters[labels[i]].append(model_names[i])

            # flatten and sort
            clusters = [clusters[i] for i in range(len(centroids))]
            centroids = centroids.reshape(-1).tolist()
            centroids, clusters = zip(*sorted(zip(centroids, clusters)))
            
            print(f"[Clustering SUCC] Task {task_id}: {len(model_names)} models, get {len(centroids)} clusters")
            self.outputs[task_id]["centroids"] = centroids
            self.outputs[task_id]["clusters"] = clusters
                
    def cal_cv(self):
        """ Calculate the coefficient of variation of every task. """
        for task_id, model_results in self.model_results.items():
            avgs = np.array([], dtype=float)
            for model_name, task_result in model_results.items():
                if task_result[0] == "success":
                    avgs = np.append(avgs, task_result[1])
            if len(avgs) == 0:
                self.outputs[task_id]["cv"] = 0
                continue
            cv = np.std(avgs) / np.mean(avgs)
            self.outputs[task_id]["cv"] = cv

    def save_results_to_excel(self, output_dir: str):
        """ Save the results to excel. """
        excel_output = {task_id: {} for task_id in self.model_results.keys()}
        for task_id, task_result in self.outputs.items():
            excel_output[task_id]["cv"] = task_result["cv"]
            for i in range(len(task_result["centroids"])):
                excel_output[task_id][f"clusters{i}"] = [f"{task_result['centroids'][i]:.2e}"] + task_result["clusters"][i]
        df = pd.DataFrame.from_dict(excel_output, orient="index")
        df.to_excel(os.path.join(output_dir, "task_cv.xlsx"))

    def ranking_all_models_by_perf(self, output_dir: str):
        """
        Rank all models by their performance.
        """
        model_scores = {task_id: {} for task_id in self.model_results.keys()}
        for task_id, model_results in self.model_results.items():
            # filter out low CV tasks
            if self.outputs[task_id]["cv"] < 0.2:
                continue

            # calculate the score of each model
            # assume the centroids are sorted
            # the score of a model is the number of models that have a higher result than it
            for model_name, task_result in model_results.items():
                if task_result[0] != "success":
                    model_scores[task_id][model_name] = 0.0
                else:
                    centroids = self.outputs[task_id]["centroids"]
                    clusters = self.outputs[task_id]["clusters"]

                    battled_clusters = 0
                    for controid in centroids:
                        if task_result[1] > controid:
                            break
                        battled_clusters += 1
                    battled_models = sum([len(cluster) for cluster in clusters[:battled_clusters]])
                    model_scores[task_id][model_name] = battled_models / len(model_results) * 100
        
        # save the results to excel
        df = pd.DataFrame.from_dict(model_scores, orient="index")
        df = df.transpose()
        df.to_excel(os.path.join(output_dir, "model_scores.xlsx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval"
    )
    parser.add_argument("--data-dir", type=str, default="./evalperf/data/humaneval")
    parser.add_argument("--output-dir", type=str, default="./evalperf/data/results")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_model_results = AllModelResults(data_dir)
    all_model_results.clustering(k, 777)
    all_model_results.cal_cv()
    all_model_results.save_results_to_excel(output_dir)
    all_model_results.ranking_all_models_by_perf(output_dir)
