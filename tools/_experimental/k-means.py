import json
import numpy as np
from collections import namedtuple


input_path = "./model_avg_data.json"
k = 5


def k_means_clustering(data, k, max_iterations=10000000):
    # Randomly initialize k centroids from the data
    centroids = np.random.choice(data, size=k, replace=False)

    for _ in range(max_iterations):
        # Compute the distances between data points and centroids
        distances = np.abs(data[:, np.newaxis] - centroids)

        # Assign data points to the nearest centroid
        clusters = np.argmin(distances, axis=1)

        # Calculate new centroids as the mean of data points in each cluster
        new_centroids = np.array([data[clusters==i].mean() for i in range(k)])

        # If centroids do not change, stop the iteration
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return clusters, centroids


if __name__ == "__main__":

    data = json.load(open(input_path, "r"))

    selected_models = []
    
    for task_id, task_results in data.items():
        selected_models = output = '\n'.join(f'- {item}' for item in task_results.keys())
        print("models are ")
        print(selected_models)
        break

    print("{}\t{}\t{}".format("task_id", "k-means-cv", "groups"))
    
    for task_id, task_results in data.items():
        # print("---processing task: ", task_id, "---")

        avgs = np.array([], dtype=float)

        for _, avg in task_results.items():
            # Select correct models
            if avg["plus"] > 0:      
                avgs = np.append(avgs, avg["plus"])

        if len(avgs) >= k:
            clusters, centroids = k_means_clustering(avgs, k)
        else:
            centroids = avgs

        if len(centroids) > 0:
            mean = np.mean(centroids)
            std = np.std(centroids)
            cv = np.abs(std / mean)
            print("{}\t{}\t{}".format(task_id, cv, len(centroids)))
        else:

            print("{}\t{}".format(task_id, "No data"))