# Basic implementation of the k-means clustering algorithm.
# Manuel Germán Morales
import copy
import random

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_biclusters, make_moons #Try generating different sets!

# Euclidean distance
def dist(pointA, pointB):
    return np.sqrt(sum((pointB - pointA) ** 2))

def draw_cluster_plot(data, centroids, sol, text, real_centroids = None):
    colors = ["#880000", "#008800", "#000088", "#888800", "#008888", "#880088", "#808080", "#888000", "#000888"]
    #center_col = ["#ff0088", "#88ff00", "#0880ff", "#ffff88", "#88ffff", "#ff88ff", "#f8f8f8", "#fff888", "#888fff"]
    fig = plt.figure()
    ax = fig.add_subplot()
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Point',
                              markerfacecolor='#000000', markersize=10),
                       Line2D([0], [0], marker=(5,1), color='w', label='Apprx. centroid',
                              markerfacecolor='#fd55f4', markersize=15),
                       Line2D([0], [0], marker=">", color='w', label='Real centroid',
                              markerfacecolor='#4c2882', markersize=10)
                       ]
    for indp, point in enumerate(data):
        ax.scatter(point[1], point[0], color=colors[sol[indp]])
    for indc, center in enumerate(centroids):
        ax.scatter(center[1], center[0], color='#fd55f4', marker=(5,1), s=150)
    if real_centroids is not None:
        for indc, center in enumerate(real_centroids):
            ax.scatter(center[1], center[0], color="#4c2882", marker=">", s=100)
    ax.title.set_text(text)
    ax.legend(handles=legend_elements)
    plt.show()

# Cluster elements
def do_clusters(data, centroids):
    result = []
    # For every point in the dataset, get its cluster
    for index, point in enumerate(data):
        # Aux vars needed
        mindist = float('inf')
        mincluster = -1
        # Calculate the distance between this point and the centroids.
        for indexC, centroid in enumerate(centroids):
            distpc = dist(point, centroid)
            # Minimize mindist value, save the correspondent cluster
            if distpc < mindist:
                mindist = distpc
                mincluster = indexC
        # Save the cluster with the minimun distance to the point.
        result.append(mincluster)
    return result

# Should I stop?
def stopCondition(newarr, oldarr, it, maxIt = None):
    if oldarr is None:
        return False
    # Max iterations
    if maxIt is not None and it >= maxIt:
        return True
    # Have the centroids changed? (or "moved"?)
    for i in range(len(newarr)):
        err = dist(oldarr[i], newarr[i])
        if err > 0.00001:
            return False
    return True

# Main Algorithm
def kmeans(data, n_clusters, max_it = None):
    it = 0
    old_centroids = None
    # Select N initial random centroids from our data
    centroids = random.choices(data, k=n_clusters)
    # print(f'Centroides iniciales {centroids}')
    result = None
    while not stopCondition(centroids, old_centroids, it, max_it):
        # 1) assignment phase
        result = do_clusters(data, centroids)
        draw_cluster_plot(data, centroids, result, 'It ' + str(it))

        # 2) Update phase
        # 2.1) Save the old centroids
        old_centroids = copy.deepcopy(centroids)
        for center in centroids:
            center.fill(0)

        # 2.2) Update the centroids: Every centroid moves to the mean value of its cluster.
        for index in range(len(centroids)):
            points = [data[i] for i in range(len(result)) if result[i] == index]
            if not points:
                # If a centroid do not has points, reinit it.
                centroids[index] = random.choice(data)
            else:
                # Else, find the mean value
                centroids[index] = np.mean(points, axis=0)
        #print(f'Nuevos centroides {centroids}')
        it += 1
    return result, centroids

if __name__ == '__main__':
    #features = make_moons(n_samples=1000, noise=0.5)
    #features = make_circles(n_samples=1000, noise=0.1)
    features = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=3, random_state=1245, return_centers=True)

    # Draw initial dataset
    fig = plt.figure()
    ax = fig.add_subplot()

    for ind, point in enumerate(features[0]):
        ax.scatter(point[1], point[0])
    plt.show()

    # Execute k-means....
    result, centroids = kmeans(features[0], 5)

    # Draw dataset clustered! :D
    draw_cluster_plot(features[0], centroids, result, "Final", real_centroids=features[2])