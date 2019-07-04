import random
import numpy as np
from collections import Counter

def read_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            line_data = line[:-1].split(',')
            data.append(line_data)

    return data[1:]


def standardize_data(data):
    temp = np.array(data, dtype='float64').T
    result = []
    for row in temp:
        ro_std = np.std(row)
        ro_avg = np.average(row)
        t = (row-ro_avg)/ro_std
        result.append(t)

    return np.array(result).T.tolist()



def select_specified_attrs(data, attr_list):
    result = []
    labels = []
    for row in data:
        temp = []
        for j in range(len(row)):
            if j in attr_list:
                temp.append(row[j])
        result.append(temp)
        labels.append(row[1])
    return result, labels

        
def select_not_specified_attrs(data, attr_list):
    result = []
    labels = []
    for row in data:
        temp = []
        for j in range(len(row)):
            if j not in attr_list:
                temp.append(row[j])
        result.append(temp)
        labels.append(row[1])
    return result, labels


def euclidean_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (float(x[i]) - float(y[i]))**2
    return sum ** (1/2)


def calculate_centroids(clusters):
    new_centroids = []
    for c in clusters:
        temp = [0 for i in range(len(c[0]))]
        for point in c:
            for j in range(len(point)):
                temp[j] += float(point[j])
        
        for i in range(len(temp)):
            temp[i] = temp[i]/len(c)
        new_centroids.append(temp)
    return new_centroids
    

def mykmeans(X, k):
    labels = X[-1]
    X = standardize_data(X[0])
    centroids = random.sample(X, k)
    
    while True:
        k_clusters = [[] for i in range(k)]
        k_dist = [[] for i in range(k)]
        for p, point in enumerate(X):
            dist_from_centroid = []
            for c in centroids:
                dist_from_centroid.append(euclidean_distance(point, c))

            min_idx = dist_from_centroid.index(min(dist_from_centroid))
            k_clusters[min_idx].append(point)
            k_dist[min_idx].append(labels[p])

        new_centroids = calculate_centroids(k_clusters)
        if new_centroids != centroids:
            centroids = new_centroids
        else:
            break

    pos_distribution = []
    for cluster in k_dist:
        distr = Counter(cluster)
        pos_distribution.append(distr)

    print("K_cluster centroids:", centroids)
    print("distribution in each cluster:", pos_distribution)
    
    return centroids, k_clusters, pos_distribution


def run():
    filename = "NBAstats.csv"
    data = read_data(filename)
    kk = [3,5]
    exclude_colm = [0,1,3]
    X = select_not_specified_attrs(data, exclude_colm)
    for k in kk:
        result = mykmeans(X, k)

    print("#####################################")
    print("####     use attributes {2P%, 3P%, FT%, TRB, AST, STL, BLK}   ####")
    print("#####################################")
    select_only = [15,12,19,22,23,24,25]
    X = select_not_specified_attrs(data, select_only)
    for k in kk:
        result = mykmeans(X, k)

if __name__ == "__main__":
    run()

    






    

    
