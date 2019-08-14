"""
Online streaming feature selection using adapted Neighborhood Rough Set
author: Peng Zhou, Xuegang Hu, Peipei Li, Xindong Wu
https://doi.org/10.1016/j.ins.2018.12.074
"""


import heapq
from tools import *


# for continuous data
def in_delta_neighborhood(universe, x, y, radius, distance, attributes, display=False):
    """
    if the sample y is the neighborhood of the sample x by the limitation of the radius, return True, else return False
    Applies to attributes whose attribute values are discrete data
    :param universe: the universe of objects(feature vector/sample/instance)
    :param x: feature vector, sample, instance (use the index of the sample in the universe)
    :param y: the same as above
    :param radius: the radius
    :param distance: the method to calculate the distance
    :param attributes: the feature(s)/attribute(s) of object
    :param display: default is Fault ,if is True, the distance will display
    :return: True/False
    """
    if display:
        print(x, y, distance(universe, x, y, attributes))
    if distance(universe, x, y, attributes) <= radius:
        return True
    else:
        return False


def generate_delta_neighborhood(universe, attributes, radius, distance, display_distance=False):
    """
    generate the delta neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param radius: radius
    :param distance: the method to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: list, the delta_neighborhood(raw_universe/attributes)
    """
    elementary_sets = []
    for i in range(len(universe)):
        flag = True
        for elementary_single_set in elementary_sets:
            if in_delta_neighborhood(
                    universe, i, elementary_single_set[0], radius, distance, attributes, display=display_distance):
                elementary_single_set.append(i)
                flag = False
                break
        if flag:
            elementary_sets.append([i])
    return elementary_sets


def generate_delta_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_delta_neighborhood(np.array(data), [0, 1, 2, 3], 33, euclidean_distance)
    print("result:", result)
    # result: [[0], [1, 4], [2, 3, 6], [5, 7]]
    return


# for continuous data
def generate_distance_matrix(universe, attributes, distance, display_distance=False):
    """
    generate the distance triangle matrix
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param distance: the method to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: the distance triangle matrix
    """
    matrix = np.triu(np.zeros(len(universe)**2).reshape(len(universe), len(universe)))
    for i in range(len(universe)):
        for j in range(i, len(universe)):
            matrix[i][j] = distance(universe, i, j, attributes)
    matrix += matrix.T - np.diag(matrix.diagonal())
    if display_distance:
        print(matrix)
    return matrix


# for continuous data
def generate_distance_triangle_matrix_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_distance_matrix(np.array(data), [0, 1, 2, 3], euclidean_distance)
    print("result:", result[0])
    k_nearest_index = heapq.nsmallest(3, range(len(result[0])), result[0].take)
    print(k_nearest_index)
    print(type(k_nearest_index))
    print(k_nearest_index[1:])
    print(k_nearest_index.pop(0))
    print(k_nearest_index.pop(0))
    print(k_nearest_index.pop(0))
    print(k_nearest_index.pop(0))
    return


# for continuous data
def generate_k_nearest_neighborhood(universe, attributes, k, distance, display_distance=False):
    """
    generate the k nearest neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param k: k
    :param distance: the method to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: list, the k_nearest_neighborhood(raw_universe/attributes)
    """
    distance = generate_distance_matrix(universe, attributes, distance, display_distance)
    universe_index = list(np.arange(universe.shape[0]))
    elementary_sets = []  # R-elementary sets
    i = 0
    while i < len(universe_index):
        k_nearest_index = \
            heapq.nsmallest(k + 1, range(len(distance[universe_index[i]])), distance[universe_index[i]].take)
        elementary_sets.append(k_nearest_index)
        i += 1
    return elementary_sets


def generate_k_nearest_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_k_nearest_neighborhood(
        np.array(data), [0, 1, 2, 3], 2, euclidean_distance)
    print(result)
    return


def generate_gap_neighborhood(
        universe, attributes, gap_weight=1., distance=euclidean_distance, display_distance=False):
    """
    generate the gap neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param gap_weight: the weight of gap
    :param distance: the method to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: list, the k_nearest_neighborhood(raw_universe/attributes)
    """
    distance_matrix = generate_distance_matrix(universe, attributes, distance, display_distance)
    universe_index = list(np.arange(universe.shape[0]))
    elementary_sets = []  # R-elementary sets
    i = 0
    while i < len(universe_index):
        distance_sort = heapq.nsmallest(universe.shape[0], range(len(distance_matrix[universe_index[i]])),
                                        distance_matrix[universe_index[i]].take)
        distance_max = distance(universe, distance_sort[0], distance_sort[-1], attributes)
        distance_min = distance(universe, distance_sort[0], distance_sort[1], attributes)
        distance_mean = (distance_max - distance_min) / (universe.shape[0] - 1)
        gap = distance_mean * gap_weight
        j = 1
        while j < len(distance_sort):
            if distance(universe, distance_sort[j], distance_sort[j - 1], attributes) >= gap:
                break
        elementary_sets.append(distance_sort[:j])
        i += 1
    return elementary_sets


def generate_gap_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_gap_neighborhood(
        np.array(data), [0, 1], 1.5, standardized_euclidean_distance)
    print(result)
    return


if __name__ == '__main__':
    # generate_delta_neighborhood_test()
    # generate_distance_triangle_matrix_test()
    # generate_k_nearest_neighborhood_test()
    generate_gap_neighborhood_test()
    pass
