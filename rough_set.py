"""
partition
lower and upper approximations
positive, boundary and negative regions
"""


import pandas as pd
import numpy as np
import time
import math


# for discrete data
def is_indiscernible(x, y, attributes):
    """
    if the two feature vector is indistinguishable, return True, else return False
    Applies to attributes whose attribute values are discrete data
    :param x: feature vector, sample, instance
    :param y: the same as above
    :param attributes: the feature(s)/attribute(s) of object
    :return: True/False
    """
    flag = True
    for attribute in attributes:
        if x[attribute] == y[attribute]:
            pass
        else:
            flag = False
            break
    return flag


# 伪代码
# 输入：样本集，特征
# 输出：基本集
# 1 基本集置空集
# 2 for x in 样本集
# 	flag = True
# 	for y in 基本集(y为等价类)
# 		如果 x 与y[0]相比为不可分辨关系
# 			将x加入该等价类
# 			Flag = False
# 			break
# 	if flag
# 		为 x 创建等价类加入到基本集中
def partition(universe, attributes):
    """
    calculate the partition of universe on attributes
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :return: list, the partition(universe/attributes)
    """
    elementary_sets = []
    for i in range(len(universe)):
        flag = True
        for elementary_single_set in elementary_sets:
            if is_indiscernible(universe[i], universe[elementary_single_set[0]], attributes):
                elementary_single_set.append(i)
                flag = False
                break
        if flag:
            elementary_sets.append([i])
    return elementary_sets


# partition实现方式2
# 伪代码
# 输入：样本集，特征
# 输出：基本集
# 1 基本集置空集
# 2 for x in 样本集
#     为 x 创建等价类
#     for y in 样本集-x
#         如果 y 和 x为不可分辨关系
#             将y加入到x的等价类中
# 	    将y从样本集中移除
def partition2(raw_universe, attributes):
    """
    Method 2 to calculate the partition of raw_universe on attributes
    :param raw_universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :return: list, the partition(raw_universe/attributes)
    """
    universe = list(np.arange(raw_universe.shape[0]))
    elementary_sets = []  # R-elementary sets
    i = 0
    while i < len(universe):
        IND_IS_R = [universe[i]]  # Equivalence class
        j = i + 1
        while j < len(universe):
            if is_indiscernible(raw_universe[universe[i]], raw_universe[universe[j]], attributes):
                IND_IS_R.append(universe[j])
                universe.remove(universe[j])
                j -= 1
            j += 1
        universe.remove(universe[i])
        elementary_sets.append(IND_IS_R)
    return elementary_sets


def partition_test_by_mushroom():
    """
    to test the partition and partition by mushroom
    calculate their time usage
    :return: None

    result:
    The time used: 37.04471778869629 seconds
    The time used: 0.01600813865661621 seconds
    The time used: 29.257173776626587 seconds
    The time used: 0.30783629417419434 seconds
    """
    start_time = time.time()
    data = pd.read_csv("mushroom.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    result = partition(attributes, np.arange(attributes.shape[1]))
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    result = partition(labels, np.arange(1))
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))

    start_time = time.time()
    data = pd.read_csv("mushroom.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    result = partition2(attributes, np.arange(attributes.shape[1]))
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    result = partition2(labels, np.arange(1))
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))
    return None


def partition_test():
    """
    check partition and partition2 result to confirm it's correct
    print the partition result
    :return: None

    result:
    The time used: 37.04471778869629 seconds
    The time used: 0.01600813865661621 seconds
    The time used: 29.257173776626587 seconds
    The time used: 0.30783629417419434 seconds
    """
    start_time = time.time()
    data = pd.read_csv("mushroom_little.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    result = partition(attributes, np.arange(attributes.shape[1]))
    print(len(result))
    print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    result = partition(labels, np.arange(1))
    print(len(result))
    print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))

    start_time = time.time()
    data = pd.read_csv("mushroom_little.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    result = partition2(attributes, np.arange(attributes.shape[1]))
    print(len(result))
    print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    result = partition2(labels, np.arange(1))
    print(len(result))
    print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))
    return None


# unused
def element_is_include(element, set):
    """
    judge if the element is included by(belong to) the mylist2
    :param element: a object's serial number
    :param set: list, a set of objects' serial number
    :return: True/False
    """
    flag = True
    try:
        set.index(element)
    except ValueError:
        flag = False
    return flag


def set_is_include(set1, set2):
    """
    judge if the mylist1 is included by(belong to) the mylist2
    :param set1: a set of objects' serial number
    :param set2: list, a set of objects' serial number
    :return: True/False
    """
    for element in set2:
        flag = True
        for x1 in set1:
            try:
                element.index(x1)
            except ValueError:
                flag = False
                break
        if flag:
            return True
        else:
            continue
    return False


def feature_subset_low_approximations_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset lower approximations of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' serial number
    :param feature_subset: features' index
    :return: list, lower_approximations is composed by a set of objects' serial number
    """
    lower_approximations = []
    partition_1 = partition(universe, feature_subset)
    for x in partition_1:
        if set_is_include(x, [sample_subset]):
            lower_approximations.extend(x)
    lower_approximations.sort()
    return lower_approximations


def features_lower_approximations_of_universe(universe, attributes, labels):
    """
    get the features lower approximations of U/R
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param labels: labels' index
    :return: list, lower_approximations is composed by a set of objects' serial number
    """
    lower_approximations = []
    partition_1 = partition(universe, attributes)
    partition_2 = partition(universe, labels)
    for x in partition_1:
        if set_is_include(x, partition_2):
            lower_approximations.extend(x)
    lower_approximations.sort()
    return lower_approximations


def features_lower_approximations_of_universe_test():
    """
    test features_lower_approximations_of_universe
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    result = features_lower_approximations_of_universe(np.array(data), np.arange(4), np.arange(4, 5))
    print("approximation result:", result)
    print("\t\t\t\t\t", "[0, 1, 3, 4, 6, 7]")
    result = partition(np.array(data), np.arange(4))
    print("partition by attributes:", result)
    result = partition(np.array(data), np.arange(4, 5))
    print("partition by label:", result)
    return None


def feature_subset_low_approximations_of_sample_subset_test():
    """
    test feature_subset_low_approximations_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    print(data.shape)
    del data[4]
    print(data.shape)
    # result = feature_subset_low_approximations_of_sample_subset(np.array(data), [i for i in range(8)], np.arange(4))
    result = feature_subset_low_approximations_of_sample_subset(np.array(data), [1, 2, 3], [i for i in range(4)])
    print("approximation result:", result)
    result = partition(np.array(data), np.arange(4))
    print("partition by attributes:", result)
    return None


def is_contain(x, y):
    """
    judge that the intersection of x and y is not an empty set
    :param x: the set of objects' serial number
    :param y: the set of objects' serial number
    :return: True/False
    """
    intersection = [i for i in x if i in y]
    if len(intersection) > 0:
        return True
    else:
        return False


def features_upper_approximations_of_universe(universe, attributes, labels):
    """
    get the features upper approximations of U/R
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param labels: labels' index
    :return: list, upper_approximations is composed by a set of objects' serial number
    """
    upper_approximations = list(np.arange(len(universe)))
    upper_approximations.sort()
    return upper_approximations


def features_upper_approximations_of_universe_test():
    """
    test upper_features_lower_approximations_of_universe
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    result = features_upper_approximations_of_universe(np.array(data), np.arange(4), np.arange(4, 5))
    print("result:", result)
    print(len(result))
    return None


def feature_subset_upper_approximations_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset upper approximations of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' serial number
    :param feature_subset: features' index
    :return: list, upper_approximations is composed by a set of objects' serial number
    """
    upper_approximations = []
    partition_1 = partition(universe, feature_subset)
    for x in partition_1:
        if is_contain(x, sample_subset):
            upper_approximations.extend(x)
    upper_approximations.sort()
    return upper_approximations


def feature_subset_positive_region_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset positive_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' serial number
    :param feature_subset: features' index
    :return: list, positive_region is composed by a set of objects' serial number
    """
    positive_region = feature_subset_low_approximations_of_sample_subset(universe, sample_subset, feature_subset)
    return positive_region


def feature_subset_positive_region_of_sample_subset_test():
    """
    test feature_subset_positive_region_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    del data[4]
    result = feature_subset_low_approximations_of_sample_subset(np.array(data), [0, 1, 4, 6, 7], [0, 3])
    print("result:", result)
    print(len(result))
    return None


def feature_subset_boundary_region_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset boundary_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' serial number
    :param feature_subset: features' index
    :return: list, boundary_region is composed by a set of objects' serial number
    """
    upper_approximations = feature_subset_upper_approximations_of_sample_subset(universe, sample_subset, feature_subset)
    lower_approximations = feature_subset_low_approximations_of_sample_subset(universe, sample_subset, feature_subset)
    boundary_region = [i for i in upper_approximations if i not in lower_approximations]
    return boundary_region


def feature_subset_boundary_region_of_sample_subset_test():
    """
    test feature_subset_boundary_region_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    del data[4]
    result = feature_subset_boundary_region_of_sample_subset(np.array(data), [0, 1, 4, 6, 7], [0, 3])
    print("result:", result)
    print(len(result))
    return None


def feature_subset_negative_region_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset negative_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' serial number
    :param feature_subset: features' index
    :return: list, negative_region is composed by a set of objects' serial number
    """
    upper_approximations = feature_subset_upper_approximations_of_sample_subset(universe, sample_subset, feature_subset)
    return [i for i in np.arange(len(universe)) if i not in upper_approximations]


def feature_subset_negative_region_of_sample_subset_test():
    """
    test feature_subset_negative_region_of_sample_subset
    :return:
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    del data[4]
    result = feature_subset_negative_region_of_sample_subset(np.array(data), [0, 1, 4, 6, 7], [0, 3])
    print("result:", result)
    print(len(result))
    return None


def feature_subset_upper_approximations_of_sample_subset_test():
    """
    test feature_subset_upper_approximations_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    print(data.shape)
    del data[4]
    print(data.shape)
    # result = feature_subset_low_approximations_of_sample_subset(np.array(data), [i for i in range(8)], np.arange(4))
    result = feature_subset_upper_approximations_of_sample_subset(np.array(data), [1, 2, 3, 4], [i for i in range(4)])
    print("result:", result)
    return None


def dependency(universe, features_1, features_2):
    """
    to calculate the dependency between attributes
    :param universe: the universe of objects(feature vector/sample/instance)
    :param features_1: list, a set of features' serial number
    :param features_2: list, a set of features' serial number
    :return: float number(0-->1, 1 represent that features_1 completely depends on features_2,
              All values of attributes from D are uniquely determined by the values of attributes from C.),
              the dependency of features_1 to features_2, POS_features_1(features_2)
    """
    partition_2 = partition(universe, features_2)
    positive_region_size = 0
    for y in partition_2:
        positive_region_size += len(feature_subset_positive_region_of_sample_subset(universe, y, features_1))
        # print(feature_subset_positive_region_of_sample_subset(universe, y, features_1))
    dependency_degree = positive_region_size/len(universe)
    return dependency_degree


def dependency_test():
    """
    test dependency
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    # result = feature_subset_low_approximations_of_sample_subset(np.array(data), [i for i in range(8)], np.arange(4))
    result = dependency(np.array(data), [0, 3], [4])
    print("dependency:", result)
    return None


def mean_positive_region(universe, sample_subset, feature_subset):
    """
    [important] Only applicable for continuous values!!!!!!
    Not applicable to discrete values.
    don't consider the condition that the positive region is empty

    get the feature_subset lower approximations of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of features' serial number
    :param feature_subset: features' index
    :return: list(object), the mean of all object attributes values in positive region
    """
    positive_region = feature_subset_positive_region_of_sample_subset(universe, sample_subset, feature_subset)
    total = []
    mean = []
    for i in range(len(feature_subset)):
        total.append(0)
        mean.append(0)
    for x in positive_region:
        for i in range(len(feature_subset)):
            test1 = universe[x]
            test2 = feature_subset[i]
            test3 = test1[test2]
            total[i] += (universe[x])[feature_subset[i]]
    for i in range(len(feature_subset)):
        mean[i] = total[i]/len(positive_region)
    return mean


def mean_positive_region_test():
    """
    test mean_positive_region
    :return: None
    """
    data = pd.read_csv("real_value_data.csv", header=None)
    result = mean_positive_region(np.array(data), [0, 1, 2, 3, 4], [i for i in range(5)])
    print("result:", result)
    return None


# for continuous data
def proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(universe, features_1, features_2):
    """
    proximity_of_objects_in_boundary_region_from_mean_positive_region
    don't consider the condition that the positive region is empty
    :param universe: the universe of objects(feature vector/sample/instance)
    :param features_1: list, a set of features' serial number
    :param features_2: list, a set of features' serial number
    :return: float
    """
    partition_2 = partition(universe, features_2)
    boundary = []
    positive = []
    for subset in partition_2:
        boundary.extend(feature_subset_boundary_region_of_sample_subset(universe, subset, features_1))
        positive.extend(feature_subset_positive_region_of_sample_subset(universe, subset, features_1))
    if len(positive) == 0:
        return 0
    mean = mean_positive_region(universe, positive, features_1)
    proximity_of_object_in_boundary_from_mean = 0
    for y in boundary:
        proximity_of_object_in_boundary_from_mean += distance(mean, universe[y]) if len(positive) > 0 else 0
    return 1/proximity_of_object_in_boundary_from_mean if len(boundary) > 0 else 1


def proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance_test():
    """
    test proximity_of_objects_in_boundary_region_from_mean_positive_region
    :return: None
    """
    data = np.array(pd.read_csv("approximation_data.csv", header=None))
    proximity = proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(data, [0, 1, 2], [3, 4])
    print("proximity:", proximity)
    return None


# for continuous data
def distance(x, y):
    """
    calculate the distance of two objects
    :param x: feature vector, sample, instance
    :param y: feature vector, sample, instance
    :return: float, distance
    """
    total = 0
    for i in range(len(x)):
        total += (x[i] - y[i])**2
    return math.sqrt(total)


def impurity_rate(subset_a, subset_b):
    """
    the noise portion of subset_a to subset_b
    impurity rate of subset_a with respect to subset_b, the noise information
    :param subset_a: a set of objects' serial number
    :param subset_b: a set of objects' serial number
    :return: float, impurity rate of subset_a with respect to subset_b
    """
    difference = [i for i in subset_a if i not in subset_b]
    return len(difference)/len(subset_a)


def related_information_of_subset_b(subset_a, subset_b):
    """
    related_information in subset_b from subset_a, the useful information
    generated by impurity_rate
    :param subset_a: a set of objects' serial number
    :param subset_b: a set of objects' serial number
    :return: float, related_information in subset_b
    """
    impurity = impurity_rate(subset_a, subset_b)
    if impurity > 0.5:
        return 0
    else:
        return 1 - impurity


def related_information_of_subset_b_test():
    """
    test related_information_of_subset_b_test and impurity_rate
    :return: None
    """
    print(impurity_rate([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 5, 25, 7, 96, 2]))
    print(related_information_of_subset_b([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 5, 25, 7, 96, 2]))
    print(impurity_rate([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 25, 7, 96, 2, 59, 85, 75]))
    print(related_information_of_subset_b([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 25, 7, 96, 2, 59, 85, 75]))
    return


def proximity_of_boundary_region_to_positive_region_based_portion(universe, sample_subset, feature_subset):
    """
    a noise measure function
    to describe the information contain by the boundary of partition(universe, sample_subset)
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' serial number
    :param feature_subset: list, a set of features' serial number
    :return: float, the proximity
    """
    partition_1 = partition(universe, feature_subset)
    total = 0
    for elementary_set in partition_1:
        related_information = related_information_of_subset_b(elementary_set, sample_subset)
        if related_information != 1:
            total += related_information
    return total/(len(partition_1))


def proximity_of_boundary_region_to_positive_region_based_portion_test():
    """
    test proximity_of_boundary_region_to_positive_region_based_portion
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    proximity = proximity_of_boundary_region_to_positive_region_based_portion(np.array(data), [0, 1, 2], [i for i in range(2)])
    print(proximity)  # 1/2 / 5
    proximity = proximity_of_boundary_region_to_positive_region_based_portion(np.array(data), [0, 1, 2], [i for i in range(3)])
    print(proximity)  # 1/2 / 7 = 1/14
    return None


def noisy_dependency_of_feature_subset_d_on_feature_subset_c(universe, feature_subset_c, feature_subset_d):
    """
    :param universe: the universe of objects(feature vector/sample/instance)
    :param feature_subset_c: list, a set of features' serial number
    :param feature_subset_d: list, a set of features' serial number
    :return: noisy dependency of feature subset a on feature subset b
    """
    partition_d = partition(universe, feature_subset_d)
    total_dependency = 0
    for p in partition_d:
        the_dependency = proximity_of_boundary_region_to_positive_region_based_portion(universe, p, feature_subset_c)
        total_dependency += the_dependency
    return total_dependency


def noisy_dependency_of_feature_subset_a_on_feature_subset_b_test():
    """
    test noisy_dependency_of_feature_subset_a_on_feature_subset_b
    :return: None
    """
    data = pd.read_csv("example_of_noisy_data.csv", header=None)
    the_dependency = noisy_dependency_of_feature_subset_d_on_feature_subset_c(np.array(data), [0], [1])
    print(the_dependency)  # 0.75
    return None


# notice the data's type
def proximity_combine_noisy_dependency(universe, feature_subset_c, feature_subset_d):
    """
    :param universe: the universe of objects(feature vector/sample/instance)
    :param feature_subset_c: list, a set of features' serial number
    :param feature_subset_d: list, a set of features' serial number
    :return: float, the combined measure value
    """
    proximity = proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(
        universe, feature_subset_c, feature_subset_d)
    noisy_dependency = noisy_dependency_of_feature_subset_d_on_feature_subset_c(
        universe, feature_subset_c, feature_subset_d)
    return proximity + noisy_dependency


def noise_resistant_evaluation_measure(universe, feature_subset_c, feature_subset_d):
    """
    :param universe: the universe of objects(feature vector/sample/instance)
    :param feature_subset_c: list, a set of features' serial number
    :param feature_subset_d: list, a set of features' serial number
    :return: float, the noise resistant evaluation measure value
    """
    combined_value = proximity_combine_noisy_dependency(universe, feature_subset_c, feature_subset_d)
    dependency_value = dependency(universe, feature_subset_c, feature_subset_d)
    return (combined_value + dependency_value)/2


# confirm the function of the above function
if __name__ == '__main__':
    # check_partition_result()
    # print("lower approximations:\n")
    # features_lower_approximations_of_universe_test()
    # feature_subset_low_approximations_of_sample_subset_test()
    # print("\nupper approximations:\n")
    # features_upper_approximations_of_universe_test()
    # feature_subset_upper_approximations_of_sample_subset_test()
    # feature_subset_positive_region_of_sample_subset_test()
    # feature_subset_boundary_region_of_sample_subset_test()
    # feature_subset_negative_region_of_sample_subset_test()
    # dependency_test()
    # mean_positive_region_test()
    # proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance_test()
    # related_information_of_subset_b_test()
    # proximity_of_boundary_region_to_positive_region_based_portion_test()
    noisy_dependency_of_feature_subset_a_on_feature_subset_b_test()
    pass
