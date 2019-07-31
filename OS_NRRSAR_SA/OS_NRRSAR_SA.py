"""
Online streaming feature selection using rough sets
author: S.Eskandari M.M.Javidi
https://doi.org/10.1016/j.ijar.2015.11.006
"""

from rough_set import *
import copy
import random


def quick_reduct(universe, raw_conditional_features, decision_features):
    """
    to get a reduct by Sequential Forward Selection method using dependency based on positive region
    :param universe: the universe of objects(feature vector/sample/instance)
    :param raw_conditional_features: list, a set of features' serial number
    :param decision_features: list, a set of features' serial number
    :return: candidate_features
    """
    destination_dependency = dependency(universe, raw_conditional_features, decision_features)
    contitional_features = copy.deepcopy(raw_conditional_features)
    candidate_features = []
    candidate_dependency = 0
    current_dependency = 0
    # the dependency is monotonic with positive,
    # because more conditional features can divide the universe into more partitions
    while destination_dependency != candidate_dependency:
        for i in range(len(contitional_features)):
            candidate_features.append(contitional_features[i])
            result = dependency(universe, candidate_features, [decision_features])
            candidate_features.remove(contitional_features[i])
            if result > current_dependency:
                current_dependency = result
                index = i
        feature = contitional_features[index]
        candidate_features.append(feature)
        contitional_features.remove(feature)
        candidate_dependency = current_dependency
    return candidate_features


def quick_reduct_test():
    """
    test quick reduct
    :return: None
    """
    # data = pd.read_csv("approximation_data.csv", header=None)
    # result = quick_reduct(np.array(data), [0, 1, 2, 3], [4])
    # reduct: [0, 2]
    data = pd.read_csv("arcene_train.csv", header=None)
    result = quick_reduct(np.array(data), [i for i in range(0, 10000)], [10000])
    # reduct: [3355]
    print("reduct:", result)
    return None


def noise_resistant_assisted_quick_reduct(universe, raw_conditional_features, decision_features):
    """
    to get a reduct by Sequential Forward Selection method using dependency based on positive region
    :param universe: the universe of objects(feature vector/sample/instance)
    :param raw_conditional_features: list, a set of features' serial number
    :param decision_features: list, a set of features' serial number
    :return: candidate_features
    """
    destination_dependency = dependency(universe, raw_conditional_features, decision_features)
    contitional_features = copy.deepcopy(raw_conditional_features)
    candidate_features = []
    candidate_dependency = 0
    # the dependency is monotonic with positive,
    # because more conditional features can divide the universe into more partitions
    count = 0
    while destination_dependency != candidate_dependency:
        count += 1
        print(count)
        noise_resistant_increase = 0
        test_features = copy.deepcopy(candidate_features)
        count1 = 0
        for i in range(len(contitional_features)):
            count1 += 1
            if count1%1000 == 0:
                print(count1, end="-")
            test_features.append(contitional_features[i])
            result = noise_resistant_evaluation_measure(universe, test_features, [decision_features]) - \
                     noise_resistant_evaluation_measure(universe, candidate_features, [decision_features])
            test_features.remove(contitional_features[i])
            if result > noise_resistant_increase:
                noise_resistant_increase = result
                index = i
        feature = contitional_features[index]
        candidate_features.append(feature)
        contitional_features.remove(feature)
        candidate_dependency = dependency(universe, candidate_features, [decision_features])
    return candidate_features


def noise_resistant_assisted_quick_reduct_test():
    """
    test noise_resistant_assisted_quick_reduct
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    result = noise_resistant_assisted_quick_reduct(np.array(data), [0, 1, 2, 3], [4])
    # reduct: [2, 0]
    # data = pd.read_csv("arcene_train.csv", header=None)
    # result = noise_resistant_assisted_quick_reduct(np.array(data), [i for i in range(0, 10000)], [10000])\
    # reduct: [3355]
    print("reduct:", result)
    return None


def significance_of_feature_subset(universe, candidate_features, decision_features, features):
    """
    significance of features belongs to candidate_features + features
    :param universe: the universe of objects(feature vector/sample/instance)
    :param candidate_features: list, a set of features' serial number
    :param decision_features: list, a set of features' serial number
    :param features:
    :return: the significance
    """
    test_features = copy.deepcopy(candidate_features)
    for feature in features:
        test_features.append(feature)
    test_features_dependency = dependency(universe, test_features, decision_features)
    candidate_features_dependency = dependency(universe, candidate_features, decision_features)
    return (test_features_dependency - candidate_features_dependency) / test_features_dependency


def subsets_recursive(items):
    # the power set of the empty set has one element, the empty set
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


def heuristic_non_significant(universe, raw_candidate_features, decision_features, feature):
    """
    non_significance in candidate_features + feature due to the feature
    :param universe:
    :param raw_candidate_features:
    :param decision_features:
    :param feature:
    :return:
    """
    result = []
    max_size = 0
    candidate_features = copy.deepcopy(raw_candidate_features)
    test_candidate_features = copy.deepcopy(candidate_features)
    candidate_features.append(feature)
    subsets = subsets_recursive(test_candidate_features)
    subsets.remove([])
    for item in subsets:
        if significance_of_feature_subset(universe, [i for i in candidate_features if i not in item],
                                          decision_features, item) == 0:
            if len(item) > max_size:
                max_size = len(item)
                result = item
    return result


def sequential_backward_elimination_non_significant(universe, raw_candidate_features, decision_features, feature):
    """
    non_significance in candidate_features + feature due to the feature
    :param universe:
    :param raw_candidate_features:
    :param decision_features:
    :param feature:
    :return:
    """
    result = []
    candidate_features = copy.deepcopy(raw_candidate_features)
    test_candidate_features = copy.deepcopy(candidate_features)
    candidate_features.append(feature)
    while len(test_candidate_features) != 0:
        g = test_candidate_features[random.randint(0, len(test_candidate_features) - 1)]
        if significance_of_feature_subset(universe, candidate_features, decision_features, [g]) == 0:
            result.append(g)
            candidate_features.remove(g)
        test_candidate_features.remove(g)
    return result


def online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis(
        universe, decision_features, get_new_feature):
    start = time.time()
    temp = start
    candidate_features = []
    count = 0
    for feature in get_new_feature():
        count += 1
        if count%1000 == 0:
            print(count)
            print("{}  ".format(time.time() - temp), "{}".format(time.time() - start))
            temp = time.time()
        test_features = copy.deepcopy(candidate_features)
        test_features.append(feature)
        noise_resistant_evaluation_measure(universe, [feature], decision_features)
        if dependency(universe, candidate_features, decision_features) != 1:
            if (dependency(universe, test_features, decision_features) -
                dependency(universe, candidate_features, decision_features)) > 0 or \
                    noise_resistant_evaluation_measure(universe, [feature], decision_features) > 0:
                candidate_features.append(feature)
        else:
            non_significant_subset = heuristic_non_significant(universe, candidate_features, decision_features, feature)
            if len(non_significant_subset) > 1:
                candidate_features.append(feature)
                candidate_features = [i for i in candidate_features if i not in non_significant_subset]
            if len(non_significant_subset) == 1:
                noise_resistant_measure_a = noise_resistant_evaluation_measure(universe, [feature], decision_features)
                noise_resistant_measure_b = noise_resistant_evaluation_measure(universe, non_significant_subset,
                                                                               decision_features)
                if noise_resistant_measure_a > noise_resistant_measure_b:
                    candidate_features.remove(non_significant_subset[0])
                    candidate_features.append(feature)
                elif noise_resistant_measure_a < noise_resistant_measure_b:
                    candidate_features.append(feature)
                else:
                    flag = random.randint(0, 1)
                    if flag == 1:
                        candidate_features.remove(non_significant_subset[0])
                        candidate_features.append(feature)
                    else:
                        pass
    print('The time used: {} seconds'.format(time.time() - start))
    return candidate_features


def get_new_approximation_data_feature():
    """
    to transfer a feature to the algorithm through yield
    :return: None
    """
    conditional_features = [0, 1, 2, 3]
    for i in range(len(conditional_features)):
        yield conditional_features[i]
    return None


def get_new_arcene_feature():
    """
    to transfer a feature to the algorithm through yield
    :return: None
    """
    conditional_features = [i for i in range(0, 10000)]
    for i in range(len(conditional_features)):
        yield conditional_features[i]
    return None


def get_new_example_feature():
    """
    to transfer a feature to the algorithm through yield
    :return: None
    """
    conditional_features = [i for i in range(0, 5)]
    for i in range(len(conditional_features)):
        yield conditional_features[i]
    return None


def online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis_test():
    # universe = np.array(pd.read_csv("approximation_data.csv", header=None))
    # decision_features = [4]
    # result =
    # online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis(
    #     universe, decision_features, get_new_approximation_data_feature)
    # print("approximation_data reduct:", result)
    # print('The time used: {} seconds'.format(time.time() - start))
    # approximation_data reduct: [0, 1, 2]
    # The time used: 0.006995677947998047 seconds

    # universe = np.array(pd.read_csv("arcene_train.csv", header=None))
    # decision_features = [10000]
    # result = online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis(
    #     universe, decision_features, get_new_arcene_feature)
    # print("arcene_train reduct:", result)
    # print('The time used: {} seconds'.format(time.time() - start))
    # arcene_train reduct: [9997, 9998, 9999]
    # The time used: 374.79046154022217 seconds

    universe = np.array(pd.read_csv("example_data.csv", header=None))
    decision_features = [5]
    result = online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis(
        universe, decision_features, get_new_example_feature)
    print("example_data reduct:", result)
    pass


if __name__ == '__main__':
    # quick_reduct_test()
    # noise_resistant_assisted_quick_reduct_test()

    online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis_test()
    pass
