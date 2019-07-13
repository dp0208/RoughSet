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
    data = pd.read_csv("approximation_data.csv", header=None)
    result = quick_reduct(np.array(data), [0, 1, 2, 3], [4])
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
    while destination_dependency != candidate_dependency:
        noise_resistant_increase = 0
        test_features = copy.deepcopy(candidate_features)
        for i in range(len(contitional_features)):
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
    print("reduct:", result)
    return None


def get_new_feature():
    """
    to transfer a feature to the algorithm through yield
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    conditional_features = [0, 1, 2, 3]
    for i in range(len(conditional_features)):
        yield conditional_features[i]
    return None


def significance_of_feature_subset(universe, candidate_features, decision_features, features):
    """
    significance of features belongs to candidate_features + features
    :param universe:
    :param candidate_features:
    :param decision_features:
    :param features:
    :return: the significance
    """
    test_features = copy.deepcopy(candidate_features)
    for feature in features:
        test_features.append(feature)
    test_features_dependency = dependency(universe, test_features, decision_features)
    candidate_features_dependency = dependency(universe, candidate_features, decision_features)
    return (test_features_dependency - candidate_features_dependency) / test_features_dependency


def non_significant(universe, raw_candidate_features, decision_features, feature):
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
    candidate_features.append(feature)
    test_candidate_features = copy.deepcopy(candidate_features)
    while len(test_candidate_features) != 0:
        g = test_candidate_features[random.randint(0, len(test_candidate_features) - 1)]
        if significance_of_feature_subset(universe, candidate_features, decision_features, g) == 0:
            result.append(g)
            candidate_features.remove(g)
        test_candidate_features.remove(g)
    return result


def online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis():
    universe = np.array(pd.read_csv("approximation_data.csv", header=None))
    candidate_features = []
    decision_features = [4]
    for feature in get_new_feature():
        test_features = copy.deepcopy(candidate_features)
        test_features.append(feature)
        if dependency(universe, candidate_features, decision_features) != 1:
            if (dependency(universe, test_features, decision_features) -
                dependency(universe, candidate_features, decision_features)) > 0:
                candidate_features.append(feature)
        else:
            non_significant_subset = non_significant(universe, candidate_features, decision_features, feature)
            if len(non_significant_subset) > 1:
                candidate_features.append(feature)
                candidate_features = [i for i in candidate_features if i not in non_significant_subset]
            if len(non_significant_subset) == 1:
                noise_resistant_measure_a = noise_resistant_evaluation_measure(universe, feature, decision_features)
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
    return candidate_features


if __name__ == '__main__':
    # quick_reduct_test()
    # noise_resistant_assisted_quick_reduct_test()
    re = online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis()
    print(re)
    pass
