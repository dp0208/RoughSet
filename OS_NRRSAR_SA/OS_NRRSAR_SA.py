"""
Online streaming feature selection using rough sets
author: S.Eskandari M.M.Javidi
https://doi.org/10.1016/j.ijar.2015.11.006
"""

from rough_set import *
import math
import copy
import random


class QuickReduction:
    def __init__(self):
        return

    @staticmethod
    def run(universe, raw_conditional_features, decision_features):
        """
        to get a reduction by Sequential Forward Selection method using dependency based on positive region
        :param universe: the universe of objects(feature vector/sample/instance)
        :param raw_conditional_features: list, a set of features' serial number
        :param decision_features: list, a set of features' serial number
        :return: candidate_features
        """
        destination_dependency = dependency(universe, raw_conditional_features, decision_features)
        conditional_features = copy.deepcopy(raw_conditional_features)
        candidate_features = []
        candidate_dependency = 0
        current_dependency = 0
        # the dependency is monotonic with positive,
        # because more conditional features can divide the universe into more partitions
        while destination_dependency != candidate_dependency:
            index = 0
            for i in range(len(conditional_features)):
                candidate_features.append(conditional_features[i])
                result = dependency(universe, candidate_features, [decision_features])
                candidate_features.remove(conditional_features[i])
                if result > current_dependency:
                    current_dependency = result
                    index = i
            feature = conditional_features[index]
            candidate_features.append(feature)
            conditional_features.remove(feature)
            candidate_dependency = current_dependency
        return candidate_features


def quick_reduction_test():
    """
    test quick reduction
    :return: None
    """
    algorithm = QuickReduction()

    # data = pd.read_csv("approximation_data.csv", header=None)
    # result = algorithm.run(np.array(data), [0, 1, 2, 3], [4])
    # reduction: [0, 2]

    data = pd.read_csv("arcene_train.csv", header=None)
    result = algorithm.run(np.array(data), [i for i in range(0, 10000)], [10000])
    print("reduction:", result)
    # reduction: [3355]
    return None


class NoiseResistantDependencyMeasure:
    def __init__(self):
        return

    @staticmethod
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
                total[i] += (universe[x])[feature_subset[i]]
        for i in range(len(feature_subset)):
            mean[i] = total[i] / len(positive_region)
        return mean

    # for continuous data
    @staticmethod
    def proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(
            universe, features_1, features_2, distance):
        """
        proximity_of_objects_in_boundary_region_from_mean_positive_region
        don't consider the condition that the positive region is empty
        :param universe: the universe of objects(feature vector/sample/instance)
        :param features_1: list, a set of features' serial number
        :param features_2: list, a set of features' serial number
        :param distance: the method to calcalate the distance of objects
        :return: float
        """
        partition_2 = partition(universe, features_2)
        boundary = []
        positive = []
        for subset in partition_2:
            boundary.extend(feature_subset_boundary_region_of_sample_subset(universe, subset, features_1))
            boundary = list(set(boundary))
            positive.extend(feature_subset_positive_region_of_sample_subset(universe, subset, features_1))
        if len(boundary) == 0:
            return 1
        if len(positive) == 0:
            return 1 / len(boundary)
        mean = NoiseResistantDependencyMeasure.mean_positive_region(universe, positive, features_1)
        proximity_of_object_in_boundary_from_mean = 0
        for y in boundary:
            proximity_of_object_in_boundary_from_mean += distance(mean, universe[y], features_1)
        return 1 / proximity_of_object_in_boundary_from_mean

    @staticmethod
    def variation_of_euclidean_distance(x, y, features):
        """
        calculate the distance of two objects
        :param x: feature vector, sample, instance
        :param y: feature vector, sample, instance
        :param features: list, a set of features' serial number
        :return: float, distance
        """
        total = 0
        for i in range(len(features)):
            if x[i] == y[features[i]]:
                continue
            total += 1
        return math.sqrt(total)

    @staticmethod
    def impurity_rate(subset_a, subset_b):
        """
        the noise portion of subset_a to subset_b
        impurity rate of subset_a with respect to subset_b, the noise information
        :param subset_a: a set of objects' serial number
        :param subset_b: a set of objects' serial number
        :return: float, impurity rate of subset_a with respect to subset_b
        """
        difference = [i for i in subset_a if i not in subset_b]
        return len(difference) / len(subset_a)

    @staticmethod
    def related_information_of_subset_b(subset_a, subset_b):
        """
        related_information in subset_b from subset_a, the useful information
        generated by impurity_rate
        :param subset_a: a set of objects' serial number
        :param subset_b: a set of objects' serial number
        :return: float, related_information in subset_b
        """
        impurity = NoiseResistantDependencyMeasure.impurity_rate(subset_a, subset_b)
        if impurity > 0.5:
            return 0
        else:
            return 1 - impurity

    @staticmethod
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
            related_information = NoiseResistantDependencyMeasure.related_information_of_subset_b(
                elementary_set, sample_subset)
            if related_information != 1:
                total += related_information
        return total / (len(partition_1))

    @staticmethod
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
            the_dependency = NoiseResistantDependencyMeasure. \
                proximity_of_boundary_region_to_positive_region_based_portion(universe, p, feature_subset_c)
            total_dependency += the_dependency
        return total_dependency

    # notice the data's type
    @staticmethod
    def proximity_combine_noisy_dependency(universe, feature_subset_c, feature_subset_d):
        """
        :param universe: the universe of objects(feature vector/sample/instance)
        :param feature_subset_c: list, a set of features' serial number
        :param feature_subset_d: list, a set of features' serial number
        :return: float, the combined measure value
        """
        proximity = \
            NoiseResistantDependencyMeasure. \
                proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(
                    universe, feature_subset_c, feature_subset_d,
                    distance=NoiseResistantDependencyMeasure.variation_of_euclidean_distance)
        noisy_dependency = NoiseResistantDependencyMeasure.noisy_dependency_of_feature_subset_d_on_feature_subset_c(
            universe, feature_subset_c, feature_subset_d)
        return proximity + noisy_dependency

    @staticmethod
    def noise_resistant_evaluation_measure(universe, feature_subset_c, feature_subset_d):
        """
        :param universe: the universe of objects(feature vector/sample/instance)
        :param feature_subset_c: list, a set of features' serial number
        :param feature_subset_d: list, a set of features' serial number
        :return: float, the noise resistant evaluation measure value
        """
        combined_value = NoiseResistantDependencyMeasure.proximity_combine_noisy_dependency(
            universe, feature_subset_c, feature_subset_d)
        dependency_value = dependency(universe, feature_subset_c, feature_subset_d)
        return (combined_value + dependency_value) / 2


class NoiseResistantDependencyMeasureTest:
    @staticmethod
    def mean_positive_region_test():
        """
        test mean_positive_region
        :return: None
        """
        data = pd.read_csv("real_value_data.csv", header=None)
        result = NoiseResistantDependencyMeasure.mean_positive_region(
            np.array(data), [0, 1, 2, 3, 4], [i for i in range(5)])
        print("result:", result)
        return None

    @staticmethod
    def proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance_test():
        """
        test proximity_of_objects_in_boundary_region_from_mean_positive_region
        :return: None
        """
        data = np.array(pd.read_csv("approximation_data.csv", header=None))
        proximity = NoiseResistantDependencyMeasure. \
            proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(
                data, [0, 1, 2], [3, 4], distance=NoiseResistantDependencyMeasure.variation_of_euclidean_distance)
        print("proximity:", proximity)
        return None

    @staticmethod
    def related_information_of_subset_b_test():
        """
        test related_information_of_subset_b_test and impurity_rate
        :return: None
        """
        print(NoiseResistantDependencyMeasure.impurity_rate([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 5, 25, 7, 96, 2]))
        print(NoiseResistantDependencyMeasure.related_information_of_subset_b(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 5, 25, 7, 96, 2]))
        print(NoiseResistantDependencyMeasure.impurity_rate(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 25, 7, 96, 2, 59, 85, 75]))
        print(NoiseResistantDependencyMeasure.related_information_of_subset_b(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 25, 7, 96, 2, 59, 85, 75]))
        return

    @staticmethod
    def proximity_of_boundary_region_to_positive_region_based_portion_test():
        """
        test proximity_of_boundary_region_to_positive_region_based_portion
        :return: None
        """
        data = pd.read_csv("approximation_data.csv", header=None)
        proximity = NoiseResistantDependencyMeasure.proximity_of_boundary_region_to_positive_region_based_portion(
            np.array(data), [0, 1, 2], [i for i in range(2)])
        print(proximity)  # 1/2 / 5
        proximity = NoiseResistantDependencyMeasure.proximity_of_boundary_region_to_positive_region_based_portion(
            np.array(data), [0, 1, 2], [i for i in range(3)])
        print(proximity)  # 1/2 / 7 = 1/14
        return None

    @staticmethod
    def noisy_dependency_of_feature_subset_a_on_feature_subset_b_test():
        """
        test noisy_dependency_of_feature_subset_a_on_feature_subset_b
        :return: None
        """
        data = pd.read_csv("example_of_noisy_data.csv", header=None)
        the_dependency = NoiseResistantDependencyMeasure.noisy_dependency_of_feature_subset_d_on_feature_subset_c(
            np.array(data), [0], [1])
        print(the_dependency)  # 0.75
        return None


class NoiseResistantAssistedQuickReduct:
    def __init__(self):
        return

    @staticmethod
    def run(universe, raw_conditional_features, decision_features):
        """
        to get a reduction by Sequential Forward Selection method using dependency based on positive region
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
            index = 0
            for i in range(len(contitional_features)):
                count1 += 1
                if count1 % 1000 == 0:
                    print(count1, end="-")
                test_features.append(contitional_features[i])
                result = NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                    universe, test_features, [decision_features]) - \
                         NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                             universe, candidate_features, [decision_features])
                test_features.remove(contitional_features[i])
                if result > noise_resistant_increase:
                    noise_resistant_increase = result
                    index = i
            feature = contitional_features[index]
            candidate_features.append(feature)
            contitional_features.remove(feature)
            candidate_dependency = dependency(universe, candidate_features, [decision_features])
        return candidate_features


def noise_resistant_assisted_quick_reduction_test():
    """
    test noise_resistant_assisted_quick_reduct
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    algorithm = NoiseResistantAssistedQuickReduct()
    result = algorithm.run(np.array(data), [0, 1, 2, 3], [4])
    # reduction: [2, 0]
    # data = pd.read_csv("arcene_train.csv", header=None)
    # result = algorithm.run(np.array(data), [i for i in range(0, 10000)], [10000])\
    # reduction: [3355]
    print("reduction:", result)
    return None


class OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis:
    def __init__(self, universe, conditional_features, decision_features):
        self.universe = universe
        self.conditional_features = conditional_features
        self.decision_features = decision_features
        return

    @staticmethod
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

    @staticmethod
    def subsets_recursive(items):
        # the power set of the empty set has one element, the empty set
        result = [[]]
        for x in items:
            result.extend([subset + [x] for subset in result])
        return result

    @staticmethod
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
        subsets = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis.subsets_recursive(
            test_candidate_features)
        subsets.remove([])
        for item in subsets:
            if OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis. \
                    significance_of_feature_subset(universe, [i for i in candidate_features if i not in item],
                                                   decision_features, item) == 0:
                if len(item) > max_size:
                    max_size = len(item)
                    result = item
        return result

    @staticmethod
    def sequential_backward_elimination_non_significant(universe, raw_candidate_features, decision_features,
                                                        feature):
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
            if OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis. \
                    significance_of_feature_subset(universe, candidate_features, decision_features, [g]) == 0:
                result.append(g)
                candidate_features.remove(g)
            test_candidate_features.remove(g)
        return result

    def get_new_feature(self):
        """
        to transfer a feature to the algorithm through yield
        :return: None
        """
        conditional_features = self.conditional_features
        for i in range(len(conditional_features)):
            yield conditional_features[i]
        return None

    def run(self, non_significant="SEB3"):
        """
        :param non_significant: the non-significant implementation method, max subset, SEB1, SEB3, default is SEB3
        :return: the reduction
        """
        start = time.time()
        temp = start
        candidate_features = []
        count = 0
        for feature in self.get_new_feature():
            count += 1
            if count % 1000 == 0:
                print(count)
                print("{}  ".format(time.time() - temp), "{}".format(time.time() - start))
                temp = time.time()
            test_features = copy.deepcopy(candidate_features)
            test_features.append(feature)
            NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                self.universe, [feature], self.decision_features)
            non_significant_subset = []
            if dependency(self.universe, candidate_features, self.decision_features) != 1:
                if (dependency(self.universe, test_features, self.decision_features) -
                    dependency(self.universe, candidate_features, self.decision_features)) > 0 or \
                        NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                            self.universe, [feature], self.decision_features) > 0:
                    candidate_features.append(feature)
            else:
                if non_significant == "max subset":
                    non_significant_subset = self.heuristic_non_significant(self.universe, candidate_features,
                                                                            self.decision_features, feature)
                elif non_significant == "SEB1":
                    non_significant_subset = self.sequential_backward_elimination_non_significant(
                        self.universe, candidate_features, self.decision_features, feature)
                elif non_significant == "SEB3":
                    max_length = 0
                    for i in range(0, 3):
                        temp_non_significant_subset1 = self.sequential_backward_elimination_non_significant(
                            self.universe, candidate_features, self.decision_features, feature)
                        temp_length = len(temp_non_significant_subset1)
                        if temp_length > max_length:
                            max_length = temp_length
                            non_significant_subset = temp_non_significant_subset1
                else:
                    print("wrong parameter of " + non_significant)
                    return
                if len(non_significant_subset) > 1:
                    candidate_features.append(feature)
                    candidate_features = [i for i in candidate_features if i not in non_significant_subset]
                if len(non_significant_subset) == 1:
                    noise_resistant_measure_a = NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                        self.universe, [feature], self.decision_features)
                    noise_resistant_measure_b = NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                        self.universe, non_significant_subset, self.decision_features)
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


def online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis_test():
    universe = np.array(pd.read_csv("approximation_data.csv", header=None))
    conditional_features = [0, 1, 2, 3]
    decision_features = [4]
    algorithm = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis(
        universe, conditional_features, decision_features)
    result = algorithm.run(non_significant="max subset")
    print("approximation_data reduction:", result)
    # approximation_data reduction: [0, 1, 2]
    # The time used: 0.006995677947998047 seconds

    universe = np.array(pd.read_csv("example_data.csv", header=None))
    conditional_features = [i for i in range(0, 5)]
    decision_features = [5]
    algorithm = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis(
        universe, conditional_features, decision_features)
    result = algorithm.run(non_significant="max subset")
    print("example_data reduction:", result)
    # The time used: 0.011988639831542969 seconds
    # example_data reduction: [2, 4]

    universe = np.array(pd.read_csv("example_data.csv", header=None))
    conditional_features = [i for i in range(0, 5)]
    decision_features = [5]
    algorithm = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis(
        universe, conditional_features, decision_features)
    result = algorithm.run(non_significant="SEB1")
    print("example_data reduction:", result)
    # The time used: 0.009993553161621094 seconds
    # example_data reduction: [3, 4]

    universe = np.array(pd.read_csv("arcene_train.csv", header=None))
    conditional_features = [i for i in range(0, 10000)]
    decision_features = [10000]
    algorithm = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis(
        universe, conditional_features, decision_features)
    result = algorithm.run(non_significant="max subset")
    print("arcene_train reduction:", result)
    # arcene_train reduction: [9997, 9998, 9999]
    # The time used: 374.79046154022217 secondsprint('The time used: {} seconds'.format(time.time() - start))
    return


def main():
    quick_reduction_test()
    noise_resistant_assisted_quick_reduction_test()

    online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis_test()
    NoiseResistantDependencyMeasureTest.mean_positive_region_test()
    NoiseResistantDependencyMeasureTest. \
        proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance_test()
    NoiseResistantDependencyMeasureTest.related_information_of_subset_b_test()
    NoiseResistantDependencyMeasureTest.proximity_of_boundary_region_to_positive_region_based_portion_test()
    NoiseResistantDependencyMeasureTest.noisy_dependency_of_feature_subset_a_on_feature_subset_b_test()
    pass


if __name__ == '__main__':
    main()
