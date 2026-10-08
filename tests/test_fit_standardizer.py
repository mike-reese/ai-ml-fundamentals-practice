import evalkit.fit_standardizer as fit_standardizer
import torch
import pytest
from sklearn.preprocessing import StandardScaler 

@pytest.fixture
def fit_standardizer_inputs():
    partition = ["training", "validation"]
    training_family = ["f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12"]
    validation_family = ["f13", "f14", "f15", "f16", "f17", "f18"]
    cases = ["a", "b"]

    training_features = [[7, 0.9], [7, 0.88], [5, 0.74], [5, 0.76], [2, 0.66], [2, 0.64], [6, 0.81], [6, 0.83], [6, 0.30], [6, 0.28],[8, 0.38], [8, 0.36], [1, 0.18], [1, 0.16], [2, 0.28], [2, 0.30], [0, 0.12], [0, 0.14], [3, 0.50], [3, 0.52]]
    validation_features = [[6, 0.8], [6, 0.77], [1, 0.58], [1, 0.60], [5, 0.80], [5, 0.82], [7, 0.33], [7, 0.31], [1, 0.21], [1, 0.23], [4, 0.56], [4, 0.54]]

    training_tensor = torch.tensor(training_features, dtype=torch.float64)
    validation_tensor = torch.tensor(validation_features, dtype=torch.float64)
    
    training_family_expanded = [family for family in training_family for _ in range(2)]
    validation_family_expanded = [family for family in validation_family for _ in range(2)]

    training_family_cases = [family + case for family in training_family for case in cases]
    validation_family_cases = [family + case for family in validation_family for case in cases]

    partition_training = [partition[0]] * len(training_family_expanded)
    partition_validation = [partition[1]] * len(validation_family_expanded)

    partitions = partition_training + partition_validation
    families = training_family_expanded + validation_family_expanded
    case_families = training_family_cases + validation_family_cases

    dataset_labels = [partitions, families, case_families]
    dataset_features = torch.cat([training_tensor, validation_tensor], dim=0)

    training_label_index = [i for i, row in enumerate(dataset_labels[0]) if row=="training"]
    validation_label_index = [i for i, row in enumerate(dataset_labels[0]) if row=="validation"]
    
    training_features = dataset_features[training_label_index]
    validation_features = dataset_features[validation_label_index]

    return dataset_labels, dataset_features, training_features, validation_features

    
def test_statistics(fit_standardizer_inputs):
    dataset_labels, dataset_features, training_features, validation_features = fit_standardizer_inputs
    
    per_feature_mean, per_feature_standard_deviation = fit_standardizer.fit_standardizer(training_features)

    # Tolerance - 1e-4 is to match our tensor float to our rounded reference number.
    torch.testing.assert_close(per_feature_mean, torch.tensor([4.000, 0.4870], dtype=torch.float64), atol=1e-4, rtol=0)
    torch.testing.assert_close(per_feature_standard_deviation, torch.tensor([2.6077, 0.2626], dtype=torch.float64), atol=1e-4, rtol=0)

def test_library(fit_standardizer_inputs):
    dataset_labels, dataset_features, training_features, validation_features = fit_standardizer_inputs
    
    per_feature_mean, per_feature_standard_deviation = fit_standardizer.fit_standardizer(training_features)

    scaler = StandardScaler()
    scaler = scaler.fit(training_features.numpy())
  
    # Tolerance - 1e-9 is where we achieve the same formula as in float64, only rounding error remains.
    torch.testing.assert_close(torch.from_numpy(scaler.mean_), per_feature_mean, atol=1e-9, rtol=0)
    torch.testing.assert_close(torch.from_numpy(scaler.scale_), per_feature_standard_deviation, atol=1e-9, rtol=0)

def test_transform_one(fit_standardizer_inputs):
    dataset_labels, dataset_features, training_features, validation_features = fit_standardizer_inputs

    index = dataset_labels[2].index("f13a")
    validation_features_one = dataset_features[index]

    per_feature_mean, per_feature_standard_deviation = fit_standardizer.fit_standardizer(training_features)
    validation_one_feature_matrix = fit_standardizer.apply_standardization(validation_features_one, per_feature_mean, per_feature_standard_deviation)
    # Reason for 1e-3 -- our notebook prediction cell asked for 3 decimals
    torch.testing.assert_close(validation_one_feature_matrix, torch.tensor([0.7670, 1.1917], dtype=torch.float64), atol=1e-3, rtol=0)

def test_transform_sets(fit_standardizer_inputs):
    dataset_labels, dataset_features, training_features, validation_features = fit_standardizer_inputs

    training_per_feature_mean, training_per_feature_standard_deviation = fit_standardizer.fit_standardizer(training_features)

    training_standardized_feature_set = fit_standardizer.apply_standardization(training_features, training_per_feature_mean, training_per_feature_standard_deviation) 
    standardized_training_per_feature_mean, standardized_training_per_feature_standard_deviation = fit_standardizer.fit_standardizer(training_standardized_feature_set)

    validation_standardized_feature_set = fit_standardizer.apply_standardization(validation_features,training_per_feature_mean,training_per_feature_standard_deviation)
    validation_per_feature_mean, validation_per_feature_standard_deviation = fit_standardizer.fit_standardizer(validation_standardized_feature_set)
    
    # Tolerance - 1e-9 is where we achieve the same formula as in float64, only rounding error remains.
    torch.testing.assert_close(standardized_training_per_feature_mean, torch.tensor([0, 0], dtype=torch.float64), atol=1e-9, rtol=0)
    torch.testing.assert_close(standardized_training_per_feature_standard_deviation, torch.tensor([1, 1], dtype=torch.float64), atol=1e-9, rtol=0)
    
    # Tolerance - 1e-3 is what our notebook's prediction cell asked for (3 decimals)
    torch.testing.assert_close(validation_per_feature_mean[1], torch.tensor(0.2240, dtype=torch.float64), atol=1e-3, rtol=0)
    torch.testing.assert_close(validation_per_feature_standard_deviation[0], torch.tensor(0.8856, dtype=torch.float64), atol=1e-3, rtol=0)

def test_leakage(fit_standardizer_inputs):
    dataset_labels, dataset_features, training_features, _ = fit_standardizer_inputs
    
    leakage_per_feature_mean, leakage_per_feature_standard_deviation = fit_standardizer.fit_standardizer(dataset_features)
    correct_per_feature_mean, correct_per_feature_standard_deviation = fit_standardizer.fit_standardizer(training_features)

    torch.testing.assert_close(leakage_per_feature_mean, torch.tensor([4.0000, 0.5091], dtype=torch.float64), atol=1e-3, rtol=0)
    torch.testing.assert_close(leakage_per_feature_standard_deviation, torch.tensor([2.5000, 0.2487], dtype=torch.float64), atol=1e-3, rtol=0)
    
    leakage_standardized_feature_set = fit_standardizer.apply_standardization(training_features, leakage_per_feature_mean, leakage_per_feature_standard_deviation)
    correct_standardized_feature_set = fit_standardizer.apply_standardization(training_features, correct_per_feature_mean, correct_per_feature_standard_deviation)

    shifts = leakage_standardized_feature_set - correct_standardized_feature_set
    max_difference = shifts.abs().max()

    # Tolerance 1e-3 because the reference number only has 3 decimals
    torch.testing.assert_close(max_difference,torch.tensor(0.1670, dtype=torch.float64), atol=1e-3, rtol=0)

def test_group_split(fit_standardizer_inputs):
    dataset_labels, dataset_features, _, _ = fit_standardizer_inputs
    
    partitions = dataset_labels[0]
    family_labels = dataset_labels[1]
    family_cases = dataset_labels[2]

    overlap = fit_standardizer.check_group_split(partitions, family_labels)
    assert overlap == set()

    f13_alteration_base = list(partitions)
    f13_alteration_base_index = family_cases.index("f13a")
    f13_alteration_base[f13_alteration_base_index] = "training"

    f13_partitions = f13_alteration_base

    overlap_f13 = fit_standardizer.check_group_split(f13_partitions, family_labels)
    assert overlap_f13 == {"f13"}

    test_groups_all_in_one = [["P1", "P1", "P1", "P2", "P3"], ["G1", "G1", "G1", "G2", "G2"], ["C1", "C2", "C3", "C1", "C2"]]
    test_partitions = test_groups_all_in_one[0]
    test_groups = test_groups_all_in_one[1]

    overlap_test = fit_standardizer.check_group_split(test_partitions,test_groups)
    assert overlap_test == {"G2"}


