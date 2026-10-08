import torch
import evalkit.fit_standardizer as fit_standardizer
import json
from pathlib import Path 


def setup_data():
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



dataset_labels, dataset_features, training_features, validation_features = setup_data()
training_set_per_feature_mean, training_set_per_feature_standard_deviation = fit_standardizer.fit_standardizer(training_features)

print(f"Training Set Per-Feature-Mean: {training_set_per_feature_mean}")
print(f"Training Set Per-Feature-Standard-Deviation: {training_set_per_feature_standard_deviation}")


standardized_training_feature_set = fit_standardizer.apply_standardization(training_features, training_set_per_feature_mean, training_set_per_feature_standard_deviation)
standardized_validation_feature_set = fit_standardizer.apply_standardization(validation_features, training_set_per_feature_mean, training_set_per_feature_standard_deviation)

standardized_training_per_feature_mean, standardized_training_per_feature_standard_deviation = fit_standardizer.fit_standardizer(standardized_training_feature_set)
standardized_validation_per_feature_mean, standardized_validation_per_feature_standard_deviation = fit_standardizer.fit_standardizer(standardized_validation_feature_set)

family_cases = dataset_labels[2]
f13a_index = family_cases.index("f13a")
standardized_dataset = torch.cat([standardized_training_feature_set,standardized_validation_feature_set], dim=0)
f13a_features_standardized = standardized_dataset[f13a_index]

print("")
print(f"Standardized Training Set Per-Feature Mean: {standardized_training_per_feature_mean}, Standardized Training Set Per-Feature Standard Deviation: {standardized_training_per_feature_standard_deviation}")
print(f"Standardized Validation Set Per-Feature Mean: {standardized_validation_per_feature_mean}, Standardized Validation Set Per-Feature Standard Deviation: {standardized_validation_per_feature_standard_deviation}")
print(f"Case f13a features: {f13a_features_standardized}")
print(f"Case f13a overlap calculation: ({dataset_features[f13a_index][0]:.4f} - {training_set_per_feature_mean[0]:.4f}) / {training_set_per_feature_standard_deviation[0]:.4f} = {f13a_features_standardized[0]:.4f}")
print(f"Case f13a cosine calculation: ({dataset_features[f13a_index][1]:.4f} - {training_set_per_feature_mean[1]:.4f}) / {training_set_per_feature_standard_deviation[1]:.4f} = {f13a_features_standardized[1]:.4f}")

wrong_per_feature_mean, wrong_per_feature_standard_deviation = fit_standardizer.fit_standardizer(dataset_features)
leakage_standardized_feature_set = fit_standardizer.apply_standardization(training_features, wrong_per_feature_mean, wrong_per_feature_standard_deviation)
shifts = leakage_standardized_feature_set - standardized_training_feature_set
max_difference = shifts.abs().max()

print("")
print(f"Data Leakage -- Training on both sets together. Incorrect Per-Feature Mean: {wrong_per_feature_mean}, Incorrect Per-Feature Standard Deviation: {wrong_per_feature_standard_deviation}, Correct Per-Feature Mean: {training_set_per_feature_mean}, Correct Per-Feature Standard Deviation: {training_set_per_feature_standard_deviation}")
print(f"Largest Single Item Change: {max_difference}")

partitions = dataset_labels[0]
family_labels = dataset_labels[1]
family_cases = dataset_labels[2]

overlap = fit_standardizer.check_group_split(partitions, family_labels)

f13_alteration_base = list(partitions)
f13_alteration_base_index = family_cases.index("f13a")
f13_alteration_base[f13_alteration_base_index] = "training"

f13_partitions = f13_alteration_base

overlap_f13 = fit_standardizer.check_group_split(f13_partitions, family_labels)

print("")
print(f"Overlap on assigned partitions: {overlap}")
print(f"Overlap when a group is split across partitions: {overlap_f13}")

json_schema = {
    "Training Mean": training_set_per_feature_mean.tolist(),
    "Training Std Dev": training_set_per_feature_standard_deviation.tolist(),
    "f13a Raw Features": dataset_features[f13a_index].tolist(),
    "f13a Standardized Features": standardized_dataset[f13a_index].tolist(),
    "Standardized Training Set Mean": standardized_training_per_feature_mean.tolist(),
    "Standardized Training Set Std Dev": standardized_training_per_feature_standard_deviation.tolist(),
    "Standardized Validation Set Mean": standardized_validation_per_feature_mean.tolist(),
    "Standardized Validation Set Std Dev": standardized_validation_per_feature_standard_deviation.tolist(),
    "Data Leakage Example Mean": wrong_per_feature_mean.tolist(),
    "Data Leakage Example Std Dev": wrong_per_feature_standard_deviation.tolist(),
    "Largest Change from Data Leakage": max_difference.item(),
    "Overlap Check on Assigned Partitions": list(overlap),
    "Overlap Check with forced group split": list(overlap_f13)
}

path = Path(__file__).parent / "results" / "F04A-L1.json"
with open (path, "w") as f:
    json.dump(json_schema, f, indent=2)
