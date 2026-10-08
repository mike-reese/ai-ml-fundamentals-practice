import torch
import collections


def fit_standardizer(feature_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit Standardizer: Return the per-feature-mean and per-feature-standard-deviation of a training set feature matrix
    
    We take a feature matrix of size (N, F), manually calculate the mean and standard deviation of each feature, and return those two values as tensors with size (F, ) for each

    Parameters:
    ------ 
    feature_matrix: The torch tensor of the training set feature matrix with size (N, F)
    
    Returns:
    ------ 
    per_feature_mean: A tensor of shape (F, ) of the mean of each feature in the training set.
    per_feature_standard_deviation: A tensor of shape (F, ) of the standard deviation of each feature of the training set. The standard deviation is the population definition: The square root of the mean of the squared deviations  
    """
    if feature_matrix.ndim != 2:
       raise ValueError(f"Feature Matrix must be a tensor of size (N, F). Current shape: {feature_matrix.shape}")

    mean = feature_matrix.mean(dim=0)
    deviation = feature_matrix - mean
    dev_squared = deviation * deviation
    variance = dev_squared.sum(dim=0) / feature_matrix.shape[0]
    std_dev = variance.sqrt()

    per_feature_mean = mean
    per_feature_standard_deviation = std_dev

    return per_feature_mean, per_feature_standard_deviation

def apply_standardization(feature_matrix: torch.Tensor, per_feature_mean: torch.Tensor, per_feature_standard_deviation: torch.Tensor) -> torch.Tensor: 
    """
    Apply Standardization: We create a Z-score for each feature in a feature matrix from a supplied training set per_feature_mean and per_feature_standard_deviation and return a standardized feature matrix

    We take in a feature matrix (of any dataset) of shape (N, F), find the Z-score for each feature, and return it.

    Parameters:
    ------ 
    feature_matrix: a tensor of shape (N, F) containing the feature matrix of the dataset.
    per_feature_mean: A tensor of shape (F, ) of the mean of each feature in the training set.
    per_feature_standard_deviation: A tensor of shape (F, ) of the standard deviation of each feature of the training set.
    
    Returns:
    ------ 
    standardized_feature_matrix: a tensor of shape (N, F) containing the standardized feature matrix. Applying the standardization is (X - mean) / standard_deviation for each feature, returning a feature matrix with standardized values in place of the originals.

    """
    if feature_matrix.ndim > 2 or feature_matrix.ndim == 0:
        raise ValueError(f"Feature Matrix must be a tensor of size (F, ) or (N, F). Current shape: {feature_matrix.shape}")
    if not (feature_matrix.shape[-1] == len(per_feature_mean) == len(per_feature_standard_deviation)):
        raise ValueError(f"Feature Matrix, per-feature-mean, and per-feature-standard-deviation must be all compatible sizes ((F, ) or (N, F)), (F, ), (F, ). Current sizes: Feature Matrix: {feature_matrix.shape}, per-feature-mean: {per_feature_mean.shape}, per-feature-standard-deviation: {per_feature_standard_deviation.shape}")
    if (per_feature_standard_deviation == 0).any():
        raise ValueError("Standard Deviation cannot be zero - check the feature matrix used in fit_standardizer")
    
    standardized_feature_matrix = (feature_matrix - per_feature_mean) / per_feature_standard_deviation

    return standardized_feature_matrix

def check_group_split(partitions: list[str], group_labels: list[str]) -> set:
    """
    Check Group Split: We verify if our grouping labels are split across partitions, which would compromise our analysis as the group is the independent unit. 

    We take in a list of partitions and group labels of equal size, create a set of partitions for each group, and use set comprehension to identify overlapping labels.

    Parameters:
    ----- 
    partitions: a list containing the full dataset's partition names. A view computed from the full dataset.
    group_labels: a list containing the dataset's group labels. A view computed from the full dataset.
    
    Returns:
    ----- 
    overlapping_labels: a set of the overlapping labels that appear in more than one partition. An empty set means the split is clean.
    """   

    if len(partitions) != len(group_labels):
        raise ValueError(f"Partitions and Group Labels must be the same size. Paritions: {len(partitions)}, Group Labels: {len(group_labels)}" )
    grouped = zip(group_labels, partitions)
    
    items = collections.defaultdict(set)

    for group, partition in grouped:
        items[group].add(partition)

    overlapping_labels = {group for group, partition in items.items() if len(partition) > 1}

    return overlapping_labels

