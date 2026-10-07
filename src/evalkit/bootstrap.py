import numpy
from numpy.typing import ArrayLike


def paired_bootstrap(paired_differences:ArrayLike, group_labels:ArrayLike, resamples:int, interval_level:float, seed:int) -> tuple[float, float, float]:
    """
    Paired Bootstrap: Calculate the paired differences from a grouped set of data using resampling with replacement. Return the mean difference of the dataset and the resampling low and high intervals.    

    Take a grouped dataset, resample it with replacement, and return the mean_difference of the original dataset and the low and high intervals of the resampled distribution.

    Parameters:
    ---- 
    paired_differences: The numpy array of paired differences from two different model runs over the same data
    group_labels: The numpy array of labels for the grouping of the dataset
    resamples: The number of resamples with replacement to execute
    interval_level: The float of the center interval percentage to return from the analysis, trimming 1/2 the remainder from either side.
    seed: A fixed integer to create a deterministic resampling distribution.

    Returns:
    ----- 
    mean_differences: The float mean difference of the initial paired_differences dataset. The sum of the differences d over the n cases divided by n.
    percentile_interval_start: The low bound of the interval range of the distribution of means after resampling with replacement. Quantile of m at (1-interval_level) / 2
    percentile_interval_end: The high bound of the interval range of the distribution of means after resampling with replacement.
    """
    paired_differences = numpy.asarray(paired_differences)
    group_labels = numpy.asarray(group_labels)
    if len(paired_differences) != len(group_labels):
        raise ValueError(f"Input array lengths do not match. \npaired_differences: {len(paired_differences)} \ngroup_labels: {len(group_labels)} ")

    if resamples < 1:
        raise ValueError(f"Number of resamples must be greater than zero. Resamples: {resamples}")
    
    if interval_level <= 0 or interval_level >= 1:
        raise ValueError(f"Interval Level must be between 0 and 1. \ninterval_level: {interval_level}")

    mean_differences = paired_differences.mean()
    
    drawn_resamples = draw_resamples(group_labels, resamples, seed)

    groups, group_index, group_counts = numpy.unique(group_labels, return_inverse=True, return_counts=True)
    if not numpy.all(group_counts[0] == group_counts):
        raise ValueError(f"Group Sizes must be identical. \nGroup Sizes: {group_counts}")

    n_groups = len(groups)
    sorted_group_index = numpy.argsort(group_index)

    reshaped_groups = sorted_group_index.reshape(n_groups, -1)
    position_table = reshaped_groups[drawn_resamples]

    values_table = paired_differences[position_table]
    values_table_means = values_table.mean(axis=(1,2))

    interval = numpy.quantile(values_table_means, [(1-interval_level)/2, 1 - (1-interval_level)/2])
    percentile_interval_start = interval[0]
    percentile_interval_end = interval[1]
    return mean_differences, percentile_interval_start, percentile_interval_end

def draw_resamples(group_labels:ArrayLike, resamples:int, seed:int) -> numpy.ndarray:
    """
    Draw Resamples: Draw resamples with replacement from a provided grouping of data.

    Take a group label array, create a seeded generator, and return an array of size resamples after resampling the labels with replacement.

    Paramters:
    ----- 
    group_labels: The numpy array of labels for the grouping of the dataset
    resamples: The number of resamples with replacement to execute
    seed: A fixed integer to create a deterministic resampling distribution

    Returns:
    ----- 
    drawn_array: The full numpy array of resampled label indices. Size (resamples, n_groups). Returns indexes, not group labels.
    """
    generator = numpy.random.default_rng(seed)

    n_groups = len(numpy.unique(group_labels))

    drawn_array = generator.integers(0,n_groups, (resamples,n_groups)) 
    return drawn_array
