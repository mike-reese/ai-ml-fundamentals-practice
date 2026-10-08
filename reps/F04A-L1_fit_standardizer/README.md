# F04A-L1 Fit Standardizer

## Mechanism
We standardize the input features by taking their respective Z-scores so that the gradients of the feature weights have comparable scale and one fixed learning rate suits all of them.

## Task
Standardize the input features with their Z-scores and validate that there is no overlap between grouped features across partitions. Cases are not independent of family groups, so a family group must exist entirely in one partition only and may not be split.

## Interface
Fit Standardizer
name:
  fit_standardizer
arguments:
  feature_matrix: a tensor of shape (N, F) containing the feature matrix of the training set. 
returns:
  per_feature_mean: A tensor of shape (F, ) of the mean of each feature in the training set.
  per_feature_standard_deviation: A tensor of shape (F, ) of the standard deviation of each feature of the training set. The standard deviation is the population definition: The square root of the mean of the squared deviations 

Apply Standardization
name:
  apply_standardization
arguments:
  feature_matrix: a tensor of shape (N, F) containing the feature matrix of the dataset.
  per_feature_mean: A tensor of shape (F, ) of the mean of each feature in the training set.
  per_feature_standard_deviation: A tensor of shape (F, ) of the standard deviation of each feature of the training set.
returns:
  standardized_feature_matrix: a tensor of shape (N, F) containing the standardized feature matrix. Applying the standardization is (X - mean) / standard_deviation for each feature, returning a feature matrix with standardized values in place of the originals.

Check Group Split
name:
  check_group_split
arguments:
  partitions: a list containing the full dataset's partition names.
  group_labels: a list containing the dataset's group labels.
returns:
  overlapping_labels: a set of the overlapping labels that appear in more than one partition. An empty set means the split is clean.

## Pass Check
tests/test_fit_standardizer.py passes

Tests will assert the following:

1. **Statistics.** On the training set, the mean equals `[4.0000, 0.4870]` and the standard deviation equals `[2.6077, 0.2626]`, within `1e-4` on each entry. Reason for the tolerance: the notebook printed four decimals. The sample standard deviation, which divides by `N − 1`, gives `2.6754` for `overlap` and fails this check; that is intended.
2. **Against a library.** The mean and standard deviation agree with `sklearn.preprocessing.StandardScaler` fitted on the same matrix, attributes `mean_` and `scale_`, within `1e-9`. Reason for the tolerance: same formula in float64, so only rounding error remains.
3. **Transform, one case.** Validation case `f13a` standardizes to `[0.7670, 1.1917]` within `1e-3`. Reason: the notebook's prediction cell asked for three decimals.
4. **Transform, the sets.** The standardized training set has per-feature mean `0` and standard deviation `1`, within `1e-9`. The standardized validation set has cosine mean `+0.2240` and overlap standard deviation `0.8856`, within `1e-3`. Reason for the second pair: the validation set's own statistics were never used, so it does not come out at 0 and 1. (Its overlap mean happens to be `0.0000`, because the validation overlap mean equals the training one. That is a coincidence of this fixture, not a property of the method.)
5. **Leakage, measured.** Fitted on the 32 training and validation cases together, the mean is `[4.0000, 0.5091]` and the standard deviation `[2.5000, 0.2487]`, within `1e-3`; the largest change of any standardized training value against the correct version is `0.1670`, within `1e-3`.
6. **Group split, exact.** On the 32 cases with their true partitions, `check_group_split` returns the empty set. On the designed split with `f13a` training and `f13b` validation, it returns exactly `{"f13"}`. A group whose cases all sit in one partition is never reported, however many cases it has.

## Result
All Tests Passed.

We successfully demonstrated the accuracy of our standardization process by comparing it to StandardScaler in our tests and validating that we catch data leakage.

Training Set Per-Feature-Mean: tensor([4.0000, 0.4870], dtype=torch.float64)
Training Set Per-Feature-Standard-Deviation: tensor([2.6077, 0.2626], dtype=torch.float64)

Standardized Training Set Per-Feature Mean: tensor([2.2204e-17, 2.8866e-16], dtype=torch.float64), Standardized Training Set Per-Feature Standard Deviation: tensor([1.0000, 1.0000], dtype=torch.float64)
Standardized Validation Set Per-Feature Mean: tensor([0.0000, 0.2240], dtype=torch.float64), Standardized Validation Set Per-Feature Standard Deviation: tensor([0.8856, 0.8325], dtype=torch.float64)
Case f13a features: tensor([0.7670, 1.1917], dtype=torch.float64)
Case f13a overlap calculation: (6.0000 - 4.0000) / 2.6077 = 0.7670
Case f13a cosine calculation: (0.8000 - 0.4870) / 0.2626 = 1.1917

Data Leakage -- Training on both sets together. Incorrect Per-Feature Mean: tensor([4.0000, 0.5091], dtype=torch.float64), Incorrect Per-Feature Standard Deviation: tensor([2.5000, 0.2487], dtype=torch.float64), Correct Per-Feature Mean: tensor([4.0000, 0.4870], dtype=torch.float64), Correct Per-Feature Standard Deviation: tensor([2.6077, 0.2626], dtype=torch.float64)
Largest Single Item Change: 0.1670402459162661

Overlap on assigned partitions: set()
Overlap when a group is split across partitions: {'f13'}

## Why It Matters
When our data's independent unit is at the group level, instead of the individual case level, it's necessary for us to ensure our partitions are correctly segmented at the level of our independent units to prevent data leakage.

Additionally, the standardization is necessary regardless. We standardized the features using a Z-score because our raw values for one of our features was significantly larger than the other. During the training cycles, using a static learning rate would cause the weights for the features with larger raw values to move much more per-step than the weights for the features with the smaller raw values. By standardizing them with a Z-score, we have a reversible, standardized value entry for each of them that allows both feature weight gradients to move at a comparable scale during the training process.

## Toolkit
We added fit_standardizer.py which includes fit_standardizer, allowing us to retrieve a per-feature-mean and per-feature-standard-deviation of our training set, we added apply_standardization, which allows us to use those training set values to standardize our validation and test datasets without leakage, and we added check_group_split which allows us to proactively identify data leakage in the dataset partitioning process at the grain of our independent units. 
