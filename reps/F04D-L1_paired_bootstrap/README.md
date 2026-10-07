# F04D-L1 -- Paired bootstrap by group

## Mechanism
Resample the paired differences of two separate models run over the same dataset in order to infer the general performance of each model across different makeups of the test set data. We resample the data, grouped at the appropriate granularity to resample at the independent unit, with replacement to provide us a confidence interval so that we can determine if the changes we made to a model are supported as established independent of the families in the test set provided. 

## Task
Analyze the paired differences with resampling to create a 95% percentile bootstrap confidence interval to determine if the changes we made to a model are established as being more effective for these families of the test set data.
Take the paired differences of each case for each model, resample them at the group level, and take the 95% interval of the resulting distribution. If the 95% interval lies above 0, we can say that for these families of data, one model change is better ('Established').

Inputs - a prewritten designation of paired differences at the case level, grouped, for two models.
Outputs - a mean difference of the source data and an interval showing the confidence interval of this dataset with resampling and an analysis over that distribution determining whether we can establish that one model is more effective independent of the test dataset families.

## Interface
Paired Bootstrap:
Name: 
  paired_bootstrap
arguments:
  paired_differences: the array of paired differences.
  group_labels: the array of group labels of each difference. One value per case, same length as paired_differences.
  resamples: the integer number of resamples to take.
  interval_level: the float of the central percentile of interval to take from the final distribution.
  seed: an integer of a seed to create a deterministic and recreatable resampling distribution across all resamples for a given bootstrap.
returns:
  mean_difference: float - mean of the difference from the paired differences.
  percentile_interval_start: float - the lower value of the percentile interval from the resample distribution.
  percentile_interval_end: float - the upper value of the percentile interval from the resample distribution.

Draw:
Name:
  draw_resamples
arguments:
  group_labels: the array of group labels of each difference. One value per case, same length as paired_differences.
  resamples: the integer number of resamples to take.
  seed: an integer of a seed to create a deterministic and recreatable resampling distribution across all resamples for a given bootstrap.
returns:
  drawn_array: a 2D numpy array of (n_resamples, n_groups) that holds the group numbers 0 to n_groups - 1, where group numbers follow the sorted distinct labels, drawn in that resample as integers.

## Pass Check
tests/test_bootstrap.py passes.

Tests will assert the following:

1. **Mean.** The mean difference equals `+1.0458` within `1e-4`. Reason for the tolerance: the inputs are rounded to four decimals.
2. **Structure, exact.** In a group resample, the number of groups drawn equals the number of groups in the data, and the two cases of a drawn group always appear together.
3. **Determinism, exact.** The same seed gives the same interval twice.
4. **Against a library.** The 95% group interval agrees with `scipy.stats.bootstrap` using `method="percentile"` on the eight group means, within `0.15` at each end. Reason for the tolerance: two different random number streams give different resamples, and with 2000 resamples of 8 units the ends move by about that much.
5. **Against the notebook.** The group interval is within `0.15` of `[−0.0575, +2.5603]` at each end. The case interval is within `0.15` of `[+0.2015, +2.0581]`.
6. **Verdict.** The group interval contains 0, the case interval does not, and the group interval is the wider one.

## Result
Result: Not Established

Settings:
Resamples: 2000
Seed: 1
Interval Level: 0.95

Results:
Group-Level:
Mean Difference: +1.0458
95% Center Interval: [-0.0561, +2.6500]
Result crosses the 0 boundary therefore the claim is not established.

Case-Level:
Mean Difference: +1.0458
95% Center Interval: [+0.2006, +2.0698]
Result does not cross the 0 boundary therefore the claim is established.

Case-Level results were there to show perspective of an analysis error - cases are not independent. Sampling at the case level would skew the results and errantly show an established claim.

## Why It Matters
This type of analysis, a paired bootstrap, is a mechanism we can use to differentiate if the performance change was due to the model change or the test dataset. By resampling the families with replacement, we are drawing a different test set of the same size from the same kind of data. The distribution of mean differences from that process shows how much the measured difference depends on which families were in each sample set.

There can be certain families of data that have an outsized impact on model performance between changes. Resampling in this way allows us to get a confidence interval to identify the impact of that dataset separate from the model change itself.

We also showed with this process that resampling at the case level is an analysis design error. Cases are not independent - they share a source. Resampling cases gives us a skewed confidence interval that actually reaches the opposite verdict. Resampling data must be approached at the appropriate level of granularity to resample at the independent unit in the resampling process.

## Toolkit
This added paired_bootstrap, a function for performing a paired_bootstrap on grouped datasets, and draw_resamples, a function for resampling a grouping index for a dataset. Both serve to facilitate grouped families of data rather than flat, denormalized data.
