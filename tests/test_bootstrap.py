import pytest
import evalkit.bootstrap as bootstrap
import scipy.stats
import numpy

@pytest.fixture
def f04_paired_differences():
    LABELS = ["f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28"]
    VALUES = [[4.0820, 5.2069], [-0.2537, -0.1221], [3.6559, 4.2919], [-0.0017, -0.0023], [-0.0028, -0.0020], [-0.0173, -0.0239], [-0.0050, -0.0036], [-0.0402, -0.0291]]
    LABELS = numpy.repeat(LABELS,2)
    VALUES = numpy.array(VALUES).ravel()
    return LABELS,VALUES

@pytest.fixture
def f04_test_structure_paired_differences():
    LABELS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    VALUES = [[1, -1], [2, -2],[3, -3],[4, -4],[5,-5],[6,-6],[7,-7],[8,-8] ]
    LABELS = numpy.repeat(LABELS,2)
    VALUES = numpy.array(VALUES).ravel()
    return LABELS,VALUES

def test_f04_validate_mean(f04_paired_differences):
    LABELS, VALUES = f04_paired_differences
    mean_difference, percentile_start, percentile_end = bootstrap.paired_bootstrap(paired_differences=VALUES, group_labels=LABELS,resamples=1,interval_level=0.95,seed=1)
    assert mean_difference == pytest.approx(1.0458, abs=1e-4)

def test_f04_validate_structure(f04_test_structure_paired_differences):
    LABELS, VALUES = f04_test_structure_paired_differences
    resample_array = bootstrap.draw_resamples(group_labels=LABELS, resamples=5, seed=1) 
    assert resample_array.shape[1] == len(set(LABELS))

    mean_difference, percentile_start, percentile_end = bootstrap.paired_bootstrap(paired_differences=VALUES,group_labels=LABELS, resamples=2000, interval_level=0.95, seed=1)
    assert [percentile_start, percentile_end] == [0,0]

def test_f04_validate_determinism(f04_paired_differences):
    LABELS, VALUES = f04_paired_differences
    seed=2
    mean_difference_a, percentile_start_a, percentile_end_a = bootstrap.paired_bootstrap(paired_differences=VALUES, group_labels=LABELS, resamples=2000, interval_level=0.95, seed=seed)
    mean_difference_b, percentile_start_b, percentile_end_b = bootstrap.paired_bootstrap(paired_differences=VALUES, group_labels=LABELS, resamples=2000, interval_level=0.95, seed=seed)
    interval_a = [percentile_start_a, percentile_end_a]
    interval_b = [percentile_start_b, percentile_end_b]
    assert interval_a == interval_b

def test_f04_validate_library(f04_paired_differences):
    LABELS, VALUES = f04_paired_differences
    mean_difference, percentile_start, percentile_end = bootstrap.paired_bootstrap(paired_differences=VALUES, group_labels=LABELS, resamples=2000, interval_level=0.95, seed=1)
    interval = [percentile_start, percentile_end]
    
    value_averages = VALUES.reshape(-1,2).mean(axis=1)
    
    scikit_interval = scipy.stats.bootstrap((value_averages, ),statistic=numpy.mean,n_resamples=2000,method="percentile",rng=1)
   
    # Reason for 0.15 -- two different random number streams give different resamples, with 2000 resamples of 8 units the ends move by about 0.15
    assert interval == pytest.approx([scikit_interval.confidence_interval.low, scikit_interval.confidence_interval.high], abs=0.15)


def test_f04_validate_notebook(f04_paired_differences):
    LABELS, VALUES = f04_paired_differences
    mean_difference, percentile_start, percentile_end = bootstrap.paired_bootstrap(paired_differences=VALUES, group_labels=LABELS, resamples=2000, interval_level=0.95, seed=1)
    group_interval = [percentile_start, percentile_end]
    assert group_interval == pytest.approx([-0.0575, 2.5603], abs=0.15)
   
    case_labels = numpy.arange(16).astype(str)
    case_mean_difference, case_percentile_start, case_percentile_end = bootstrap.paired_bootstrap(paired_differences=VALUES, group_labels=case_labels, resamples=2000,interval_level=0.95,seed=1)
    case_interval = [case_percentile_start, case_percentile_end]
    assert case_interval == pytest.approx([0.2015, 2.0581], abs=0.15)

def test_f04_validate_verdict(f04_paired_differences):
    LABELS, VALUES = f04_paired_differences
    mean_difference, percentile_start, percentile_end = bootstrap.paired_bootstrap(paired_differences=VALUES, group_labels=LABELS, resamples=2000, interval_level=0.95, seed=1)
    group_interval = [percentile_start, percentile_end]
    assert percentile_start <= 0 <= percentile_end

    case_labels = numpy.arange(16).astype(str)
    case_mean_difference, case_percentile_start, case_percentile_end = bootstrap.paired_bootstrap(paired_differences=VALUES, group_labels=case_labels, resamples=2000,interval_level=0.95,seed=1)
    case_interval = [case_percentile_start, case_percentile_end]
    assert  case_percentile_end >= case_percentile_start >= 0

    assert (group_interval[1] - group_interval[0]) > (case_interval[1] - case_interval[0])
