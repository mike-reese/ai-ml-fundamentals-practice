import json, numpy
from evalkit.bootstrap import paired_bootstrap 
from pathlib import Path

if __name__ == "__main__":    
    LABELS = ["f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28"]
    VALUES = [[4.0820, 5.2069], [-0.2537, -0.1221], [3.6559, 4.2919], [-0.0017, -0.0023], [-0.0028, -0.0020], [-0.0173, -0.0239], [-0.0050, -0.0036], [-0.0402, -0.0291]]
    LABELS = numpy.repeat(LABELS,2)
    VALUES = numpy.array(VALUES).ravel()

    resamples = 2000
    seed = 1 
    interval_level = 0.95

    grouped_mean_difference, grouped_interval_low, grouped_interval_high = paired_bootstrap(VALUES,LABELS,resamples,interval_level,seed)

    case_labels = numpy.arange(16).astype(str)
    case_mean_difference, case_interval_low, case_interval_high = paired_bootstrap(VALUES, case_labels, resamples,interval_level,seed)


    print(f"Settings: Resamples: {resamples}, Seed: {seed}, Interval Level: {interval_level}")
    print(f"{'Unit':<20} {'Mean Difference':<20} {'Lower':<20} {'Upper':<20}")
    print(f"{'Group-Level':<20} {grouped_mean_difference:>+10.4f} [{grouped_interval_low:>+10.4f},{grouped_interval_high:>+10.4f}]")
    print(f"{'Case-Level':<20} {case_mean_difference:>+10.4f} [{case_interval_low:>+10.4f}, {case_interval_high:>+10.4f}]")
    print(f"Verdict: Supported if the group interval lies entirely above 0. \nVerdict: {'Established' if grouped_interval_high >= grouped_interval_low > 0 else 'Not Established'}")


    json_schema = {
        'Settings': {
            'Resamples': resamples,
            'Seed': seed,
            'Interval Level': interval_level
        },
        'Mean Difference': grouped_mean_difference,
        'Group-Level': {
                'Interval Low': grouped_interval_low, 
                'Interval High': grouped_interval_high
            },
        'Case-Level':{
                'Interval Low': case_interval_low,
                'Interval High': case_interval_high
        },
    'Verdict': 'Established' if grouped_interval_high >= grouped_interval_low > 0 else 'Not Established'
    }

    path = Path(__file__).parent / "results" / "F04D-L1.json"
    with open (path, "w") as f:
        json.dump(json_schema, f, indent=2)
