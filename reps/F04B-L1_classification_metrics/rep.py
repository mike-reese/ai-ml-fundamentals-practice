import evalkit.classification_metrics as classification_metrics
import numpy
from pathlib import Path
import json 

def validation_cases():
    cases = ["f13a", "f13b", "f14a", "f14b", "f15a", "f15b", "f16a", "f16b", "f17a", "f17b", "f18a", "f18b"]
    case_labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    probabilities = [0.585180, 0.453601, 0.842832, 0.884210, 0.800398, 0.850974, 0.000123, 0.000086, 0.007693, 0.010919, 0.140861, 0.103250]
    rule_baseline_predicted_class = [1,1,0,0,1,1,1,1,0,0,1,1]
    majority_baseline_predicted_class = [0,0,0,0,0,0,0,0,0,0,0,0]

    case_labels = numpy.array(case_labels)
    probabilities = numpy.array(probabilities)
    rule_baseline_predicted_class = numpy.array(rule_baseline_predicted_class)
    majority_baseline_predicted_class = numpy.array(majority_baseline_predicted_class)

    decision_threshold = 0.5
    classifier_predicted_class = (probabilities > decision_threshold).astype(int)

    return cases, case_labels, probabilities, classifier_predicted_class, rule_baseline_predicted_class, majority_baseline_predicted_class, decision_threshold


def assign_outcome_labels(case_labels, predicted_class):
    outcomes = {
        (0,0): "TN",
        (0,1): "FP",
        (1,0): "FN",
        (1,1): "TP"
    }
    
    outcome_labels = []
    for case_label, predicted_c in zip(case_labels, predicted_class):
        outcome_labels.append(outcomes[case_label, predicted_c])

    return outcome_labels


cases, case_labels, probabilities, classifier_predicted_class, rule_baseline_predicted_class, majority_baseline_predicted_class, decision_threshold = validation_cases()

classifier_metrics = classification_metrics.generate_classification_metrics(case_labels, classifier_predicted_class)
majority_baseline_metrics = classification_metrics.generate_classification_metrics(case_labels, majority_baseline_predicted_class)
rule_baseline_metrics = classification_metrics.generate_classification_metrics(case_labels, rule_baseline_predicted_class)

classifier_outcome_labels = assign_outcome_labels(case_labels, classifier_predicted_class)

print("1. Classifier decisions")
print(f"{'Case ID':<8} | {'Label':>6} | {'Probability':>12} | {'Predicted':>10} | {'Outcome':<8}")
for case, case_label, probability, predicted_class, outcome_label in zip(cases, case_labels, probabilities, classifier_predicted_class, classifier_outcome_labels):
    print(f"{case:<8} | {case_label:>6d} | {probability:>12.6f} | {predicted_class:>10d} | {outcome_label:<8}")

print("")
print("2. Confusion matrix and per-class metrics")
predicted_systems = [("classifier", classifier_metrics), ("rule baseline", rule_baseline_metrics), ("majority baseline", majority_baseline_metrics)]
labels = ["Class 0", "Class 1"]
for sys, metrics in predicted_systems:
    matrix = metrics.confusion_matrix
    # Accuracy is the diagonal sum over the case count. Not a toolkit function, so computed here from the matrix.
    accuracy = numpy.trace(matrix) / matrix.sum()

    print("")
    print(f"--- {sys} ---")
    print(f"{'':<8} {'Pred 0':>8} {'Pred 1':>8}")
    for i, label in enumerate(labels):
        print(f"{label:<8} {matrix[i, 0]:>8d} {matrix[i, 1]:>8d}")
    print("")
    print(f"{'':<8} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Support':>8}")
    for i, label in enumerate(labels):
        print(f"{label:<8} | {metrics.precision[i]:>10.4f} | {metrics.recall[i]:>10.4f} | {metrics.f1[i]:>10.4f} | {metrics.support[i]:>8d}")
    print(f"{'Macro-F1':<8} | {metrics.macro_f1:>10.4f}")
    print(f"{'Accuracy':<8} | {accuracy:>10.4f}")

print("")
print("3. Each case's per-example loss and classifier's log loss")
log_loss, per_example_loss = classification_metrics.calculate_log_loss(case_labels, probabilities)

print(f"----- Per-Example Loss -----")
print(f"{'Case':<20} | {'Loss':<20}")
for i , case in enumerate(cases):
    print(f"{case:<20} | {per_example_loss[i]:>10.4f}")

print("----- Log Loss -----")
print(f"Log Loss: {log_loss:.4f} | Calculation: {per_example_loss.sum():.4f} / {per_example_loss.size}")

print("")
print("4. Largest losses and correct predictions with the lowest confidence")
max_loss = per_example_loss.max()
max_loss_index = numpy.argmax(per_example_loss == max_loss)
max_loss_case = cases[max_loss_index]
print(f"-- Largest Loss --")
print(f"{'Case':<20} | {'Loss':<20}")
print(f"{max_loss_case:<20} | {max_loss:.4f}")

print(f"-- Correct Prediction with lowest confidence -- ")
p_target = numpy.where(case_labels == 1, probabilities, 1 - probabilities)
matching_predictions = (case_labels == classifier_predicted_class)
mask = numpy.where(matching_predictions, p_target, numpy.inf)
min_index = numpy.argmin(mask)
print(f"{'Case':<20} | {'Confidence':<20} | {'Probability':<20} | {'Loss':<20}")
print(f"{cases[min_index]:<20} | {p_target[min_index]:<20.4f} | {probabilities[min_index]:<20.4f} | {per_example_loss[min_index]:<20.4f}")

# Results file
json_schema = {
    "Decision Threshold": decision_threshold,
    "Classifier Decisions": [
        {
            "Case ID": case,
            "Label": case_label.item(),
            "Probability": probability.item(),
            "Predicted Class": predicted_class.item(),
            "Outcome": outcome_label,
        }
        for case, case_label, probability, predicted_class, outcome_label
        in zip(cases, case_labels, probabilities, classifier_predicted_class, classifier_outcome_labels)
    ],
    "Systems": {
        sys: {
            "Confusion Matrix": metrics.confusion_matrix.tolist(),
            "Per Class": {
                label: {
                    "Precision": metrics.precision[i].item(),
                    "Recall": metrics.recall[i].item(),
                    "F1": metrics.f1[i].item(),
                    "Support": metrics.support[i].item(),
                }
                for i, label in enumerate(labels)
            },
            "Macro-F1": metrics.macro_f1.item(),
            "Accuracy": (numpy.trace(metrics.confusion_matrix) / metrics.confusion_matrix.sum()).item(),
        }
        for sys, metrics in predicted_systems
    },
    "Per-Example Loss": dict(zip(cases, per_example_loss.tolist())),
    "Log Loss": log_loss.item(),
    "Largest Loss": {
        "Case ID": max_loss_case,
        "Probability": probabilities[max_loss_index].item(),
        "Per-Example Loss": max_loss.item(),
    },
    "Lowest Confidence Correct Prediction": {
        "Case ID": cases[min_index],
        "Confidence": p_target[min_index].item(),
        "Probability": probabilities[min_index].item(),
        "Per-Example Loss": per_example_loss[min_index].item(),
    },
}

path = Path(__file__).parent / "results" / "F04B-L1.json"
with open(path, "w") as f:
    json.dump(json_schema, f, indent=2)
