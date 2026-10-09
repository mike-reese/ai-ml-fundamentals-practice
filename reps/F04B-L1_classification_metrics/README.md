# F04B-L1 - Classification Metrics.
Confusion Matrix, precision, recall, F1, macro-F1, log loss

## Mechanism
Analyze the outputs of different classifiers of a case being assigned either label 1 or label 0 by using a confusion matrix, calculating accuracy/precision/recall, calculating the F1 and macro-F1, and returning the log loss. This analysis compares the outputs of 3 separate classification processes -- a classification model, a majority baseline, and a rule-based baseline.

## Task
Create a generate_classification_metrics function, generate_confusion_matrix function, and calculate_log_loss function that takes in the supplied data and outputs the relevant classification metrics for that function (defined in the interface). 

Validate passing by running a suite of 7 tests against these functions to assert code integrity, safe value handling, and edge case handling. Demonstrate understanding of the fundamental concepts and utility of each process with a writeup.

## Interface
Classification Metrics:
name:
  generate_classification_metrics
arguments:
  case_labels: A numpy integer array of the known correct case labels from a set of cases
  predicted_classes: A numpy integer array of the predicted classes of labels from a set of cases.
returns:
  ClassificationMetrics: A dataclass containing the following metrics from the function call:
    confusion_matrix: A numpy integer array of shape (2, 2) in scikit-learn's layout. row = label (0, then 1), column = predicted class (0, then 1)
    precision: A numpy float array of shape (2, ) containing the precision calculation for each classification of our analysis dataset 
    recall: A numpy float array of shape (2, ) containing the recall calculation for each classification of our analysis dataset
    f1: A numpy float array of shape (2, ) containing the f1 calculation for each classification of our analysis dataset
    support: A numpy integer array of shape (2, ) containing the support calculation for each classification of our analysis dataset
    macro_f1: A float, scalar, that is the mean of the f1 scores of each classification of our analysis dataset 

Confusion Matrix:
name:
  generate_confusion_matrix
arguments:
  case_labels: A numpy integer array of the known correct case labels from a set of cases. Shape (N, ) and values of 0 or 1
  predicted_classes: A numpy integer array of the predicted classes of labels from a set of cases. Shape (N, ) and values of 0 or 1
returns:
  confusion_matrix: A numpy integer array of shape (2, 2) in scikit-learn's layout. row = label (0, then 1), column = predicted class (0, then 1)


Log Loss:
name:
  calculate_log_loss
arguments:
  case_labels: An array of the known correct case labels from a set of cases, shape (N, )
  probabilities: An array of the probabilities of label 1, shape (N, )
returns:
  log_loss: Float, the mean over the cases of -ln(p_target), where p_target is p when the label is 1 and 1-p when the label is 0. Probabilities are clipped to [eps, 1 - eps], eps as the dtype's machine epsilon, before the log.
  per_example_loss: A numpy float array of the per-example loss for each case

## Pass Check 
All tests in test_classification_metrics.py must pass.

Here is our test criteria:

1. **Confusion matrices, exact.** Classifier `[[6, 2], [1, 3]]`. Majority baseline `[[8, 0], [4, 0]]`. Rule baseline `[[2, 6], [2, 2]]`. Integers, so the comparison is exact.
2. **Per-class scores, classifier.** Class 0: precision `0.8571`, recall `0.7500`, F1 `0.8000`, support `8`. Class 1: precision `0.6000`, recall `0.7500`, F1 `0.6667`, support `4`. Macro-F1 `0.7333`. Within `1e-4`; support exact. Reason for the tolerance: the notebook printed four decimals.
3. **Zero division.** Majority baseline, class 1: precision, recall and F1 are `0`, with no exception and no `nan`. Its macro-F1 is `0.4000` within `1e-4`. Rule baseline macro-F1 `0.3333` within `1e-4`.
4. **Against a library.** For all three systems, your per-class precision, recall, F1 and support agree with `sklearn.metrics.precision_recall_fscore_support(y, predicted, labels=[0, 1], zero_division=0)`, and macro-F1 with `sklearn.metrics.f1_score(..., average="macro", zero_division=0)`, within `1e-12`. Reason for the tolerance: the same divisions in float64, so only rounding error remains.
5. **Per-example loss.** `f13a` `0.5358`, `f13b` `0.7905`, `f15b` `1.9036`, within `1e-4`. Reason: the notebook printed four decimals.
6. **Log loss.** The classifier's log loss is `0.4513` within `1e-4` of the notebook, and agrees with `sklearn.metrics.log_loss(y, p)` within `1e-12`. Reason for the second tolerance: no probability here is within `eps` of 0 or 1, so the clip does not act and only rounding error remains. (The six-decimal inputs move the log loss by about `2e-7` from the notebook's unrounded value.)
7. **The clip.** For labels `[1, 0]` and probabilities `[0.0, 0.0]`, your `log_loss` returns `18.0218` within `1e-4` and agrees with scikit-learn within `1e-12`. Without the clip the first case is `−ln(0)`, which has no finite value (STATE 9). With it, that case costs `−ln(eps) ≈ 36.0437` and the second `−ln(1 − eps) ≈ 2.2e-16`; their mean is `18.0218`. Reason for the tolerance: the same formula in float64. Also test that labels `[1, 0]` with probabilities `[1.0, 1.0]` give the same `18.0218`: the clip acts at both ends.

## Result
All tests passed.

We successfully demonstrated the ability to create a confusion matrix and generate the necessary classification metrics and log loss for our model outputs.

Our tests asserted that our logic was equivalent to sklearn's functions at a tolerance of 1e-12.

Model Results from analysis:
The classifier model had the highest Macro-F1 and accuracy rating. From our analysis of these metrics, we can identify it as the best-performing model compared to the rule-based baseline and the majority baseline. It also showed that our classifier was successful in that it outperformed the majority baseline at a minimum.

What this does not establish is generalization - these were comparison metrics between models only on 12 data points. It shows us that over these 12 cases our classifier performed best, but it does not conclusively support a finding for widespread ability. 

## Why It Matters 
This gives us a succinct analysis of the behavior of our different models, allowing us to very quickly compare performance and diagnose the specific nature of where the model either excelled or may have had issues.

A confusion matrix allows us to classify the outputs into a standardized format that we can analyze. The classification metrics we can create from that allow us to identify the precision (how often a model was correct at predicting a given class), the recall (how much "coverage" the model achieved on identifying all of the instances of a given class), the f1 score (the harmonic mean of the precision and recall, allowing us to see, for each class, a comparable standardized singular correctness score that balances the model's ability to be correct and identify all instances of a class), and the support, which gives us a raw count of all instances of a given class.

Analyzing these side-by-side for separate models allows us to have a single readout to quickly intuit a given model's strengths or weaknesses for a given classification of data. It's an effective starting point for further, targeted analysis if necessary.

## Toolkit
This added generate_classification_metrics, a function that outputs all of the necessary classification metrics for a given model's predicted classes and known classifications datasets; generate_confusion_matrix, a function that generates a confusion matrix for those same datasets; and calculate_log_loss, a function that allows us to calculate the log_loss and per_example_loss of a given classifier's predictions against the known classifications. 
