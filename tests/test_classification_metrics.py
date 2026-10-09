import numpy
import pytest
import sklearn
import evalkit.classification_metrics as classification_metrics 

@pytest.fixture
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

    return cases, case_labels, probabilities, classifier_predicted_class, rule_baseline_predicted_class, majority_baseline_predicted_class

def test_confusion_matrix(validation_cases):
    _, case_labels, probabilities, classifier_predicted_class, rule_baseline_predicted_class, majority_baseline_predicted_class = validation_cases
        
    classifier_confusion_matrix = classification_metrics.generate_confusion_matrix(case_labels, classifier_predicted_class)
    majority_baseline_confusion_matrix = classification_metrics.generate_confusion_matrix(case_labels, majority_baseline_predicted_class)
    rule_baseline_confusion_matrix = classification_metrics.generate_confusion_matrix(case_labels, rule_baseline_predicted_class)

    correct_classifier_matrix = numpy.array([[6,2],[1,3]])
    correct_baseline_matrix = numpy.array([[8,0],[4,0]])
    correct_rules_based_matrix = numpy.array([[2,6],[2,2]])
    
    numpy.testing.assert_array_equal(classifier_confusion_matrix,correct_classifier_matrix)
    numpy.testing.assert_array_equal(majority_baseline_confusion_matrix,correct_baseline_matrix)
    numpy.testing.assert_array_equal(rule_baseline_confusion_matrix,correct_rules_based_matrix)

def test_per_class_scores(validation_cases):
    _, case_labels, _, classifier_predicted_class, _, _ = validation_cases
    
    metrics = classification_metrics.generate_classification_metrics(case_labels, classifier_predicted_class)
    
    precision = metrics.precision
    recall = metrics.recall
    f1 = metrics.f1
    support = metrics.support
    macro_f1 = metrics.macro_f1

    # correct answers
    correct_precision = numpy.array([0.8571, 0.6000])
    correct_recall = numpy.array([0.7500, 0.7500])
    correct_f1 = numpy.array([0.8000,0.6667])
    correct_support = numpy.array([8,4])
    correct_macro_f1 = 0.7333 
    
    # Tolerance at 1e-4 because this was our printed value in our workbook.
    numpy.testing.assert_allclose(precision, correct_precision, atol=1e-4, rtol=0)
    numpy.testing.assert_allclose(recall,correct_recall,atol=1e-4,rtol=0)
    numpy.testing.assert_allclose(f1,correct_f1,atol=1e-4,rtol=0)
    numpy.testing.assert_array_equal(support,correct_support)
    numpy.testing.assert_allclose(macro_f1, numpy.array(correct_macro_f1),atol=1e-4,rtol=0)

    
def test_zero_division(validation_cases):
    _, case_labels, probabilities, classifier_predicted_class, rule_baseline_predicted_class, majority_baseline_predicted_class = validation_cases
    
    majority_baseline_metrics = classification_metrics.generate_classification_metrics(case_labels,majority_baseline_predicted_class)
    rule_baseline_metrics = classification_metrics.generate_classification_metrics(case_labels,rule_baseline_predicted_class)

    class_1_majority_baseline_precision = majority_baseline_metrics.precision[1]
    class_1_majority_baseline_recall = majority_baseline_metrics.recall[1]
    class_1_majority_baseline_f1 = majority_baseline_metrics.f1[1]
    class_1_majority_baseline_macro_f1 = majority_baseline_metrics.macro_f1

    rule_baseline_macro_f1 = rule_baseline_metrics.macro_f1

    numpy.testing.assert_array_equal(class_1_majority_baseline_precision,0)
    numpy.testing.assert_array_equal(class_1_majority_baseline_recall,0)
    numpy.testing.assert_array_equal(class_1_majority_baseline_f1,0)
    # Tolerance at 1e-4 because this was our printed value in our notebook.
    numpy.testing.assert_allclose(class_1_majority_baseline_macro_f1,numpy.array(0.4000),atol=1e-4,rtol=0) 
    numpy.testing.assert_allclose(rule_baseline_macro_f1,numpy.array(0.3333),atol=1e-4,rtol=0)

def test_library(validation_cases):
    _, case_labels, probabilities, classifier_predicted_class, rule_baseline_predicted_class, majority_baseline_predicted_class = validation_cases
    
    predicted_systems = [("classifier", classifier_predicted_class), ("rule baseline", rule_baseline_predicted_class), ("majority baseline", majority_baseline_predicted_class)]
    
    for name, prediction in predicted_systems:
        metrics = classification_metrics.generate_classification_metrics(case_labels, prediction)
        sklearn_metrics = sklearn.metrics.precision_recall_fscore_support(case_labels, prediction, labels=[0,1], zero_division=0)
        sklearn_macro_f1 = sklearn.metrics.f1_score(case_labels, prediction, labels=[0,1], average="macro", zero_division=0)
       
        # Tolerance is at 1e-12 because this is the equal at float64, only the rounding error reamins.
        numpy.testing.assert_allclose(metrics.precision,sklearn_metrics[0],atol=1e-12,rtol=0)
        numpy.testing.assert_allclose(metrics.recall,sklearn_metrics[1],atol=1e-12,rtol=0)
        numpy.testing.assert_allclose(metrics.f1,sklearn_metrics[2],atol=1e-12,rtol=0)
        numpy.testing.assert_array_equal(metrics.support,sklearn_metrics[3])
        numpy.testing.assert_allclose(metrics.macro_f1,sklearn_macro_f1,atol=1e-12,rtol=0)

def test_per_example_loss(validation_cases):
    cases, case_labels, probabilities, classifier_predicted_class, rule_baseline_predicted_class, majority_baseline_predicted_class = validation_cases
 
    log_loss, per_example_loss = classification_metrics.calculate_log_loss(case_labels, probabilities)

    f13a_index = cases.index("f13a")
    f13b_index = cases.index("f13b")
    f15b_index = cases.index("f15b")

    f13a_loss = per_example_loss[f13a_index]
    f13b_loss = per_example_loss[f13b_index]
    f15b_loss = per_example_loss[f15b_index]

    # Tolerance is at 1e-4 because this is how we presented the data in our workbook
    numpy.testing.assert_allclose(f13a_loss, 0.5358, atol=1e-4, rtol=0)
    numpy.testing.assert_allclose(f13b_loss, 0.7905, atol=1e-4, rtol=0)
    numpy.testing.assert_allclose(f15b_loss, 1.9036, atol=1e-4, rtol=0)
        
def test_log_loss(validation_cases):
    _, case_labels, probabilities, classifier_predicted_class, rule_baseline_predicted_class, majority_baseline_predicted_class = validation_cases

    log_loss, per_example_loss = classification_metrics.calculate_log_loss(case_labels, probabilities)

    sklearn_loss = sklearn.metrics.log_loss(case_labels, probabilities)

    # Tolerance is at 1e-4 for our data because it matches our workbook. 1e-12 for scikit compare because it is float64 equivalent, only rounding error remains.
    numpy.testing.assert_allclose(log_loss, 0.4513, atol=1e-4, rtol=0)
    numpy.testing.assert_allclose(log_loss, sklearn_loss, atol=1e-12, rtol=0)

def test_clip():
    labels = numpy.array([1, 0])
    probabilities = numpy.array([0.0, 0.0])

    log_loss, per_example_loss = classification_metrics.calculate_log_loss(labels, probabilities)
    sklearn_loss = sklearn.metrics.log_loss(labels, probabilities)

    # Tolerance is at 1e-4 for our data because it matches our workbook. 1e-12 for scikit compare because it is float64 equivalent, only rounding error remains.
    numpy.testing.assert_allclose(log_loss, 18.0218, atol=1e-4, rtol=0)
    numpy.testing.assert_allclose(log_loss, sklearn_loss, atol=1e-12, rtol=0)

    probabilities = numpy.array([1.0,1.0])

    log_loss, per_example_loss = classification_metrics.calculate_log_loss(labels, probabilities)
    sklearn_loss = sklearn.metrics.log_loss(labels, probabilities)

    # Tolerance is at 1e-4 for our data because it matches our workbook. 1e-12 for scikit compare because it is float64 equivalent, only rounding error remains.
    numpy.testing.assert_allclose(log_loss, 18.0218, atol=1e-4, rtol=0)
    numpy.testing.assert_allclose(log_loss, sklearn_loss, atol=1e-12, rtol=0)
