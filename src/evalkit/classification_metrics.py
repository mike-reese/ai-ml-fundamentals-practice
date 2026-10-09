import numpy
from numpy.typing import ArrayLike
from dataclasses import dataclass

@dataclass(frozen=True)
class ClassificationMetrics:
    confusion_matrix: numpy.ndarray
    precision: numpy.ndarray 
    recall: numpy.ndarray 
    f1: numpy.ndarray 
    support: numpy.ndarray 
    macro_f1: numpy.float64 


def generate_classification_metrics(case_labels: ArrayLike, predicted_classes: ArrayLike) -> ClassificationMetrics:
    """
    Generate Classification Metrics: We generate the classification metrics (confusion_matrix, precision, recall, f1, support, macro_f1) for our known case labels and predicted classes.

    Take in an array of known case labels, an array of predicted classes, generate the confusion matrix, and use that confusion matrix to calculate our necessary classification metrics to analyze model performance.
    Zero Division is handled with numpy.divide and a where clause. Returns 0 for any division calculation where the denominator would be 0.
   
    Parameters:
    ----- 
    case_labels: A numpy integer array of the known correct case labels from a set of cases
    predicted_classes: A numpy integer array of the predicted classes of labels from a set of cases.
    
    Returns:
    ----- 
     ClassificationMetrics: A dataclass containing the following metrics from the function call:
        confusion_matrix: A numpy integer array of shape (2, 2) in scikit-learn's layout. row = label (0, then 1), column = predicted class (0, then 1)
        precision: A numpy float array of shape (2, ) containing the precision calculation for each classification of our analysis dataset 
        recall: A numpy float array of shape (2, ) containing the recall calculation for each classification of our analysis dataset
        f1: A numpy float array of shape (2, ) containing the f1 calculation for each classification of our analysis dataset
        support: A numpy integer array of shape (2, ) containing the support calculation for each classification of our analysis dataset
        macro_f1: A float, scalar, that is the mean of the f1 scores of each classification of our analysis dataset 
    """
    if case_labels.shape != predicted_classes.shape:
        raise ValueError(f"Arrays must be the same shape. Current shapes: case_labels: {case_labels.shape}, predicted_classes: {predicted_classes.shape}")
    if case_labels.ndim != 1 or predicted_classes.ndim != 1:
        raise ValueError(f"Arrays must be shape (N, ). Current shapes: case_labels: {case_labels.shape}, predicted_classes: {predicted_classes.shape}")
    if not numpy.isin(case_labels, [0,1]).all() or not numpy.isin(predicted_classes, [0,1]).all():
        raise ValueError(f"Classification labels must be [0,1] only. case_labels: {case_labels}, predicted_classes: {predicted_classes}")

    confusion_matrix = generate_confusion_matrix(case_labels, predicted_classes)

    # Gives us the matching label vs predicted_classes values - our numerators 
    numerators = numpy.diag(confusion_matrix)

    precision_denominator = confusion_matrix.sum(axis=0)
    recall_denominator = confusion_matrix.sum(axis=1)

    precision = numpy.divide(numerators, precision_denominator, out=numpy.zeros(precision_denominator.shape), where=(precision_denominator != 0))
    recall = numpy.divide(numerators, recall_denominator, out=numpy.zeros(recall_denominator.shape), where=(recall_denominator !=0))
    
    f1_numerator = 2 * precision * recall 
    f1_denominator = precision + recall 
    f1 = numpy.divide(f1_numerator, f1_denominator, out=numpy.zeros(f1_denominator.shape), where=(f1_denominator != 0))

    # recall_denominator and support are both row sums 
    support = recall_denominator
   
    macro_f1 = f1.mean()

    metrics = ClassificationMetrics(
        confusion_matrix=confusion_matrix,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        macro_f1=macro_f1
    )
    return metrics

def generate_confusion_matrix(case_labels: ArrayLike, predicted_classes: ArrayLike) -> numpy.ndarray:
    """
    Generate Confusion Matrix: We generate a confusion matrix for an array of known case labels (classes) and predicted classes

    We take in two arrays of equal shape and generate the row and column level occurences of the matching values for what the true class was compared to what the predicted class was
    
    Parameters:
    ----- 
    case_labels: A numpy integer array of the known correct case labels from a set of cases. Shape (N, ) and values of 0 or 1
    predicted_classes: A numpy integer array of the predicted classes of labels from a set of cases. Shape (N, ) and values of 0 or 1
    
    Returns:
    ----- 
    confusion_matrix: A numpy integer array of shape (2, 2) in scikit-learn's layout. row = label (0, then 1), column = predicted class (0, then 1)
    """
    if case_labels.shape != predicted_classes.shape:
        raise ValueError(f"Arrays must be the same shape. Current shapes: case_labels: {case_labels.shape}, predicted_classes: {predicted_classes.shape}")
    if case_labels.ndim != 1 or predicted_classes.ndim != 1:
        raise ValueError(f"Arrays must be shape (N, ). Current shapes: case_labels: {case_labels.shape}, predicted_classes: {predicted_classes.shape}")
    if not numpy.isin(case_labels, [0,1]).all() or not numpy.isin(predicted_classes, [0,1]).all():
        raise ValueError(f"Classification labels must be [0,1] only. case_labels: {case_labels}, predicted_classes: {predicted_classes}")


    labels = [0,1]
    classes = numpy.unique(labels)
    c = len(classes)
    confusion_matrix = numpy.zeros([c,c], dtype=numpy.int64)
    
    numpy.add.at(confusion_matrix, (case_labels, predicted_classes), 1)
  
    return confusion_matrix

def calculate_log_loss(case_labels: ArrayLike, probabilities: ArrayLike) -> tuple[numpy.float64, numpy.ndarray]:
    """
    Calculate Log Loss: We calculate the log loss and per-example log loss for a given set of known case labels (classes) and probabilities from our classifier

    We take in two arrays of equal length, clip the probabilities to prevent float64 overflow and underflow when we take the -ln of it, calculate the p_target, and then take -ln(p_target) as the per_example_loss and the average of that per_example_loss as the log_loss

    Parameters:
    ----- 
    case_labels: An array of the known correct case labels from a set of cases, shape (N, )
    probabilities: An array of the probabilities of label 1, shape (N, )
    
    Returns:
    ----- 
    log_loss: Float, the mean over the cases of -ln(p_target), where p_target is p when the label is 1 and 1-p when the label is 0. Probabilities are clipped to [eps, 1 - eps], eps as the dtype's machine epsilon, before the log.
    per_example_loss: A numpy float array of the per-example loss for each case

    """
    if case_labels.shape != probabilities.shape:
        raise ValueError(f"Arrays must be the same shape. Current shapes: case_labels: {case_labels.shape}, probabilities: {probabilities.shape}")
    if case_labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError(f"Arrays must be shape (N, ). Current shapes: case_labels: {case_labels.shape}, probabilities: {probabilities.shape}")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"Probabilities must be between 0 and 1")
    if probabilities.dtype != float:
        raise ValueError(f"Probabilities must be a float. Probabilities Type: {probabilities.dtype}")
    if not numpy.isin(case_labels, [0,1]).all():
        raise ValueError(f"Classification labels must be [0,1] only. case_labels: {case_labels}")

    eps = numpy.finfo(probabilities.dtype).eps
    clipped_probabilities = numpy.clip(probabilities, eps, 1-eps)

    # p_target is p when label ==1 , 1-p when label == 0
    p_target = numpy.where(case_labels == 1, clipped_probabilities, 1 - clipped_probabilities)

    # Log Loss
    per_example_loss = -numpy.log(p_target)
    log_loss = per_example_loss.mean()

    return log_loss, per_example_loss

