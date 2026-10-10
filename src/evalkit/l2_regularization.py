import torch
import numpy
from numpy.typing import ArrayLike


def l2_step(parameter: torch.Tensor, gradient: torch.Tensor, learning_rate: float, weight_decay: float) -> torch.Tensor:
    """
    L2 Step: Compute a single step of a parameter update, taking in a parameter, the gradient of that parameter, the learning rate constant, and the weight decay constant

    Take a parameter tensor of any shape, the gradient of the data loss with respect to the parameter, the learning rate and weight decay, and return the updated parameter after one optimizer step with the same shape and dtype, computed as parameter - lr * (gradient + weight_decay * parameter).

    Parameters:
    ----- 
    parameter: A tensor of any shape of the parameter of the model
    gradient: A tensor of any shape (same shape as parameter) of the gradient of the data loss with respect to the parameter
    learning_rate: Float. A scalar learning rate to apply to scale the parameter deltas.
    weight_decay: Float. The weight decay to apply to the L2 penalty calculation to scale the parameter deltas.
    
    Returns:
    ----- 
    updated_parameters: A tensor of any shape (same shape as the parameter in the input) holding the updated values of the same parameter provided as an argument.
    """
    if parameter.shape != gradient.shape:
        raise ValueError(f"parameter and gradient must be the same shape. parameter: {parameter.shape}, gradient: {gradient.shape}")
        
    l2_penalty, l2_penalty_gradient = calculate_l2_penalty(parameter, weight_decay)
    parameter_delta = generate_parameter_delta(gradient, learning_rate, l2_penalty_gradient)

    updated_parameters = parameter + parameter_delta
    
    return updated_parameters

def calculate_l2_penalty(parameter: torch.Tensor, weight_decay: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate L2 Penalty: Compute a single L2 penalty and L2 Penalty Gradient for a given parameter tensor and weight decay hyperparameter

    Take a parameter tensor of any shape and a weight decay hyperparameter and compute the l2 penalty as (weight_decay / 2) * parameter^2 and the l2 penalty gradient as weight_decay * parameter. Return both values.

    Parameters:
    ----- 
    parameter: A tensor of any shape of the parameter of the model
    weight_decay: Float. The weight decay value to apply to the L2 penalty calculation to scale the parameter deltas. 
    
    Returns:
    ----- 
    l2_penalty: A tensor of any shape (same shape as the parameter input) of the L2 penalties for reporting. 
    l2_penalty_gradient: A tensor of any shape (same shape as the parameter input) of the calculation result of the derivative of the L2 function to apply to the parameter deltas.
    """
    
    l2_penalty = (weight_decay / 2) * parameter**2
    l2_penalty_gradient = weight_decay * parameter

    return l2_penalty, l2_penalty_gradient

def generate_parameter_delta(gradient: torch.Tensor, learning_rate: float, l2_penalty_gradient:torch.Tensor) -> torch.Tensor:
    """
    Generate Parameter Delta: Compute a single delta to apply to the parameter given the gradient of the data loss with respect to the parameter, learning_rate, and l2_penalty_gradient.

    Take a given gradient of the data loss with respect to the parameter, learning_rate hyperparameter, and l2_penalty_gradient, and compute the parameter delta as -lr * (gradient + weight_decay * parameter). 

    Parameters:
    ----- 
    gradient: a tensor of any shape (same shape as parameter) of the gradient of the data loss with respect to the parameter
    learning_rate: float. a scalar learning rate to apply to scale the parameter deltas.
    l2_penalty_gradient:  a tensor of any shape (same shape as the parameter) of the l2 derivative calculation results which we apply to the parameter delta. 
    
    Returns:
    ----- 
    parameter_delta: a tensor of any shape (same shape as the parameter input) of the delta to add to the parameter value. 
    """
    if gradient.shape != l2_penalty_gradient.shape:
        raise ValueError(f"gradient and l2_penalty_gradient must be the same shape. gradient: {gradient.shape}, l2_penalty_gradient: {l2_penalty_gradient.shape}")

    parameter_delta = -learning_rate * (gradient + l2_penalty_gradient)

    return parameter_delta 

def select_on_validation(hyperparameters: ArrayLike, validation_loss: ArrayLike) -> tuple[numpy.float64, numpy.int64, numpy.float64]:
    """
    Select On Validation: Return the selected hyperparameter, its index, and the associated validation loss from an analysis of an array model performance outputs on validation sets using an array of the hyperparameters used and the associated validation losses realized.

    Take in a numpy array of hyperparameters, a numpy array of validation losses (same shape), and select the hyperparameter value that returned the lowest validation loss. Return the hyperparameter, its index, and the associated validation loss.
    For the loss, lower is better. When there is a tie, we return the first of the tied entries.

    Parameters:
    ----- 
    hyperparameters: A numpy float64 array of shape (K, ) of the hyperparameter value candidates from the validation dataset runs to select from.
    validation_loss: A numpy float64 array of shape (K, ) of the validation loss from the validation dataset runs with the applied hyperparameter values. 
    
    Returns:
    ----- 
    selected_hyperparameter: A numpy float64 scalar. The selected hyperparameter value from our selection over the input data.
    hyperparameter_index: A numpy int64 scalar. The index of the selected_hyperparameter from the hyperparameters input tensor.
    selected_validation_loss: A numpy float 64 scalar. The validation loss of the selected_hyperparameter from the hyperparameters input tensor.
    """
    if hyperparameters.shape != validation_loss.shape:
        raise ValueError(f"hyperparameters and validation_loss must match shapes. hyperparameters: {hyperparameters.shape}, validation_loss: {validation_loss.shape}")
    if not (hyperparameters.ndim == 1 and validation_loss.ndim == 1):
        raise ValueError(f"hyperparameters and validation_loss must both be shape (K, ). hyperparameters: {hyperparameters.shape}, validation_loss: {validation_loss.shape}")

    hyperparameter_index = numpy.argmin(validation_loss)
    selected_hyperparameter = hyperparameters[hyperparameter_index]
    selected_validation_loss = validation_loss[hyperparameter_index]
    return selected_hyperparameter, hyperparameter_index, selected_validation_loss


