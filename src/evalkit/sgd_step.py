import torch

def compute_gradient(weight: torch.Tensor, bias: torch.Tensor, batch: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Gradient: Run the forward pass, the mean squared error loss, and the backward pass of a linear regression model on one batch, and return the loss and the gradients of the parameters.

    Take the weight and bias of a linear model, a batch of B examples with F features, and the targets of those examples. Compute the predictions as y_hat = batch @ weight + bias, the loss as the mean of (y_hat - targets)^2 over the B examples, and the gradients of that loss with respect to the weight and the bias. The mean divides by the batch size B, not the size of the full data set. The arguments are not changed.

    Parameters:
    -----
    weight: A float tensor of shape (F,) containing the weight parameter values.
    bias: A float tensor of shape () containing the bias parameter value.
    batch: A float tensor of shape (B, F) containing B examples of the batch, each with F features.
    targets: A float tensor of shape (B,) containing the targets of the batch inputs.

    Returns:
    -----
    loss: A float tensor of shape () containing the mean squared error of the model's predictions compared to the targets.
    weight_gradient: A float tensor of shape (F,) containing the gradient of the loss with respect to the weight.
    bias_gradient: A float tensor of shape () containing the gradient of the loss with respect to the bias.

    Raises:
    -----
    ValueError: If batch is not of shape (B, F), targets is not of shape (B,), or weight is not of shape (F,). For example, targets of shape (B, 1) would broadcast against predictions of shape (B,) to (B, B) without an error.
    """
    if batch.ndim != 2:
        raise ValueError(f"batch must be shape (B, F). batch: {batch.shape}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be shape (B, ). targets: {targets.shape}")
    if weight.ndim != 1:
        raise ValueError(f"weight must be shape (F, ). weight: {weight.shape}")
    if batch.shape[0] != targets.shape[0]:
        raise ValueError(f"batch and targets must have the same number of examples B. batch: {batch.shape}, targets: {targets.shape}")
    if batch.shape[1] != weight.shape[0]:
        raise ValueError(f"batch and weight must have the same number of features F. batch: {batch.shape}, weight: {weight.shape}")

    y_hat = forward_pass(weight, bias, batch)
    errors = y_hat - targets
    loss = mse_loss(y_hat, targets)
    weight_gradient, bias_gradient = backward(batch, errors)

    return loss, weight_gradient, bias_gradient

def forward_pass(weight: torch.Tensor, bias: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Forward Pass: Compute the predictions of a linear regression model for a batch of examples.

    Take the weight and bias of a linear model and a batch of B examples with F features, and return the predictions as y_hat = batch @ weight + bias.

    Parameters:
    -----
    weight: A float tensor of shape (F,) containing the weight parameter values.
    bias: A float tensor of shape () containing the bias parameter value.
    batch: A float tensor of shape (B, F) containing B examples of the batch, each with F features.

    Returns:
    -----
    y_hat: A float tensor of shape (B,) containing the output values from the model.
    """
    y_hat = []
    for item in batch:
        y_hat.append(item @ weight + bias)
    
    y_hat = torch.tensor(y_hat)
    return y_hat

def mse_loss(y_hat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MSE Loss: Compute the mean squared error of a model's predictions compared to the targets.

    Take the predictions and the targets of a batch of B examples, and return the loss as the mean of (y_hat - targets)^2 over the B examples.

    Parameters:
    -----
    y_hat: A float tensor of shape (B,) containing the output values from the model.
    targets: A float tensor of shape (B,) containing the targets of the batch inputs.

    Returns:
    -----
    loss: A float tensor of shape () containing the mean squared error of the model's predictions compared to the targets.
    """
    squared_error = 0.0
    for idx, item in enumerate(y_hat):
        squared_error += (y_hat[idx] - targets[idx])**2
    loss = squared_error/len(y_hat)

    return loss 

def backward(batch: torch.Tensor, error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward: Compute the gradients of the mean squared error loss with respect to the weight and bias of a linear regression model.

    Take a batch of B examples with F features and the error of each prediction, y_hat - targets, and return the gradient of the mean squared error loss with respect to the weight and the bias: weight_gradient = (2 / B) * batch^T @ error and bias_gradient = (2 / B) * sum(error). The gradients need the per-example errors, not the averaged loss, because averaging discards them.

    Parameters:
    -----
    batch: A float tensor of shape (B, F) containing B examples of the batch, each with F features.
    error: A float tensor of shape (B,) containing the error of each prediction, y_hat - targets.

    Returns:
    -----
    weight_gradient: A float tensor of shape (F,) containing the gradient of the loss with respect to the weight.
    bias_gradient: A float tensor of shape () containing the gradient of the loss with respect to the bias.
    """
    B, F = batch.shape 

    weight_gradient = (2 / B) * (batch.T @ error)
    bias_gradient = (2 / B) * error.sum()

    return weight_gradient, bias_gradient

def sgd_step(parameters: list[torch.Tensor], gradients: list[torch.Tensor], lr: float) -> list[torch.Tensor]:
    """
    SGD Step: Compute one stochastic gradient descent update of a list of parameters from their gradients and a learning rate.

    Take a list of parameter tensors of any shapes, a list of their gradients in the same order, and the learning rate, and return a new list of updated parameters, each computed as p - lr * g. It knows nothing about the model. The arguments are not changed, and each result keeps the shape and dtype of its parameter.

    Parameters:
    -----
    parameters: A list of float tensors, each of any shape, holding the model's parameters.
    gradients: A list of float tensors, the same length and order as parameters, each the same shape as its parameter, from our compute_gradient function.
    lr: Float. The scalar learning rate value for the step.

    Returns:
    -----
    updated_parameters: A list of float tensors, the same length, order and shapes as parameters, each equal to p - lr * g after one step.

    Raises:
    -----
    ValueError: If parameters and gradients have different lengths, or a parameter and its gradient have different shapes.
    """
    if len(parameters) != len(gradients):
        raise ValueError(f"parameters and gradients must be the same length. parameters: {len(parameters)}, gradients: {len(gradients)}")
    for idx, (p, g) in enumerate(zip(parameters, gradients)):
        if p.shape != g.shape:
            raise ValueError(f"parameter and gradient {idx} must be the same shape. parameter: {p.shape}, gradient: {g.shape}")

    updated_parameters = [p - lr * g for p, g in zip(parameters, gradients)]
    return updated_parameters
