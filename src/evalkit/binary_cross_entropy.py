import torch

def forward_pass(weight: torch.Tensor, bias: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Forward Pass: Compute the logits of a logistic regression model for a batch of examples.

    Take the weight and bias of a logistic regression model and a batch of B examples with F features, and return the logits as z = inputs @ weight + bias. It returns logits, not probabilities or predicted classes: pass the logits to sigmoid for probabilities, or to binary_cross_entropy for the loss.

    Parameters:
    -----
    weight: A float tensor of shape (F,) containing the weight parameter values.
    bias: A float tensor of shape () containing the bias parameter value.
    inputs: A float tensor of shape (B, F) containing B examples of the batch, each with F features.

    Returns:
    -----
    logits: A float tensor of shape (B,) containing the logit z of each example.
    """
    logits = inputs @ weight + bias
    return logits

def sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid: Compute the probability of class 1 from each logit.

    Take logits of any shape and return sigma(z) = 1 / (1 + e^(-z)) for each entry. For z < 0 it uses the equal form e^z / (1 + e^z), so every exp argument is at or below 0 and nothing overflows: in float32, e^(-z) for z = -100 would be inf. Both forms are written with e^(-|z|).

    Parameters:
    -----
    logits: A float tensor of any shape containing the logits z.

    Returns:
    -----
    probabilities: A float tensor of the same shape as logits containing sigma(z), each in [0, 1].
    """
    # exp(-|x|) is always in (0, 1], so neither branch can overflow
    exp_neg_abs = torch.exp(-torch.abs(logits))
    probabilities = torch.where(logits >= 0, 1 / (1 + exp_neg_abs), exp_neg_abs / (1 + exp_neg_abs))
    return probabilities

def binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Binary Cross-Entropy: Compute the binary cross-entropy loss of N examples from their logits, and its gradient with respect to the logits.

    Take the logits and the 0/1 targets of N examples. Return the per-example losses l = max(z, 0) - y * z + ln(1 + e^(-|z|)), the loss L = (1 / N) * sum(l), and the gradient dL/dz = (sigma(z) - y) / N. This is the stable form of -[y * ln(p) + (1 - y) * ln(1 - p)] written in the logit: it never takes ln(0) or exp of a positive number, so it stays finite in float32 where sigma(z) rounds to exactly 1.0. It knows nothing about the weight or bias, so any model that produces logits can call it. The arguments are not changed.

    Parameters:
    -----
    logits: A float tensor of shape (N,) containing the logit z of each example.
    targets: A float tensor of shape (N,), in the same order as logits, containing the target class of each example, 0 or 1.

    Returns:
    -----
    loss: A float tensor of shape () containing the mean binary cross-entropy over the N examples.
    per_example_losses: A float tensor of shape (N,), in the same order as logits, containing the loss l of each example.
    logit_gradients: A float tensor of shape (N,), in the same order as logits, containing the gradient of the loss with respect to each logit.

    Raises:
    -----
    ValueError: If logits and targets have different shapes. For example, targets of shape (N, 1) would broadcast against logits of shape (N,) to (N, N) without an error.
    """
    # A shape mismatch would broadcast silently, e.g. (B,) against (B, 1) gives (B, B)
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have the same shape. logits: {logits.shape}, targets: {targets.shape}")

    # Stable form of -[y*log(p) + (1-y)*log(1-p)]: never takes log of 0 or exp of a large positive number
    per_example_losses = torch.clamp(logits, min=0) - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))
    loss = per_example_losses.mean()
    # d(mean loss)/dz = (p - y) / N
    logit_gradients = (sigmoid(logits) - targets) / logits.numel()
    return loss, per_example_losses, logit_gradients

def logistic_regression_gradient(weight: torch.Tensor, bias: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Logistic-Regression Gradient: Run the forward pass, the binary cross-entropy loss, and the backward pass of a logistic regression model on one batch, and return the loss and the gradients of the parameters.

    Take the weight and bias of a logistic regression model, a batch of B examples with F features, and the 0/1 targets of those examples. Compute the logits as z = inputs @ weight + bias, the loss and dL/dz from binary_cross_entropy, and the gradients by the chain rule: weight_gradient = inputs^T @ dL/dz and bias_gradient = sum(dL/dz). The mean divides by the batch size B. The arguments are not changed.

    Parameters:
    -----
    weight: A float tensor of shape (F,) containing the weight parameter values.
    bias: A float tensor of shape () containing the bias parameter value.
    inputs: A float tensor of shape (B, F) containing B examples of the batch, each with F features.
    targets: A float tensor of shape (B,), in the same order as inputs, containing the target class of each example, 0 or 1.

    Returns:
    -----
    loss: A float tensor of shape () containing the mean binary cross-entropy of the model's logits compared to the targets.
    weight_gradient: A float tensor of shape (F,) containing the gradient of the loss with respect to the weight.
    bias_gradient: A float tensor of shape () containing the gradient of the loss with respect to the bias.

    Raises:
    -----
    ValueError: From binary_cross_entropy, if the logits (B,) and targets have different shapes.
    """
    logits = forward_pass(weight, bias, inputs)
    loss, _, logit_gradients = binary_cross_entropy(logits, targets)
    # Chain rule through z = inputs @ weight + bias
    weight_gradient = inputs.T @ logit_gradients
    bias_gradient = logit_gradients.sum()
    return loss, weight_gradient, bias_gradient
