import torch
from evalkit.binary_cross_entropy import forward_pass
from evalkit.softmax import softmax

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cross-Entropy: Compute the softmax cross-entropy loss of N examples over C classes from their logits, and its gradient with respect to the logits.

    Take the logits and the target class index of N examples. Return the per-example losses l = m + ln(sum_k e^(z_k - m)) - z_t, where m is the max logit of the row and t is its target, the loss L = (1 / N) * sum(l), and the gradient dL/dz = (softmax(z) - onehot(t)) / N. This is the stable form of -ln(p_t) written in the logits: every exp argument is at or below 0, so nothing overflows, and it never takes ln(0), so it stays finite in float32 where p_t rounds to exactly 0.0. It knows nothing about the weight or bias, so any model that produces logits can call it. The arguments are not changed.

    Parameters:
    -----
    logits: A float tensor of shape (N, C) containing the logit of each class for each example.
    targets: An int64 tensor of shape (N,), in the same order as logits, containing the target class index of each example, in [0, C).

    Returns:
    -----
    loss: A float tensor of shape () containing the mean cross-entropy over the N examples.
    per_example_losses: A float tensor of shape (N,), in the same order as logits, containing the loss l of each example.
    logit_gradients: A float tensor of shape (N, C), the same shape as logits, containing the gradient of the loss with respect to each logit. Each row sums to 0.

    Raises:
    -----
    ValueError: If logits is not of shape (N, C), targets is not of shape (N,), or targets is not int64. For example, float targets cannot index a class.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be shape (N, C). logits: {logits.shape}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be shape (N, ). targets: {targets.shape}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"logits and targets must have the same number of examples N. logits: {logits.shape}, targets: {targets.shape}")
    if targets.dtype != torch.int64:
        raise ValueError(f"targets must be int64 class indices. targets: {targets.dtype}")

    N, C = logits.shape

    # Shift by the row max m so every exp argument is <= 0: ln(sum e^z) = m + ln(sum e^(z - m))
    m = torch.amax(logits, dim=-1, keepdim=True)
    log_sum_exp = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    per_example_losses = log_sum_exp - target_logits
    loss = per_example_losses.mean()

    # d(mean loss)/dz = (p - onehot(t)) / N
    one_hot = torch.zeros_like(logits)
    one_hot[torch.arange(N), targets] = 1.0
    logit_gradients = (softmax(logits, dim=-1) - one_hot) / N
    return loss, per_example_losses, logit_gradients

def classifier_gradient(weight: torch.Tensor, bias: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Classifier Gradient: Run the forward pass, the cross-entropy loss, and the backward pass of a linear softmax classifier on one batch, and return the loss and the gradients of the parameters.

    Take the weight and bias of a linear classifier over C classes, a batch of N examples with F features, and the target class of each example. Compute the logits as Z = inputs @ weight + bias, the loss and dL/dZ from cross_entropy, and the gradients by the chain rule: weight_gradient = inputs^T @ dL/dZ and bias_gradient = the sum of dL/dZ over the N examples. The mean divides by the batch size N. The arguments are not changed.

    Parameters:
    -----
    weight: A float tensor of shape (F, C) containing the weight parameter values.
    bias: A float tensor of shape (C,) containing the bias of each class.
    inputs: A float tensor of shape (N, F) containing N examples of the batch, each with F features.
    targets: An int64 tensor of shape (N,), in the same order as inputs, containing the target class index of each example, in [0, C).

    Returns:
    -----
    loss: A float tensor of shape () containing the mean cross-entropy of the model's logits compared to the targets.
    weight_gradient: A float tensor of shape (F, C) containing the gradient of the loss with respect to the weight.
    bias_gradient: A float tensor of shape (C,) containing the gradient of the loss with respect to the bias.

    Raises:
    -----
    ValueError: If inputs is not of shape (N, F) or weight is not of shape (F, C), and from cross_entropy, if the logits (N, C) and targets do not match.
    """
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be shape (N, F). inputs: {inputs.shape}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shape (F, C). weight: {weight.shape}")

    logits = forward_pass(weight, bias, inputs)
    loss, _, logit_gradients = cross_entropy(logits, targets)
    # Chain rule through Z = inputs @ weight + bias
    weight_gradient = inputs.T @ logit_gradients
    bias_gradient = logit_gradients.sum(dim=0)
    return loss, weight_gradient, bias_gradient

def scorer_gradient(weight: torch.Tensor, bias: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Scorer Gradient: Run the forward pass, the cross-entropy loss, and the backward pass of a linear candidate scorer on one batch, and return the loss and the gradients of the parameters.

    Take the weight and bias of a linear scorer, a batch of N queries with C candidates of F features each, and the index of the correct candidate of each query. One weight scores every candidate: the logits are Z = inputs @ weight + bias, shape (N, C), so the candidates of a query compete through the softmax. Compute the loss and dL/dZ from cross_entropy, and the gradients by the chain rule: weight_gradient = the sum of dL/dZ[n, c] * inputs[n, c] over both the N queries and the C candidates, and bias_gradient = the sum of every entry of dL/dZ. Each row of dL/dZ sums to 0, so bias_gradient is 0 up to rounding: a bias shared by all candidates cannot change which one wins. The mean divides by the number of queries N, not N * C. The arguments are not changed.

    Parameters:
    -----
    weight: A float tensor of shape (F,) containing the weight parameter values.
    bias: A float tensor of shape () containing the bias parameter value.
    inputs: A float tensor of shape (N, C, F) containing N queries of the batch, each with C candidates of F features.
    targets: An int64 tensor of shape (N,), in the same order as inputs, containing the index of the correct candidate of each query, in [0, C).

    Returns:
    -----
    loss: A float tensor of shape () containing the mean cross-entropy of the model's logits compared to the targets.
    weight_gradient: A float tensor of shape (F,) containing the gradient of the loss with respect to the weight.
    bias_gradient: A float tensor of shape () containing the gradient of the loss with respect to the bias.

    Raises:
    -----
    ValueError: If inputs is not of shape (N, C, F) or weight is not of shape (F,), and from cross_entropy, if the logits (N, C) and targets do not match.
    """
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shape (N, C, F). inputs: {inputs.shape}")
    if weight.ndim != 1:
        raise ValueError(f"weight must be shape (F, ). weight: {weight.shape}")

    logits = forward_pass(weight, bias, inputs)
    loss, _, logit_gradients = cross_entropy(logits, targets)
    # Chain rule through Z[n, c] = inputs[n, c] @ weight + bias: sum over queries and candidates
    weight_gradient = (logit_gradients.unsqueeze(-1) * inputs).sum(dim=(0, 1))
    bias_gradient = logit_gradients.sum()
    return loss, weight_gradient, bias_gradient
