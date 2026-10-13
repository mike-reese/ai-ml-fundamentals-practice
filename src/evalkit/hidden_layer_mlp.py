import torch
from evalkit.binary_cross_entropy import binary_cross_entropy

def mlp_forward_pass(W_hidden: torch.Tensor, b_hidden: torch.Tensor, w_out: torch.Tensor, b_out: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MLP Forward Pass: Compute the hidden pre-activations, the hidden activations and the logits of a one-hidden-layer MLP with a ReLU and one output logit.

    Take the parameters of the hidden layer and the output layer and a batch of N examples with F features. Compute hidden_pre = inputs @ W_hidden + b_hidden, hidden = max(0, hidden_pre) entry by entry, and z = hidden @ w_out + b_out. It returns logits, not probabilities: pass them to binary_cross_entropy for the loss. Without the ReLU the two layers would collapse into one straight line in the inputs. The arguments are not changed.

    Parameters:
    -----
    W_hidden: A float tensor of shape (F, H) containing the hidden layer's weight, one column per hidden unit.
    b_hidden: A float tensor of shape (H,) containing the bias of each hidden unit.
    w_out: A float tensor of shape (H,) containing the output layer's weight.
    b_out: A float tensor of shape () containing the output layer's bias.
    inputs: A float tensor of shape (N, F) containing N examples of the batch, each with F features.

    Returns:
    -----
    hidden_pre: A float tensor of shape (N, H) containing the pre-activation of each hidden unit for each example.
    hidden: A float tensor of shape (N, H) containing max(0, hidden_pre).
    logits: A float tensor of shape (N,) containing the logit z of each example.

    Raises:
    -----
    ValueError: If inputs is not of shape (N, F), W_hidden is not of shape (F, H), or b_hidden and w_out are not of shape (H,). For example, w_out of shape (H, 1) would give logits of shape (N, 1), which broadcast against targets (N,) to (N, N).
    """
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be shape (N, F). inputs: {inputs.shape}")
    if W_hidden.ndim != 2:
        raise ValueError(f"W_hidden must be shape (F, H). W_hidden: {W_hidden.shape}")
    if inputs.shape[1] != W_hidden.shape[0]:
        raise ValueError(f"inputs and W_hidden must have the same number of features F. inputs: {inputs.shape}, W_hidden: {W_hidden.shape}")
    H = W_hidden.shape[1]
    if b_hidden.shape != (H,):
        raise ValueError(f"b_hidden must be shape (H, ). b_hidden: {b_hidden.shape}, H: {H}")
    if w_out.shape != (H,):
        raise ValueError(f"w_out must be shape (H, ). w_out: {w_out.shape}, H: {H}")

    hidden_pre = inputs @ W_hidden + b_hidden
    hidden = torch.clamp(hidden_pre, min=0)
    logits = hidden @ w_out + b_out
    return hidden_pre, hidden, logits

def mlp_backward_pass(W_hidden: torch.Tensor, b_hidden: torch.Tensor, w_out: torch.Tensor, b_out: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MLP Backward Pass: Run the forward pass, the binary cross-entropy loss, and the backward pass of a one-hidden-layer MLP on one batch, and return the loss, the gradient at the hidden pre-activations and the gradients of the four parameters.

    Take the MLP's parameters, a batch of N examples with F features, and the 0/1 targets of those examples. Get the logits from mlp_forward_pass and the loss and delta = dL/dz from binary_cross_entropy. Then apply the chain rule right to left: dL/db_out = sum(delta), dL/dw_out = hidden^T @ delta, dL/dhidden = delta[:, None] * w_out, dL/dhidden_pre = dL/dhidden * (hidden_pre > 0), dL/db_hidden = the sum of dL/dhidden_pre over the N examples, and dL/dW_hidden = inputs^T @ dL/dhidden_pre. The mask is 0 where the pre-activation is at or below 0, as in PyTorch's ReLU backward. The mean divides by the batch size N. The arguments are not changed.

    Parameters:
    -----
    W_hidden: A float tensor of shape (F, H) containing the hidden layer's weight, one column per hidden unit.
    b_hidden: A float tensor of shape (H,) containing the bias of each hidden unit.
    w_out: A float tensor of shape (H,) containing the output layer's weight.
    b_out: A float tensor of shape () containing the output layer's bias.
    inputs: A float tensor of shape (N, F) containing N examples of the batch, each with F features.
    targets: A float tensor of shape (N,), in the same order as inputs, containing the target class of each example, 0 or 1.

    Returns:
    -----
    loss: A float tensor of shape () containing the mean binary cross-entropy of the model's logits compared to the targets.
    hidden_pre_gradient: A float tensor of shape (N, H) containing the gradient of the loss with respect to each hidden pre-activation.
    W_hidden_gradient: A float tensor of shape (F, H) containing the gradient of the loss with respect to W_hidden.
    b_hidden_gradient: A float tensor of shape (H,) containing the gradient of the loss with respect to b_hidden.
    w_out_gradient: A float tensor of shape (H,) containing the gradient of the loss with respect to w_out.
    b_out_gradient: A float tensor of shape () containing the gradient of the loss with respect to b_out.

    Raises:
    -----
    ValueError: From mlp_forward_pass, if a parameter or the inputs have the wrong shape, and from binary_cross_entropy, if the logits (N,) and targets have different shapes.
    """
    hidden_pre, hidden, logits = mlp_forward_pass(W_hidden, b_hidden, w_out, b_out, inputs)
    loss, _, logit_gradients = binary_cross_entropy(logits, targets)

    # Output layer: z = hidden @ w_out + b_out
    b_out_gradient = logit_gradients.sum()
    w_out_gradient = hidden.T @ logit_gradients
    hidden_gradient = logit_gradients[:, None] * w_out

    # ReLU: the derivative is 1 where hidden_pre > 0 and 0 at or below 0
    hidden_pre_gradient = hidden_gradient * (hidden_pre > 0).to(hidden_gradient.dtype)

    # Hidden layer: hidden_pre = inputs @ W_hidden + b_hidden
    b_hidden_gradient = hidden_pre_gradient.sum(dim=0)
    W_hidden_gradient = inputs.T @ hidden_pre_gradient
    return loss, hidden_pre_gradient, W_hidden_gradient, b_hidden_gradient, w_out_gradient, b_out_gradient
