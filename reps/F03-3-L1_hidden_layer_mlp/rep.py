import evalkit.hidden_layer_mlp as hidden_layer_mlp
import evalkit.binary_cross_entropy as binary_cross_entropy
import evalkit.sgd_step as sgd_step
import torch
import json
from pathlib import Path

def mlp_fit():
    inputs = torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0], dtype=torch.float64)
    W_hidden = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    b_hidden = torch.tensor([-1.25, 3.25], dtype=torch.float64)
    w_out = torch.tensor([0.5, -1.0], dtype=torch.float64)
    b_out = torch.tensor(1.0, dtype=torch.float64)
    lr = 0.2
    steps = 1000

    return inputs, targets, [W_hidden, b_hidden, w_out, b_out], lr, steps

def baseline_fit():
    weight = torch.tensor([0.5], dtype=torch.float64)
    bias = torch.tensor(0.0, dtype=torch.float64)
    lr = 0.5
    steps = 500

    return weight, bias, lr, steps

def hazard_case():
    inputs = torch.tensor([[1.25], [2.0], [0.0]], dtype=torch.float64)
    targets = torch.tensor([1, 0, 1], dtype=torch.float64)
    W_hidden = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    b_hidden = torch.tensor([-1.25, -3.0], dtype=torch.float64)
    w_out = torch.tensor([0.5, -1.0], dtype=torch.float64)
    b_out = torch.tensor(1.0, dtype=torch.float64)

    return inputs, targets, [W_hidden, b_hidden, w_out, b_out]

def accuracy(logits, targets):
    """Fraction of examples where the predicted class (p > 0.5) equals the target."""
    probabilities = binary_cross_entropy.sigmoid(logits)
    return ((probabilities > 0.5).to(targets.dtype) == targets).to(targets.dtype).mean().item()

def rounded(values, digits=10):
    """Round every entry of a list (or a list of lists) for printing."""
    if isinstance(values, list):
        return [rounded(v, digits) for v in values]
    return round(values, digits)

def pair_text(values, digits=10):
    """Format a list of numbers as [a, b] with a fixed number of decimals."""
    return "[" + ", ".join(f"{v:.{digits}f}" for v in values) + "]"

def parameters_dict(parameters):
    W_hidden, b_hidden, w_out, b_out = parameters
    return {
        "W_hidden": W_hidden.tolist(),
        "b_hidden": b_hidden.tolist(),
        "w_out": w_out.tolist(),
        "b_out": b_out.item(),
    }

def gradient_table(title, parameters, inputs, targets):
    """Print one row per example (y, hidden_pre, hidden, z, p, loss, dL/dz, dL/dhidden_pre), then the loss and the four gradients. Return the same numbers."""
    hidden_pre, hidden, logits = hidden_layer_mlp.mlp_forward_pass(*parameters, inputs)
    probabilities = binary_cross_entropy.sigmoid(logits)
    _, per_example_losses, logit_gradients = binary_cross_entropy.binary_cross_entropy(logits, targets)
    loss, grad_hidden_pre, grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out = hidden_layer_mlp.mlp_backward_pass(*parameters, inputs, targets)

    print(title)
    W_hidden, b_hidden, w_out, b_out = parameters
    print(f"W_hidden = {rounded(W_hidden.tolist())}, b_hidden = {rounded(b_hidden.tolist())}, w_out = {rounded(w_out.tolist())}, b_out = {b_out.item():.10f}")
    print(f"{'Example':<7} | {'x':>4} | {'y':>1} | {'hidden_pre':<18} | {'hidden':<18} | {'z':>8} | {'p':>12} | {'Loss':>12} | {'dL/dz':>13} | {'dL/dhidden_pre':<31}")
    rows = []
    for i in range(inputs.shape[0]):
        x = inputs[i, 0].item()
        y = targets[i].item()
        print(f"{'ex' + str(i):<7} | {x:>4.2f} | {y:>1.0f} | {pair_text(hidden_pre[i].tolist(), 4):<18} | {pair_text(hidden[i].tolist(), 4):<18} | {logits[i].item():>8.4f} | {probabilities[i].item():>12.10f} | {per_example_losses[i].item():>12.10f} | {logit_gradients[i].item():>13.10f} | {pair_text(grad_hidden_pre[i].tolist()):<31}")
        rows.append({
            "x": inputs[i].tolist(),
            "y": y,
            "hidden_pre": hidden_pre[i].tolist(),
            "hidden": hidden[i].tolist(),
            "z": logits[i].item(),
            "p": probabilities[i].item(),
            "Loss": per_example_losses[i].item(),
            "dL/dz": logit_gradients[i].item(),
            "dL/dhidden_pre": grad_hidden_pre[i].tolist(),
        })
    print(f"  Loss:          {loss.item():.10f}")
    print(f"  dL/dW_hidden:  {rounded(grad_W_hidden.tolist())}")
    print(f"  dL/db_hidden:  {rounded(grad_b_hidden.tolist())}")
    print(f"  dL/dw_out:     {rounded(grad_w_out.tolist())}")
    print(f"  dL/db_out:     {grad_b_out.item():.10f}")
    print(f"  Accuracy:      {accuracy(logits, targets):.4f}")
    print("")

    return {
        **parameters_dict(parameters),
        "Examples": rows,
        "Loss": loss.item(),
        "dL/dW_hidden": grad_W_hidden.tolist(),
        "dL/db_hidden": grad_b_hidden.tolist(),
        "dL/dw_out": grad_w_out.tolist(),
        "dL/db_out": grad_b_out.item(),
        "Accuracy": accuracy(logits, targets),
    }

inputs, targets, initial_parameters, lr, steps = mlp_fit()
baseline_weight, baseline_bias, baseline_lr, baseline_steps = baseline_fit()
hazard_inputs, hazard_targets, hazard_parameters = hazard_case()

# 1. The initialization: per-example table and the four gradients
initialization = gradient_table("1. The initialization", initial_parameters, inputs, targets)

# 2. Train the MLP with sgd_step; each row is the state after that many steps
print(f"2. Training: {steps} full-batch steps, lr = {lr}")
print(f"{'Step':>4} | {'W_hidden':<30} | {'b_hidden':<30} | {'w_out':<30} | {'b_out':>13} | {'Loss':>12} | {'Accuracy':>8}")
report_steps = (0, 1, 2, 10, 100, steps)
training_results = []
first_perfect_step = None
parameters = initial_parameters
for step in range(steps + 1):
    loss, _, grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out = hidden_layer_mlp.mlp_backward_pass(*parameters, inputs, targets)
    _, _, logits = hidden_layer_mlp.mlp_forward_pass(*parameters, inputs)
    step_accuracy = accuracy(logits, targets)
    if first_perfect_step is None and step_accuracy == 1.0:
        first_perfect_step = step
    if step in report_steps:
        training_results.append({"Step": step, **parameters_dict(parameters), "Loss": loss.item(), "Accuracy": step_accuracy})
        W_hidden, b_hidden, w_out, b_out = parameters
        print(f"{step:>4} | {pair_text(W_hidden[0].tolist()):<30} | {pair_text(b_hidden.tolist()):<30} | {pair_text(w_out.tolist()):<30} | {b_out.item():>13.10f} | {loss.item():>12.10f} | {step_accuracy:>8.4f}")
    if step < steps:
        parameters = sgd_step.sgd_step(parameters, [grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out], lr)
trained_parameters = parameters
print(f"  First step with accuracy 1.0: {first_perfect_step}")

# Where each trained unit bends: hidden_pre[j] = 0 at x = -b_hidden[j] / W_hidden[0, j]
W_hidden, b_hidden, w_out, b_out = trained_parameters
bends = (-b_hidden / W_hidden[0]).tolist()
print(f"  Unit bends:   {pair_text(bends, 4)}")

# Where the trained logit changes sign, on a grid of x in [0, 4.5]
grid = torch.linspace(0.0, 4.5, 4501, dtype=torch.float64).unsqueeze(-1)
_, _, grid_logits = hidden_layer_mlp.mlp_forward_pass(*trained_parameters, grid)
sign_change = (grid_logits[:-1] > 0) != (grid_logits[1:] > 0)
# The logit is linear between grid points away from a bend, so interpolate to z = 0
x0, x1 = grid[:-1, 0][sign_change], grid[1:, 0][sign_change]
z0, z1 = grid_logits[:-1][sign_change], grid_logits[1:][sign_change]
crossings = (x0 - z0 * (x1 - x0) / (z1 - z0)).tolist()
print(f"  Logit sign changes at x = {pair_text(crossings, 4)}")
print("")

trained_model = gradient_table(f"2b. The trained MLP after {steps} steps", trained_parameters, inputs, targets)

# 3. Baselines: F03-1-L1's single-layer logistic regression, and always predicting the majority class
print(f"3. Baselines beside the trained MLP")
weight, bias = baseline_weight, baseline_bias
for _ in range(baseline_steps):
    _, weight_gradient, bias_gradient = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)
    weight, bias = sgd_step.sgd_step([weight, bias], [weight_gradient, bias_gradient], baseline_lr)
baseline_loss, _, _ = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)
baseline_accuracy = accuracy(binary_cross_entropy.forward_pass(weight, bias, inputs), targets)
majority_class = float(targets.mean().item() > 0.5)
majority_accuracy = (targets == majority_class).to(targets.dtype).mean().item()
mlp_loss = trained_model["Loss"]
mlp_accuracy = trained_model["Accuracy"]

print(f"{'Model':<36} | {'Loss':>12} | {'Accuracy':>8}")
print(f"{'Majority class (always ' + str(int(majority_class)) + ')':<36} | {'-':>12} | {majority_accuracy:>8.4f}")
print(f"{f'Single layer, {baseline_steps} steps, lr = {baseline_lr}':<36} | {baseline_loss.item():>12.10f} | {baseline_accuracy:>8.4f}")
print(f"{f'MLP, {steps} steps, lr = {lr}':<36} | {mlp_loss:>12.10f} | {mlp_accuracy:>8.4f}")
print(f"  Single layer: weight = {rounded(weight.tolist())}, bias = {bias.item():.10f}")
print("")

baseline_results = {
    "Majority Class": {"Class": majority_class, "Accuracy": majority_accuracy},
    "Single Layer": {"weight": weight.tolist(), "bias": bias.item(), "Steps": baseline_steps, "lr": baseline_lr, "Loss": baseline_loss.item(), "Accuracy": baseline_accuracy},
    "MLP": {"Steps": steps, "lr": lr, "Loss": mlp_loss, "Accuracy": mlp_accuracy},
}

# 4. The hazard: a dead unit and a pre-activation of exactly 0
hazard_results = gradient_table("4. The hazard: unit 1 dead everywhere, unit 0 exactly 0 at ex0", hazard_parameters, hazard_inputs, hazard_targets)

# 5. Results file
json_schema = {
    "Initialization": initialization,
    "Training": training_results,
    "First Step With Accuracy 1.0": first_perfect_step,
    "Unit Bends": bends,
    "Logit Sign Changes": crossings,
    "Trained Model": trained_model,
    "Baselines": baseline_results,
    "Hazard Case": hazard_results,
}

path = Path(__file__).parent / "results" / "F03-3-L1.json"
with open(path, "w") as f:
    json.dump(json_schema, f, indent=2, allow_nan=False)
