import evalkit.binary_cross_entropy as binary_cross_entropy
import evalkit.sgd_step as sgd_step
import torch
import json
from pathlib import Path

def logistic_fit():
    inputs = torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 0, 1, 0, 0, 1, 1], dtype=torch.float64)
    weight = torch.tensor([0.5], dtype=torch.float64)
    bias = torch.tensor(0.0, dtype=torch.float64)
    lr = 0.5
    steps = 500
    unseen_weight = torch.tensor([-1.0], dtype=torch.float64)
    unseen_bias = torch.tensor(2.0, dtype=torch.float64)

    return inputs, targets, weight, bias, lr, steps, unseen_weight, unseen_bias

def saturated_case():
    logits = torch.tensor([40.0, -40.0, 100.0, -100.0], dtype=torch.float32)
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.float32)

    return logits, targets

def accuracy(probabilities, targets):
    """Fraction of examples where the predicted class (p > 0.5) equals the target."""
    return ((probabilities > 0.5).to(targets.dtype) == targets).to(targets.dtype).mean()

def example_table(title, weight, bias, inputs, targets):
    """Print one row per example (x, y, z, p, loss, dL/dz), then the loss and both gradients. Return the same numbers."""
    logits = binary_cross_entropy.forward_pass(weight, bias, inputs)
    probabilities = binary_cross_entropy.sigmoid(logits)
    loss, per_example_losses, logit_gradients = binary_cross_entropy.binary_cross_entropy(logits, targets)
    _, weight_gradient, bias_gradient = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)

    print(title)
    print(f"weight = {[round(v, 10) for v in weight.tolist()]}, bias = {bias.item():.10f}")
    print(f"{'Example':<7} | {'x':>4} | {'y':>3} | {'z':>13} | {'p':>12} | {'Loss':>12} | {'dL/dz':>13}")
    rows = []
    for i in range(inputs.shape[0]):
        x = inputs[i, 0].item()
        y = targets[i].item()
        print(f"{'ex' + str(i):<7} | {x:>4.1f} | {y:>3.0f} | {logits[i].item():>13.10f} | {probabilities[i].item():>12.10f} | {per_example_losses[i].item():>12.10f} | {logit_gradients[i].item():>13.10f}")
        rows.append({
            "x": inputs[i].tolist(),
            "y": y,
            "z": logits[i].item(),
            "p": probabilities[i].item(),
            "Loss": per_example_losses[i].item(),
            "dL/dz": logit_gradients[i].item(),
        })
    print(f"  Loss:       {loss.item():.10f}")
    print(f"  dL/dweight: {[round(v, 10) for v in weight_gradient.tolist()]}")
    print(f"  dL/dbias:   {bias_gradient.item():.10f}")
    print(f"  Accuracy:   {accuracy(probabilities, targets).item():.4f}")
    print("")

    return {
        "weight": weight.tolist(),
        "bias": bias.item(),
        "Examples": rows,
        "Loss": loss.item(),
        "dL/dweight": weight_gradient.tolist(),
        "dL/dbias": bias_gradient.item(),
        "Accuracy": accuracy(probabilities, targets).item(),
    }

inputs, targets, weight, bias, lr, steps, unseen_weight, unseen_bias = logistic_fit()
saturated_logits, saturated_targets = saturated_case()

# 1. Per-example tables at the initialization and the unseen start
initialization = example_table("1a. The initialization", weight, bias, inputs, targets)
unseen_start = example_table("1b. The unseen start: negative logits use the e^z/(1 + e^z) branch", unseen_weight, unseen_bias, inputs, targets)

# 2. Train 500 full-batch steps; each row is the state after that many steps
print(f"2. Training: {steps} full-batch steps, lr = {lr}")
print(f"{'Step':>4} | {'weight':>13} | {'bias':>14} | {'Loss':>12} | {'Accuracy':>8}")
report_steps = (0, 1, 2, 5, 10, 50, 100, 200, 500)
training_results = []
train_weight, train_bias = weight, bias
for step in range(steps + 1):
    loss, weight_gradient, bias_gradient = binary_cross_entropy.logistic_regression_gradient(train_weight, train_bias, inputs, targets)
    if step in report_steps:
        probabilities = binary_cross_entropy.sigmoid(binary_cross_entropy.forward_pass(train_weight, train_bias, inputs))
        step_accuracy = accuracy(probabilities, targets).item()
        training_results.append({
            "Step": step,
            "weight": train_weight.tolist(),
            "bias": train_bias.item(),
            "Loss": loss.item(),
            "Accuracy": step_accuracy,
        })
        print(f"{step:>4} | {train_weight.item():>13.10f} | {train_bias.item():>14.10f} | {loss.item():>12.10f} | {step_accuracy:>8.4f}")
    if step < steps:
        train_weight, train_bias = sgd_step.sgd_step([train_weight, train_bias], [weight_gradient, bias_gradient], lr)
print("")

trained_model = example_table(f"2b. The trained model after {steps} steps", train_weight, train_bias, inputs, targets)

# 3. Saturated logits: our stable functions beside the probability route
print("3. Saturated logits (float32): stable form against the probability route")
saturated_probabilities = binary_cross_entropy.sigmoid(saturated_logits)
saturated_loss, saturated_per_example_losses, saturated_logit_gradients = binary_cross_entropy.binary_cross_entropy(saturated_logits, saturated_targets)
# Probability route: sigma first, then -ln p_target
p_target = torch.where(saturated_targets == 1, saturated_probabilities, 1 - saturated_probabilities)
route_per_example_losses = -torch.log(p_target)
route_loss = route_per_example_losses.mean()

print(f"{'Example':<7} | {'z':>6} | {'y':>3} | {'p':>12} | {'Stable Loss':>11} | {'Route Loss':>10} | {'dL/dz':>7}")
saturated_rows = []
for i in range(saturated_logits.shape[0]):
    print(f"{'ex' + str(i):<7} | {saturated_logits[i].item():>6.1f} | {saturated_targets[i].item():>3.0f} | {saturated_probabilities[i].item():>12.4e} | {saturated_per_example_losses[i].item():>11.4f} | {route_per_example_losses[i].item():>10.4f} | {saturated_logit_gradients[i].item():>7.4f}")
    saturated_rows.append({
        "z": saturated_logits[i].item(),
        "y": saturated_targets[i].item(),
        "p": saturated_probabilities[i].item(),
        "Stable Loss": saturated_per_example_losses[i].item(),
        "Probability Route Loss": route_per_example_losses[i].item(),
        "dL/dz": saturated_logit_gradients[i].item(),
    })
print(f"  Stable loss:            {saturated_loss.item():.4f}")
print(f"  Probability route loss: {route_loss.item():.4f}")

saturated_results = {
    "Examples": saturated_rows,
    "Stable Loss": saturated_loss.item(),
    "Probability Route Loss": route_loss.item(),
}

# Results file
json_schema = {
    "Initialization": initialization,
    "Unseen Start": unseen_start,
    "Training": training_results,
    "Trained Model": trained_model,
    "Saturated Case": saturated_results,
}

path = Path(__file__).parent / "results" / "F03-1-L1.json"
with open(path, "w") as f:
    json.dump(json_schema, f, indent=2)
