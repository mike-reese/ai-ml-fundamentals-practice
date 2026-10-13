import evalkit.cross_entropy as cross_entropy
import evalkit.sgd_step as sgd_step
import evalkit.softmax as softmax
from evalkit.binary_cross_entropy import forward_pass
import torch
import json
import math
from pathlib import Path

def classifier_fit():
    inputs = torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 0, 1, 1, 2, 1, 2], dtype=torch.int64)
    weight = torch.tensor([[-0.5, 0.25, 1.0]], dtype=torch.float64)
    bias = torch.tensor([1.0, 0.0, -2.5], dtype=torch.float64)
    lr = 0.5
    steps = 300

    return inputs, targets, weight, bias, lr, steps

def scorer_fit():
    # (query, candidate, feature); features: overlap, cosine, version_ok, accepted
    inputs = torch.tensor([
        [[0.5, 0.8, 1.0, 1.0], [0.9, 0.6, 0.0, 1.0], [0.2, 0.3, 1.0, 0.0]],
        [[0.1, 0.2, 1.0, 1.0], [0.7, 0.9, 1.0, 1.0], [0.4, 0.4, 0.0, 0.0]],
        [[0.8, 0.7, 0.0, 0.0], [0.3, 0.5, 1.0, 1.0], [0.6, 0.9, 1.0, 0.0]],
        [[0.2, 0.1, 1.0, 0.0], [0.9, 0.8, 0.0, 1.0], [0.6, 0.7, 1.0, 1.0]],
        [[0.4, 0.9, 1.0, 1.0], [0.5, 0.3, 1.0, 1.0], [0.9, 0.9, 0.0, 0.0]],
        [[0.4, 0.9, 1.0, 1.0], [0.5, 0.3, 1.0, 1.0], [0.1, 0.2, 0.0, 1.0]],
    ], dtype=torch.float64)
    targets = torch.tensor([0, 1, 1, 2, 0, 1], dtype=torch.int64)
    weight = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    bias = torch.tensor(-1.0, dtype=torch.float64)
    lr = 2.0
    steps = 200

    return inputs, targets, weight, bias, lr, steps

def saturated_case():
    logits = torch.tensor([[100.0, 0.0, -100.0], [100.0, 0.0, -100.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([2, 0, 1], dtype=torch.int64)

    return logits, targets

def accuracy(logits, targets):
    """Fraction of examples where the predicted class (the argmax of the logits) equals the target."""
    return (logits.argmax(dim=-1) == targets).to(logits.dtype).mean()

def rounded(values, digits=10):
    """Round every entry of a list (or a list of lists) for printing."""
    if isinstance(values, list):
        return [rounded(v, digits) for v in values]
    return round(values, digits)

def json_safe(value):
    """Strict JSON has no inf or nan, so write them as the strings "inf", "-inf" and "nan"."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value

def example_table(title, label, first_label, gradient_function, weight, bias, inputs, targets):
    """Print one row per example (y, Z, P, loss, dL/dZ), then the loss and both gradients. Return the same numbers."""
    logits = forward_pass(weight, bias, inputs)
    probabilities = softmax.softmax(logits, dim=-1)
    loss, per_example_losses, logit_gradients = cross_entropy.cross_entropy(logits, targets)
    _, weight_gradient, bias_gradient = gradient_function(weight, bias, inputs, targets)

    print(title)
    print(f"weight = {rounded(weight.tolist())}, bias = {rounded(bias.tolist())}")
    print(f"{'Example':<7} | {'y':>1} | {'Z':<26} | {'P':<40} | {'Loss':>12} | {'dL/dZ':<43}")
    rows = []
    for i in range(inputs.shape[0]):
        y = targets[i].item()
        z = logits[i].tolist()
        p = probabilities[i].tolist()
        dz = logit_gradients[i].tolist()
        z_text = "[" + ", ".join(f"{v:.4f}" for v in z) + "]"
        p_text = "[" + ", ".join(f"{v:.10f}" for v in p) + "]"
        dz_text = "[" + ", ".join(f"{v:.10f}" for v in dz) + "]"
        print(f"{label + str(i + first_label):<7} | {y:>1} | {z_text:<26} | {p_text:<40} | {per_example_losses[i].item():>12.10f} | {dz_text:<43}")
        rows.append({
            "y": y,
            "Z": z,
            "P": p,
            "Loss": per_example_losses[i].item(),
            "dL/dZ": dz,
        })
    print(f"  Loss:       {loss.item():.10f}")
    print(f"  dL/dweight: {rounded(weight_gradient.tolist())}")
    print(f"  dL/dbias:   {rounded(bias_gradient.tolist())}")
    print(f"  Accuracy:   {accuracy(logits, targets).item():.4f}")
    print("")

    return {
        "weight": weight.tolist(),
        "bias": bias.tolist(),
        "Examples": rows,
        "Loss": loss.item(),
        "dL/dweight": weight_gradient.tolist(),
        "dL/dbias": bias_gradient.tolist(),
        "Accuracy": accuracy(logits, targets).item(),
    }

def training_table(title, gradient_function, weight, bias, inputs, targets, lr, steps):
    """Train with full-batch sgd_step and print the state after steps 0, 1, 2, 10, 50, 100 and the last. Return the same numbers."""
    print(f"{title}: {steps} full-batch steps, lr = {lr}")
    print(f"{'Step':>4} | {'weight':<52} | {'bias':<42} | {'Loss':>12} | {'Accuracy':>8}")
    report_steps = (0, 1, 2, 10, 50, 100, steps)
    rows = []
    train_weight, train_bias = weight, bias
    for step in range(steps + 1):
        loss, weight_gradient, bias_gradient = gradient_function(train_weight, train_bias, inputs, targets)
        if step in report_steps:
            step_accuracy = accuracy(forward_pass(train_weight, train_bias, inputs), targets).item()
            rows.append({
                "Step": step,
                "weight": train_weight.tolist(),
                "bias": train_bias.tolist(),
                "Loss": loss.item(),
                "Accuracy": step_accuracy,
            })
            weight_text = str(rounded(train_weight.flatten().tolist(), 10))
            bias_text = str(rounded(train_bias.tolist(), 10))
            print(f"{step:>4} | {weight_text:<52} | {bias_text:<42} | {loss.item():>12.10f} | {step_accuracy:>8.4f}")
        if step < steps:
            train_weight, train_bias = sgd_step.sgd_step([train_weight, train_bias], [weight_gradient, bias_gradient], lr)
    print("")

    return rows

classifier_inputs, classifier_targets, classifier_weight, classifier_bias, classifier_lr, classifier_steps = classifier_fit()
scorer_inputs, scorer_targets, scorer_weight, scorer_bias, scorer_lr, scorer_steps = scorer_fit()
saturated_logits, saturated_targets = saturated_case()

# 1. Per-example tables at the initialization
classifier_initialization = example_table("1a. Classifier at the initialization", "ex", 0, cross_entropy.classifier_gradient, classifier_weight, classifier_bias, classifier_inputs, classifier_targets)
scorer_initialization = example_table("1b. Scorer at the initialization", "q", 1, cross_entropy.scorer_gradient, scorer_weight, scorer_bias, scorer_inputs, scorer_targets)

# 2. Train each model; each row is the state after that many steps
classifier_training = training_table("2a. Classifier training", cross_entropy.classifier_gradient, classifier_weight, classifier_bias, classifier_inputs, classifier_targets, classifier_lr, classifier_steps)
scorer_training = training_table("2b. Scorer training", cross_entropy.scorer_gradient, scorer_weight, scorer_bias, scorer_inputs, scorer_targets, scorer_lr, scorer_steps)

# 3. Saturated logits: our stable cross-entropy beside the probability route
print("3. Saturated logits (float32): stable form against the probability route")
saturated_probabilities = softmax.softmax(saturated_logits, dim=-1)
saturated_loss, saturated_per_example_losses, saturated_logit_gradients = cross_entropy.cross_entropy(saturated_logits, saturated_targets)
# Probability route: softmax first, then -ln P[t]
p_target = saturated_probabilities[torch.arange(saturated_logits.shape[0]), saturated_targets]
route_per_example_losses = -torch.log(p_target)
route_loss = route_per_example_losses.mean()

print(f"{'Row':<3} | {'Z':<18} | {'t':>1} | {'P':<34} | {'Stable Loss':>11} | {'Route Loss':>10} | {'dL/dZ':<34}")
saturated_rows = []
for i in range(saturated_logits.shape[0]):
    z = saturated_logits[i].tolist()
    p = saturated_probabilities[i].tolist()
    dz = saturated_logit_gradients[i].tolist()
    z_text = "[" + ", ".join(f"{v:.0f}" for v in z) + "]"
    p_text = "[" + ", ".join(f"{v:.4e}" for v in p) + "]"
    dz_text = "[" + ", ".join(f"{v:.4e}" for v in dz) + "]"
    print(f"{i:<3} | {z_text:<18} | {saturated_targets[i].item():>1} | {p_text:<34} | {saturated_per_example_losses[i].item():>11.7f} | {route_per_example_losses[i].item():>10.7f} | {dz_text:<34}")
    saturated_rows.append({
        "Z": z,
        "t": saturated_targets[i].item(),
        "P": p,
        "Stable Loss": saturated_per_example_losses[i].item(),
        "Probability Route Loss": route_per_example_losses[i].item(),
        "dL/dZ": dz,
    })
print(f"  Stable loss:            {saturated_loss.item():.6f}")
print(f"  Probability route loss: {route_loss.item():.6f}")

saturated_results = {
    "Examples": saturated_rows,
    "Stable Loss": saturated_loss.item(),
    "Probability Route Loss": route_loss.item(),
}

# Results file
json_schema = {
    "Classifier Initialization": classifier_initialization,
    "Scorer Initialization": scorer_initialization,
    "Classifier Training": classifier_training,
    "Scorer Training": scorer_training,
    "Saturated Case": saturated_results,
}

path = Path(__file__).parent / "results" / "F03-2-L1.json"
with open(path, "w") as f:
    json.dump(json_safe(json_schema), f, indent=2, allow_nan=False)
