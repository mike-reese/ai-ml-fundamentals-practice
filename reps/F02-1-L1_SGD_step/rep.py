import evalkit.sgd_step as sgd_step
import torch
import torch.nn.functional as F
import json
from pathlib import Path

def line_fit():
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float64)
    targets = torch.tensor([2.1, 2.3, 3.1, 3.6, 3.8, 4.6], dtype=torch.float64)
    weight = torch.tensor([1.5], dtype=torch.float64)
    bias = torch.tensor(0.0, dtype=torch.float64)
    lr = 0.05
    batches = torch.tensor([[2, 5], [3, 0], [1, 4]], dtype=torch.int64)
    unseen_weight = torch.tensor([0.0], dtype=torch.float64)
    unseen_bias = torch.tensor(1.0, dtype=torch.float64)
    unseen_lr = 0.02

    return inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr

def two_features():
    inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 1.0]], dtype=torch.float64)
    targets = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    weight = torch.tensor([0.5, -1.0], dtype=torch.float64)
    bias = torch.tensor(0.25, dtype=torch.float64)
    lr = 0.1

    return inputs, targets, weight, bias, lr

def full_loss(weight, bias, inputs, targets):
    """Loss on every example at the given parameters."""
    loss, _, _ = sgd_step.compute_gradient(weight, bias, inputs, targets)
    return loss

inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit()
two_f_inputs, two_f_targets, two_f_weight, two_f_bias, two_f_lr = two_features()

print("1. One full-batch step of the line fit from the initialization")
print(f"weight = {weight.tolist()}, bias = {bias.item()}, lr = {lr}")
loss, weight_gradient, bias_gradient = sgd_step.compute_gradient(weight, bias, inputs, targets)
new_weight, new_bias = sgd_step.sgd_step([weight, bias], [weight_gradient, bias_gradient], lr)
loss_after = full_loss(new_weight, new_bias, inputs, targets)
print(f"  Loss:               {loss.item():.10f}")
print(f"  dL/dweight:         {[round(v, 10) for v in weight_gradient.tolist()]}")
print(f"  dL/dbias:           {bias_gradient.item():.10f}")
print(f"  weight after:       {[round(v, 10) for v in new_weight.tolist()]}")
print(f"  bias after:         {new_bias.item():.10f}")
print(f"  Loss on all six after: {loss_after.item():.10f}")

full_batch_step = {
    "Loss": loss.item(),
    "dL/dweight": weight_gradient.tolist(),
    "dL/dbias": bias_gradient.item(),
    "weight After": new_weight.tolist(),
    "bias After": new_bias.item(),
    "Loss On All Six After": loss_after.item(),
}

print("")
print("2. Batch gradients at the initialization: the batch size divides")
print(f"{'Batch':<10} | {'Loss':>14} | {'dL/dweight':>14} | {'dL/dbias':>14}")
batch_results = {}
batch_weight_gradients = []
batch_bias_gradients = []
for batch in batches:
    batch_loss, batch_weight_gradient, batch_bias_gradient = sgd_step.compute_gradient(weight, bias, inputs[batch], targets[batch])
    batch_weight_gradients.append(batch_weight_gradient)
    batch_bias_gradients.append(batch_bias_gradient)
    batch_results[str(batch.tolist())] = {
        "Loss": batch_loss.item(),
        "dL/dweight": batch_weight_gradient.tolist(),
        "dL/dbias": batch_bias_gradient.item(),
    }
    print(f"{str(batch.tolist()):<10} | {batch_loss.item():>14.10f} | {batch_weight_gradient.item():>14.10f} | {batch_bias_gradient.item():>14.10f}")

mean_weight_gradient = torch.stack(batch_weight_gradients).mean(dim=0)
mean_bias_gradient = torch.stack(batch_bias_gradients).mean(dim=0)
print(f"{'Mean':<10} | {'':>14} | {mean_weight_gradient.item():>14.10f} | {mean_bias_gradient.item():>14.10f}")
print(f"{'Full batch':<10} | {'':>14} | {weight_gradient.item():>14.10f} | {bias_gradient.item():>14.10f}")

batch_gradients = {
    "Batches": batch_results,
    "Mean Of Batch Gradients": {"dL/dweight": mean_weight_gradient.tolist(), "dL/dbias": mean_bias_gradient.item()},
    "Full-Batch Gradient": {"dL/dweight": weight_gradient.tolist(), "dL/dbias": bias_gradient.item()},
}

print("")
print("3. One epoch over the Table 1 batches, each step from the previous step's result")
print(f"{'Step':<4} | {'Batch':<8} | {'Batch Loss':>13} | {'dL/dweight':>11} | {'dL/dbias':>11} | {'weight':>10} | {'bias':>11} | {'Loss On All Six':>15}")
epoch_results = []
epoch_weight, epoch_bias = weight, bias
for step, batch in enumerate(batches, start=1):
    batch_loss, batch_weight_gradient, batch_bias_gradient = sgd_step.compute_gradient(epoch_weight, epoch_bias, inputs[batch], targets[batch])
    epoch_weight, epoch_bias = sgd_step.sgd_step([epoch_weight, epoch_bias], [batch_weight_gradient, batch_bias_gradient], lr)
    epoch_loss_after = full_loss(epoch_weight, epoch_bias, inputs, targets)
    epoch_results.append({
        "Step": step,
        "Batch": batch.tolist(),
        "Batch Loss": batch_loss.item(),
        "dL/dweight": batch_weight_gradient.tolist(),
        "dL/dbias": batch_bias_gradient.item(),
        "weight After": epoch_weight.tolist(),
        "bias After": epoch_bias.item(),
        "Loss On All Six After": epoch_loss_after.item(),
    })
    print(f"{step:<4} | {str(batch.tolist()):<8} | {batch_loss.item():>13.10f} | {batch_weight_gradient.item():>11.6f} | {batch_bias_gradient.item():>11.6f} | {epoch_weight.item():>10.6f} | {epoch_bias.item():>11.7f} | {epoch_loss_after.item():>15.10f}")

print("")
print("4. The two-feature step")
print(f"weight = {two_f_weight.tolist()}, bias = {two_f_bias.item()}, lr = {two_f_lr}")
two_f_loss, two_f_weight_gradient, two_f_bias_gradient = sgd_step.compute_gradient(two_f_weight, two_f_bias, two_f_inputs, two_f_targets)
two_f_new_weight, two_f_new_bias = sgd_step.sgd_step([two_f_weight, two_f_bias], [two_f_weight_gradient, two_f_bias_gradient], two_f_lr)
two_f_loss_after = full_loss(two_f_new_weight, two_f_new_bias, two_f_inputs, two_f_targets)
print(f"  Loss:         {two_f_loss.item():.10f}")
print(f"  dL/dweight:   {[round(v, 10) for v in two_f_weight_gradient.tolist()]}")
print(f"  dL/dbias:     {two_f_bias_gradient.item():.10f}")
print(f"  weight after: {[round(v, 10) for v in two_f_new_weight.tolist()]}")
print(f"  bias after:   {two_f_new_bias.item():.10f}")
print(f"  Loss after:   {two_f_loss_after.item():.10f}")

two_feature_step = {
    "Loss": two_f_loss.item(),
    "dL/dweight": two_f_weight_gradient.tolist(),
    "dL/dbias": two_f_bias_gradient.item(),
    "weight After": two_f_new_weight.tolist(),
    "bias After": two_f_new_bias.item(),
    "Loss After": two_f_loss_after.item(),
}

print("")
print("5. float32: max difference from torch.optim.SGD on the check 6 data")
g = torch.Generator().manual_seed(0)
lib_inputs = torch.randn(32, 8, generator=g)
true_weight = torch.randn(8, generator=g)
lib_targets = lib_inputs @ true_weight + 0.5 + 0.1 * torch.randn(32, generator=g)
lib_lr = 0.05

ours_weight = torch.zeros(8)
ours_bias = torch.tensor(0.0)
lib_weight = torch.zeros(8, requires_grad=True)
lib_bias = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([lib_weight, lib_bias], lr=lib_lr)

library_results = {}
print(f"{'Steps':>5} | {'Max Diff weight':>15} | {'Max Diff bias':>13} | {'Our Loss':>10}")
for step in range(1, 201):
    opt.zero_grad()
    lib_loss = F.mse_loss(lib_inputs @ lib_weight + lib_bias, lib_targets)
    lib_loss.backward()
    opt.step()

    ours_loss, ours_weight_gradient, ours_bias_gradient = sgd_step.compute_gradient(ours_weight, ours_bias, lib_inputs, lib_targets)
    ours_weight, ours_bias = sgd_step.sgd_step([ours_weight, ours_bias], [ours_weight_gradient, ours_bias_gradient], lib_lr)

    if step in (1, 200):
        weight_difference = (ours_weight - lib_weight.detach()).abs().max().item()
        bias_difference = (ours_bias - lib_bias.detach()).abs().item()
        ours_loss_after = full_loss(ours_weight, ours_bias, lib_inputs, lib_targets).item()
        library_results[f"After {step} Steps"] = {
            "Max Difference weight": weight_difference,
            "Max Difference bias": bias_difference,
            "Our Loss After": ours_loss_after,
        }
        print(f"{step:>5} | {weight_difference:>15.3e} | {bias_difference:>13.3e} | {ours_loss_after:>10.6f}")

# Results file
json_schema = {
    "Full-Batch Step": full_batch_step,
    "Batch Gradients At The Initialization": batch_gradients,
    "One Epoch": epoch_results,
    "Two-Feature Step": two_feature_step,
    "float32 Against torch.optim.SGD": library_results,
}

path = Path(__file__).parent / "results" / "F02-1-L1.json"
with open(path, "w") as f:
    json.dump(json_schema, f, indent=2)
