import evalkit.l2_regularization as l2_regularization
import torch
import numpy
from pathlib import Path
import json

def parameter_validation():
    w = torch.tensor(2.0, dtype=torch.float64)
    grad_w = torch.tensor(0.1, dtype=torch.float64)
    lr = 0.3
    weight_decay = 0.01
    h = 0.001

    return w, grad_w, lr, weight_decay, h

def multi_parameter():
    w = torch.tensor([2.0, -1.5, 0.0], dtype=torch.float64)
    grad_w = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    
    return w, grad_w

def hyperparameter_selection():
    weight_decay = numpy.array([0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    training_loss = numpy.array([0.0099, 0.0285, 0.0738, 0.2083, 0.2790, 0.3862, 0.6367])
    validation_loss = numpy.array([1.7445, 1.1305, 0.6264, 0.3746, 0.4244, 0.4900, 0.6488])
    sum_squared_parameters = numpy.array([251.61, 139.12, 66.26, 15.39, 6.83, 3.01, 0.13])
    validation_accuracy = numpy.array([0.6667, 0.6667, 0.6667, 0.8333, 0.8333, 0.6667, 0.6667])

    return weight_decay, training_loss, validation_loss, sum_squared_parameters, validation_accuracy

w, grad_w, lr, weight_decay, h = parameter_validation()
multi_w, multi_grad_w = multi_parameter()
h_weight_decay, h_training_loss, h_validation_loss, h_sum_squared_parameters, h_validation_accuracy = hyperparameter_selection()

w_h = w + h
l2_penalty, l2_penalty_gradient = l2_regularization.calculate_l2_penalty(w, weight_decay)
l2_penalty_h, l2_penalty_gradient_h = l2_regularization.calculate_l2_penalty(w_h, weight_decay)
finite_difference = (l2_penalty_h - l2_penalty)/h
print("1. Penalty and finite difference check")
print("Finite Difference Calculation = (penalty(w+h)-penalty(w))/h")
print(f"{'Penalty':<20} | {'Finite Difference Check':<20} | {'Exact Derivative':<20}")
print(f"{l2_penalty:.4f} | {finite_difference:.6f} | {l2_penalty_gradient:.4f}")

updated_parameter = l2_regularization.l2_step(w, grad_w, lr, weight_decay)
decay_then_update = (1 - lr * weight_decay) * w - lr * grad_w
weight_decay_0 = 0.0
updated_parameter_wd_0 = l2_regularization.l2_step(w, grad_w, lr, weight_decay_0)
decay_then_update_wd_0 = (1 - lr * weight_decay_0) * w - lr * grad_w
print("")
print("2. Optimizer step - equivalent formulas")
optimizer_steps = [
    ("w - lr * (grad_w + weight_decay * w)", weight_decay, updated_parameter),
    ("(1 - lr * weight_decay) * w - lr * grad_w", weight_decay, decay_then_update),
    ("w - lr * (grad_w + weight_decay * w)", weight_decay_0, updated_parameter_wd_0),
    ("(1 - lr * weight_decay) * w - lr * grad_w", weight_decay_0, decay_then_update_wd_0),
]
print(f"w = {w.item()}, grad_w = {grad_w.item()}, lr = {lr}")
print(f"{'Formula':<44} | {'Weight Decay':>12} | {'Updated w':>10}")
for formula, wd, result in optimizer_steps:
    print(f"{formula:<44} | {float(wd):>12.4f} | {result.item():>10.4f}")

print("")
print("3. Table 2 - hyperparameter selection")
selected_weight_decay, selected_index, selected_validation_loss = l2_regularization.select_on_validation(h_weight_decay, h_validation_loss)
training_weight_decay, training_index, training_selected_loss = l2_regularization.select_on_validation(h_weight_decay, h_training_loss)
generalization_gap = h_validation_loss - h_training_loss
print("Generalization Gap = validation loss - training loss")
print(f"{'Weight Decay':>12} | {'Train Loss':>10} | {'Val Loss':>10} | {'Sum w^2':>10} | {'Val Acc':>10} | {'Gap':>10} | {'Chosen By':<20}")
for i in range(h_weight_decay.size):
    chosen_by = []
    if i == selected_index:
        chosen_by.append("validation loss")
    if i == training_index:
        chosen_by.append("training loss")
    print(f"{h_weight_decay[i]:>12.4f} | {h_training_loss[i]:>10.4f} | {h_validation_loss[i]:>10.4f} | {h_sum_squared_parameters[i]:>10.2f} | {h_validation_accuracy[i]:>10.4f} | {generalization_gap[i]:>10.4f} | {', '.join(chosen_by):<20}")

# Results file
json_schema = {
    "Penalty and Finite Difference Check": {
        "w": w.item(),
        "Weight Decay": weight_decay,
        "h": h,
        "Penalty": l2_penalty.item(),
        "Finite Difference": finite_difference.item(),
        "Exact Derivative": l2_penalty_gradient.item(),
    },
    "Optimizer Step": {
        "w": w.item(),
        "grad_w": grad_w.item(),
        "lr": lr,
        "Steps": [
            {
                "Formula": formula,
                "Weight Decay": float(wd),
                "Updated w": result.item(),
            }
            for formula, wd, result in optimizer_steps
        ],
    },
    "Table 2": [
        {
            "Weight Decay": h_weight_decay[i].item(),
            "Training Loss": h_training_loss[i].item(),
            "Validation Loss": h_validation_loss[i].item(),
            "Sum Squared Parameters": h_sum_squared_parameters[i].item(),
            "Validation Accuracy": h_validation_accuracy[i].item(),
            "Generalization Gap": generalization_gap[i].item(),
        }
        for i in range(h_weight_decay.size)
    ],
    "Selected By Validation Loss": {
        "Index": selected_index.item(),
        "Weight Decay": selected_weight_decay.item(),
        "Validation Loss": selected_validation_loss.item(),
    },
    "Selected By Training Loss": {
        "Index": training_index.item(),
        "Weight Decay": training_weight_decay.item(),
        "Training Loss": training_selected_loss.item(),
    },
}

path = Path(__file__).parent / "results" / "F04C-L1.json"
with open(path, "w") as f:
    json.dump(json_schema, f, indent=2)
