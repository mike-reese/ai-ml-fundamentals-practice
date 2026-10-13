import torch
import torch.nn as nn
import pytest
import evalkit.hidden_layer_mlp as hidden_layer_mlp
import evalkit.binary_cross_entropy as binary_cross_entropy
import evalkit.sgd_step as sgd_step

@pytest.fixture
def setup():
    inputs = torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0], dtype=torch.float64)
    W_hidden = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    b_hidden = torch.tensor([-1.25, 3.25], dtype=torch.float64)
    w_out = torch.tensor([0.5, -1.0], dtype=torch.float64)
    b_out = torch.tensor(1.0, dtype=torch.float64)
    lr = 0.2
    steps = 1000

    return inputs, targets, W_hidden, b_hidden, w_out, b_out, lr, steps

@pytest.fixture
def baseline():
    weight = torch.tensor([0.5], dtype=torch.float64)
    bias = torch.tensor(0.0, dtype=torch.float64)
    lr = 0.5
    steps = 500

    return weight, bias, lr, steps

@pytest.fixture
def hazard_case():
    inputs = torch.tensor([[1.25], [2.0], [0.0]], dtype=torch.float64)
    targets = torch.tensor([1, 0, 1], dtype=torch.float64)
    W_hidden = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    b_hidden = torch.tensor([-1.25, -3.0], dtype=torch.float64)
    w_out = torch.tensor([0.5, -1.0], dtype=torch.float64)
    b_out = torch.tensor(1.0, dtype=torch.float64)

    return inputs, targets, W_hidden, b_hidden, w_out, b_out

@pytest.fixture
def library_check():
    g = torch.Generator().manual_seed(0)
    inputs = torch.randn(64, 8, generator=g)
    W_hidden = torch.randn(8, 4, generator=g)
    b_hidden = torch.randn(4, generator=g)
    w_out = torch.randn(4, generator=g)
    b_out = torch.randn((), generator=g)
    targets = torch.randint(0, 2, (64,), generator=g).float()
    lr = 0.5

    return inputs, targets, W_hidden, b_hidden, w_out, b_out, lr

@pytest.fixture
def known_results():
    # Per-example table at the initialization
    known_hidden_pre = torch.tensor([[-0.75, 2.75], [-0.25, 2.25], [0.25, 1.75], [0.75, 1.25], [1.25, 0.75], [1.75, 0.25], [2.25, -0.25], [2.75, -0.75]], dtype=torch.float64)
    known_hidden = torch.tensor([[0.0, 2.75], [0.0, 2.25], [0.25, 1.75], [0.75, 1.25], [1.25, 0.75], [1.75, 0.25], [2.25, 0.0], [2.75, 0.0]], dtype=torch.float64)
    known_z = torch.tensor([-1.75, -1.25, -0.625, 0.125, 0.875, 1.625, 2.125, 2.375], dtype=torch.float64)
    known_p = torch.tensor([0.1480471980, 0.2227001388, 0.3486451353, 0.5312093734, 0.7057850278, 0.8354835371, 0.8933094061, 0.9149009550], dtype=torch.float64)
    known_per_example_loss = torch.tensor([0.1602241504, 0.2519290813, 0.4287006783, 0.6325990353, 0.3484445810, 1.8047446354, 2.2378222788, 2.4639394655], dtype=torch.float64)
    known_grad_z = torch.tensor([0.0185058998, 0.0278375174, 0.0435806419, -0.0585988283, -0.0367768715, 0.1044354421, 0.1116636758, 0.1143626194], dtype=torch.float64)
    known_grad_hidden_pre = torch.tensor([
        [0.0, -0.0185058998], [0.0, -0.0278375174], [0.0217903210, -0.0435806419], [-0.0292994142, 0.0585988283],
        [-0.0183884358, 0.0367768715], [0.0522177211, -0.1044354421], [0.0558318379, 0.0], [0.0571813097, 0.0],
    ], dtype=torch.float64)

    # K1 to K3
    known_loss = torch.tensor(1.0410504883, dtype=torch.float64)
    known_accuracy = 0.625
    known_grad_W_hidden = torch.tensor([[0.5089053982, -0.2066279211]], dtype=torch.float64)
    known_grad_b_hidden = torch.tensor([0.1393333397, -0.0989838013], dtype=torch.float64)
    known_grad_w_out = torch.tensor([0.6694774473, 0.1150694332], dtype=torch.float64)
    known_grad_b_out = torch.tensor(0.3250100964, dtype=torch.float64)

    # K4: after step 1
    known_step1 = (
        torch.tensor([[0.8982189204, -0.9586744158]], dtype=torch.float64),
        torch.tensor([-1.2778666679, 3.2697967603], dtype=torch.float64),
        torch.tensor([0.3661045105, -1.0230138866], dtype=torch.float64),
        torch.tensor(0.9349979807, dtype=torch.float64),
    )
    known_step1_loss = torch.tensor(0.8902428654, dtype=torch.float64)

    # K5: after step 1000
    known_step1000 = (
        torch.tensor([[1.9378204634, -2.4878914537]], dtype=torch.float64),
        torch.tensor([-4.8187267141, 4.9520113475], dtype=torch.float64),
        torch.tensor([-4.9676969173, -4.4631425941], dtype=torch.float64),
        torch.tensor(2.4420026680, dtype=torch.float64),
    )
    known_step1000_loss = torch.tensor(0.0382387308, dtype=torch.float64)
    known_step1000_accuracy = 1.0

    # K6: single-layer baseline after 500 steps
    known_baseline_loss = torch.tensor(0.5623351450, dtype=torch.float64)
    known_baseline_accuracy = 0.75

    # K7 and K8: hazard case
    known_hazard_hidden_pre = torch.tensor([[0.0, -4.25], [0.75, -5.0], [-1.25, -3.0]], dtype=torch.float64)
    known_hazard_grad_z = torch.tensor([-0.0896471405, 0.2660622592, -0.0896471405], dtype=torch.float64)
    known_hazard_grad_hidden_pre = torch.tensor([[0.0, 0.0], [0.1330311296, 0.0], [0.0, 0.0]], dtype=torch.float64)
    known_hazard_grad_W_hidden = torch.tensor([[0.2660622592, 0.0]], dtype=torch.float64)
    known_hazard_grad_b_hidden = torch.tensor([0.1330311296, 0.0], dtype=torch.float64)
    known_hazard_grad_w_out = torch.tensor([0.1995466944, 0.0], dtype=torch.float64)
    known_hazard_grad_b_out = torch.tensor(0.0867679783, dtype=torch.float64)

    return {
        "hidden_pre": known_hidden_pre, "hidden": known_hidden, "z": known_z, "p": known_p,
        "per_example_loss": known_per_example_loss, "grad_z": known_grad_z, "grad_hidden_pre": known_grad_hidden_pre,
        "loss": known_loss, "accuracy": known_accuracy,
        "grad_W_hidden": known_grad_W_hidden, "grad_b_hidden": known_grad_b_hidden,
        "grad_w_out": known_grad_w_out, "grad_b_out": known_grad_b_out,
        "step1": known_step1, "step1_loss": known_step1_loss,
        "step1000": known_step1000, "step1000_loss": known_step1000_loss, "step1000_accuracy": known_step1000_accuracy,
        "baseline_loss": known_baseline_loss, "baseline_accuracy": known_baseline_accuracy,
        "hazard_hidden_pre": known_hazard_hidden_pre, "hazard_grad_z": known_hazard_grad_z,
        "hazard_grad_hidden_pre": known_hazard_grad_hidden_pre,
        "hazard_grad_W_hidden": known_hazard_grad_W_hidden, "hazard_grad_b_hidden": known_hazard_grad_b_hidden,
        "hazard_grad_w_out": known_hazard_grad_w_out, "hazard_grad_b_out": known_hazard_grad_b_out,
    }

def accuracy(logits, targets):
    """Fraction of examples where the predicted class (p > 0.5) equals the target."""
    probabilities = binary_cross_entropy.sigmoid(logits)
    return ((probabilities > 0.5).to(targets.dtype) == targets).to(targets.dtype).mean().item()

def test_known_values(setup, known_results):
    inputs, targets, W_hidden, b_hidden, w_out, b_out, lr, steps = setup
    k = known_results

    hidden_pre, hidden, logits = hidden_layer_mlp.mlp_forward_pass(W_hidden, b_hidden, w_out, b_out, inputs)
    probabilities = binary_cross_entropy.sigmoid(logits)
    _, per_example_losses, logit_gradients = binary_cross_entropy.binary_cross_entropy(logits, targets)
    loss, grad_hidden_pre, grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out = hidden_layer_mlp.mlp_backward_pass(W_hidden, b_hidden, w_out, b_out, inputs, targets)

    # Short decimals: exact up to float64 rounding
    torch.testing.assert_close(hidden_pre, k["hidden_pre"], atol=1e-12, rtol=0)
    torch.testing.assert_close(hidden, k["hidden"], atol=1e-12, rtol=0)
    torch.testing.assert_close(logits, k["z"], atol=1e-12, rtol=0)

    # Rounded to ten decimals
    torch.testing.assert_close(probabilities, k["p"], atol=1e-10, rtol=0)
    torch.testing.assert_close(per_example_losses, k["per_example_loss"], atol=1e-10, rtol=0)
    torch.testing.assert_close(logit_gradients, k["grad_z"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_hidden_pre, k["grad_hidden_pre"], atol=1e-10, rtol=0)
    torch.testing.assert_close(loss, k["loss"], atol=1e-10, rtol=0)
    assert accuracy(logits, targets) == k["accuracy"]

    # Shapes exact
    assert grad_W_hidden.shape == (1, 2)
    assert grad_b_hidden.shape == (2,)
    assert grad_w_out.shape == (2,)
    assert grad_b_out.shape == ()
    torch.testing.assert_close(grad_W_hidden, k["grad_W_hidden"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_b_hidden, k["grad_b_hidden"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_w_out, k["grad_w_out"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_b_out, k["grad_b_out"], atol=1e-10, rtol=0)

def test_against_library(library_check):
    inputs, targets, W_hidden, b_hidden, w_out, b_out, lr = library_check

    # Our functions
    hidden_pre, hidden, logits = hidden_layer_mlp.mlp_forward_pass(W_hidden, b_hidden, w_out, b_out, inputs)
    loss, grad_hidden_pre, grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out = hidden_layer_mlp.mlp_backward_pass(W_hidden, b_hidden, w_out, b_out, inputs, targets)

    # Library: nn.Linear stores its weight as (out, in)
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    model.load_state_dict({
        "0.weight": W_hidden.T,
        "0.bias": b_hidden,
        "2.weight": w_out.unsqueeze(0),
        "2.bias": b_out.reshape(1),
    })
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Run the layers one at a time so hidden_pre.grad holds dL/dhidden_pre
    model.train()
    optimizer.zero_grad()
    torch_hidden_pre = model[0](inputs)
    torch_hidden = model[1](torch_hidden_pre)
    torch_logits = model[2](torch_hidden).squeeze(-1)
    torch_hidden_pre.retain_grad()
    torch_loss = loss_fn(torch_logits, targets)
    torch_loss.backward()

    torch.testing.assert_close(hidden_pre, torch_hidden_pre.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(hidden, torch_hidden.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(logits, torch_logits.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(loss, torch_loss.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(grad_hidden_pre, torch_hidden_pre.grad, atol=1e-6, rtol=0)
    torch.testing.assert_close(grad_W_hidden, model[0].weight.grad.T, atol=1e-6, rtol=0)
    torch.testing.assert_close(grad_b_hidden, model[0].bias.grad, atol=1e-6, rtol=0)
    torch.testing.assert_close(grad_w_out, model[2].weight.grad[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(grad_b_out, model[2].bias.grad[0], atol=1e-6, rtol=0)

    # Library step and our step
    optimizer.step()
    new_W_hidden, new_b_hidden, new_w_out, new_b_out = sgd_step.sgd_step(
        [W_hidden, b_hidden, w_out, b_out],
        [grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out],
        lr,
    )

    torch.testing.assert_close(new_W_hidden, model[0].weight.detach().T, atol=1e-6, rtol=0)
    torch.testing.assert_close(new_b_hidden, model[0].bias.detach(), atol=1e-6, rtol=0)
    torch.testing.assert_close(new_w_out, model[2].weight.detach()[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(new_b_out, model[2].bias.detach()[0], atol=1e-6, rtol=0)

def test_hazard(hazard_case, known_results):
    inputs, targets, W_hidden, b_hidden, w_out, b_out = hazard_case
    k = known_results

    hidden_pre, hidden, logits = hidden_layer_mlp.mlp_forward_pass(W_hidden, b_hidden, w_out, b_out, inputs)
    _, _, logit_gradients = binary_cross_entropy.binary_cross_entropy(logits, targets)
    loss, grad_hidden_pre, grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out = hidden_layer_mlp.mlp_backward_pass(W_hidden, b_hidden, w_out, b_out, inputs, targets)

    torch.testing.assert_close(hidden_pre, k["hazard_hidden_pre"], atol=1e-10, rtol=0)
    torch.testing.assert_close(logit_gradients, k["hazard_grad_z"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_hidden_pre, k["hazard_grad_hidden_pre"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_W_hidden, k["hazard_grad_W_hidden"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_b_hidden, k["hazard_grad_b_hidden"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_w_out, k["hazard_grad_w_out"], atol=1e-10, rtol=0)
    torch.testing.assert_close(grad_b_out, k["hazard_grad_b_out"], atol=1e-10, rtol=0)

    # Unit 1 is dead: every gradient that reaches it is exactly 0
    assert (grad_hidden_pre[:, 1] == 0).all()
    assert (grad_W_hidden[:, 1] == 0).all()
    assert grad_b_hidden[1] == 0
    assert grad_w_out[1] == 0

    # Unit 0's pre-activation is exactly 0 at the first example: the mask gives 0 there
    assert hidden_pre[0, 0] == 0
    assert grad_hidden_pre[0, 0] == 0

def test_training(setup, baseline, known_results):
    inputs, targets, W_hidden, b_hidden, w_out, b_out, lr, steps = setup
    k = known_results

    # MLP: 1000 full-batch steps, float64
    parameters = [W_hidden, b_hidden, w_out, b_out]
    for step in range(1, steps + 1):
        _, _, grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out = hidden_layer_mlp.mlp_backward_pass(*parameters, inputs, targets)
        parameters = sgd_step.sgd_step(parameters, [grad_W_hidden, grad_b_hidden, grad_w_out, grad_b_out], lr)

        # Known step values are measured at the updated parameters
        if step == 1:
            loss, *_ = hidden_layer_mlp.mlp_backward_pass(*parameters, inputs, targets)
            for parameter, known in zip(parameters, k["step1"]):
                torch.testing.assert_close(parameter, known, atol=1e-9, rtol=0)
            torch.testing.assert_close(loss, k["step1_loss"], atol=1e-9, rtol=0)

    loss, *_ = hidden_layer_mlp.mlp_backward_pass(*parameters, inputs, targets)
    _, _, logits = hidden_layer_mlp.mlp_forward_pass(*parameters, inputs)
    for parameter, known in zip(parameters, k["step1000"]):
        torch.testing.assert_close(parameter, known, atol=1e-9, rtol=0)
    torch.testing.assert_close(loss, k["step1000_loss"], atol=1e-9, rtol=0)
    assert accuracy(logits, targets) == k["step1000_accuracy"]

    # Single-layer baseline: F03-1-L1's logistic regression, 500 full-batch steps
    weight, bias, baseline_lr, baseline_steps = baseline
    for _ in range(baseline_steps):
        _, weight_gradient, bias_gradient = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)
        weight, bias = sgd_step.sgd_step([weight, bias], [weight_gradient, bias_gradient], baseline_lr)

    baseline_loss, _, _ = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)
    baseline_logits = binary_cross_entropy.forward_pass(weight, bias, inputs)
    torch.testing.assert_close(baseline_loss, k["baseline_loss"], atol=1e-9, rtol=0)
    assert accuracy(baseline_logits, targets) == k["baseline_accuracy"]
