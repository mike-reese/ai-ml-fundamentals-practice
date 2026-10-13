import evalkit.binary_cross_entropy as binary_cross_entropy
from evalkit import sgd_step
import torch
import pytest
import torch.nn.functional as F 

@pytest.fixture
def logistic_fit():
    inputs = torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 0, 1, 0, 0, 1, 1], dtype=torch.float64)
    weight = torch.tensor([0.5], dtype=torch.float64)
    bias = torch.tensor(0.0, dtype=torch.float64)
    lr = torch.tensor(0.5, dtype=torch.float64)
    steps = torch.tensor(500, dtype=torch.int64)
    unseen_weight = torch.tensor([-1.0], dtype=torch.float64)
    unseen_bias = torch.tensor(2.0, dtype=torch.float64)
    saturated_logits = torch.tensor([40.0, -40.0, 100.0, -100.0], dtype=torch.float32)
    saturated_targets = torch.tensor([0, 1, 1, 0], dtype=torch.float32)

    return inputs, targets, weight, bias, lr, steps, unseen_weight, unseen_bias, saturated_logits, saturated_targets

@pytest.fixture
def known_results():
    known_z = torch.tensor([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], dtype=torch.float64)
    known_p = torch.tensor([0.5621765009, 0.6224593312, 0.6791786992, 0.7310585786, 0.7772998612, 0.8175744762, 0.8519528020, 0.8807970780], dtype=torch.float64)
    known_per_example_loss = torch.tensor([0.8259394199, 0.9740769842, 1.1368710061, 0.3132616875, 1.5019290813, 1.7014132780, 0.1602241504, 0.1269280110], dtype=torch.float64)
    known_loss = torch.tensor(0.8425804523, dtype=torch.float64)
    known_grad_z = torch.tensor([0.0702720626, 0.0778074164, 0.0848973374, -0.0336176777, 0.0971624826, 0.1021968095, -0.0185058998, -0.0149003653], dtype=torch.float64)
    known_grad_weight = torch.tensor([0.5981786235], dtype=torch.float64)
    known_grad_bias = torch.tensor(0.3653121659, dtype=torch.float64)
    known_step1_weight = torch.tensor([0.2009106883], dtype=torch.float64)
    known_step1_bias = torch.tensor(-0.1826560830, dtype=torch.float64)
    known_step1_loss = torch.tensor(0.6732824348, dtype=torch.float64)
    known_step500_weight = torch.tensor([1.5589395535], dtype=torch.float64)
    known_step500_bias = torch.tensor(-4.3571551186, dtype=torch.float64)
    known_step500_loss = torch.tensor(0.4362370264, dtype=torch.float64)
    known_step500_p = torch.tensor([0.0271808546, 0.0574206796, 0.1172487413, 0.2245620315, 0.3870317913, 0.5792422525, 0.7500976804, 0.8674505927], dtype=torch.float64)
    known_unseen_z = torch.tensor([1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0], dtype=torch.float64)
    known_unseen_p = torch.tensor([0.8175744762, 0.7310585786, 0.6224593312, 0.5, 0.3775406688, 0.2689414214, 0.1824255238, 0.1192029220], dtype=torch.float64)
    known_unseen_per_example_loss = torch.tensor([1.7014132780, 1.3132616875, 0.9740769842, 0.6931471806, 0.4740769842, 0.3132616875, 1.7014132780, 2.1269280110], dtype=torch.float64)
    known_unseen_loss = torch.tensor(1.1621973864, dtype=torch.float64)
    known_unseen_grad_z = torch.tensor([0.1021968095, 0.0913823223, 0.0778074164, -0.0625, 0.0471925836, 0.0336176777, -0.1021968095, -0.1100996347], dtype=torch.float64)
    known_unseen_grad_weight = torch.tensor([-0.4450610286], dtype=torch.float64)
    known_unseen_grad_bias = torch.tensor(0.0774003653, dtype=torch.float64)
    known_saturated_p = torch.tensor([1.0, 4.2484e-18, 1.0, 3.7835e-44], dtype=torch.float32)
    known_saturated_per_example_loss = torch.tensor([40.0, 40.0, 0.0, 0.0], dtype=torch.float32)
    known_saturated_loss = torch.tensor(20.0, dtype=torch.float32)
    known_saturated_grad_z = torch.tensor([0.25, -0.25, 0.0, 0.0], dtype=torch.float32)

    return (
        known_z, known_p, known_per_example_loss, known_loss, known_grad_z, known_grad_weight, known_grad_bias,
        known_step1_weight, known_step1_bias, known_step1_loss,
        known_step500_weight, known_step500_bias, known_step500_loss, known_step500_p,
        known_unseen_z, known_unseen_p, known_unseen_per_example_loss, known_unseen_loss, known_unseen_grad_z, known_unseen_grad_weight, known_unseen_grad_bias,
        known_saturated_p, known_saturated_per_example_loss, known_saturated_loss, known_saturated_grad_z,
    )

def test_known_values(logistic_fit, known_results):
    inputs, targets, weight, bias, lr, steps, unseen_weight, unseen_bias, saturated_logits, saturated_targets = logistic_fit
    (
        known_z, known_p, known_per_example_loss, known_loss, known_grad_z, known_grad_weight, known_grad_bias,
        known_step1_weight, known_step1_bias, known_step1_loss,
        known_step500_weight, known_step500_bias, known_step500_loss, known_step500_p,
        known_unseen_z, known_unseen_p, known_unseen_per_example_loss, known_unseen_loss, known_unseen_grad_z, known_unseen_grad_weight, known_unseen_grad_bias,
        known_saturated_p, known_saturated_per_example_loss, known_saturated_loss, known_saturated_grad_z,
    ) = known_results


    logits = binary_cross_entropy.forward_pass(weight, bias, inputs)
    probabilities = binary_cross_entropy.sigmoid(logits)
    loss, per_example_losses, logit_gradient = binary_cross_entropy.binary_cross_entropy(logits, targets)
    lr_loss, weight_gradient, bias_gradient = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)

    torch.testing.assert_close(logits, known_z, atol=1e-12, rtol=0)
    torch.testing.assert_close(probabilities, known_p, atol=1e-10, rtol=0)
    torch.testing.assert_close(per_example_losses, known_per_example_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(loss, known_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(logit_gradient, known_grad_z, atol=1e-10, rtol=0)
    torch.testing.assert_close(weight_gradient, known_grad_weight, atol=1e-10, rtol=0)
    torch.testing.assert_close(bias_gradient, known_grad_bias, atol=1e-10, rtol=0)

    # Unseen start: logits run from 1.5 down to -2.0, so the negative-logit sigmoid branch is used
    unseen_logits = binary_cross_entropy.forward_pass(unseen_weight, unseen_bias, inputs)
    unseen_probabilities = binary_cross_entropy.sigmoid(unseen_logits)
    unseen_loss, unseen_per_example_losses, unseen_logit_gradient = binary_cross_entropy.binary_cross_entropy(unseen_logits, targets)
    unseen_lr_loss, unseen_weight_gradient, unseen_bias_gradient = binary_cross_entropy.logistic_regression_gradient(unseen_weight, unseen_bias, inputs, targets)

    torch.testing.assert_close(unseen_logits, known_unseen_z, atol=1e-12, rtol=0)
    torch.testing.assert_close(unseen_probabilities, known_unseen_p, atol=1e-10, rtol=0)
    torch.testing.assert_close(unseen_per_example_losses, known_unseen_per_example_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(unseen_loss, known_unseen_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(unseen_logit_gradient, known_unseen_grad_z, atol=1e-10, rtol=0)
    torch.testing.assert_close(unseen_weight_gradient, known_unseen_grad_weight, atol=1e-10, rtol=0)
    torch.testing.assert_close(unseen_bias_gradient, known_unseen_grad_bias, atol=1e-10, rtol=0)

def test_against_library(logistic_fit, known_results):

    # Torch Library 
    g = torch.Generator().manual_seed(0)
    inputs = torch.randn(64, 8, generator=g)
    weight = 2 * torch.randn(8, generator=g)
    targets = torch.randint(0, 2, (64,), generator=g).float()
    bias = torch.tensor(0.5)    
    lr = 0.5

    # Our functions 
    logits = binary_cross_entropy.forward_pass(weight, bias, inputs)
    probabilities = binary_cross_entropy.sigmoid(logits)
    loss, per_example_losses, logit_gradient = binary_cross_entropy.binary_cross_entropy(logits, targets)
    lr_loss, weight_gradient, bias_gradient = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)

    # Torch functions
    model = torch.nn.Linear(8, 1)
    model.load_state_dict({
        "weight": weight.unsqueeze(0),
        "bias": bias.unsqueeze(0)
    }) 

    loss_fn = torch.nn.BCEWithLogitsLoss()
    per_example_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Library Training through to the backward pass (no step yet)
    model.train()
    optimizer.zero_grad()
    torch_logits = model(inputs).squeeze(-1)
    torch_loss = loss_fn(torch_logits, targets)
    torch_loss.backward()

    torch_per_example_loss = per_example_loss_fn(torch_logits,targets)

    with torch.no_grad():
        probs = torch.sigmoid(torch_logits)
    # Checks
    torch.testing.assert_close(logits, torch_logits.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(probabilities, probs, atol=1e-6, rtol=0)
    torch.testing.assert_close(per_example_losses, torch_per_example_loss.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(loss, torch_loss.detach(), atol=1e-6, rtol=0)
    torch.testing.assert_close(weight_gradient, model.weight.grad[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(bias_gradient, model.bias.grad[0], atol=1e-7, rtol=0)
    
    # Library Step
    optimizer.step()

    # Our step
    parameters = [weight, bias]
    gradients = [weight_gradient, bias_gradient]
    new_weight, new_bias = sgd_step.sgd_step(parameters, gradients, lr)

    torch.testing.assert_close(new_weight, model.weight.detach()[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(new_bias, model.bias.detach()[0], atol=1e-6, rtol=0)


def test_hazard(logistic_fit, known_results):
    inputs, targets, weight, bias, lr, steps, unseen_weight, unseen_bias, saturated_logits, saturated_targets = logistic_fit
    (
        known_z, known_p, known_per_example_loss, known_loss, known_grad_z, known_grad_weight, known_grad_bias,
        known_step1_weight, known_step1_bias, known_step1_loss,
        known_step500_weight, known_step500_bias, known_step500_loss, known_step500_p,
        known_unseen_z, known_unseen_p, known_unseen_per_example_loss, known_unseen_loss, known_unseen_grad_z, known_unseen_grad_weight, known_unseen_grad_bias,
        known_saturated_p, known_saturated_per_example_loss, known_saturated_loss, known_saturated_grad_z,
    ) = known_results
   
    probabilities = binary_cross_entropy.sigmoid(saturated_logits)
    loss, per_example_losses, logit_gradient = binary_cross_entropy.binary_cross_entropy(saturated_logits, saturated_targets)

    torch.testing.assert_close(probabilities, known_saturated_p, atol=1e-7, rtol=0)
    torch.testing.assert_close(per_example_losses, known_saturated_per_example_loss, atol=1e-6, rtol=0)
    torch.testing.assert_close(loss, known_saturated_loss, atol=1e-6, rtol=0)
    torch.testing.assert_close(logit_gradient, known_saturated_grad_z, atol=1e-7, rtol=0)

    # The probability route gives inf here; every stable output must be finite
    for output in (probabilities, per_example_losses, loss, logit_gradient):
        assert torch.isfinite(output).all()

def test_training(logistic_fit, known_results):
    inputs, targets, weight, bias, lr, steps, unseen_weight, unseen_bias, saturated_logits, saturated_targets = logistic_fit
    (
        known_z, known_p, known_per_example_loss, known_loss, known_grad_z, known_grad_weight, known_grad_bias,
        known_step1_weight, known_step1_bias, known_step1_loss,
        known_step500_weight, known_step500_bias, known_step500_loss, known_step500_p,
        known_unseen_z, known_unseen_p, known_unseen_per_example_loss, known_unseen_loss, known_unseen_grad_z, known_unseen_grad_weight, known_unseen_grad_bias,
        known_saturated_p, known_saturated_per_example_loss, known_saturated_loss, known_saturated_grad_z,
    ) = known_results

    # Loop 1 - float64, our functions against known values

    for i in range(500):
        # Our functions 
        logits = binary_cross_entropy.forward_pass(weight, bias, inputs)
        probabilities = binary_cross_entropy.sigmoid(logits)
        loss, per_example_losses, logit_gradient = binary_cross_entropy.binary_cross_entropy(logits, targets)
        lr_loss, weight_gradient, bias_gradient = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)

        # Our step
        parameters = [weight, bias]
        gradients = [weight_gradient, bias_gradient]
        weight, bias = sgd_step.sgd_step(parameters, gradients, lr)

        # Known step values are measured at the updated parameters
        logits = binary_cross_entropy.forward_pass(weight, bias, inputs)
        probabilities = binary_cross_entropy.sigmoid(logits)
        loss, per_example_losses, logit_gradient = binary_cross_entropy.binary_cross_entropy(logits, targets)

        if i == 0:
            torch.testing.assert_close(weight, known_step1_weight, atol=1e-9, rtol=0)
            torch.testing.assert_close(bias, known_step1_bias, atol=1e-9, rtol=0)
            torch.testing.assert_close(loss, known_step1_loss, atol=1e-9, rtol=0)

    torch.testing.assert_close(weight, known_step500_weight, atol=1e-9, rtol=0)
    torch.testing.assert_close(bias, known_step500_bias, atol=1e-9, rtol=0)
    torch.testing.assert_close(loss, known_step500_loss, atol=1e-9, rtol=0)
    torch.testing.assert_close(probabilities, known_step500_p, atol=1e-9, rtol=0)


    # Loop 2 - float32 against library
    inputs, targets, weight, bias, lr, steps, unseen_weight, unseen_bias, saturated_logits, saturated_targets = logistic_fit

    inputs = inputs.float()
    targets = targets.float()
    weight = weight.float()
    bias = bias.float()

    # Library instantiation    
    model = torch.nn.Linear(1,1)
    model.load_state_dict({
        "weight": weight.unsqueeze(0),
        "bias": bias.unsqueeze(0)
    }) 

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()

    # Steps 0 to 500: compare both sides before each update, 501 comparisons; step 500 has no update
    for i in range(501):
        # Our forward pass, loss and backward pass at the current parameters
        loss, weight_gradient, bias_gradient = binary_cross_entropy.logistic_regression_gradient(weight, bias, inputs, targets)

        # Torch forward pass and loss at the current parameters
        optimizer.zero_grad()
        torch_logits = model(inputs).squeeze(-1)
        torch_loss = loss_fn(torch_logits, targets)

        # Compare before this step's update
        torch.testing.assert_close(weight, model.weight.detach()[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(bias, model.bias.detach()[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(loss, torch_loss.detach(), atol=1e-6, rtol=0)

        if i < 500:
            # Torch backward pass and step
            torch_loss.backward()
            optimizer.step()

            # Our step
            parameters = [weight, bias]
            gradients = [weight_gradient, bias_gradient]
            weight, bias = sgd_step.sgd_step(parameters, gradients, lr.item())
        
