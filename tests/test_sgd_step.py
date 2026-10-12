import evalkit.sgd_step as sgd_step
import torch
import pytest
import torch.nn.functional as F 



@pytest.fixture
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

@pytest.fixture
def two_features():
    inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 1.0]], dtype=torch.float64)
    targets = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    weight = torch.tensor([0.5, -1.0], dtype=torch.float64)
    bias = torch.tensor(0.25, dtype=torch.float64)
    lr = 0.1

    return inputs, targets, weight, bias, lr

@pytest.fixture
def known_results():
    # Each row: (loss, grad_weight, grad_bias, weight_after, bias_after), cut short where the table is blank.
    line_full_step = (
        torch.tensor(3.1866666667, dtype=torch.float64),
        torch.tensor([8.3333333333], dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor([1.0833333333], dtype=torch.float64),
        torch.tensor(-0.05, dtype=torch.float64),
    )
    line_full_loss_after = torch.tensor(1.3625462963, dtype=torch.float64)
    line_batch_2_5 = (
        torch.tensor(4.21, dtype=torch.float64),
        torch.tensor([14.3], dtype=torch.float64),
        torch.tensor(2.8, dtype=torch.float64),
    )
    line_batch_3_0 = (
        torch.tensor(2.61, dtype=torch.float64),
        torch.tensor([2.7], dtype=torch.float64),
        torch.tensor(-1.2, dtype=torch.float64),
    )
    line_batch_1_4 = (
        torch.tensor(2.74, dtype=torch.float64),
        torch.tensor([8.0], dtype=torch.float64),
        torch.tensor(1.4, dtype=torch.float64),
    )
    line_epoch_step_1 = (
        torch.tensor(4.21, dtype=torch.float64),
        torch.tensor([14.3], dtype=torch.float64),
        torch.tensor(2.8, dtype=torch.float64),
        torch.tensor([0.785], dtype=torch.float64),
        torch.tensor(-0.14, dtype=torch.float64),
    )
    line_epoch_step_2 = (
        torch.tensor(3.4679125, dtype=torch.float64),
        torch.tensor([-4.155], dtype=torch.float64),
        torch.tensor(-3.625, dtype=torch.float64),
        torch.tensor([0.99275], dtype=torch.float64),
        torch.tensor(0.04125, dtype=torch.float64),
    )
    line_epoch_step_3 = (
        torch.tensor(0.82390303125, dtype=torch.float64),
        torch.tensor([-0.417], dtype=torch.float64),
        torch.tensor(-1.05375, dtype=torch.float64),
        torch.tensor([1.0136], dtype=torch.float64),
        torch.tensor(0.0939375, dtype=torch.float64),
    )
    line_unseen_full_step = (
        torch.tensor(5.8116666667, dtype=torch.float64),
        torch.tensor([-14.1666666667], dtype=torch.float64),
        torch.tensor(-4.5, dtype=torch.float64),
        torch.tensor([0.2833333333], dtype=torch.float64),
        torch.tensor(1.09, dtype=torch.float64),
    )
    two_features_step = (
        torch.tensor(6.5625, dtype=torch.float64),
        torch.tensor([0.0, -6.0], dtype=torch.float64),
        torch.tensor(-3.5, dtype=torch.float64),
        torch.tensor([0.5, -0.4], dtype=torch.float64),
        torch.tensor(0.6, dtype=torch.float64),
    )
    two_features_loss_after = torch.tensor(2.86, dtype=torch.float64)

    return (
        line_full_step, line_full_loss_after,
        line_batch_2_5, line_batch_3_0, line_batch_1_4,
        line_epoch_step_1, line_epoch_step_2, line_epoch_step_3,
        line_unseen_full_step,
        two_features_step, two_features_loss_after,
    )

def checked(function, *args):
    """Call function, then assert every tensor argument is exactly as it was before the call (check 8)."""
    def copy(arg):
        if isinstance(arg, torch.Tensor):
            return arg.clone()
        if isinstance(arg, list):
            return [copy(a) for a in arg]
        return arg

    def assert_same(arg, before):
        if isinstance(arg, torch.Tensor):
            assert torch.equal(arg, before)
        elif isinstance(arg, list):
            assert len(arg) == len(before)
            for a, b in zip(arg, before):
                assert_same(a, b)

    args_before = [copy(a) for a in args]
    result = function(*args)
    for arg, before in zip(args, args_before):
        assert_same(arg, before)
    return result

def test_full_batch(line_fit, two_features, known_results):
    inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit
    (line_full_step, line_full_loss_after,
    line_batch_2_5, line_batch_3_0, line_batch_1_4,
    line_epoch_step_1, line_epoch_step_2, line_epoch_step_3,
    line_unseen_full_step,
    two_features_step, two_features_loss_after) = known_results

    loss, weight_gradient, bias_gradient = sgd_step.compute_gradient(weight, bias, inputs, targets)

    correct_loss, grad_weight, grad_bias, weight_after, bias_after = line_full_step

    # Reason for tolerance: Rounding error is at most 5e-11
    torch.testing.assert_close(loss, correct_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(weight_gradient, grad_weight, atol=1e-10, rtol=0)
    torch.testing.assert_close(bias_gradient, grad_bias, atol=1e-10, rtol=0)

def test_one_step_full_batch(line_fit, two_features, known_results):
    inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit
    (line_full_step, line_full_loss_after,
    line_batch_2_5, line_batch_3_0, line_batch_1_4,
    line_epoch_step_1, line_epoch_step_2, line_epoch_step_3,
    line_unseen_full_step,
    two_features_step, two_features_loss_after) = known_results

    loss, weight_gradient, bias_gradient = checked(sgd_step.compute_gradient, weight, bias, inputs, targets)
    parameters = [weight, bias]
    gradients = [weight_gradient, bias_gradient]
    new_weight, new_bias = checked(sgd_step.sgd_step, parameters, gradients, lr)    

    correct_loss, grad_weight, grad_bias, weight_after, bias_after = line_full_step
    
    # Reason for tolerance: Table 3 is rounded to ten decimals (error at most 5e-11);
    # float64 arithmetic is accurate to about 1e-15 here.
    torch.testing.assert_close(new_weight,weight_after,atol=1e-10,rtol=0)
    torch.testing.assert_close(new_bias,bias_after,atol=1e-10,rtol=0)

    loss, weight_gradient, bias_gradient = checked(sgd_step.compute_gradient, new_weight, new_bias, inputs, targets)
    # Reason for tolerance: Table 3 is rounded to ten decimals (error at most 5e-11);
    # float64 arithmetic is accurate to about 1e-15 here.
    torch.testing.assert_close(loss, line_full_loss_after, atol=1e-10, rtol=0)

def test_batch_sizes(line_fit, two_features, known_results):
    inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit
    (line_full_step, line_full_loss_after,
    line_batch_2_5, line_batch_3_0, line_batch_1_4,
    line_epoch_step_1, line_epoch_step_2, line_epoch_step_3,
    line_unseen_full_step,
    two_features_step, two_features_loss_after) = known_results

    # Full batch
    fb_loss, fb_weight_gradient, fb_bias_gradient = checked(sgd_step.compute_gradient, weight, bias, inputs, targets)

    batch_weight_gradients = []
    batch_bias_gradients = []
    target_answer_list = [line_batch_2_5, line_batch_3_0, line_batch_1_4]
    for idx, target in enumerate(target_answer_list):
        batch = batches[idx]

        batch_inputs = inputs[batch]
        batch_targets = targets[batch]

        loss, weight_gradient, bias_gradient = checked(sgd_step.compute_gradient, weight, bias, batch_inputs, batch_targets)

        correct_loss, grad_weight, grad_bias = target

        # Reason for tolerance: the batch values are short decimals, exact in Table 3,
        # so only float64 rounding (about 1e-15) remains.
        torch.testing.assert_close(loss, correct_loss, atol=1e-12, rtol=0)
        torch.testing.assert_close(weight_gradient, grad_weight, atol=1e-12, rtol=0)
        torch.testing.assert_close(bias_gradient, grad_bias, atol=1e-12, rtol=0)
        batch_weight_gradients.append(weight_gradient)
        batch_bias_gradients.append(bias_gradient)

    batch_weight_gradient_tensors = torch.stack(batch_weight_gradients)
    batch_bias_gradient_tensors = torch.stack(batch_bias_gradients)
    mean_weight = batch_weight_gradient_tensors.mean(dim=0)
    mean_bias = batch_bias_gradient_tensors.mean(dim=0)

    # Reason for tolerance: averaging three short-decimal gradients adds only float64 rounding.
    torch.testing.assert_close(mean_weight, fb_weight_gradient, atol=1e-12, rtol=0)
    torch.testing.assert_close(mean_bias, fb_bias_gradient, atol=1e-12, rtol=0)

def test_one_epoch(line_fit, two_features, known_results):
    inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit
    (line_full_step, line_full_loss_after,
    line_batch_2_5, line_batch_3_0, line_batch_1_4,
    line_epoch_step_1, line_epoch_step_2, line_epoch_step_3,
    line_unseen_full_step,
    two_features_step, two_features_loss_after) = known_results

    target_answer_list = [line_epoch_step_1, line_epoch_step_2, line_epoch_step_3]
    for idx, target in enumerate(target_answer_list):
        batch = batches[idx]

        batch_inputs = inputs[batch]
        batch_targets = targets[batch]

        loss, weight_gradient, bias_gradient = checked(sgd_step.compute_gradient, weight, bias, batch_inputs, batch_targets)
        parameters = [weight,bias]
        gradients = [weight_gradient,bias_gradient]
        new_weight, new_bias = checked(sgd_step.sgd_step, parameters, gradients, lr)
        correct_loss, grad_weight, grad_bias, weight_after, bias_after = target 

        # Reason for tolerance: the epoch rows are exact decimals in Table 3; three float64
        # steps add rounding of about 1e-15, far below 1e-10.
        torch.testing.assert_close(loss, correct_loss, atol=1e-10, rtol=0)
        torch.testing.assert_close(new_weight, weight_after, atol=1e-10, rtol=0)
        torch.testing.assert_close(new_bias, bias_after, atol=1e-10, rtol=0)

        # The next batch starts from where this step left off.
        weight, bias = new_weight, new_bias

def test_unseen_start_two_features(line_fit, two_features, known_results):
    inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit
    two_f_inputs, two_f_targets, two_f_weight, two_f_bias, two_f_lr = two_features
    (line_full_step, line_full_loss_after,
    line_batch_2_5, line_batch_3_0, line_batch_1_4,
    line_epoch_step_1, line_epoch_step_2, line_epoch_step_3,
    line_unseen_full_step,
    two_features_step, two_features_loss_after) = known_results

    
    fb_loss, fb_weight_gradient, fb_bias_gradient = checked(sgd_step.compute_gradient, unseen_weight, unseen_bias, inputs, targets)
    unseen_parameters = [unseen_weight, unseen_bias]
    unseen_gradients = [fb_weight_gradient, fb_bias_gradient]
    unseen_new_weight, unseen_new_bias = checked(sgd_step.sgd_step, unseen_parameters, unseen_gradients, unseen_lr)


    correct_loss, grad_weight, grad_bias, weight_after, bias_after = line_unseen_full_step
    # Reason for tolerance: Table 3 is rounded to ten decimals (error at most 5e-11);
    # float64 arithmetic is accurate to about 1e-15 here.
    torch.testing.assert_close(fb_loss, correct_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(fb_weight_gradient, grad_weight, atol=1e-10, rtol=0)
    torch.testing.assert_close(fb_bias_gradient, grad_bias, atol=1e-10, rtol=0)
    torch.testing.assert_close(unseen_new_weight, weight_after, atol=1e-10, rtol=0)
    torch.testing.assert_close(unseen_new_bias, bias_after, atol=1e-10, rtol=0)


    two_f_loss, two_f_weight_gradient, two_f_bias_gradient = checked(sgd_step.compute_gradient, two_f_weight, two_f_bias, two_f_inputs, two_f_targets)
    
    two_f_parameters = [two_f_weight, two_f_bias]
    two_f_gradients = [two_f_weight_gradient, two_f_bias_gradient]
    
    two_f_new_weight, two_f_new_bias = checked(sgd_step.sgd_step, two_f_parameters, two_f_gradients, two_f_lr)
   
    correct_loss, grad_weight, grad_bias, weight_after, bias_after = two_features_step
    # Reason for tolerance: the two-feature rows are exact decimals in Table 3,
    # so only float64 rounding (about 1e-15) remains.
    torch.testing.assert_close(two_f_loss, correct_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(two_f_weight_gradient, grad_weight, atol=1e-10, rtol=0)
    torch.testing.assert_close(two_f_bias_gradient, grad_bias, atol=1e-10, rtol=0)
    torch.testing.assert_close(two_f_new_weight, weight_after, atol=1e-10, rtol=0)
    torch.testing.assert_close(two_f_new_bias, bias_after, atol=1e-10, rtol=0)

    two_f_loss_after, _, _ = checked(sgd_step.compute_gradient, two_f_new_weight, two_f_new_bias, two_f_inputs, two_f_targets)
    # Reason for tolerance: same as the two-feature row above.
    torch.testing.assert_close(two_f_loss_after, two_features_loss_after, atol=1e-10, rtol=0)

    # No tolerance: the first feature's gradient is exactly zero, so 0.5 - lr * 0.0 is exactly 0.5.
    assert two_f_weight_gradient[0] == 0.0
    assert two_f_new_weight[0] == 0.5

def test_against_library(line_fit, two_features, known_results):
    inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit
    two_f_inputs, two_f_targets, two_f_weight, two_f_bias, two_f_lr = two_features
    (line_full_step, line_full_loss_after,
    line_batch_2_5, line_batch_3_0, line_batch_1_4,
    line_epoch_step_1, line_epoch_step_2, line_epoch_step_3,
    line_unseen_full_step,
    two_features_step, two_features_loss_after) = known_results

    g = torch.Generator().manual_seed(0)
    inputs = torch.randn(32,8, generator=g)
    true_weight = torch.randn(8,generator=g)
    targets = inputs @ true_weight + 0.5 + 0.1 * torch.randn(32, generator=g)

    weight = torch.zeros(8)
    bias = torch.tensor(0.0)

    lib_weight = torch.zeros(8, requires_grad=True)
    lib_bias = torch.tensor(0.0, requires_grad=True)

    # Our loss
    loss, weight_gradient, bias_gradient = checked(sgd_step.compute_gradient, weight, bias, inputs, targets)
    # Our step
    parameters = [weight, bias]
    gradients = [weight_gradient, bias_gradient]
    new_weight, new_bias = checked(sgd_step.sgd_step, parameters, gradients, lr)

    # Torch loss
    lib_loss = F.mse_loss(inputs @ lib_weight + lib_bias, targets)
    lib_loss.backward()

    # Compare values
    # Reason for tolerance: a different order of the same float32 sums can move a gradient
    # by a few spacings (measured up to 4.8e-7).
    torch.testing.assert_close(loss, lib_loss.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(weight_gradient, lib_weight.grad, atol=1e-5, rtol=0)
    torch.testing.assert_close(bias_gradient, lib_bias.grad, atol=1e-5, rtol=0)

    # Torch step
    opt = torch.optim.SGD([lib_weight, lib_bias], lr=lr)
    opt.step()

    # Compare
    # Reason for tolerance: the gradient error is scaled by lr, and the parameters stay below 2
    # in magnitude, where the float32 spacing is at most 2.4e-7 (measured up to 3.0e-8).
    torch.testing.assert_close(new_weight, lib_weight.detach(), atol=1e-6, rtol=0)
    torch.testing.assert_close(new_bias, lib_bias.detach(), atol=1e-6, rtol=0)
    weight, bias = new_weight, new_bias

    # 199 more steps, 200 in total.
    for _ in range(199):
        # Torch
        opt.zero_grad()
        lib_loss = F.mse_loss(inputs @ lib_weight + lib_bias, targets)
        lib_loss.backward()
        opt.step()

        # Ours
        loss, weight_gradient, bias_gradient = checked(sgd_step.compute_gradient, weight, bias, inputs, targets)
        parameters = [weight, bias]
        gradients = [weight_gradient, bias_gradient]
        new_weight, new_bias = checked(sgd_step.sgd_step, parameters, gradients, lr)

        # Reason for tolerance: small differences can add up over 200 steps; 1e-5 is the
        # notebook's tolerance (measured up to 1.2e-7).
        torch.testing.assert_close(new_weight, lib_weight.detach(), atol=1e-5, rtol=0)
        torch.testing.assert_close(new_bias, lib_bias.detach(), atol=1e-5, rtol=0)
        weight, bias = new_weight, new_bias

def test_shape_hazard(line_fit, two_features, known_results):
    inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit
    two_f_inputs, two_f_targets, two_f_weight, two_f_bias, two_f_lr = two_features
    (line_full_step, line_full_loss_after,
    line_batch_2_5, line_batch_3_0, line_batch_1_4,
    line_epoch_step_1, line_epoch_step_2, line_epoch_step_3,
    line_unseen_full_step,
    two_features_step, two_features_loss_after) = known_results

    test_tar_error = torch.tensor([[1.0], [1.1], [1.2], [1.3], [1.4], [1.5]], dtype=torch.float64)
    test_input_error = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=torch.float64)
    test_wrong_tar_shape = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4], dtype=torch.float64)
    test_weight_error = torch.tensor([2.0, 2.1], dtype=torch.float64)
    test_grad_error = torch.tensor([1.0], dtype=torch.float64)
    test_grad_error_2 = torch.tensor([1.0, 1.1, 1.2], dtype=torch.float64)

    # Targets (6, 1): (6,) - (6, 1) would broadcast to (6, 6) without an error.
    with pytest.raises(ValueError):
        sgd_step.compute_gradient(weight, bias, inputs, test_tar_error)

    # Inputs (6,): no feature axis.
    with pytest.raises(ValueError):
        sgd_step.compute_gradient(weight, bias, test_input_error, targets)

    # Inputs (6, 1) with targets (5,): batch sizes disagree.
    with pytest.raises(ValueError):
        sgd_step.compute_gradient(weight, bias, inputs, test_wrong_tar_shape)

    # Weight (2,) with inputs (6, 1): feature counts disagree.
    with pytest.raises(ValueError):
        sgd_step.compute_gradient(test_weight_error, bias, inputs, targets)

    # Two parameters and one gradient.
    with pytest.raises(ValueError):
        sgd_step.sgd_step([weight, bias], [test_grad_error], lr)

    # Parameter (2,) with gradient (3,).
    with pytest.raises(ValueError):
        sgd_step.sgd_step([test_weight_error], [test_grad_error_2], lr)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_any_shape_arguments_unchanged(line_fit, dtype):
    inputs, targets, weight, bias, lr, batches, unseen_weight, unseen_bias, unseen_lr = line_fit

    g = torch.Generator().manual_seed(0)
    parameters = [
        torch.randn(8, generator=g, dtype=dtype),
        torch.randn((), generator=g, dtype=dtype),
        torch.randn(2, 3, generator=g, dtype=dtype),
    ]
    gradients = [torch.randn(p.shape, generator=g, dtype=dtype) for p in parameters]
    parameters_before = [p.clone() for p in parameters]
    gradients_before = [gr.clone() for gr in gradients]

    updated = sgd_step.sgd_step(parameters, gradients, lr)

    assert len(updated) == 3
    for new_p, p, gr in zip(updated, parameters_before, gradients_before):
        assert new_p.shape == p.shape
        assert new_p.dtype == dtype
        # Reason for tolerance: the expected value is computed the same way, one multiply and
        # one subtract per entry with no sums, so only rounding in the last bit can differ.
        torch.testing.assert_close(new_p, p - lr * gr, atol=1e-12, rtol=0)

    # The tensors passed in are exactly unchanged.
    for p, p_before in zip(parameters, parameters_before):
        assert torch.equal(p, p_before)
    for gr, gr_before in zip(gradients, gradients_before):
        assert torch.equal(gr, gr_before)

    # compute_gradient also leaves its arguments alone and keeps the input dtype.
    weight, bias, inputs, targets = (t.to(dtype) for t in (weight, bias, inputs, targets))
    arguments_before = [t.clone() for t in (weight, bias, inputs, targets)]

    results = sgd_step.compute_gradient(weight, bias, inputs, targets)

    for t, t_before in zip((weight, bias, inputs, targets), arguments_before):
        assert torch.equal(t, t_before)
    for r in results:
        assert r.dtype == dtype
