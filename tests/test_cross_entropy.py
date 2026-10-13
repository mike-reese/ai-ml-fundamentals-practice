import torch
import torch.nn as nn
import pytest
import evalkit.cross_entropy as cross_entropy
import evalkit.sgd_step as sgd_step
import evalkit.softmax as softmax
from evalkit.binary_cross_entropy import forward_pass
import torch.nn.functional as F

@pytest.fixture
def setup():
    classifier_inputs = torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]], dtype=torch.float64)
    classifier_targets = torch.tensor([0, 0, 0, 1, 1, 2, 1, 2], dtype=torch.int64)
    classifier_weight = torch.tensor([[-0.5, 0.25, 1.0]], dtype=torch.float64)
    classifier_bias = torch.tensor([1.0, 0.0, -2.5], dtype=torch.float64)
    classifier_lr = torch.tensor(0.5, dtype=torch.float64)
    classifier_steps = torch.tensor(300, dtype=torch.int64)
    scorer_inputs = torch.tensor([
        [[0.5, 0.8, 1.0, 1.0], [0.9, 0.6, 0.0, 1.0], [0.2, 0.3, 1.0, 0.0]],
        [[0.1, 0.2, 1.0, 1.0], [0.7, 0.9, 1.0, 1.0], [0.4, 0.4, 0.0, 0.0]],
        [[0.8, 0.7, 0.0, 0.0], [0.3, 0.5, 1.0, 1.0], [0.6, 0.9, 1.0, 0.0]],
        [[0.2, 0.1, 1.0, 0.0], [0.9, 0.8, 0.0, 1.0], [0.6, 0.7, 1.0, 1.0]],
        [[0.4, 0.9, 1.0, 1.0], [0.5, 0.3, 1.0, 1.0], [0.9, 0.9, 0.0, 0.0]],
        [[0.4, 0.9, 1.0, 1.0], [0.5, 0.3, 1.0, 1.0], [0.1, 0.2, 0.0, 1.0]],
    ], dtype=torch.float64)
    scorer_targets = torch.tensor([0, 1, 1, 2, 0, 1], dtype=torch.int64)
    scorer_weight = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    scorer_bias = torch.tensor(-1.0, dtype=torch.float64)
    scorer_lr = torch.tensor(2.0, dtype=torch.float64)
    scorer_steps = torch.tensor(200, dtype=torch.int64)
    saturated_logits = torch.tensor([[100.0, 0.0, -100.0], [100.0, 0.0, -100.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    saturated_targets = torch.tensor([2, 0, 1], dtype=torch.int64)

    return (
        classifier_inputs, classifier_targets, classifier_weight, classifier_bias, classifier_lr, classifier_steps,
        scorer_inputs, scorer_targets, scorer_weight, scorer_bias, scorer_lr, scorer_steps,
        saturated_logits, saturated_targets,
    )

@pytest.fixture
def library_check():
    g = torch.Generator().manual_seed(0)
    classifier_inputs = torch.randn(64, 8, generator=g)
    classifier_weight = 2 * torch.randn(8, 5, generator=g)
    classifier_bias = torch.randn(5, generator=g)
    classifier_targets = torch.randint(0, 5, (64,), generator=g)
    classifier_lr = torch.tensor(0.5, dtype=torch.float32)

    g = torch.Generator().manual_seed(0)
    scorer_inputs = torch.randn(32, 5, 8, generator=g)
    scorer_weight = 2 * torch.randn(8, generator=g)
    scorer_targets = torch.randint(0, 5, (32,), generator=g)
    scorer_bias = torch.tensor(0.5, dtype=torch.float32)
    scorer_lr = torch.tensor(0.5, dtype=torch.float32)

    return (
        classifier_inputs, classifier_weight, classifier_bias, classifier_targets, classifier_lr,
        scorer_inputs, scorer_weight, scorer_targets, scorer_bias, scorer_lr,
    )

@pytest.fixture
def known_classifier_init():
    known_y = torch.tensor([0, 0, 0, 1, 1, 2, 1, 2], dtype=torch.int64)
    known_z = torch.tensor([
        [0.75, 0.125, -2.0],
        [0.5, 0.25, -1.5],
        [0.25, 0.375, -1.0],
        [0.0, 0.5, -0.5],
        [-0.25, 0.625, 0.0],
        [-0.5, 0.75, 0.5],
        [-0.75, 0.875, 1.0],
        [-1.0, 1.0, 1.5],
    ], dtype=torch.float64)
    known_p = torch.tensor([
        [0.6253168442, 0.3347079873, 0.0399751684],
        [0.5224289002, 0.4068680366, 0.0707030632],
        [0.4132823577, 0.4683102643, 0.1184073779],
        [0.3071958857, 0.5064803911, 0.1863237232],
        [0.2135428577, 0.5122626855, 0.2741944568],
        [0.1387227148, 0.4841898505, 0.3770874347],
        [0.0845092677, 0.4291734741, 0.4863172582],
        [0.0486108240, 0.3591881058, 0.5922010702],
    ], dtype=torch.float64)
    known_per_example_loss = torch.tensor([0.4694968069, 0.6492663805, 0.8836242447, 0.6802696706, 0.6689177279, 0.9752781961, 0.8458940734, 0.5239090562], dtype=torch.float64)
    known_grad_z = torch.tensor([
        [-0.0468353945, 0.0418384984, 0.0049968961],
        [-0.0596963875, 0.0508585046, 0.0088378829],
        [-0.0733397053, 0.0585387830, 0.0148009222],
        [0.0383994857, -0.0616899511, 0.0232904654],
        [0.0266928572, -0.0609671643, 0.0342743071],
        [0.0173403393, 0.0605237313, -0.0778640707],
        [0.0105636585, -0.0713533157, 0.0607896573],
        [0.0060763530, 0.0448985132, -0.0509748662],
    ], dtype=torch.float64)

    return known_y, known_z, known_p, known_per_example_loss, known_grad_z

@pytest.fixture
def known_scorer_init():
    known_y = torch.tensor([0, 1, 1, 2, 0, 1], dtype=torch.int64)
    known_z = torch.tensor([
        [0.3, 0.5, -0.5],
        [-0.7, 0.6, -0.2],
        [0.5, -0.2, 0.5],
        [-0.7, 0.7, 0.3],
        [0.3, -0.2, 0.8],
        [0.3, -0.2, -0.7],
    ], dtype=torch.float64)
    known_p = torch.tensor([
        [0.3744292216, 0.4573288841, 0.1682418943],
        [0.1582774867, 0.5807670544, 0.2609554589],
        [0.4005470987, 0.1989058027, 0.4005470987],
        [0.1286424830, 0.5216709930, 0.3496865240],
        [0.3071958857, 0.1863237232, 0.5064803911],
        [0.5064803911, 0.3071958857, 0.1863237232],
    ], dtype=torch.float64)
    known_per_example_loss = torch.tensor([0.9823524882, 0.5434055416, 1.6149239197, 1.0507181715, 1.1802696706, 1.1802696706], dtype=torch.float64)
    known_grad_z = torch.tensor([
        [-0.1042617964, 0.0762214807, 0.0280403157],
        [0.0263795811, -0.0698721576, 0.0434925765],
        [0.0667578498, -0.1335156996, 0.0667578498],
        [0.0214404138, 0.0869451655, -0.1083855793],
        [-0.1154673524, 0.0310539539, 0.0844133985],
        [0.0844133985, -0.1154673524, 0.0310539539],
    ], dtype=torch.float64)

    return known_y, known_z, known_p, known_per_example_loss, known_grad_z

@pytest.fixture
def known_training():
    # K1
    known_classifier_loss = torch.tensor(0.7120820195, dtype=torch.float64)
    known_scorer_loss = torch.tensor(1.0919899104, dtype=torch.float64)
    # K2
    known_classifier_grad_weight = torch.tensor([[0.0637067065, -0.0047832429, -0.0589234636]], dtype=torch.float64)
    known_scorer_grad_weight = torch.tensor([0.0885638131, -0.0046811995, -0.3888844248, -0.3109024041], dtype=torch.float64)
    # K3
    known_classifier_grad_bias = torch.tensor([-0.0807987935, 0.0626475994, 0.0181511941], dtype=torch.float64)
    known_scorer_grad_bias = torch.tensor(0.0, dtype=torch.float64)
    # K4
    known_classifier_step1_weight = torch.tensor([[-0.5318533533, 0.2523916215, 1.0294617318]], dtype=torch.float64)
    known_classifier_step1_bias = torch.tensor([1.0403993967, -0.0313237997, -2.5090755970], dtype=torch.float64)
    known_classifier_step1_loss = torch.tensor(0.7039060342, dtype=torch.float64)
    known_scorer_step1_weight = torch.tensor([0.8228723738, 1.0093623990, 0.7777688496, 0.6218048082], dtype=torch.float64)
    known_scorer_step1_bias = torch.tensor(-1.0, dtype=torch.float64)
    known_scorer_step1_loss = torch.tensor(0.7039478310, dtype=torch.float64)
    # K5
    known_classifier_step300_weight = torch.tensor([[-2.5846193964, 1.0434140547, 2.2912053417]], dtype=torch.float64)
    known_classifier_step300_bias = torch.tensor([5.0660071983, -1.2262320856, -5.3397751127], dtype=torch.float64)
    known_classifier_step300_loss = torch.tensor(0.4205052552, dtype=torch.float64)
    known_scorer_step200_weight = torch.tensor([3.9161594343, 0.8167489064, 5.8565391548, 5.5084382350], dtype=torch.float64)
    known_scorer_step200_bias = torch.tensor(-1.0, dtype=torch.float64)
    known_scorer_step200_loss = torch.tensor(0.2470848441, dtype=torch.float64)
    # K6
    known_saturated_p = torch.tensor([[1.0, 3.7835e-44, 0.0], [1.0, 3.7835e-44, 0.0], [1/3, 1/3, 1/3]], dtype=torch.float32)
    known_saturated_per_example_loss = torch.tensor([200.0, 0.0, 1.0986123], dtype=torch.float32)
    known_saturated_loss = torch.tensor(67.032875, dtype=torch.float32)
    known_saturated_grad_z = torch.tensor([[0.3333333, 1.26e-44, -0.3333333], [0.0, 1.26e-44, 0.0], [0.1111111, -0.2222222, 0.1111111]], dtype=torch.float32)

    return (
        known_classifier_loss, known_scorer_loss,
        known_classifier_grad_weight, known_scorer_grad_weight,
        known_classifier_grad_bias, known_scorer_grad_bias,
        known_classifier_step1_weight, known_classifier_step1_bias, known_classifier_step1_loss,
        known_scorer_step1_weight, known_scorer_step1_bias, known_scorer_step1_loss,
        known_classifier_step300_weight, known_classifier_step300_bias, known_classifier_step300_loss,
        known_scorer_step200_weight, known_scorer_step200_bias, known_scorer_step200_loss,
        known_saturated_p, known_saturated_per_example_loss, known_saturated_loss, known_saturated_grad_z,
    )

def test_known_values(setup, known_classifier_init, known_scorer_init, known_training):
    (
        classifier_inputs, classifier_targets, classifier_weight, classifier_bias, _, _,
        scorer_inputs, scorer_targets, scorer_weight, scorer_bias, _, _,
        _, _,
    ) = setup
    _, known_classifier_z, known_classifier_p, known_classifier_per_example_loss, known_classifier_grad_z = known_classifier_init
    _, known_scorer_z, known_scorer_p, known_scorer_per_example_loss, known_scorer_grad_z = known_scorer_init
    (
        known_classifier_loss, known_scorer_loss,
        known_classifier_grad_weight, known_scorer_grad_weight,
        known_classifier_grad_bias, known_scorer_grad_bias,
        *_,
    ) = known_training

    # Classifier
    c_logits = forward_pass(classifier_weight, classifier_bias, classifier_inputs)
    c_p = softmax.softmax(c_logits, dim=-1)
    c_loss, c_per_example_loss, c_logit_gradient = cross_entropy.cross_entropy(c_logits, classifier_targets)
    c_gradient_loss, c_weight_gradient, c_bias_gradient = cross_entropy.classifier_gradient(classifier_weight, classifier_bias, classifier_inputs, classifier_targets)
    
    # Scorer
    s_logits = forward_pass(scorer_weight, scorer_bias, scorer_inputs)
    s_p = softmax.softmax(s_logits, dim=-1)
    s_loss, s_per_example_loss, s_logit_gradient = cross_entropy.cross_entropy(s_logits, scorer_targets)
    s_gradient_loss, s_weight_gradient, s_bias_gradient = cross_entropy.scorer_gradient(scorer_weight, scorer_bias, scorer_inputs, scorer_targets)

    # Classifier checks
    torch.testing.assert_close(c_logits, known_classifier_z, atol=1e-12, rtol=0)
    torch.testing.assert_close(c_p, known_classifier_p, atol=1e-10, rtol=0)
    torch.testing.assert_close(c_per_example_loss, known_classifier_per_example_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(c_logit_gradient, known_classifier_grad_z, atol=1e-10, rtol=0)
    torch.testing.assert_close(c_loss, known_classifier_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(c_gradient_loss, known_classifier_loss, atol=1e-10, rtol=0)
    assert c_weight_gradient.shape == (1, 3)
    torch.testing.assert_close(c_weight_gradient, known_classifier_grad_weight, atol=1e-10, rtol=0)
    torch.testing.assert_close(c_bias_gradient, known_classifier_grad_bias, atol=1e-10, rtol=0)
    torch.testing.assert_close(c_logit_gradient.sum(dim=-1), torch.zeros(8, dtype=torch.float64), atol=1e-15, rtol=0)

    # Scorer checks
    torch.testing.assert_close(s_logits, known_scorer_z, atol=1e-12, rtol=0)
    torch.testing.assert_close(s_p, known_scorer_p, atol=1e-10, rtol=0)
    torch.testing.assert_close(s_per_example_loss, known_scorer_per_example_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(s_logit_gradient, known_scorer_grad_z, atol=1e-10, rtol=0)
    torch.testing.assert_close(s_loss, known_scorer_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(s_gradient_loss, known_scorer_loss, atol=1e-10, rtol=0)
    assert s_weight_gradient.shape == (4,)
    torch.testing.assert_close(s_weight_gradient, known_scorer_grad_weight, atol=1e-10, rtol=0)
    torch.testing.assert_close(s_bias_gradient, known_scorer_grad_bias, atol=1e-15, rtol=0)
    torch.testing.assert_close(s_logit_gradient.sum(dim=-1), torch.zeros(6, dtype=torch.float64), atol=1e-15, rtol=0)

def test_against_library(library_check):
    (
        classifier_inputs, classifier_weight, classifier_bias, classifier_targets, classifier_lr,
        scorer_inputs, scorer_weight, scorer_targets, scorer_bias, scorer_lr,
    ) = library_check

    # Classifier
    c_logits = forward_pass(classifier_weight, classifier_bias, classifier_inputs)
    c_p = softmax.softmax(c_logits, dim=-1)
    c_loss, c_per_example_loss, c_logit_gradient = cross_entropy.cross_entropy(c_logits, classifier_targets)
    c_gradient_loss, c_weight_gradient, c_bias_gradient = cross_entropy.classifier_gradient(classifier_weight, classifier_bias, classifier_inputs, classifier_targets)
    
    # Scorer
    s_logits = forward_pass(scorer_weight, scorer_bias, scorer_inputs)
    s_p = softmax.softmax(s_logits, dim=-1)
    s_loss, s_per_example_loss, s_logit_gradient = cross_entropy.cross_entropy(s_logits, scorer_targets)
    s_gradient_loss, s_weight_gradient, s_bias_gradient = cross_entropy.scorer_gradient(scorer_weight, scorer_bias, scorer_inputs, scorer_targets)

    # Steps
    c_parameters = [classifier_weight, classifier_bias]
    c_gradients = [c_weight_gradient, c_bias_gradient]
    new_c_weights, new_c_bias = sgd_step.sgd_step(c_parameters, c_gradients, classifier_lr)

    s_parameters = [scorer_weight, scorer_bias]
    s_gradients = [s_weight_gradient, s_bias_gradient]
    new_s_weights, new_s_bias = sgd_step.sgd_step(s_parameters, s_gradients, scorer_lr)

    # Library
    classifier_model = torch.nn.Linear(8, 5)
    scorer_model = torch.nn.Linear(8, 1)

    # Param loads
    classifier_model.load_state_dict({
        "weight": classifier_weight.T,
        "bias": classifier_bias
    }) 

    scorer_model.load_state_dict({
        "weight": scorer_weight.unsqueeze(0),
        "bias": scorer_bias.reshape(1)
    }) 
    
    loss_fn = torch.nn.CrossEntropyLoss()
    per_example_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    classifier_optimizer = torch.optim.SGD(classifier_model.parameters(), lr=classifier_lr)
    scorer_optimizer = torch.optim.SGD(scorer_model.parameters(), lr=scorer_lr)

    classifier_model.train()
    scorer_model.train()
    classifier_optimizer.zero_grad()
    scorer_optimizer.zero_grad()

    classifier_logits = classifier_model(classifier_inputs).squeeze(-1)
    scorer_logits = scorer_model(scorer_inputs).squeeze(-1)
    
    classifier_loss = loss_fn(classifier_logits, classifier_targets)
    scorer_loss = loss_fn(scorer_logits, scorer_targets)

    classifier_loss.backward()
    scorer_loss.backward()

    classifier_optimizer.step()
    scorer_optimizer.step()

    with torch.no_grad():
        classifier_probs = torch.softmax(classifier_logits, dim=-1)
        scorer_probs = torch.softmax(scorer_logits, dim=-1)
        classifier_per_example = per_example_loss_fn(classifier_logits, classifier_targets)
        scorer_per_example = per_example_loss_fn(scorer_logits, scorer_targets)

    # Classifier checks
    torch.testing.assert_close(c_logits, classifier_logits.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(c_p, classifier_probs, atol=1e-6, rtol=0)
    torch.testing.assert_close(c_per_example_loss, classifier_per_example, atol=1e-5, rtol=0)
    torch.testing.assert_close(c_loss, classifier_loss.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(c_gradient_loss, classifier_loss.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(c_weight_gradient, classifier_model.weight.grad.T, atol=1e-6, rtol=0)
    torch.testing.assert_close(c_bias_gradient, classifier_model.bias.grad, atol=1e-6, rtol=0)
    torch.testing.assert_close(new_c_weights, classifier_model.weight.detach().T, atol=1e-6, rtol=0)
    torch.testing.assert_close(new_c_bias, classifier_model.bias.detach(), atol=1e-6, rtol=0)

    # Scorer checks
    torch.testing.assert_close(s_logits, scorer_logits.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(s_p, scorer_probs, atol=1e-6, rtol=0)
    torch.testing.assert_close(s_per_example_loss, scorer_per_example, atol=1e-5, rtol=0)
    torch.testing.assert_close(s_loss, scorer_loss.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(s_gradient_loss, scorer_loss.detach(), atol=1e-5, rtol=0)
    torch.testing.assert_close(s_weight_gradient, scorer_model.weight.grad[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(s_bias_gradient, scorer_model.bias.grad[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(new_s_weights, scorer_model.weight.detach()[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(new_s_bias, scorer_model.bias.detach()[0], atol=1e-6, rtol=0)


def test_hazard(setup, known_training):
    *_, saturated_logits, saturated_targets = setup
    *_, known_saturated_p, known_saturated_per_example_loss, known_saturated_loss, known_saturated_grad_z = known_training

    loss, per_example_loss, logit_gradient = cross_entropy.cross_entropy(saturated_logits, saturated_targets)

    torch.testing.assert_close(per_example_loss, known_saturated_per_example_loss, atol=1e-4, rtol=0)
    torch.testing.assert_close(loss, known_saturated_loss, atol=1e-4, rtol=0)
    torch.testing.assert_close(logit_gradient, known_saturated_grad_z, atol=1e-7, rtol=0)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(per_example_loss).all()
    assert torch.isfinite(logit_gradient).all()

def test_training(setup, known_training):
    (
        classifier_inputs, classifier_targets, classifier_weight, classifier_bias, classifier_lr, classifier_steps,
        scorer_inputs, scorer_targets, scorer_weight, scorer_bias, scorer_lr, scorer_steps,
        _, _,
    ) = setup
    (
        _, _, _, _, _, _,
        known_classifier_step1_weight, known_classifier_step1_bias, known_classifier_step1_loss,
        known_scorer_step1_weight, known_scorer_step1_bias, known_scorer_step1_loss,
        known_classifier_step300_weight, known_classifier_step300_bias, known_classifier_step300_loss,
        known_scorer_step200_weight, known_scorer_step200_bias, known_scorer_step200_loss,
        _, _, _, _,
    ) = known_training

    # Classifier
    c_weight, c_bias = classifier_weight, classifier_bias
    for step in range(int(classifier_steps)):
        _, c_weight_gradient, c_bias_gradient = cross_entropy.classifier_gradient(c_weight, c_bias, classifier_inputs, classifier_targets)
        c_weight, c_bias = sgd_step.sgd_step([c_weight, c_bias], [c_weight_gradient, c_bias_gradient], classifier_lr)

        if step == 0:
            c_loss, _, _ = cross_entropy.classifier_gradient(c_weight, c_bias, classifier_inputs, classifier_targets)
            torch.testing.assert_close(c_weight, known_classifier_step1_weight, atol=1e-9, rtol=0)
            torch.testing.assert_close(c_bias, known_classifier_step1_bias, atol=1e-9, rtol=0)
            torch.testing.assert_close(c_loss, known_classifier_step1_loss, atol=1e-9, rtol=0)

    c_loss, _, _ = cross_entropy.classifier_gradient(c_weight, c_bias, classifier_inputs, classifier_targets)
    torch.testing.assert_close(c_weight, known_classifier_step300_weight, atol=1e-9, rtol=0)
    torch.testing.assert_close(c_bias, known_classifier_step300_bias, atol=1e-9, rtol=0)
    torch.testing.assert_close(c_loss, known_classifier_step300_loss, atol=1e-9, rtol=0)

    # Scorer
    s_weight, s_bias = scorer_weight, scorer_bias
    for step in range(int(scorer_steps)):
        _, s_weight_gradient, s_bias_gradient = cross_entropy.scorer_gradient(s_weight, s_bias, scorer_inputs, scorer_targets)
        s_weight, s_bias = sgd_step.sgd_step([s_weight, s_bias], [s_weight_gradient, s_bias_gradient], scorer_lr)

        torch.testing.assert_close(s_bias, torch.tensor(-1.0, dtype=torch.float64), atol=1e-12, rtol=0)

        if step == 0:
            s_loss, _, _ = cross_entropy.scorer_gradient(s_weight, s_bias, scorer_inputs, scorer_targets)
            torch.testing.assert_close(s_weight, known_scorer_step1_weight, atol=1e-9, rtol=0)
            torch.testing.assert_close(s_bias, known_scorer_step1_bias, atol=1e-9, rtol=0)
            torch.testing.assert_close(s_loss, known_scorer_step1_loss, atol=1e-9, rtol=0)

    s_loss, _, _ = cross_entropy.scorer_gradient(s_weight, s_bias, scorer_inputs, scorer_targets)
    torch.testing.assert_close(s_weight, known_scorer_step200_weight, atol=1e-9, rtol=0)
    torch.testing.assert_close(s_bias, known_scorer_step200_bias, atol=1e-9, rtol=0)
    torch.testing.assert_close(s_loss, known_scorer_step200_loss, atol=1e-9, rtol=0)
