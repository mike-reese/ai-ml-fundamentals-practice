import pytest
import torch
import numpy
import evalkit.l2_regularization as l2_regularization

@pytest.fixture
def parameter_validation():
    w = torch.tensor(2.0, dtype=torch.float64)
    grad_w = torch.tensor(0.1, dtype=torch.float64)
    lr = 0.3
    weight_decay = 0.01
    h = 0.001

    return w, grad_w, lr, weight_decay, h

@pytest.fixture
def multi_parameter():
    w = torch.tensor([2.0, -1.5, 0.0], dtype=torch.float64)
    grad_w = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    
    return w, grad_w

@pytest.fixture
def hyperparameter_selection():
    weight_decay = numpy.array([0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    training_loss = numpy.array([0.0099, 0.0285, 0.0738, 0.2083, 0.2790, 0.3862, 0.6367])
    validation_loss = numpy.array([1.7445, 1.1305, 0.6264, 0.3746, 0.4244, 0.4900, 0.6488])
    sum_squared_parameters = numpy.array([251.61, 139.12, 66.26, 15.39, 6.83, 3.01, 0.13])
    validation_accuracy = numpy.array([0.6667, 0.6667, 0.6667, 0.8333, 0.8333, 0.6667, 0.6667])

    return weight_decay, training_loss, validation_loss, sum_squared_parameters, validation_accuracy

def test_one_step(parameter_validation):
    w, grad_w, lr, weight_decay, h = parameter_validation
    updated_parameter = l2_regularization.l2_step(parameter=w, gradient=grad_w, learning_rate=lr, weight_decay=weight_decay)
    
    # Reason for tolerance - this is precise to the float64 rounding error.
    torch.testing.assert_close(updated_parameter, torch.tensor(1.9640, dtype=torch.float64), atol=1e-12, rtol=0)

def test_without_weight_decay(parameter_validation):
    w, grad_w, lr, weight_decay, h = parameter_validation
    weight_decay = 0.0
    updated_parameter = l2_regularization.l2_step(parameter=w, gradient=grad_w, learning_rate=lr, weight_decay=weight_decay)
    
    # Reason for tolerance - this is precise to the float64 rounding error.
    torch.testing.assert_close(updated_parameter, torch.tensor(1.9700, dtype=torch.float64), atol=1e-12, rtol=0)

def test_decay_then_update(parameter_validation):
    w, grad_w, lr, weight_decay, h = parameter_validation
    rearranged_update = (1 - lr * weight_decay) * w - lr * grad_w
    
    updated_parameter = l2_regularization.l2_step(parameter=w, gradient=grad_w, learning_rate=lr, weight_decay=weight_decay)
    
    # Reason for tolerance - same arithmetic in float64 rearranged. Precise to rounding error.
    torch.testing.assert_close(rearranged_update, updated_parameter, atol=1e-12, rtol=0)

def test_any_shape_parameter(parameter_validation, multi_parameter):
    _, _, lr, weight_decay, h = parameter_validation
    w, grad_w = multi_parameter

    updated_parameters = l2_regularization.l2_step(parameter=w, gradient=grad_w, learning_rate=lr, weight_decay=weight_decay)

    # Reason for tolerance - this is precise to the float64 rounding error.
    torch.testing.assert_close(updated_parameters, torch.tensor([1.9640, -1.4355, -0.0150], dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(w, torch.tensor([2.0, -1.5, 0.0], dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(grad_w, torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64), atol=1e-12, rtol=0)

def test_against_library(parameter_validation, multi_parameter):
    _, _, lr, weight_decay, h = parameter_validation
    _, grad_w = multi_parameter
    w = torch.tensor([2.0, -1.5, 0.0], dtype=torch.float64, requires_grad=True)
    w.grad = grad_w

    updated_parameters = l2_regularization.l2_step(parameter=w, gradient=grad_w, learning_rate=lr, weight_decay=weight_decay)

    SGD = torch.optim.SGD([w], lr=lr, weight_decay=weight_decay)
    SGD.step()
    torch_parameters = w.detach()
    
    # Reason for tolerance -- it's the same two operations in float64 so this is precise to the rounding error.
    torch.testing.assert_close(updated_parameters, torch_parameters, atol=1e-12, rtol=0)

def test_selection_on_validation_set(hyperparameter_selection):
    weight_decay, training_loss, validation_loss, sum_squared_parameters, validation_accuracy = hyperparameter_selection

    selected_hyperparameter, hyperparameter_index, selected_validation_loss = l2_regularization.select_on_validation(hyperparameters=weight_decay, validation_loss=validation_loss)

    numpy.testing.assert_allclose(selected_hyperparameter,0.01)
    numpy.testing.assert_array_equal(hyperparameter_index,3)

    # Reason for tolerance -- float64 precise to the rounding error.
    numpy.testing.assert_allclose(selected_validation_loss,0.3746,atol=1e-12,rtol=0)
    

def test_training_set_cannot_select(hyperparameter_selection):
    weight_decay, training_loss, validation_loss, sum_squared_parameters, validation_accuracy = hyperparameter_selection

    selected_hyperparameter, hyperparameter_index, selected_validation_loss = l2_regularization.select_on_validation(hyperparameters=weight_decay, validation_loss=training_loss)

    numpy.testing.assert_allclose(selected_hyperparameter,0.0)
    numpy.testing.assert_array_equal(hyperparameter_index,0)

def test_ties(hyperparameter_selection):
    values = numpy.array([0.0, 0.01, 0.1])
    losses = numpy.array([0.5, 0.4, 0.4])
 
    selected_hyperparameter, hyperparameter_index, selected_validation_loss = l2_regularization.select_on_validation(hyperparameters=values, validation_loss=losses)
    numpy.testing.assert_array_equal(hyperparameter_index,1)
    numpy.testing.assert_allclose(selected_hyperparameter,0.01)

