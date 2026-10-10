# F04C-L1 -- L2 Regularization, Model Selection

## Mechanism
We will compute an L2 Regularization step from a provided parameter and hyperparameter input table to demonstrate the understanding of the details of this mechanism, and then use an input table with varying hyperparameter selections from a model run on a validation dataset and the resulting validation loss from those runs to identify the correct hyperparameter to select from a list of options. 

## Task
Create the functions l2_step, calculate_l2_penalty, generate_parameter_delta, and select_on_validation and write a test suite of 8 tests asserting that our code accurately and robustly applies an L2 Regularization process to a given parameter, matches the library invocation of these processes within 1e-12 precision, and correctly selects the hyperparameter selection with the lowest validation loss from a list of choices.

## Interface
L2 Step:
name:
  l2_step
arguments:
  parameter: A tensor of any shape of the parameter of the model
  gradient: A tensor of any shape (same shape as parameter) of the gradient of the data loss with respect to the parameter
  learning_rate: Float. A scalar learning rate to apply to scale the parameter deltas.
  weight_decay: Float. The weight decay to apply to the L2 penalty calculation to scale the parameter deltas.
returns:
  updated_parameters: A tensor of any shape (same shape as the parameter in the input) holding the updated values of the same parameter provided as an argument.

Calculate L2 Penalty:
name:
  calculate_l2_penalty
arguments:
  parameter: A tensor of any shape of the parameter of the model
  weight_decay: Float. The weight decay value to apply to the L2 penalty calculation to scale the parameter deltas. 
returns:
  l2_penalty: A tensor of any shape (same shape as the parameter input) of the L2 penalties for reporting. 
  l2_penalty_gradient: A tensor of any shape (same shape as the parameter input) of the calculation result of the derivative of the L2 function to apply to the parameter deltas.


Generate Parameter Delta:
name:
  generate_parameter_delta
arguments:
  gradient: a tensor of any shape (same shape as parameter) of the gradient of the data loss with respect to the parameter
  learning_rate: float. a scalar learning rate to apply to scale the parameter deltas.
  l2_penalty_gradient:  a tensor of any shape (same shape as the parameter) of the l2 derivative calculation results which we apply to the parameter delta. 
returns:
  parameter_delta: a tensor of any shape (same shape as the parameter input) of the delta to add to the parameter value. 

Select On Validation:
name:
  select_on_validation
arguments:
  hyperparameters: A numpy float64 array of shape (K, ) of the hyperparameter value candidates from the validation dataset runs to select from.
  validation_loss: A numpy float64 array of shape (K, ) of the validation loss from the validation dataset runs with the applied hyperparameter values. 
returns:
  selected_hyperparameter: A numpy float64 scalar. The selected hyperparameter value from our selection over the input data.
  hyperparameter_index: A numpy int64 scalar. The index of the selected_hyperparameter from the hyperparameters input tensor.
  selected_validation_loss: A numpy float 64 scalar. The validation loss of the selected_hyperparameter from the hyperparameters input tensor.

## Pass Check 
All tests must pass.

Here is our test criteria:

1. **one step, one weight.** `l2_step` on `w = 2.0`, `grad_w = 0.1`, `lr = 0.3`, `weight_decay = 0.01` returns `1.9640`, within `1e-12`. reason for the tolerance: the exact value is `2.0 − 0.3 · 0.12 = 1.964`, and the notebook's four printed decimals are the whole number, so only float64 rounding error remains.
2. **without weight decay.** the same step with `weight_decay = 0.0` returns `1.9700`, within `1e-12`. same reason.
3. **decay then update.** for the table 1 numbers, `(1 − lr · weight_decay) · w − lr · grad_w` computed in the test equals the function's result within `1e-12`. reason: the same arithmetic in float64, regrouped.
4. **any shape, every parameter.** for the parameter tensor `[2.0, −1.5, 0.0]` and gradient `[0.1, −0.2, 0.05]`, with `lr = 0.3` and `weight_decay = 0.01`, the function returns `[1.9640, −1.4355, −0.0150]`, within `1e-12`, with shape `(3,)`. the arguments are unchanged after the call. reason for the values: the step is applied entrywise; the third entry shows that a parameter at 0 is not decayed and moves by `−lr · grad` alone.
5. **against a library.** for the tensors of check 4, one `step()` of `torch.optim.SGD([parameter], lr=0.3, weight_decay=0.01)` with `parameter.grad` set to the gradient gives the same tensor as your function, within `1e-12`. reason for the tolerance: pytorch adds `weight_decay · parameter` to the gradient and then subtracts `lr` times the sum, the same two operations in float64.
6. **selection on the validation set.** `select_on_validation` on table 2's weight-decay column and validation-loss column returns value `0.01`, index `3`, loss `0.3746`. the value and index are exact; the loss within `1e-12`, since it is an input returned unchanged.
7. **the training set cannot select.** the same function on the weight-decay column and the training-loss column returns value `0.0`, index `0`. exact.
8. **ties.** on values `[0.0, 0.01, 0.1]` and losses `[0.5, 0.4, 0.4]` the function returns index `1`, value `0.01`. exact. this also checks the argument order: the minimum over the values would be index `0`.

## Result 
All tests passed.

Numbers from `results/F04C-L1.json`:

One optimizer step on `w = 2.0`, `grad_w = 0.1`, `lr = 0.3`:

| Formula | Weight Decay | Updated w |
|---|---|---|
| `w − lr · (grad_w + weight_decay · w)` | 0.01 | 1.9640 |
| `(1 − lr · weight_decay) · w − lr · grad_w` | 0.01 | 1.9640 |
| `w − lr · (grad_w + weight_decay · w)` | 0.0 | 1.9700 |
| `(1 − lr · weight_decay) · w − lr · grad_w` | 0.0 | 1.9700 |

Selection on Table 2:

| Selected By | Index | Weight Decay | Loss |
|---|---|---|---|
| Validation loss | 3 | 0.01 | 0.3746 (validation) |
| Training loss | 0 | 0.0 | 0.0099 (training) |

We successfully demonstrated the ability to perform L2 regularization on provided parameters and update the parameters correctly in a way that matches torch.optim.SGD.

We also successfully demonstrated retrieving the hyperparameter value with the lowest validation loss from our list of computed model results.

Results from analysis:
This is from the notebook -- but from one training run of many optimizer steps without L2 regularization, we ended up with a very steep parameter surface. The model was extremely confident in everywhere but a narrow band, meaning loss was actually larger on the validation set because it became confidently incorrect. Applying L2 regularization with different weight decay values on our validation set allowed us to identify the weight decay hyperparameter of 0.01 which minimized the validation loss. The result was a much smoother parameter weight surface and a lower validation loss for the model. Training loss rose from 0.0099 to 0.2083; validation loss fell from 1.7445 to 0.3746. 

## Why It Matters 
Training without L2 Regularization creates overconfidence over many optimizer steps. Our validation loss actually increased over time because the weight parameter values got so large. The models "best fit" for every data point in our training set was established, and continually optimizing the parameter values began to create a large disparity between the largest values and the smallest ones. This resulted in logits that created a probability surface where the model was extremely confident nearly everywhere in our data space except for a very narrow band. For training set values where our model could not predict them correctly due to the optimization against the training set (other values also impacting the weights, fitting the line), this resulted in the probability increasing in the wrong direction for those values, creating an increase in loss, instead of a decrease.

L2 Regularization smooths that probability surface out by way of scaling the weight parameters so that larger weights get penalized more than smaller ones. Applying this function to our optimization step and testing with different hyperparameters on our validation set allowed us to identify the hyperparameter value that resulted in an overall lower validation loss.

## Toolkit
We added l2_step, calculate_l2_penalty, generate_parameter_delta, select_on_validation functions. l2_step performs a single optimizer step with L2 regularization applied. calculate_l2_penalty calculates the specific l2 penalty and its gradient that we can apply as part of the parameter delta. generate_parameter_delta allows us to isolate the specific delta we would be applying at each optimizer step to each parameter value. select_on_validation allows us to select the hyperparameter value that minimized the validation loss from a list of model optimization steps on the validation set.
