# F02-1-L1 -- SGD Step

## Mechanism
We're computing a forward pass and loss, a backward pass, and a parameter update on a linear regression model via a hand-coded SGD function and a step function.

## Task 
We're building two functions to support this - a gradient function which completes the forward pass, loss, and backward pass and returns the loss and the gradients. The loss will be the mean squared error of the linear function's outputs compared to our target. We will build supporting functions for the forward pass, loss calculation, and backwards pass to call in our gradient function.

## Interface
Gradient Function
name:
  compute_gradient
arguments:
  weight: A float tensor of shape (F, ) containing the weight parameter values 
  bias: A float tensor of shape () containing the bias parameter value 
  batch: A float tensor of shape (B, F) containing the B, number of examples in the batch set, and F, the features of those examples. 
  targets: A float tensor of shape (B, ) containing the targets of the batch inputs.
returns:
  loss: A float tensor of shape () containing the loss of the model's prediction compared to the targets for the features used as inputs.
  weight_gradient: A float tensor of shape (F, ) containing the gradient of the weights with respect to the loss.
  bias_gradient: A float tensor of shape ( ) containing the gradient of the bias with respect to the loss.

Forward Pass
name:
  forward_pass
arguments:
  weight: A float tensor of shape (F, ) containing the weight parameter values 
  bias: A float tensor of shape () containing the bias parameter value 
  batch: A float tensor of shape (B, F) containing the B, number of examples in the batch set, and F, the features of those examples. 
returns:
  y_hat: A float tensor of shape (B, ) containing the output values from the model.

MSE Loss
name:
  mse_loss
arguments:
  y_hat: A float tensor of shape (B, ) containing the output values from the model.
  targets: A float tensor of shape (B, ) containing the targets of the batch inputs for the model.
returns:
  loss: A float tensor of shape () containing the loss of the model's prediction compared to the targets for the features used as inputs.

Backward Pass
name:
  backward
arguments:
  batch: A float tensor of shape (B, F) containing the B, number of examples in the batch set, and F, the features of those examples.
  error: A float tensor of shape (B, ) containing the error of each prediction, y_hat - targets. The averaged loss cannot be used here: averaging discards the per-example errors the gradients depend on.
returns:
  weight_gradient: A float tensor of shape (F, ) containing the gradient of the weights with respect to the loss.
  bias_gradient: A float tensor of shape ( ) containing the gradient of the bias with respect to the loss.

SGD Step
name:
  sgd_step
arguments:
  parameters: A list of float tensors, each of any shape, holding the model's parameters.
  gradients: A list of float tensors, the same length and order as parameters, each the same shape as its parameter, from our compute_gradient function.
  lr: A float containing the scalar learning rate value for the step.
returns:
  updated_parameters: A list of float tensors, the same length, order and shapes as parameters, each equal to p − lr·g after one step. It knows nothing about the model.

## Pass Check 
All tests must pass.

Here is our test criteria:

1. **Loss and gradients, full batch.** At the line's initialization on all six examples, the gradient function returns the Table 3 loss and gradients within `1e-10`, with shapes `()`, `(1,)`, `()`. Reason for the tolerance: Table 3 is rounded to ten decimals (error at most `5e-11`); float64 arithmetic is accurate to about `1e-15` here.
2. **One step, full batch.** From the line's initialization with `lr = 0.05`, a step gives `weight = [1.0833333333]` and `bias = -0.05` within `1e-10`. The loss on all six at the new parameters is `1.3625462963` within `1e-10`, lower than before. Same reason.
3. **The batch size divides, not the data set size.** Each of the three line batches at the initialization gives the Table 3 loss and gradients within `1e-12`, and the mean of the three batch gradients equals the full-batch gradient within `1e-12`. Reason for the case: dividing by 6 instead of 2 gives `4.7667` for `∂L/∂weight` on `[2, 5]`, not `14.3`; the values are short decimals, so only float64 rounding remains.
4. **One epoch.** Three steps over `[2, 5]`, `[3, 0]`, `[1, 4]` in that order, each from the previous step's result, give the Table 3 epoch rows (loss, `weight` and `bias` after) within `1e-10`. Reason for the case: each batch gradient is taken at the parameters the previous step left, not at the initialization; the step-2 gradient `−4.155` shows it.
5. **Unseen start and two features.** The line's unseen start (both gradients negative, so both parameters rise) and the Table 2 step give the Table 3 rows within `1e-10`. In the two-feature step the first weight's gradient is exactly `0.0`, and that weight stays at `0.5`. Reason for the cases: the first checks the sign; the second checks that each feature gets its own column of `X` in `Xᵀ @ δ`, since transposing the wrong way or summing over the wrong axis gives a wrong shape or wrong values here.
6. **Against the library, float32, larger model.** With `torch.Generator().manual_seed(0)`, draw `inputs` `(32, 8)` with `torch.randn`, a true weight `(8,)` with `torch.randn`, and `targets = inputs @ true_weight + 0.5 + 0.1 · torch.randn(32)`, all from that generator, in that order. Start at `weight = zeros(8)`, `bias = 0.0`, `lr = 0.05`. Your loss and gradients equal `F.mse_loss(inputs @ weight + bias, targets)` and the `.grad` that `backward()` fills, within `1e-5`. One step equals one `step()` of `torch.optim.SGD([weight, bias], lr=0.05)` within `1e-6`. 200 steps of each agree in every parameter within `1e-5` at every step. Reason for the tolerances: a different order of the same float32 sums can move a gradient by a few spacings; the parameters stay below 2 in magnitude, where the spacing is at most `2.4e-7`; the 200-step tolerance is the notebook's. Measured in the course environment over two orders of the sums: gradient up to `4.8e-7`, one step up to `3.0e-8`, 200 steps up to `1.2e-7`; the loss falls to `0.0052`.
7. **The shape hazard.** The gradient function raises `ValueError` for targets of shape `(6, 1)`, for inputs of shape `(6,)`, for inputs `(6, 1)` with targets `(5,)`, and for a weight `(2,)` with inputs `(6, 1)`. `sgd_step` raises `ValueError` for two parameters and one gradient, and for a parameter `(2,)` with a gradient `(3,)`. Reason for the first case: `(6,) − (6, 1)` broadcasts to `(6, 6)` without an error, and the mean of those 36 squared entries is `7.5617` instead of `3.1867` (library comparison 2). PyTorch's `F.mse_loss` only warns here; this toolkit raises.
8. **Any shape, arguments unchanged.** `sgd_step` on parameters of shapes `(8,)`, `()` and `(2, 3)` with matching gradients returns three tensors of those shapes, each equal to `p − lr·g` within `1e-12`. After every call in checks 2 to 6, the tensors passed in equal copies taken before the call, exactly. The results keep the input dtype, float32 for float32 and float64 for float64.

## Results
All tests passed.

We hand-coded the forward pass, the mean squared error loss, the backward pass and an SGD step for a linear regression model. Every value matches Table 3. In float32 on the check 6 data, our parameters were identical to `torch.optim.SGD` after 1 step and after 200 steps: the largest difference was exactly 0.

Numbers from `results/F02-1-L1.json`:

Step 1 - one full-batch step of the line fit from `weight = [1.5]`, `bias = 0.0`, `lr = 0.05`:

| Quantity | Value |
|---|---|
| Loss | 3.1866666667 |
| dL/dweight | [8.3333333333] |
| dL/dbias | 1.0 |
| weight after | [1.0833333333] |
| bias after | -0.05 |
| Loss on all six after | 1.3625462963 |

Step 2 - batch gradients at the initialization:

| Batch | Loss | dL/dweight | dL/dbias |
|---|---|---|---|
| [2, 5] | 4.21 | 14.3 | 2.8 |
| [3, 0] | 2.61 | 2.7 | -1.2 |
| [1, 4] | 2.74 | 8.0 | 1.4 |
| Mean of the three | | 8.3333333333 | 1.0 |
| Full batch | | 8.3333333333 | 1.0 |

Step 3 - one epoch over the Table 1 batches, each step from the previous step's result:

| Step | Batch | Batch Loss | dL/dweight | dL/dbias | weight After | bias After | Loss On All Six After |
|---|---|---|---|---|---|---|---|
| 1 | [2, 5] | 4.21 | 14.3 | 2.8 | 0.785 | -0.14 | 2.2946625 |
| 2 | [3, 0] | 3.4679125 | -4.155 | -3.625 | 0.99275 | 0.04125 | 1.2565214062 |
| 3 | [1, 4] | 0.82390303125 | -0.417 | -1.05375 | 1.0136 | 0.0939375 | 1.1763345539 |

Step 4 - the two-feature step from `weight = [0.5, -1.0]`, `bias = 0.25`, `lr = 0.1`:

| Quantity | Value |
|---|---|
| Loss | 6.5625 |
| dL/dweight | [0.0, -6.0] |
| dL/dbias | -3.5 |
| weight after | [0.5, -0.4] |
| bias after | 0.6 |
| Loss after | 2.86 |

Step 5 - float32, ours against `torch.optim.SGD` on the check 6 data (32 examples, 8 features, `lr = 0.05`):

| Steps | Max Difference weight | Max Difference bias | Our Loss After |
|---|---|---|---|
| 1 | 0.000e+00 | 0.000e+00 | 3.2540553 |
| 200 | 0.000e+00 | 0.000e+00 | 0.0052183 |

Results from analysis:
The gradient follows the chain rule from each parameter to the loss. With `error = y_hat - targets`, the weight gradient is `(2 / B) * batch.T @ error` and the bias gradient is `(2 / B) * sum(error)`. Each example pushes each weight by its error times that feature's value. The transpose is what groups those pushes by feature: `batch` is `(B, F)` with one row per example, and `@` takes one dot product per row, so `batch.T` turns the feature columns into rows and gives one gradient per weight, shape `(F,)`.

The batch `[2, 5]` gives `dL/dweight = 14.3`; dividing by 6 instead of 2 would give 4.7667. Because the three batches have equal size and together cover all six examples once, the mean of their gradients equals the full-batch gradient exactly (8.3333, 1.0). Each batch gradient is a noisy estimate of the full one: they range from 2.7 to 14.3 around 8.3333.

Each step in an epoch starts from where the last one left off. The step 2 gradient is `-4.155`, not the `2.7` that batch `[3, 0]` gives at the initialization, because step 1 had already moved the weight from 1.5 to 0.785. The loss on all six does not have to fall at every step, since each step only sees two examples, but here it fell each time: 2.2947, 1.2565, 1.1763. After the epoch, which uses each example once, the loss on all six (1.1763) is lower than after one full-batch step (1.3625), which also uses each example once. Three smaller steps went further than one large one.

The two-feature step checks that each feature gets its own gradient. The first weight's gradient is exactly 0.0: examples 0 and 1 push it by `-2.25 * 1` and `0.75 * 3`, which cancel, and example 2 has feature value 0. So that weight stays at 0.5 while the second weight moves from -1.0 to -0.4. A transpose the wrong way or a sum over the wrong axis would give a wrong shape or would mix the two features.

Against the library, the parameters matched exactly in float32, after 1 step and after 200. Our code and PyTorch's did the same float32 operations in the same order on this machine, so there was no rounding difference to find. The README's measured differences (up to 3.0e-8 after one step and 1.2e-7 after 200) came from two different orders of the sums; ours is within the tolerance either way. The loss fell to 0.0052. That is below 0.01, the variance of the noise in the targets, because this is the loss on the training examples: with 9 parameters and 32 examples, the fit also absorbs part of the noise.

What this does not establish is behavior beyond plain SGD on a linear model with MSE. We did not test momentum, weight decay, learning rate schedules, other losses, or models where the backward pass is not a single matrix product. We also did not shuffle batches between epochs or test convergence over many epochs on the line fit. In practice we would let autograd compute the gradients and use `torch.optim.SGD`.

## Why It Matters
Every model we train is updated by some form of this loop: forward pass, loss, backward pass, step. Writing each part by hand shows what `loss.backward()` and `opt.step()` do for us. Autograd applies the same chain rule we worked out here, and the optimizer applies `p - lr * g` to each parameter.

Mini-batch SGD is how models are trained on data sets too large to process in one pass. The gradient of one batch is an estimate of the full gradient, and dividing by the batch size keeps that estimate on the same scale no matter how large the data set is. Get the divisor wrong and the learning rate silently means something different for every batch size.

Shape mistakes in this loop fail quietly. Targets of shape `(6, 1)` against predictions of shape `(6,)` broadcast to `(6, 6)` and give a loss of 7.5617 instead of 3.1867 with no error, and PyTorch's `F.mse_loss` only warns. Our functions check the shapes and raise `ValueError` first. The same kind of care applies to the gradient: `batch.T @ error` must come out as `(F,)`, matching the weight, or the step cannot be taken.

An optimizer step should know nothing about the model. Our `sgd_step` takes a list of parameters of any shapes and a matching list of gradients, the same way `torch.optim.SGD` takes `model.parameters()`. That is what lets one optimizer train a model with two parameters or two hundred, and it is why the step returns new tensors instead of changing the ones it was given.

## Toolkit
We added sgd_step.py, which trains a linear regression model with hand-coded gradients and matches `torch.optim.SGD` in float32. It contains five functions:

- `compute_gradient` takes the weight, bias, a batch of inputs and their targets, and returns the loss and both gradients. It checks the shapes first and raises `ValueError` if the batch is not `(B, F)`, the targets are not `(B,)`, the weight is not `(F,)`, or the sizes disagree.
- `forward_pass` returns the predictions `batch @ weight + bias`.
- `mse_loss` returns the mean of the squared errors over the batch.
- `backward` takes the batch and the per-example errors and returns `(2 / B) * batch.T @ error` and `(2 / B) * sum(error)`.
- `sgd_step` takes a list of parameters of any shapes, a matching list of gradients and the learning rate, and returns a new list with `p - lr * g` for each. It raises `ValueError` if the lists have different lengths or a parameter and its gradient have different shapes.

None of the functions change their arguments, and the results keep the input dtype. No previous toolkit functions were used.
