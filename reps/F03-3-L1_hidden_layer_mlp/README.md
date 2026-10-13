# F03-3-L1 -- Hidden Layer MLP Backwards Pass

## Mechanism
One linear layer gives a logit that is a straight line in the input, so it can draw only one decision boundary. These targets need two. We put a hidden layer with a ReLU in front of F03-1-L1's output layer (one logit, sigmoid, binary cross-entropy) and write the backward pass by hand, from `∂L/∂z` to the four parameter tensors.

## Task 
Create 2 functions -- MLP forward pass and MLP backward pass. We reuse `binary_cross_entropy` (F03-1-L1, binary_cross_entropy.py) for the loss and `∂L/∂z`, and `sgd_step` (F02-1-L1, sgd_step.py) for the optimizer step. Inside the functions we do not call `torch.relu`, `F.relu`, `nn.ReLU`, `torch.sigmoid`, any `torch.nn` layer or loss, `backward()` or `torch.autograd`. The tests may call anything.

The forward pass: `hidden_pre = inputs @ W_hidden + b_hidden`, `hidden = max(0, hidden_pre)`, `z = hidden @ w_out + b_out`. Without the ReLU the two layers collapse into one line, `z = inputs @ (W_hidden @ w_out) + (b_hidden @ w_out + b_out)`.

The backward pass, right to left, with `δ = ∂L/∂z` `(N,)`:

| operation | local derivative | gradient |
| --- | --- | --- |
| add `b_out` | 1 | `∂L/∂b_out = Σ_n δ[n]` → `()` |
| multiply by `w_out`, toward the parameter | `hidden` | `∂L/∂w_out = hiddenᵀ @ δ` → `(H,)` |
| multiply by `w_out`, toward the activations | `w_out` | `∂L/∂hidden = δ[:, None] * w_out` → `(N, H)` |
| ReLU | 1 where `hidden_pre > 0`, else 0 | `∂L/∂hidden_pre = ∂L/∂hidden * (hidden_pre > 0)` → `(N, H)` |
| add `b_hidden` | 1 | `∂L/∂b_hidden = Σ_n ∂L/∂hidden_pre[n, :]` → `(H,)` |
| multiply by `W_hidden` | `inputs` | `∂L/∂W_hidden = inputsᵀ @ ∂L/∂hidden_pre` → `(F, H)` |

## Interface
MLP Forward Pass
name:
  mlp_forward_pass
arguments:
  W_hidden: A float tensor of shape (F, H)
  b_hidden: A float tensor of shape (H, )
  w_out: A float tensor of shape (H, )
  b_out: A float tensor of shape ()
  inputs: A float tensor of shape (N, F)
returns:
  hidden_pre: A float tensor of shape (N, H)
  hidden: A float tensor of shape (N, H)
  logits: A float tensor of shape (N, )

MLP Backward Pass
name:
  mlp_backward_pass
arguments:
  W_hidden: A float tensor of shape (F, H)
  b_hidden: A float tensor of shape (H, )
  w_out: A float tensor of shape (H, )
  b_out: A float tensor of shape ()
  inputs: A float tensor of shape (N, F)
  targets: A float tensor of shape (N, ), 0.0 or 1.0
returns:
  loss: A float tensor of shape ()
  hidden_pre_gradient: A float tensor of shape (N, H)
  W_hidden_gradient: A float tensor of shape (F, H)
  b_hidden_gradient: A float tensor of shape (H, )
  w_out_gradient: A float tensor of shape (H, )
  b_out_gradient: A float tensor of shape ()

## Pass Check 
All tests must pass.

Here is our test criteria:

1. **Known values (float64).** With the Part D inputs `[[0.5], …, [4.0]]`, targets `[0, 0, 0, 1, 1, 0, 0, 0]` and the initialization `W_hidden = [[1.0, −1.0]]`, `b_hidden = [−1.25, 3.25]`, `w_out = [0.5, −1.0]`, `b_out = 1.0`, the forward pass returns the `hidden_pre`, `hidden` and `z` columns of the per-example table within `1e-12`. `p`, `ℓ`, `∂L/∂z`, `∂L/∂hidden_pre` and the loss (K1) equal the table within `1e-10`. `∂L/∂W_hidden` `(1, 2)`, `∂L/∂b_hidden` `(2,)`, `∂L/∂w_out` `(2,)` and `∂L/∂b_out` `()` have exactly those shapes and equal K2 and K3 within `1e-10`. Reason for the tolerances: `hidden_pre`, `hidden` and `z` are short decimals; the rest are rounded to ten decimals. Reason for the case: no ReLU mask gives `∂L/∂W_hidden = [[0.0521, 0.0468]]`; `δ` times `hidden` instead of `w_out` gives `[[0.0282, 0.0109]]`; a forward pass without the ReLU gives `z[ex0] = −2.125`, not `−1.75`.
2. **Against the library, one training step (float32).** With `g = torch.Generator().manual_seed(0)`, draw `inputs = torch.randn(64, 8)`, `W_hidden = torch.randn(8, 4)`, `b_hidden = torch.randn(4)`, `w_out = torch.randn(4)`, `b_out = torch.randn(())` and `targets = torch.randint(0, 2, (64,)).float()`, in that order; `lr = 0.5`. Run our functions and `sgd_step` beside `nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))`, `nn.BCEWithLogitsLoss()` and `torch.optim.SGD`, loaded with `{"0.weight": W_hidden.T, "0.bias": b_hidden, "2.weight": w_out.unsqueeze(0), "2.bias": b_out.reshape(1)}`. The layers run one at a time with `hidden_pre.retain_grad()`. `hidden_pre`, `hidden`, `z` and the loss agree within `1e-5`; `∂L/∂hidden_pre` and the four gradients within `1e-6`; the four parameters after the step within `1e-6`. Reason for the shapes: `N`, `F` and `H` all differ, so a transposed product or a wrong-axis sum fails; 55 percent of the 256 pre-activations are at or below 0, so the mask matters. Reason for the tolerances: logits reach `12.4`, where the float32 spacing is `9.5e-7`.
3. **The hazard: the ReLU mask (float64).** Inputs `[[1.25], [2.0], [0.0]]`, targets `[1, 0, 1]`, `b_hidden = [−1.25, −3.0]`, the other parameters as in check 1. Unit 1's pre-activation is below 0 for every example (a dead unit); unit 0's is exactly 0 at ex0. `hidden_pre`, `∂L/∂z`, `∂L/∂hidden_pre` and the four gradients equal K7 and K8 within `1e-10`. Every gradient entry that belongs to unit 1 is exactly 0, and `∂L/∂hidden_pre[0, 0]` is exactly 0. Reason for the case: a mask written as `hidden_pre >= 0` lets ex0 through and gives `∂L/∂W_hidden[0, 0] = 0.2100` and `∂L/∂b_hidden[0] = 0.0882` instead of `0.2661` and `0.1330`; PyTorch's ReLU backward gives 0 at exactly 0. No mask gives unit 1 `[[−0.4201]]` and `−0.0868` instead of 0.
4. **Training (float64).** From the initialization, our backward pass and `sgd_step` on the four parameters, `lr = 0.2`, 1,000 steps. The four parameters and the loss equal K4 after step 1 and K5 after step 1000 within `1e-9`, and the accuracy after step 1000 is exactly `1.0`. F03-1-L1's logistic regression from `weight = [0.5]`, `bias = 0.0`, `lr = 0.5`, 500 steps, reaches the K6 loss within `1e-9` and an accuracy of exactly `0.75`. Reason for the tolerances: `1e-9` is the table's rounding plus float64 rounding over 1,000 steps. The same run in float32 drifts `5.5e-6` by step 1000, so the check is float64.

## Results
All tests passed.

We hand-coded the forward pass and the backward pass of a one-hidden-layer MLP with a ReLU, and trained it with F02-1-L1's `sgd_step`. The loss and `∂L/∂z` come from F03-1-L1's `binary_cross_entropy`, unchanged. Every value matches the per-example table and K1 to K8. The MLP gets all eight examples right after 1,000 steps; F03-1-L1's single layer gets six, no better than always predicting class 0.

Numbers from `results/F03-3-L1.json`:

Step 1 - the initialization:

| Example | x | y | hidden_pre | hidden | z | p | Loss | dL/dz | dL/dhidden_pre |
|---|---|---|---|---|---|---|---|---|---|
| ex0 | 0.5 | 0 | [-0.75, 2.75] | [0.0, 2.75] | -1.75 | 0.1480471980 | 0.1602241504 | 0.0185058998 | [0.0, -0.0185058998] |
| ex1 | 1.0 | 0 | [-0.25, 2.25] | [0.0, 2.25] | -1.25 | 0.2227001388 | 0.2519290813 | 0.0278375174 | [0.0, -0.0278375174] |
| ex2 | 1.5 | 0 | [0.25, 1.75] | [0.25, 1.75] | -0.625 | 0.3486451353 | 0.4287006783 | 0.0435806419 | [0.0217903210, -0.0435806419] |
| ex3 | 2.0 | 1 | [0.75, 1.25] | [0.75, 1.25] | 0.125 | 0.5312093734 | 0.6325990353 | -0.0585988283 | [-0.0292994142, 0.0585988283] |
| ex4 | 2.5 | 1 | [1.25, 0.75] | [1.25, 0.75] | 0.875 | 0.7057850278 | 0.3484445810 | -0.0367768715 | [-0.0183884358, 0.0367768715] |
| ex5 | 3.0 | 0 | [1.75, 0.25] | [1.75, 0.25] | 1.625 | 0.8354835371 | 1.8047446354 | 0.1044354421 | [0.0522177211, -0.1044354421] |
| ex6 | 3.5 | 0 | [2.25, -0.25] | [2.25, 0.0] | 2.125 | 0.8933094061 | 2.2378222788 | 0.1116636758 | [0.0558318379, 0.0] |
| ex7 | 4.0 | 0 | [2.75, -0.75] | [2.75, 0.0] | 2.375 | 0.9149009550 | 2.4639394655 | 0.1143626194 | [0.0571813097, 0.0] |

Loss `1.0410504883`, `dL/dW_hidden = [[0.5089053982, -0.2066279211]]`, `dL/db_hidden = [0.1393333397, -0.0989838013]`, `dL/dw_out = [0.6694774473, 0.1150694332]`, `dL/db_out = 0.3250100964`, accuracy `0.625`.

Step 2 - training, 1,000 full-batch steps with `lr = 0.2`. Each row is measured at the parameters after that many steps:

| Step | W_hidden | b_hidden | w_out | b_out | Loss | Accuracy |
|---|---|---|---|---|---|---|
| 0 | [[1.0, -1.0]] | [-1.25, 3.25] | [0.5, -1.0] | 1.0 | 1.0410504883 | 0.625 |
| 1 | [[0.8982189204, -0.9586744158]] | [-1.2778666679, 3.2697967603] | [0.3661045105, -1.0230138866] | 0.9349979807 | 0.8902428654 | 0.5 |
| 2 | [[0.8354058356, -0.9388508602]] | [-1.2938945778, 3.2795599726] | [0.2679402749, -1.0356426347] | 0.8831908031 | 0.8157285733 | 0.5 |
| 10 | [[0.7573641797, -0.5820818528]] | [-1.3109292835, 3.3445025879] | [-0.0967101435, -0.9684178447] | 0.7434530001 | 0.5895383253 | 0.75 |
| 100 | [[0.8575420790, -0.6262253490]] | [-1.7443785866, 3.2745594890] | [-1.2227754313, -0.7275562318] | 0.8335802143 | 0.4751187068 | 0.75 |
| 1000 | [[1.9378204634, -2.4878914537]] | [-4.8187267141, 4.9520113475] | [-4.9676969173, -4.4631425941] | 2.4420026680 | 0.0382387308 | 1.0 |

Accuracy first reaches `1.0` at step 147. After step 1000, unit 0 bends at `x = 2.4867` and unit 1 at `x = 1.9904`. The logit changes sign at `x = 1.7705` and `x = 2.7403`.

The trained MLP:

| Example | x | y | hidden_pre | hidden | z | p | Loss |
|---|---|---|---|---|---|---|---|
| ex0 | 0.5 | 0 | [-3.8498, 3.7081] | [0.0, 3.7081] | -14.1076 | 0.0000007467 | 0.0000007467 |
| ex1 | 1.0 | 0 | [-2.8809, 2.4641] | [0.0, 2.4641] | -8.5557 | 0.0001924050 | 0.0001924235 |
| ex2 | 1.5 | 0 | [-1.9120, 1.2202] | [0.0, 1.2202] | -3.0038 | 0.0472541083 | 0.0484070513 |
| ex3 | 2.0 | 1 | [-0.9431, -0.0238] | [0.0, 0.0] | 2.4420 | 0.9199746509 | 0.0834091627 |
| ex4 | 2.5 | 1 | [0.0258, -1.2677] | [0.0258, 0.0] | 2.3137 | 0.9100065294 | 0.0943035044 |
| ex5 | 3.0 | 0 | [0.9947, -2.5117] | [0.9947, 0.0] | -2.4995 | 0.0758905942 | 0.0789248098 |
| ex6 | 3.5 | 0 | [1.9636, -3.7556] | [1.9636, 0.0] | -7.3128 | 0.0006665091 | 0.0006667313 |
| ex7 | 4.0 | 0 | [2.9326, -4.9996] | [2.9326, 0.0] | -12.1260 | 0.0000054166 | 0.0000054166 |

Loss `0.0382387308`, accuracy `1.0`.

Step 3 - the baselines beside the trained MLP:

| Model | Loss | Accuracy |
|---|---|---|
| Majority class (always 0) | - | 0.75 |
| Single layer (F03-1-L1), 500 steps, `lr = 0.5` | 0.5623351450 | 0.75 |
| MLP, 1,000 steps, `lr = 0.2` | 0.0382387308 | 1.0 |

The single layer ends at `weight = [-0.0000478]`, `bias = -1.0984807393`.

Step 4 - the hazard case:

| Example | x | y | hidden_pre | z | dL/dz | dL/dhidden_pre |
|---|---|---|---|---|---|---|
| ex0 | 1.25 | 1 | [0.0, -4.25] | 1.0 | -0.0896471405 | [0.0, 0.0] |
| ex1 | 2.0 | 0 | [0.75, -5.0] | 1.375 | 0.2660622592 | [0.1330311296, 0.0] |
| ex2 | 0.0 | 1 | [-1.25, -3.0] | 1.0 | -0.0896471405 | [0.0, 0.0] |

`dL/dW_hidden = [[0.2660622592, 0.0]]`, `dL/db_hidden = [0.1330311296, 0.0]`, `dL/dw_out = [0.1995466944, 0.0]`, `dL/db_out = 0.0867679783`.

Measured against the library in float32 (not in the JSON file; measured once on this machine):

| Comparison | Max Difference | Tolerance |
|---|---|---|
| hidden_pre, hidden, z | 0, 0, 0 | 1e-5 |
| Loss | 0 | 1e-5 |
| dL/dhidden_pre | 4.7e-10 | 1e-6 |
| dL/dW_hidden, dL/db_hidden | 3.0e-8, 0 | 1e-6 |
| dL/dw_out, dL/db_out | 0, 0 | 1e-6 |
| The four parameters after one step | 0, 0, 0, 0 | 1e-6 |

We also broke the mask on purpose. With `hidden_pre >= 0`, check 3 fails and checks 1, 2 and 4 pass. With no mask, all four checks fail.

Results from analysis:
The output side is the same as F03-1-L1. `binary_cross_entropy` takes the logits and returns `∂L/∂z = (σ(z) − y) / N`, and it does not know a hidden layer exists. The new work is everything after `∂L/∂z`. The output layer's gradients have the same form as logistic regression's, with `hidden` in place of `inputs`: `∂L/∂w_out = hiddenᵀ @ δ` and `∂L/∂b_out = Σ δ`. Then the gradient goes one layer further. `∂L/∂hidden = δ[:, None] * w_out` sends each example's `δ` to each hidden unit, scaled by that unit's output weight. The ReLU mask then zeroes it wherever the unit was off. The hidden layer's gradients take the same form again, with `inputs` and `∂L/∂hidden_pre` in place of `hidden` and `δ`.

The output weight decides which way each hidden unit is pushed. At the initialization, `w_out = [0.5, −1.0]`, so in every row of `∂L/∂hidden_pre` unit 1's entry is `−2` times unit 0's wherever both are on (ex2 to ex5). The same error makes unit 0 move one way and unit 1 the other. The mask decides which examples count. ex0 and ex1 send nothing to unit 0, and ex6 and ex7 send nothing to unit 1, because those units were off for those examples.

Training first made accuracy worse, from `0.625` to `0.5` at steps 1 and 2. At the initialization the logit rose with `x` past ex3, so it called ex5 to ex7 class 1. Those three had the largest losses (`1.80` to `2.46`). The first step cut `w_out[0]` from `0.5` to `0.37` and `b_out` from `1.0` to `0.93`, which lowered every logit, and ex3 (`z = 0.125`) dropped below 0. By step 10, `w_out[0]` had turned negative and accuracy was back to `0.75`. From step 10 to step 100 it stayed at `0.75`, the same as the single layer, while the units moved into place. It reached `1.0` at step 147.

The trained MLP builds a window. Both output weights are negative (`−4.97` and `−4.46`), so each active unit pushes the logit down. Unit 1 is on to the left of `x = 1.9904` and unit 0 to the right of `x = 2.4867`. Between them both are off and `z = b_out = 2.442`, so the model says class 1. The logit crosses 0 at `x = 1.7705` and `x = 2.7403`, which puts ex3 (`x = 2.0`) and ex4 (`x = 2.5`) inside and the other six outside. ex3 sits in the gap where both units are off: its `hidden` is `[0.0, 0.0]` and its `z` is exactly `b_out`. The units swapped roles during training. At the start unit 0 was on to the right and pushed the logit up (`w_out[0] = 0.5`). After training it is on to the right and pushes the logit down.

The single layer cannot build a window. Its logit is a line in `x`, so class 1 is either everything above a threshold or everything below one. The class-1 examples are in the middle, so no line gets more than six of eight right. Training made it give up on `x`. After 500 steps the weight is `−0.0000478` and the bias is `−1.0985`, close to `ln(2/6) = −1.0986`. The model predicts `p = 0.25` for every example, the share of class 1. Its loss `0.5623` is the entropy of that guess, and its accuracy `0.75` is the same as always guessing class 0.

The hazard case shows the two ways the mask zeroes a gradient. Unit 1 is dead: its pre-activation is negative for all three examples. Its activation is 0 everywhere, so `∂L/∂w_out[1]` is exactly 0, and the mask zeroes every entry of `∂L/∂hidden_pre[:, 1]`, so `∂L/∂W_hidden[:, 1]` and `∂L/∂b_hidden[1]` are exactly 0 too. No step of gradient descent will change unit 1's parameters on this data. Unit 0 is exactly 0 at ex0. The derivative of `max(0, x)` is not defined at 0, and PyTorch uses 0 there. A mask written `>= 0` uses 1 instead. That gives `∂L/∂W_hidden[0, 0] = 0.2100` instead of `0.2661`, and the library check does not catch it, because random float32 inputs never land exactly on 0. Only check 3 failed with that mask.

Against the library, the forward pass, the loss, `∂L/∂b_hidden`, both output gradients and all four parameters after the step matched exactly. `∂L/∂W_hidden` differed by `3.0e-8`, the same as the spec's measured value, from summing 64 products in another order. `∂L/∂hidden_pre` differed by `4.7e-10`. The spec measured `0` there. The difference is far below the tolerance, and we did not find its source.

The spec's rep-script notes say unit 1 bends at `x = 1.9989`. Our run gives `1.9904`. K5 gives `−b_hidden[1] / W_hidden[0, 1] = 4.9520113475 / 2.4878914537 = 1.9904`, and `hidden_pre[ex3, 1] = −0.0238` at `x = 2.0` agrees with `1.9904`, not `1.9989`. We report `1.9904`. The spec also says the logit changes sign near `x = 1.78`; our value is `1.7705`. No test checks either number.

What this does not establish is behavior beyond one hidden layer, one output logit and full-batch SGD. We did not test more than one hidden layer, other activations, softmax outputs, mini-batches, weight decay or momentum. We did not test the `ValueError` checks in `mlp_forward_pass`. We did not check that 1,000 steps reach the minimum: the gradients after step 1000 are small but not 0 (`∂L/∂W_hidden = [[−0.0031, −0.0397]]`). We did not measure the float32 drift over 1,000 steps that the spec reports (`5.5e-6`). In practice we would use `nn.Linear`, `nn.ReLU` and autograd.

## Why It Matters
A hidden layer with a nonlinearity is what lets a network draw more than one boundary. Without the ReLU, two linear layers multiply out to one, and the model can only draw a line. With it, each hidden unit adds one bend, and the output layer adds the bent pieces together. Two units were enough for one window here. More units give more bends, and that is how an MLP fits shapes a line cannot.

The backward pass here is the one autograd runs for every `nn.Linear` and `nn.ReLU` in a deep network. Each linear layer's weight gradient is (its input)ᵀ @ (the gradient at its output). Each layer passes the gradient back to the layer below it through its weight. Each ReLU zeroes the gradient where its input was at or below 0. A deeper network repeats these same steps, once per layer.

The mask is also where ReLU networks fail. A dead unit, negative for every example, gets exactly zero gradient and never changes. If a large step or a bad initialization pushes many units there, the network loses that capacity for good. This is the reason for careful initialization, smaller learning rates and variants such as Leaky ReLU, which keep a small gradient below 0. The convention at exactly 0 matters for the same reason: it decides whether a unit on its boundary learns. It cannot be checked with random float32 inputs, because they almost never land on 0, so it needs a test case built for it.

Keeping the loss separate from the model is what lets it be reused. `binary_cross_entropy` was written for logistic regression and runs here unchanged; only the backward pass after `∂L/∂z` is new. `sgd_step` takes a list of parameters of any shapes, so it updates four tensors here as easily as two in F03-1-L1.

## Toolkit
We added hidden_layer_mlp.py, which computes the forward pass and the backward pass of a one-hidden-layer MLP with a ReLU and one output logit by hand, and matches `nn.Sequential(nn.Linear, nn.ReLU, nn.Linear)` with `nn.BCEWithLogitsLoss` in float32. The training loop is in rep.py and the tests, using `sgd_step`. It contains two functions:

- `mlp_forward_pass` takes `W_hidden` `(F, H)`, `b_hidden` `(H,)`, `w_out` `(H,)`, `b_out` `()` and inputs `(N, F)`, and returns `hidden_pre = inputs @ W_hidden + b_hidden` `(N, H)`, `hidden = torch.clamp(hidden_pre, min=0)` `(N, H)` and the logits `hidden @ w_out + b_out` `(N,)`. It raises `ValueError` if the inputs are not `(N, F)`, `W_hidden` is not `(F, H)` with the same `F`, or `b_hidden` or `w_out` is not `(H,)`.
- `mlp_backward_pass` takes the same parameters, inputs and float 0/1 targets `(N,)`. It calls `mlp_forward_pass` and `binary_cross_entropy`, and returns the loss, `∂L/∂hidden_pre` `(N, H)` and the four gradients, `∂L/∂W_hidden` `(F, H)`, `∂L/∂b_hidden` `(H,)`, `∂L/∂w_out` `(H,)` and `∂L/∂b_out` `()`. The ReLU mask is `hidden_pre > 0`, so the gradient is 0 at exactly 0, as in PyTorch.

Neither function changes its arguments, and the results keep the input dtype. We reused `binary_cross_entropy` from F03-1-L1 (binary_cross_entropy.py) and `sgd_step` from F02-1-L1 (sgd_step.py).
