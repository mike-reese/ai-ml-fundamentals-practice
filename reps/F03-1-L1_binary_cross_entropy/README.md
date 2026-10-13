# F03-1-L1 -- Binary cross-entropy, Logistic Regression

## Mechanism
We're training logistic regression, a binary classifier, by hand: a forward pass to logits, a sigmoid, the binary cross-entropy loss computed from the logits, a backward pass, and an SGD step.

## Task 
Create toolkit functions that calculate the sigmoid, binary cross-entropy, and logistic-regression gradient. Write a test suite that validates the integrity of the codebase, checks against PyTorch (`nn.Linear`, `nn.BCEWithLogitsLoss`, `torch.optim.SGD`), and validates bounds and edge cases. Write a rep script that trains 500 steps and outputs the performance metrics of doing so using our functions we've created. 

## Interface
Forward Pass
name:
  forward_pass
arguments:
  weight: A float tensor of shape (F, ) containing the weight parameter values used in our logistic regression model.
  bias: A float tensor of shape () containing the bias parameter value used in our logistc regression model.
  inputs: A float tensor of shape (B, F) containing B examples, each with F features, as inputs to the logistic regression model.
returns:
  logits: A float tensor of shape (B, ) containing the logit of each example from our logistic regression model

Sigmoid 
name:
  sigmoid
arguments:
  logits: A tensor of any shape containing the logits from our logistic regression model
returns:
  probabilities: A tensor of the same shape as the logits containing the probabilities from our sigmoid function applied to those logits

Binary Cross-Entropy 
name:
  binary_cross_entropy
arguments:
  logits: A float tensor of shape (N, ) containing the logits from our logistic regression model 
  targets: A float tensor of shape (N, ) (same shape and order as logits) containing the known correct classifications for our dataset 
returns:
  loss: A float tensor of shape () containing the loss from the binary cross-entropy function
  per_example_losses: A float tensor of shape (N, ) with the same shape and order as our logits containing the per-example losses against each target 
  logit_gradients: A float tensor of the same shape and order as the logits containing the gradients of the loss with respect to the logits.

Logistic-Regression Gradient 
name:
  logistic_regression_gradient
arguments:
  weight: A float tensor of shape (F, ) containing the weight parameter values used in our logistic regression model.
  bias: A float tensor of shape () containing the bias parameter value used in our logistc regression model.
  inputs: A float tensor of shape (B, F) containing B examples, each with F features, as inputs to the logistic regression model.
  targets: A float tensor of shape (B, ) in the same order as our inputs containing the known classifications we are comparing the outputs of our logistic regression model against
returns:
  loss: A float tensor of shape () containing the loss from the binary cross-entropy function
  weight_gradient: A float tensor of shape (F, ) containing the gradient of the loss with respect to the weight parameters 
  bias_gradient: A float tensor of shape () containing the gradient of the loss with respect to the bias parameter

## Pass Check 
All tests must pass.

Here is our test criteria:

1. **Known values (float64).** At the initialization and again at the unseen start, the forward pass returns the K1 logits within `1e-12`, and sigmoid, the per-example losses, the loss, `∂L/∂z`, `∂L/∂weight` and `∂L/∂bias` return the K2 to K6 rows within `1e-10`. Reason for the tolerances: the logits are short decimals, exact up to float64 rounding; the other rows are rounded to ten decimals (error at most `5e-11`). Reason for the unseen start: its logits run from `1.5` down to `−2.0`, so it tests the sigmoid branch for negative logits, which the initialization (all logits positive) never reaches. `ex3` has `z = 0`, so its `p` is exactly `0.5` and its `∂L/∂z` exactly `−0.0625`.
2. **Against the library, one training step (float32).** With `torch.Generator().manual_seed(0)`, draw `inputs = torch.randn(64, 8)`, then `weight = 2 · torch.randn(8)`, then `targets = torch.randint(0, 2, (64,)).float()`, in that order. Set `bias = 0.5` and `lr = 0.5`. Run one step with our functions and `sgd_step`, and one step of `nn.Linear(8, 1)` + `nn.BCEWithLogitsLoss` + `torch.optim.SGD`. After `backward()`, our logits equal the library's within `1e-5`, `p` within `1e-6`, the per-example losses within `1e-5`, the loss within `1e-6`, `∂L/∂weight` within `1e-6` and `∂L/∂bias` within `1e-7`. After `step()`, `weight` and `bias` agree within `1e-6`. Reason for 8 features: with one feature, a backward pass that uses `inputs` instead of `inputsᵀ`, or sums over the wrong axis, can still give the right numbers. Reason for the factor 2: it widens the logits to about `−9.8` to `11.1`, so both sigmoid branches are used. Reason for the tolerances: at `11.1` the float32 spacing is `9.5e-7`, and an 8-term dot product summed in another order moves a few spacings.
3. **The hazard: saturated logits (float32).** For logits `[40, −40, 100, −100]` and targets `[0, 1, 1, 0]`, sigmoid returns `[1.0, 4.2484e-18, 1.0, 3.7835e-44]` within `1e-7`, the per-example losses `[40.0, 40.0, 0.0, 0.0]` within `1e-6`, the loss `20.0` within `1e-6`, and `∂L/∂z` `[0.25, −0.25, 0.0, 0.0]` within `1e-7`. Every entry of all four outputs is finite. Reason for the case: in float32 `σ(40)` rounds to exactly `1.0`, so the loss from the probability is `−ln(0) = inf`; and `e^100` overflows to `inf`, so a sigmoid written as `e^z/(1 + e^z)` for every `z` gives `nan`.
4. **Training: two separate 500-step runs.** Run 1 (float64): 500 steps of the logistic-regression gradient and `sgd_step` from `weight = [0.5]`, `bias = 0.0`, `lr = 0.5`. After step 1, `weight`, `bias` and the loss at the new parameters equal K7 within `1e-9`. After step 500, `weight`, `bias`, the loss and `p` at the new parameters equal K8 and K9 within `1e-9`. Run 2 (float32): the same loop beside `nn.Linear(1, 1)` + `nn.BCEWithLogitsLoss` + `torch.optim.SGD`. At every step from 0 to 500, before that step's update, the losses agree within `1e-6` and the parameters within `1e-5`: 501 comparisons and 500 updates, with no update at step 500. Reason for the case: some errors show only in the loop, such as taking each step's gradient at the initialization instead of at the current parameters. Reason for the tolerances: run 1's are the table's rounding plus float64 rounding over 500 steps; in run 2, `|bias|` reaches `4.4`, where the float32 spacing is `4.8e-7`.

## Results
All tests passed.

We hand-coded the forward pass, a stable sigmoid, the binary cross-entropy from the logits and its gradient, and the logistic-regression gradient, and trained the model with F02-1-L1's `sgd_step`. Every value matches K1 to K10. On the saturated logits our loss is `20.0` and finite, where the probability route gives `inf`.

Numbers from `results/F03-1-L1.json`:

Step 1a - the initialization, `weight = [0.5]`, `bias = 0.0`:

| Example | x | y | z | p | Loss | dL/dz |
|---|---|---|---|---|---|---|
| ex0 | 0.5 | 0 | 0.25 | 0.5621765009 | 0.8259394199 | 0.0702720626 |
| ex1 | 1.0 | 0 | 0.5 | 0.6224593312 | 0.9740769842 | 0.0778074164 |
| ex2 | 1.5 | 0 | 0.75 | 0.6791786992 | 1.1368710061 | 0.0848973374 |
| ex3 | 2.0 | 1 | 1.0 | 0.7310585786 | 0.3132616875 | -0.0336176777 |
| ex4 | 2.5 | 0 | 1.25 | 0.7772998612 | 1.5019290813 | 0.0971624826 |
| ex5 | 3.0 | 0 | 1.5 | 0.8175744762 | 1.7014132780 | 0.1021968095 |
| ex6 | 3.5 | 1 | 1.75 | 0.8519528020 | 0.1602241504 | -0.0185058998 |
| ex7 | 4.0 | 1 | 2.0 | 0.8807970780 | 0.1269280110 | -0.0149003653 |

Loss `0.8425804523`, `dL/dweight = [0.5981786235]`, `dL/dbias = 0.3653121659`, accuracy `0.375`.

Step 1b - the unseen start, `weight = [-1.0]`, `bias = 2.0`:

| Example | x | y | z | p | Loss | dL/dz |
|---|---|---|---|---|---|---|
| ex0 | 0.5 | 0 | 1.5 | 0.8175744762 | 1.7014132780 | 0.1021968095 |
| ex1 | 1.0 | 0 | 1.0 | 0.7310585786 | 1.3132616875 | 0.0913823223 |
| ex2 | 1.5 | 0 | 0.5 | 0.6224593312 | 0.9740769842 | 0.0778074164 |
| ex3 | 2.0 | 1 | 0.0 | 0.5 | 0.6931471806 | -0.0625 |
| ex4 | 2.5 | 0 | -0.5 | 0.3775406688 | 0.4740769842 | 0.0471925836 |
| ex5 | 3.0 | 0 | -1.0 | 0.2689414214 | 0.3132616875 | 0.0336176777 |
| ex6 | 3.5 | 1 | -1.5 | 0.1824255238 | 1.7014132780 | -0.1021968095 |
| ex7 | 4.0 | 1 | -2.0 | 0.1192029220 | 2.1269280110 | -0.1100996347 |

Loss `1.1621973864`, `dL/dweight = [-0.4450610286]`, `dL/dbias = 0.0774003653`, accuracy `0.25`.

Step 2 - training, 500 full-batch steps with `lr = 0.5`. Each row is measured at the parameters after that many steps:

| Step | weight | bias | Loss | Accuracy |
|---|---|---|---|---|
| 0 | 0.5 | 0.0 | 0.8425804523 | 0.375 |
| 1 | 0.2009106883 | -0.1826560830 | 0.6732824348 | 0.5 |
| 2 | 0.1256787005 | -0.2781991322 | 0.6531074332 | 0.625 |
| 5 | 0.1749459409 | -0.4552996011 | 0.6307778189 | 0.75 |
| 10 | 0.2701923211 | -0.7177725985 | 0.5999048977 | 0.75 |
| 50 | 0.7647020491 | -2.0921033752 | 0.4893814792 | 0.75 |
| 100 | 1.0695326256 | -2.9533047991 | 0.4550597420 | 0.75 |
| 200 | 1.3477392855 | -3.7485522855 | 0.4399219814 | 0.75 |
| 500 | 1.5589395535 | -4.3571551186 | 0.4362370264 | 0.75 |

The trained model, `weight = [1.5589395535]`, `bias = -4.3571551186`:

| Example | x | y | z | p | Loss | dL/dz |
|---|---|---|---|---|---|---|
| ex0 | 0.5 | 0 | -3.5776853419 | 0.0271808546 | 0.0275570872 | 0.0033976068 |
| ex1 | 1.0 | 0 | -2.7982155651 | 0.0574206796 | 0.0591352036 | 0.0071775849 |
| ex2 | 1.5 | 0 | -2.0187457884 | 0.1172487413 | 0.1247118183 | 0.0146560927 |
| ex3 | 2.0 | 1 | -1.2392760117 | 0.2245620315 | 1.4936033003 | -0.0969297461 |
| ex4 | 2.5 | 0 | -0.4598062349 | 0.3870317913 | 0.4894422063 | 0.0483789739 |
| ex5 | 3.0 | 0 | 0.3196635418 | 0.5792422525 | 0.8656980326 | 0.0724052816 |
| ex6 | 3.5 | 1 | 1.0991333185 | 0.7500976804 | 0.2875518404 | -0.0312377900 |
| ex7 | 4.0 | 1 | 1.8786030953 | 0.8674505927 | 0.1421967225 | -0.0165686759 |

Loss `0.4362370264`, `dL/dweight = [-0.0004426538]`, `dL/dbias = 0.0012793280`, accuracy `0.75`.

Step 3 - saturated logits (float32), our stable functions beside the probability route (`σ`, then `−ln p_target`):

| Example | z | y | p | Stable Loss | Probability Route Loss | dL/dz |
|---|---|---|---|---|---|---|
| ex0 | 40 | 0 | 1.0 | 40.0 | inf | 0.25 |
| ex1 | -40 | 1 | 4.2484e-18 | 40.0 | 40.0 | -0.25 |
| ex2 | 100 | 1 | 1.0 | 0.0 | 0.0 | 0.0 |
| ex3 | -100 | 0 | 3.7835e-44 | 0.0 | 0.0 | 0.0 |

Stable loss `20.0`; probability route loss `inf`. The JSON file stores the `inf` as `Infinity`.

Measured against the library in float32 (not in the JSON file; measured once on this machine):

| Comparison | Max Difference | Tolerance |
|---|---|---|
| Check 2: logits | 0 | 1e-5 |
| Check 2: p | 3.0e-8 | 1e-6 |
| Check 2: per-example losses | 1.8e-7 | 1e-5 |
| Check 2: loss | 0 | 1e-6 |
| Check 2: dL/dweight | 7.5e-9 | 1e-6 |
| Check 2: dL/dbias | 9.3e-10 | 1e-7 |
| Check 2: weight and bias after one step | 0 | 1e-6 |
| Check 4 run 2: weight over 500 steps | 4.8e-7 | 1e-5 |
| Check 4 run 2: bias over 500 steps | 9.5e-7 | 1e-5 |
| Check 4 run 2: loss over 500 steps | 1.2e-7 | 1e-6 |

Results from analysis:
Only sigmoid and the loss are new since F02-1-L1. The backward pass after `∂L/∂z` is the same as before: `∂L/∂weight = inputsᵀ @ ∂L/∂z` and `∂L/∂bias = Σ ∂L/∂z`. What changed is `∂L/∂z` itself. With MSE it was `(2 / B) · (ŷ − y)`; with sigmoid and binary cross-entropy together it is `(σ(z) − y) / N`. The `σ'(z)` from the sigmoid cancels against the `1 / (p(1 − p))` from the loss, which leaves this simple form. Each example pushes its logit by how far its probability is from its target.

The unseen start tests the other half of the sigmoid. Its logits run from `1.5` to `−2.0`, so ex4 to ex7 use the `e^z / (1 + e^z)` branch. The table is symmetric: `p` at `z = −1.5` is `0.1824255238`, and `1 − p` at `z = 1.5` is the same. `ex3` sits at `z = 0`, where `p` is exactly `0.5`, its loss is `ln 2 = 0.6931471806`, and its `∂L/∂z` is exactly `(0.5 − 1) / 8 = −0.0625`.

Training first moved the weight the wrong way. At the initialization the model said class 1 for every example (`p > 0.5` everywhere), and five of the eight targets are 0. So the first step pushed both the weight and the bias down, from `0.5` to `0.2009`, to lower every probability. By step 2 the weight was `0.1257`, and then it rose as the bias kept falling. After that the model could rank the examples by `x`. Accuracy reached `0.75` by step 5 and was `0.75` at every reported step after that. The loss kept falling, from `0.5999` at step 10 to `0.4362` at step 500, because the model kept growing more confident about the six examples it already got right. After 500 steps the gradients are small (`−0.0004`, `0.0013`) but not zero, so more steps would still lower the loss a little.

The trained model still gets ex3 and ex5 wrong. With one feature, `z = weight · x + bias` rises or falls with `x`, so the model can only draw one threshold on the line: here near `x = 2.79`, where `z = 0`. The targets in order of `x` are `0, 0, 0, 1, 0, 0, 1, 1`. No single threshold separates them: ex3 (`x = 2.0`, class 1) sits below ex4 and ex5 (class 0). A threshold between `x = 3.0` and `3.5` would get seven right, but training does not aim for accuracy. It lowers the mean loss, and putting the threshold lower makes ex3 less wrong at the cost of ex5. To get all eight right, the model needs more than a straight line in `x`: another feature, or a model with a hidden layer.

The saturated case shows why the loss is computed from the logits. In float32, `σ(40)` rounds to exactly `1.0`, so `1 − p` is `0` and the probability route gives `−ln(0) = inf` for ex0. One confidently wrong example makes the mean loss `inf`, and training cannot recover from that. The stable form gives `40.0`, which is correct. The gradient at the logits is `(σ(z) − y) / N`, so it is bounded by `1 / N = 0.25`, even when the loss is `40`. One optimizer step on a confidently wrong example pushes as hard as it can, but no harder: the gradient does not explode.

Against the library, the logits, the loss and the parameters after one step matched exactly. The other differences were a few float32 spacings, and they match the spec's measured values. Over 500 steps the largest drift was `9.5e-7` in the bias, which reaches `−4.36`, where the float32 spacing is `4.8e-7`.

What this does not establish is behavior beyond binary logistic regression trained by full-batch SGD. We did not test more than two classes (softmax with cross-entropy), mini-batches, class weights such as `pos_weight`, regularization, or targets other than exactly 0 and 1. We did not check that the loss reaches its minimum. In practice we would use `nn.BCEWithLogitsLoss` or `F.binary_cross_entropy_with_logits`, which take logits and use the same stable form.

## Why It Matters
Binary cross-entropy is the loss for every yes-or-no prediction: spam or not, click or not, and each label in a multilabel classifier. Logistic regression is the simplest model that uses it, and it is the last layer of most binary classifiers. A network with hidden layers still ends in `z = h @ weight + bias` and a sigmoid, so the forward pass, loss and `∂L/∂z` here are the same ones at the top of a deep model.

The loss must be computed from the logits, not from the probabilities. A trained classifier routinely produces logits of `40` or more, and in float32 the probability rounds to exactly `1.0`. The probability route then gives `inf` or `nan`, and one bad example ruins the whole batch. `F.binary_cross_entropy` hides this by clamping each `ln` at `−100`, which gives `35.0` instead of `20.0` on the saturated case: no error, just a wrong loss. This is why `nn.BCEWithLogitsLoss` takes logits, and why a model should output logits and leave the sigmoid to the loss.

The gradient `(σ(z) − y) / N` is what makes this loss train well. It is bounded by `1 / N`, so a confidently wrong example cannot cause a huge step. It also does not vanish when the sigmoid saturates: a confidently wrong example still gets the full `±1 / N`. With MSE on a sigmoid output, `σ'(z)` would multiply the gradient and shrink it toward zero exactly on those examples.

Keeping the loss separate from the model is what lets it be reused. Our `binary_cross_entropy` takes logits and targets and knows nothing about the weight or bias. An MLP or a multilabel output layer can call it unchanged; only the backward pass after `∂L/∂z` depends on the model. The same split let us reuse F02-1-L1's `sgd_step` here without changing it.

## Toolkit
We added binary_cross_entropy.py, which computes the forward pass, the binary cross-entropy loss and the gradients of a logistic regression model by hand, and matches `nn.Linear` and `nn.BCEWithLogitsLoss` in float32. The training loop is in rep.py and the tests, using `sgd_step`. It contains four functions:

- `forward_pass` takes the weight, bias and a batch of inputs and returns the logits `inputs @ weight + bias`, not probabilities or classes.
- `sigmoid` takes logits of any shape and returns `σ(z)`, using `1 / (1 + e^(−|z|))` for `z ≥ 0` and `e^(−|z|) / (1 + e^(−|z|))` for `z < 0`, so no `exp` overflows.
- `binary_cross_entropy` takes logits and targets and returns the loss, the per-example losses `max(z, 0) − y·z + ln(1 + e^(−|z|))` and `∂L/∂z = (σ(z) − y) / N`. It raises `ValueError` if the logits and targets have different shapes. It knows nothing about the model.
- `logistic_regression_gradient` takes the weight, bias, a batch of inputs and their targets, and returns the loss, `inputsᵀ @ ∂L/∂z` and `Σ ∂L/∂z`.

None of the functions change their arguments, and the results keep the input dtype. We reused `sgd_step` from F02-1-L1 (sgd_step.py) for the optimizer step.
