# F03-2-L1 -- Cross Entropy

## Mechanism
Calculate the softmax cross-entropy loss and its gradients for two linear models, a classifier and a candidate scorer, using a fixed set of input features and known targets.

## Task 
Create 3 functions -- Cross-entropy, Classifier gradient, and Scorer gradient. We will also re-use forward_pass (F03-1-L1, binary_cross_entropy.py), softmax (F01-2-L1, softmax.py) and sgd_step (F02-1-L1, sgd_step.py) from other reps. With these, we manually compute the softmax cross-entropy and gradients for two linear models with different parameter structures. The classifier has one weight column and one bias per class: `weight` `(F, C)`, `bias` `(C,)`. The scorer has one weight vector and one bias shared by every candidate: `weight` `(F,)`, `bias` `()`. Both produce logits `(N, C)` for the same loss; only the backward pass after `∂L/∂Z` differs.

## Interface
Cross Entropy
name:
  cross_entropy
arguments:
  logits: A float tensor of shape (N, C)
  targets: An int64 tensor of shape (N, )
returns:
  loss: A float tensor of shape ()
  per_example_losses: A float tensor of shape (N, )
  logit_gradients: A float tensor of shape (N, C)

Classifier Gradient:
name:
  classifier_gradient
arguments:
  weight: A float tensor of shape (F, C)
  bias: A float tensor of shape (C, )
  inputs: A float tensor of shape (N, F)
  targets: An int64 tensor of shape (N, )
returns:
  loss: A float tensor of shape ()
  weight_gradient: A float tensor of shape (F, C)
  bias_gradient: A float tensor of shape (C, )

Scorer Gradient:
name:
  scorer_gradient
arguments:
  weight: A float tensor of shape (F, )
  bias: A float tensor of shape ()
  inputs: A float tensor of shape (N, C, F)
  targets: An int64 tensor of shape (N, )
returns:
  loss: A float tensor of shape ()
  weight_gradient: A float tensor of shape (F,)
  bias_gradient: A float tensor of shape ()

## Pass Check 
All tests must pass.

Here is our test criteria:

1. **Known values (float64).** At each model's initialization, the forward pass returns the `Z` column of the per-example tables within `1e-12`. `softmax`, the per-example losses `ℓ`, `∂L/∂Z` and the loss return the `P`, `ℓ`, `∂L/∂Z` and K1 values within `1e-10`. `∂L/∂weight` has the exact shape, `(1, 3)` for the classifier and `(4,)` for the scorer, and equals K2 within `1e-10`. The classifier's `∂L/∂bias` `(3,)` equals K3 within `1e-10`. The scorer's `∂L/∂bias` `()` is `0` within `1e-15`. Each row of `∂L/∂Z` sums to `0` within `1e-15`, for both models. Reason for the tolerances: the logits are short decimals, exact up to float64 rounding; the other values are rounded to ten decimals (error at most `5e-11`). The scorer's bias gradient sums 18 entries below `0.14`, which leaves about `3e-16` of float64 rounding. Reason for the scorer's checks: a scorer that divides by `N·C = 18` instead of `N = 6` gives `∂L/∂weight[2] = −0.1296` instead of `−0.3889`, and one that sums over the query axis only gives shape `(3, 4)`.
2. **Against the library, one training step (float32).** Classifier: with `g = torch.Generator().manual_seed(0)`, draw `inputs = torch.randn(64, 8)`, then `weight = 2 · torch.randn(8, 5)`, then `bias = torch.randn(5)`, then `targets = torch.randint(0, 5, (64,))`, in that order. Scorer: with a new `g = torch.Generator().manual_seed(0)`, draw `inputs = torch.randn(32, 5, 8)`, then `weight = 2 · torch.randn(8)`, then `targets = torch.randint(0, 5, (32,))`, and set `bias = 0.5`. Both use `lr = 0.5`. Run one step with our functions and `sgd_step`, and one step of `nn.Linear` + `nn.CrossEntropyLoss` + `torch.optim.SGD`. The classifier loads `{"weight": weight.T, "bias": bias}` into `nn.Linear(8, 5)`. The scorer loads `{"weight": weight.unsqueeze(0), "bias": bias.reshape(1)}` into `nn.Linear(8, 1)`, whose logits are `model(inputs).squeeze(-1)`, `(32, 5)`. After `backward()`, our logits, `ℓ` and loss equal the library's within `1e-5`, and `P`, `∂L/∂weight` and `∂L/∂bias` within `1e-6`. After `step()`, `weight` and `bias` agree within `1e-6`. Reason for the shapes: `N`, `C` and `F` all differ, so a transposed product or a sum over the wrong axis fails. Reason for the factor 2: it spreads the logits to about `±27` (classifier) and `±14` (scorer), so the shift by the row max matters. Reason for the tolerances: at `27` the float32 spacing is `1.9e-6`, and `ℓ` moves as much; `P` is below 1, where the spacing is at most `1.2e-7`; the weight gradients sum 64 or 160 small products.
3. **The hazard: saturated logits (float32).** For logits `[[100, 0, −100], [100, 0, −100], [0, 0, 0]]` and targets `[2, 0, 1]`, `cross_entropy` returns `ℓ = [200.0, 0.0, 1.0986123]` and the loss `67.032875` within `1e-4`, and the K6 `∂L/∂Z` within `1e-7`. Every entry of the loss, `ℓ` and `∂L/∂Z` is finite. Reason for the case: row 0 is confidently wrong, row 1 confidently right, and row 2 has no preference. In float32 the probability route gives `P[0, 2] = 0.0`, so `ℓ[0] = −ln 0 = inf`. An unshifted softmax computes `e^100`, which overflows float32 to `inf`, so `P[0, 0] = inf / inf = nan`. Reason for the tolerance: `1e-4` is about six float32 spacings at `200`.
4. **Training (float64).** Each model trains from its initialization with its gradient function and `sgd_step`: the classifier for 300 steps at `lr = 0.5`, the scorer for 200 steps at `lr = 2.0`. After step 1, `weight`, `bias` and the loss at the new parameters equal K4 within `1e-9`. After the last step they equal K5 within `1e-9`. The scorer's `bias` is `−1.0` within `1e-12` after every step. Reason for the tolerances: `1e-9` is the table's rounding plus float64 rounding over 300 steps. The scorer's bias gradient is at most about `1.5e-16` per step, so 200 steps at `lr = 2.0` move the bias at most `6e-14`.

## Results
All tests passed.

We hand-coded a stable softmax cross-entropy from the logits and its gradient, and the gradients of two linear models that use it: a classifier with one weight column per class, and a scorer with one weight shared by every candidate. We reused F03-1-L1's `forward_pass`, F01-2-L1's `softmax` and F02-1-L1's `sgd_step`. Every value matches the Known results tables and K1 to K6. On the saturated logits our loss is `67.032875` and finite, where the probability route gives `inf`.

Numbers from `results/F03-2-L1.json`:

Step 1a - the classifier at the initialization, `weight = [[-0.5, 0.25, 1.0]]`, `bias = [1.0, 0.0, -2.5]`:

| Example | y | Z | P | Loss | dL/dZ |
|---|---|---|---|---|---|
| ex0 | 0 | [0.75, 0.125, -2] | [0.6253168442, 0.3347079873, 0.0399751684] | 0.4694968069 | [-0.0468353945, 0.0418384984, 0.0049968961] |
| ex1 | 0 | [0.5, 0.25, -1.5] | [0.5224289002, 0.4068680366, 0.0707030632] | 0.6492663805 | [-0.0596963875, 0.0508585046, 0.0088378829] |
| ex2 | 0 | [0.25, 0.375, -1] | [0.4132823577, 0.4683102643, 0.1184073779] | 0.8836242447 | [-0.0733397053, 0.058538783, 0.0148009222] |
| ex3 | 1 | [0, 0.5, -0.5] | [0.3071958857, 0.5064803911, 0.1863237232] | 0.6802696706 | [0.0383994857, -0.0616899511, 0.0232904654] |
| ex4 | 1 | [-0.25, 0.625, 0] | [0.2135428577, 0.5122626855, 0.2741944568] | 0.6689177279 | [0.0266928572, -0.0609671643, 0.0342743071] |
| ex5 | 2 | [-0.5, 0.75, 0.5] | [0.1387227148, 0.4841898505, 0.3770874347] | 0.9752781961 | [0.0173403393, 0.0605237313, -0.0778640707] |
| ex6 | 1 | [-0.75, 0.875, 1] | [0.0845092677, 0.4291734741, 0.4863172582] | 0.8458940734 | [0.0105636585, -0.0713533157, 0.0607896573] |
| ex7 | 2 | [-1, 1, 1.5] | [0.048610824, 0.3591881058, 0.5922010702] | 0.5239090562 | [0.006076353, 0.0448985132, -0.0509748662] |

Loss `0.7120820195`, `dL/dweight = [[0.0637067065, -0.0047832429, -0.0589234636]]`, `dL/dbias = [-0.0807987935, 0.0626475994, 0.0181511941]`, accuracy `0.625`.

Step 1b - the scorer at the initialization, `weight = [1.0, 1.0, 0.0, 0.0]`, `bias = -1.0`:

| Example | y | Z | P | Loss | dL/dZ |
|---|---|---|---|---|---|
| q1 | 0 | [0.3, 0.5, -0.5] | [0.3744292216, 0.4573288841, 0.1682418943] | 0.9823524882 | [-0.1042617964, 0.0762214807, 0.0280403157] |
| q2 | 1 | [-0.7, 0.6, -0.2] | [0.1582774867, 0.5807670544, 0.2609554589] | 0.5434055416 | [0.0263795811, -0.0698721576, 0.0434925765] |
| q3 | 1 | [0.5, -0.2, 0.5] | [0.4005470987, 0.1989058027, 0.4005470987] | 1.6149239197 | [0.0667578498, -0.1335156996, 0.0667578498] |
| q4 | 2 | [-0.7, 0.7, 0.3] | [0.128642483, 0.521670993, 0.349686524] | 1.0507181715 | [0.0214404138, 0.0869451655, -0.1083855793] |
| q5 | 0 | [0.3, -0.2, 0.8] | [0.3071958857, 0.1863237232, 0.5064803911] | 1.1802696706 | [-0.1154673524, 0.0310539539, 0.0844133985] |
| q6 | 1 | [0.3, -0.2, -0.7] | [0.5064803911, 0.3071958857, 0.1863237232] | 1.1802696706 | [0.0844133985, -0.1154673524, 0.0310539539] |

Loss `1.0919899104`, `dL/dweight = [0.0885638131, -0.0046811995, -0.3888844248, -0.3109024041]`, `dL/dbias = 0`, accuracy `0.1667`. The bias gradient is `−2.8e-17` before rounding.

Step 2a - classifier training, 300 full-batch steps with `lr = 0.5`. Each row is measured at the parameters after that many steps:

| Step | weight | bias | Loss | Accuracy |
|---|---|---|---|---|
| 0 | [[-0.5, 0.25, 1]] | [1, 0, -2.5] | 0.7120820195 | 0.625 |
| 1 | [[-0.5318533533, 0.2523916215, 1.0294617318]] | [1.0403993967, -0.0313237997, -2.509075597] | 0.7039060342 | 0.625 |
| 2 | [[-0.5579347233, 0.274059971, 1.0338747522]] | [1.081679048, -0.0554911905, -2.5261878575] | 0.6966676337 | 0.625 |
| 10 | [[-0.7119852018, 0.3624284859, 1.0995567159]] | [1.3949730305, -0.2444379364, -2.6505350942] | 0.6507547864 | 0.75 |
| 50 | [[-1.2317533687, 0.6166137637, 1.365139605]] | [2.4315418704, -0.7233039738, -3.2082378967] | 0.5471241489 | 0.75 |
| 100 | [[-1.6412940863, 0.7679385517, 1.6233555347]] | [3.2285876775, -0.9445719736, -3.7840157039] | 0.4957733561 | 0.75 |
| 300 | [[-2.5846193964, 1.0434140547, 2.2912053417]] | [5.0660071983, -1.2262320856, -5.3397751127] | 0.4205052552 | 0.75 |

Step 2b - scorer training, 200 full-batch steps with `lr = 2.0`:

| Step | weight | bias | Loss | Accuracy |
|---|---|---|---|---|
| 0 | [1, 1, 0, 0] | -1 | 1.0919899104 | 0.1667 |
| 1 | [0.8228723738, 1.009362399, 0.7777688496, 0.6218048082] | -1 | 0.703947831 | 0.6667 |
| 2 | [0.7473661508, 1.0101749408, 1.1772342223, 0.9968241748] | -1 | 0.5756445492 | 0.8333 |
| 10 | [0.7770620479, 1.020278162, 2.3801756008, 2.2718665419] | -1 | 0.375223627 | 0.8333 |
| 50 | [1.8911343174, 0.8933024418, 4.0008229977, 3.8798583463] | -1 | 0.2850401276 | 0.8333 |
| 100 | [2.8485915092, 0.808896854, 4.8787353875, 4.6554588287] | -1 | 0.261632457 | 0.8333 |
| 200 | [3.9161594343, 0.8167489064, 5.8565391548, 5.508438235] | -1 | 0.2470848441 | 0.8333 |

The scorer's bias is shown rounded. In the JSON it is `−0.9999999999999988` at step 200; the largest drift over all 200 steps is `1.6e-15`.

Step 3 - saturated logits (float32), our stable `cross_entropy` beside the probability route (`softmax`, then `−ln P[t]`):

| Row | Z | t | P | Stable Loss | Probability Route Loss | dL/dZ |
|---|---|---|---|---|---|---|
| 0 | [100, 0, -100] | 2 | [1.0, 3.7835e-44, 0.0] | 200 | inf | [0.3333333, 1.26e-44, -0.3333333] |
| 1 | [100, 0, -100] | 0 | [1.0, 3.7835e-44, 0.0] | 0 | 0 | [0.0, 1.26e-44, 0.0] |
| 2 | [0, 0, 0] | 1 | [0.3333333, 0.3333333, 0.3333333] | 1.0986123 | 1.0986123 | [0.1111111, -0.2222222, 0.1111111] |

Stable loss `67.032875`; probability route loss `inf`. The JSON file stores the `inf` as the string `"inf"`, because strict JSON has no infinity.

Measured against the library in float32 (not in the JSON file; measured once on this machine):

| Comparison | Classifier | Scorer | Tolerance |
|---|---|---|---|
| Logits | 0 | 0 | 1e-5 |
| P | 1.2e-7 | 1.2e-7 | 1e-6 |
| Per-example losses | 1.9e-6 | 9.5e-7 | 1e-5 |
| Loss | 0 | 4.8e-7 | 1e-5 |
| dL/dweight | 1.5e-8 | 7.5e-8 | 1e-6 |
| dL/dbias | 7.5e-9 | 1.3e-8 | 1e-6 |
| weight after one step | 7.5e-9 | 4.5e-8 | 1e-6 |
| bias after one step | 0 | 0 | 1e-6 |

Results from analysis:
Only the loss is new since F03-1-L1. The backward pass after `∂L/∂Z` has the same form as before: the weight gradient is the inputs times `∂L/∂Z`, and the bias gradient is a sum of `∂L/∂Z`. What changed is `∂L/∂Z` itself. With sigmoid and binary cross-entropy it was `(σ(z) − y) / N`. With softmax and cross-entropy it is `(P − onehot(t)) / N`, one row per example. The softmax Jacobian cancels against the `1 / P[t]` from `−ln P[t]`, which leaves this simple form. Each example pushes its target logit up by `(1 − P[t]) / N` and every other logit down by `P[c] / N`. Those pushes balance, so every row of `∂L/∂Z` sums to 0.

The two models share the loss and differ only after `∂L/∂Z`. The classifier gives each class its own weight column, so `∂L/∂weight = inputsᵀ @ ∂L/∂Z` has shape `(F, C)`, and `∂L/∂bias` sums over the examples only, one entry per class. The scorer uses one weight for every candidate, so its weight gradient sums `∂L/∂Z[n, c] · inputs[n, c]` over both the queries and the candidates, and its shape is `(F,)`. Summing over the queries only would give `(C, F)`, one weight per candidate slot, which is a different model.

The scorer's bias cannot learn anything. It adds the same number to every candidate's logit, and softmax does not change when every logit in a row shifts by the same amount. Its gradient is the sum of every entry of `∂L/∂Z`, and every row of `∂L/∂Z` sums to 0, so the bias gradient is exactly 0 in exact arithmetic. In float64 it is `−2.8e-17` at the initialization and at most `1.5e-16` at any step, and over 200 steps at `lr = 2.0` the bias moves by at most `1.6e-15`. A per-candidate score needs no bias; only the differences between candidates matter.

The classifier got two of the eight examples wrong after training. At the initialization it predicted classes `[0, 0, 1, 1, 1, 1, 2, 2]`, five of eight right. After 300 steps it predicts `[0, 0, 0, 1, 1, 1, 2, 2]`, six right. The targets in order of `x` are `0, 0, 0, 1, 1, 2, 1, 2`. With one feature each logit is a line in `x`, so each class wins on a single interval, in the order of the slopes (`−2.58`, `1.04`, `2.29`): class 0, then 1, then 2. The trained boundaries are `x = 1.73` (0 to 1) and `x = 3.30` (1 to 2). ex5 (`x = 3.0`, class 2) and ex6 (`x = 3.5`, class 1) are out of order, so no boundaries get both right. A boundary between `2.5` and `3.0`, or between `3.5` and `4.0`, would get seven right. Training put it between them at `3.30` instead, because it lowers the mean loss, not the error count: there both are only somewhat wrong (`P[t] = 0.41` and `0.44`). The loss was still falling at step 300 (`∂L/∂bias = [−0.0137, 0.0020, 0.0117]`), so more steps would lower it further.

The scorer learned which features matter. At the initialization it used only `overlap` and `cosine` and picked the right candidate for one query of six. After one step, `version_ok` and `accepted` gained weights of `0.78` and `0.62`, the largest gradients at the start (`−0.389` and `−0.311`), and accuracy rose to four of six. By step 2 it was five of six, and it stayed there. After 200 steps the weights are `[3.92, 0.82, 5.86, 5.51]`: `version_ok` and `accepted` matter most, then `overlap`; `cosine` ends below where it started.

The scorer cannot get q5 and q6 both right. They share their first two candidates, with different targets: `c1` is correct for q5 and `c2` for q6. The scorer computes each candidate's logit from that candidate's features alone, so `c1` and `c2` get the same two logits in both queries (`12.6665` and `12.5681`). Whichever one scores higher wins in both. Training settled near a tie, with `P ≈ 0.52` and `0.48` in both queries, which gets q5 right and q6 wrong. Even a perfect tie costs `ln 2` per query, so the loss can never go below `2 ln 2 / 6 = 0.2310`. The final loss is `0.2471`.

The saturated case shows why the loss is computed from the logits. In float32, `e^(−200)` underflows, so the probability route gives `P[0, 2] = 0.0` and `ℓ[0] = −ln 0 = inf`. One confidently wrong example makes the mean loss `inf`. Without the shift by the row max it is worse: `e^100` overflows to `inf`, and `P[0, 0]` becomes `inf / inf = nan`. The stable form `ℓ = m + ln Σ e^(z − m) − z_t` keeps every `exp` argument at or below 0. It gives `200.0` for row 0, `0.0` for row 1, `ln 3` for row 2, and the loss `67.032875`, the same as `F.cross_entropy`. The gradient `(P − onehot(t)) / N` stays bounded by `1 / N` per entry even when the loss is `200`. Row 1's middle class has `P = 3.78e-44`, a subnormal float32, so its gradient `1.26e-44` is tiny but not zero.

Against the library, the logits and the bias after one step matched exactly, and the classifier's loss did too. The other differences were a few float32 spacings, and they match the spec's measured values. The scorer's bias gradient is 0 only up to rounding: ours was `2.0e-8` and the library's `7.5e-9`.

What this does not establish is behavior beyond hard-label cross-entropy on full batches. We did not test class weights, label smoothing, `ignore_index`, soft targets, queries with different numbers of candidates (which need padding and a mask), or mini-batches. The `ValueError` checks in `cross_entropy`, `classifier_gradient` and `scorer_gradient` are not covered by the tests. In practice we would use `nn.CrossEntropyLoss` or `F.cross_entropy`, which take logits and use the same stable form.

## Why It Matters
Softmax cross-entropy is the loss for every choice among several options. It trains image and text classifiers, every token of a language model (one class per vocabulary entry), and rankers that pick one item from a list. The classifier here is the last layer of most multi-class networks: a deep model still ends in `Z = h @ weight + bias` and a softmax. The scorer is the shape of retrieval and reranking. One scoring function rates every candidate, and the softmax runs across the candidates of each query, so a list of any length can be scored.

The loss must be computed from the logits. Trained models produce logits in the tens, and a language model's softmax runs over tens of thousands of classes. In float32, the probability of a confidently wrong class rounds to `0.0` and `−ln 0` is `inf`; an unshifted `e^z` overflows past `z ≈ 88.7` and gives `nan`. Either one ruins the batch. This is why `nn.CrossEntropyLoss` takes logits, not probabilities, and why a model should output logits and leave the softmax to the loss.

The gradient `(P − onehot(t)) / N` is what makes the loss train well. It is bounded, so a confidently wrong example cannot cause a huge step, and it does not vanish when the softmax saturates: a confidently wrong example still pushes with nearly the full `1 / N`.

The scorer shows two limits of shared-weight scoring. A bias shared by every candidate has gradient 0 and learns nothing, so a ranking model gets no benefit from it. And the score depends only on the candidate's feature vector. Two queries whose candidates have the same features get the same logits, so they get the same answer. q5 and q6 have identical `c1` and `c2` with different correct answers, so no weight gets both right. Features that describe how a candidate relates to its query, such as `overlap` and `cosine`, are how a scorer tells such queries apart, but only if those values differ between the queries.

Keeping the loss separate from the model is what lets it be reused. Our `cross_entropy` takes logits and targets and knows nothing about the weights. The classifier and the scorer call it unchanged; only the backward pass after `∂L/∂Z` differs. The same split let us reuse `forward_pass`, `softmax` and `sgd_step` here without changing them.

## Toolkit
We added cross_entropy.py, which computes the softmax cross-entropy loss and the gradients of a linear classifier and a linear candidate scorer by hand, and matches `nn.Linear` and `nn.CrossEntropyLoss` in float32. The training loop is in rep.py and the tests, using `sgd_step`. It contains three functions:

- `cross_entropy` takes logits `(N, C)` and int64 targets `(N,)`, and returns the loss, the per-example losses `m + ln Σ e^(z − m) − z_t` and `∂L/∂Z = (softmax(Z) − onehot(t)) / N`. It raises `ValueError` if the logits are not `(N, C)`, the targets are not `(N,)`, the two `N` differ, or the targets are not int64. It knows nothing about the model.
- `classifier_gradient` takes the weight `(F, C)`, the bias `(C,)`, inputs `(N, F)` and targets, and returns the loss, `inputsᵀ @ ∂L/∂Z` `(F, C)` and `∂L/∂Z` summed over the examples `(C,)`.
- `scorer_gradient` takes the weight `(F,)`, the bias `()`, inputs `(N, C, F)` and targets, and returns the loss, `Σ ∂L/∂Z[n, c] · inputs[n, c]` over queries and candidates `(F,)` and the sum of every entry of `∂L/∂Z` `()`.

None of the functions change their arguments, and the results keep the input dtype. We reused `forward_pass` from F03-1-L1 (binary_cross_entropy.py), `softmax` from F01-2-L1 (softmax.py) and `sgd_step` from F02-1-L1 (sgd_step.py).
