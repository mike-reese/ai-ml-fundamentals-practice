# F01-2-L1 -- Softmax

## Mechanism
Compute a stable softmax by hand that turns a vector of logits into probabilities that are positive and sum to 1 across one chosen axis.

## Task 
Add one function to the toolkit - softmax - which performs a manual calculation of a stable softmax that exists in common ML libraries.

## Interface
Softmax:
name:
  softmax
arguments:
  logits: a floating tensor (float32 or float64) of one or more axes.
  dim: integer. May be negative. Used to select the axis to normalize over in logits.
returns:
  probabilities: a tensor of the same shape, dtype, and device as logits which contains probabilities in which every slice along dim sums to 1. Formula: `p[i] = exp(z[i] − m) / Σ_k exp(z[k] − m)`, where `z` is logits, `m` is the max of logits along dim for each slice, and `Σ_k` runs over the entries of that slice along dim. The shift `m` cancels in the quotient, so it does not change `p`. Without it, `exp` overflows `float32` once a logit is above `ln(float32 max) ≈ 88.72`.

## Pass Check 
All tests pass.

Test criteria:

1. **Known values.** The softmax of `z_orient` and of `z` match Table 2 within `1e-10`. Reason for the tolerance: Table 2 is rounded to ten decimals (error at most `5e-11`); the float64 computation itself is accurate to about `1e-16`.
2. **Each slice sums to 1.** For `z`, `Z` over `dim=-1` and `Z` over `dim=0`, every slice along the normalized axis sums to 1 within `1e-12`, and every entry is greater than 0. Reason: three float64 values summed, rounding error about `2e-16`.
3. **The axis is the argument.** `Z` over `dim=-1` and over `dim=0` match Table 2 within `1e-10`. `dim=1` gives the same result as `dim=-1` exactly. Same reason as check 1.
4. **A square input.** `G` over `dim=-1` matches Table 2 within `1e-10`, and each of its rows sums to 1 within `1e-12`. Reason for the case: on a square input, a shift or a normalizer that does not line up with its slice still broadcasts, so the shape gives no error and only the values show the mistake.
5. **Shift invariance.** The softmax of `z + 100.0` equals the softmax of `z` within `1e-12`. Reason: in float64 the rounding of `z + 100` moves each logit by at most `7.1e-15` (half the spacing near 103); the measured difference is `3e-16`.
6. **Large logits.** The softmax of `z_big` matches Table 2 within `1e-10`: no `inf` and no NaN, and the third entry is exactly `0.0`. Reason for the exact 0: `exp(−1000)` is below the smallest positive float64, so it underflows to 0; the stable softmax cannot represent that probability.
7. **Against the library, float32.** On `z`, `Z` (both axes), `z + 100`, `z_big`, `G` and `Zc` over `dim=-1`, built in float32, your result has the same shape as `torch.softmax` with the same `dim` and differs by at most `1e-6` in every entry. `z + 100` contains no NaN. Reason for the tolerance: the notebook's tolerance; probabilities are at most 1 and float32 epsilon is `1.19e-7`; the measured difference is at most `1.5e-8`.
8. **Argument unchanged, dtype kept.** After the call on `Z`, `Z` equals a copy taken before the call, exactly. The result's dtype is `torch.float32` for float32 input and `torch.float64` for float64 input.

## Results
All tests passed.

We successfully demonstrated a stable softmax that does not overflow, returns no NaN values on these inputs, and matches torch.softmax. In float32 the largest difference from torch.softmax was 1.49e-8, well inside the 1e-6 tolerance. On five of the eight inputs the difference was exactly 0.

Numbers from `results/F01-2-L1.json`:

Step 1 - the five sub-states for `z = [3.1, 1.6, 1.3]`:

| Step | Name | Operation | Value |
|---|---|---|---|
| 1 | m | max(z) | 3.1 |
| 2 | s | z - m | [0.0, -1.5, -1.8] |
| 3 | e | exp(s) | [1.0000, 0.2231, 0.1653] |
| 4 | t | sum(e) | 1.3884 |
| 5 | p | e / t | [0.7202, 0.1607, 0.1191] |

`p[0] = e[0] / t = 1.0000 / 1.3884 = 0.7202`.

Step 2 - `Z` over `dim=-1`, each row sums to 1:

| Row | c1 | c2 | c3 | Row Sum |
|---|---|---|---|---|
| r1 | 0.7202 | 0.1607 | 0.1191 | 1.0000 |
| r2 | 0.1155 | 0.8531 | 0.0315 | 1.0000 |
| Col Sum | 0.8357 | 1.0138 | 0.1505 | |

Step 2 - `Z` over `dim=0`, each column sums to 1:

| Row | c1 | c2 | c3 | Row Sum |
|---|---|---|---|---|
| r1 | 0.8320 | 0.1301 | 0.7503 | 1.7124 |
| r2 | 0.1680 | 0.8699 | 0.2497 | 1.2876 |
| Col Sum | 1.0000 | 1.0000 | 1.0000 | |

Step 3 - float32, naive softmax against ours on `z + 100 = [103.1, 101.6, 101.3]`:

| Method | p1 | p2 | p3 |
|---|---|---|---|
| Naive `exp(z) / sum(exp(z))` | NaN | NaN | NaN |
| Our softmax | 0.7202380 | 0.1607068 | 0.1190551 |

Step 3 - float32, ours against torch.softmax:

| Input | dim | Shape | Max Difference From torch.softmax |
|---|---|---|---|
| z_orient | -1 | (2,) | 0.000e+00 |
| z | -1 | (3,) | 0.000e+00 |
| Z | -1 | (2, 3) | 7.451e-09 |
| Z | 0 | (2, 3) | 0.000e+00 |
| z + 100 | -1 | (3,) | 1.490e-08 |
| z_big | -1 | (3,) | 0.000e+00 |
| G | -1 | (3, 3) | 1.490e-08 |
| Zc | -1 | (2, 3, 1) | 0.000e+00 |

Step 4 - float32, a fully masked row `z_masked = [-inf, -inf]`:

| Method | p1 | p2 |
|---|---|---|
| Our softmax | NaN | NaN |
| torch.softmax | NaN | NaN |

Results from analysis:
The shift is what makes softmax safe. After subtracting the max, the largest logit becomes 0 and its exponential is exactly 1. Every other exponential falls between 0 and 1, so the total `t` sits between 1 and the number of entries. Nothing in the calculation can overflow.

The axis decides the answer. The same `Z` gives two different sets of probabilities. Over `dim=-1` the rows sum to 1 and the column sums (0.84, 1.01, 0.15) mean nothing. Over `dim=0` it is the other way around. The axis is a choice about what competes with what: `dim=-1` makes the three candidates in a row compete, and `dim=0` makes the two rows compete for each column. A wrong axis still gives numbers that look like probabilities -- all positive, all below 1 -- so only the sums show the mistake.

The naive softmax fails on `z + 100` in float32. `103.1` is above `ln(float32 max) ≈ 88.72`, so `exp` returns `inf`, and `inf / inf` is NaN. Our version subtracts 103.1 first, so the largest exponential is `exp(0) = 1`, and the result is the same distribution as `z`. It differs from the float64 softmax of `z` by about 4.7e-7. That comes from storing `z + 100` in float32 (103.1 is stored as 103.0999985, and the spacing between float32 values near 103 is 7.6e-6), not from the softmax.

`z_big = [1000, 999, 0]` gives `[0.7311, 0.2689, 0.0]`. The third entry is exactly 0 because `exp(-1000)` underflows float64. The stable softmax trades that tiny probability for no overflow. If we took its log we would get `-inf`, which is why libraries provide a separate `log_softmax`.

`G` and `Zc` check the broadcast. On the square `G`, a max or sum taken over the wrong axis would still broadcast with no shape error, so matching Table 2 shows that each row is shifted and divided by its own values. `Zc` normalizes over an axis of length 1, so every probability is 1.0, and the output keeps the `(2, 3, 1)` shape.

What this does not establish is behavior on fully masked rows or gradients. A row of all `-inf` gives NaN in our version and in torch.softmax (Step 4), because `-inf - (-inf)` is NaN. Attention code has to handle that case before the softmax. We also did not test backpropagation through the function. In practice we would use torch.softmax or torch.nn.functional.softmax.

## Why It Matters
A stable softmax is a fundamental component of LLMs. It turns the final logits into the next-token distribution, it is the output layer of the multiclass classifiers we train (a binary classifier usually uses a sigmoid, which is the two-class softmax), and it turns attention scores into attention weights. By computing it by hand, we showed we understand both the math and the steps needed to compute it safely.

A naive softmax can overflow. In float32 that happens once a logit passes 88.72. In float16 it happens at only 11.09, which ordinary logits pass easily. Because softmax only depends on the differences between logits, subtracting the max costs nothing and removes the problem.

A poorly implemented softmax also fails as tensor shapes grow. Attention scores come from a matmul (F01-1) with shape `(batch, heads, queries, keys)` and are normalized over the last axis, the keys. Normalize over the wrong axis and the output has the right shape and looks like probabilities, but the queries compete with each other instead of the keys. The max and the sum must keep their axis (`keepdim=True`) so they broadcast back to their own slice. Without it, a square input still runs and gives wrong values.

## Toolkit
We added softmax.py which includes softmax, a stable softmax that prevents overflow and matches torch.softmax within 1.5e-8 in float32. It takes the logits and the axis to normalize over, which may be negative. It subtracts the max along that axis, takes the exponential, and divides by the sum along the same axis. The max and the sum keep their axis so they broadcast back to the slice they came from. The output has the same shape, dtype and device as the input, and the input is not changed. No previous toolkit functions were used.
