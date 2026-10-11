# F01-1-L1 -- Matrix Multiplication

## Mechanism
Handcode the matrix multiplication process to show that it is equivalent to a broadcast product then a sum over the contracted axis.

## Task 
Create a function matmul that performs matrix multiplication between two tensors manually. Split each shape, check, then contract, and retain all other axes.

## Interface
Matmul
name: 
  matmul
arguments:
  tensor_a: A torch tensor with 1, 2 or 3 axes and a floating dtype. The left operand of our matrix multiplication.
  tensor_b: A torch tensor with 1, 2 or 3 axes, the same dtype and the same device as tensor_a. The right operand of our matrix multiplication.
returns:
  new_tensor: A new torch tensor of shape (batch..., n, m) from (batch..., n, k) @ (batch..., k, m), with the dtype of the operands. The result of matrix multiplication on tensor_a and tensor_b. A 1-D operand's size-1 axis is removed from the result. Raises ValueError if the contracted lengths differ, or if an operand breaks the axis count, dtype or device requirements.

Broadcast Batch
name: 
  broadcast_batch
arguments:
  batch_a: A tuple of ints. The batch axes of the left operand.
  batch_b: A tuple of ints. The batch axes of the right operand.
returns:
  batch: A tuple of ints. The broadcast batch shape: aligned from the right, a missing or size-1 axis stretched to the other operand's length. Raises ValueError if the batch axes cannot be broadcast.

Batch Index
name: 
  batch_index
arguments:
  b: A tuple of ints. The batch index into the output.
  operand_batch: A tuple of ints. The batch axes of the operand.
returns:
  index: A tuple of ints. The batch index into the operand, read through the broadcast. Missing axes are dropped, and size-1 axes read index 0.

## Pass Check
All Tests Must Pass.

Here is our test criteria:

1. **Matrix @ vector.** `X @ w` gives shape `(3,)` and `[4.1, 2.6, 2.3]`, within `1e-12`. Reason for the tolerance: each entry is four products of short decimals summed in float64; the measured error is below `3e-16`.
2. **Matrix @ matrix.** `X @ W2` gives shape `(3, 2)` and Table 2's values within `1e-12`. Entry `[1, 1]` is `1.6` within `1e-12`. Same reason.
3. **A stack (example 2).** `XB @ w` gives shape `(2, 3)` and Table 2's values within `1e-12`. Same reason.
4. **A batch broadcast (example 3).** `XB @ W2` gives shape `(2, 3, 2)` and Table 2's values within `1e-12`. Column 0 of the result equals check 3's result within `1e-12`. Same reason.
5. **Incompatible shapes.** `XB @ mm_v`, `w @ X` and `X @ X` each raise `ValueError`. Exact. Reason for the cases: `XB` has an axis of length 3, but it is not its last axis; `w` used as a row has 4 entries against `X`'s 3 rows; `X @ X` pairs 4 columns against 3 rows.
6. **Against the library, float32.** For each of these pairs, built in float32 as the notebook built them, your result has the same shape as `torch.matmul` and differs by at most `1e-6` in every entry:
   - `X @ w` → `(3,)`; `X @ W2` → `(3, 2)`; `XB @ w` → `(2, 3)`; `XB @ W2` → `(2, 3, 2)`;
   - `w @ W2` → `(2,)`, a 1-D left operand, values `[7.5, 2.5]`;
   - `XB @ W2[None]` → `(2, 3, 2)`, a size-1 batch axis on the right stretched to 2;
   - `X @ W2[None]` → `(1, 3, 2)`, a missing batch axis on the left.

   Reason for the tolerance: float32 machine epsilon is `1.19e-7`, entries are at most about 7.5, and the library may sum the four products in another order; the notebook measured `2.38e-7` between its loops and `XB @ W2`.
7. **Arguments unchanged, dtype kept.** After the calls of check 6, `XB` and `W2` equal copies taken before the call, exactly, and the result's dtype is `torch.float32`. With the float64 inputs of check 4, the result's dtype is `torch.float64`.


## Results
All tests passed.

We successfully demonstrated that our hand-written matmul matches torch.matmul. In float64 the max difference was 0.0 on both examples. In our tests it agreed with torch.matmul within 1e-6 in float32 on every combination of 1, 2 and 3 axes that we checked.

Numbers from `results/F01-1-L1.json`:

Step 0 - shape check:

| Example | Call | Shapes | Contracted Length (left, right) | Batch Shape | Result Shape |
|---|---|---|---|---|---|
| 1 | `XB @ mm_v` | (2, 3, 4) @ (3,) | 4, 3 | - | ValueError |
| 2 | `XB @ w` | (2, 3, 4) @ (4,) | 4, 4 | (2,) | (2, 3) |
| 3 | `XB @ W2` | (2, 3, 4) @ (4, 2) | 4, 4 | (2,) | (2, 3, 2) |

Example 2 - `XB @ w`, one score per candidate for each query:

| Query | c1 | c2 | c3 |
|---|---|---|---|
| q1 | 4.1000 | 2.6000 | 2.3000 |
| q2 | 2.5000 | 4.5000 | 1.2000 |

Example 3 - `XB @ W2`, two scorers (w, w_alt) for each candidate of each query:

| Query | Candidate | w | w_alt |
|---|---|---|---|
| q1 | c1 | 4.1000 | 1.8000 |
| q1 | c2 | 2.6000 | 1.6000 |
| q1 | c3 | 2.3000 | 0.3000 |
| q2 | c1 | 2.5000 | 1.2000 |
| q2 | c2 | 4.5000 | 1.9000 |
| q2 | c3 | 1.2000 | 0.4000 |

Max difference from torch.matmul: 0.000e+00 for Example 2 and Example 3.

Results from analysis:
Example 1 fails before any arithmetic happens. The left operand contracts over 4 features and the right one only has 3. The shape check catches it and raises a ValueError that names both shapes and both k values.

The w column of Example 3 is exactly Example 2. Column 0 of W2 is w, so adding a second scorer just adds a second column to the output. One call scores every candidate for every query with both scorers, and the batch axis carries the queries with itertools.product on our side.

What this does not establish is speed. Our version performs these operations non-vectorized in Python, so it is only for understanding. In practice we would use torch.matmul or the @ operator.

## Why It Matters
Matrix multiplication is the operation under almost everything we build: linear layers, scoring candidates against a weight vector, similarity between embeddings, and attention scores. Knowing the shape rule by hand means we can read a shape error and know exactly which axis is wrong.

The rule is simple -- The last axis of the left operand and the second-to-last axis of the right operand are contracted (k) and disappear. Every other axis survives. Rows come from the left, columns come from the right, and everything to the left of those two axes is a batch axis. Batch axes are aligned from the right, and a missing or size-1 axis is stretched to match the other operand. That is how one set of weights (W2) scores a full stack of queries (XB) without copying the weights.

The 1-D case is where it's easy to get wrong. A 1-D left operand is treated as a row, a 1-D right operand as a column, and that added axis is removed from the result. If that axis gets removed in the wrong order, or not at all, the output has the wrong shape.

## Toolkit
We added matrix_multiplication.py which includes matmul, broadcast_batch and batch_index. matmul performs matrix multiplication by hand on tensors with 1, 2 or 3 axes, checks the axis count, dtype, device and contracted lengths first, and raises a ValueError with both shapes when they don't line up. broadcast_batch combines two batch shapes, aligned from the right, and raises a ValueError when they can't be broadcast. batch_index reads an output batch index back through the broadcast for one operand, so a size-1 or missing batch axis always reads from the same slice.


