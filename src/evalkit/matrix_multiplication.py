import itertools
import torch


def matmul(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """
    Matmul: A matrix multiplier done by hand with shape checks.

    We take in 2 tensors with 1, 2 or 3 axes, split each shape, check lengths and then contract, preserving the other axes.

    Shape rule: (batch..., n, k) @ (batch..., k, m) -> (batch..., n, m). The last two axes are the matrix axes (rows, columns) and every axis to their left is a batch axis.
    A 1-D left operand (k,) is used as one row (1, k), and a 1-D right operand (k,) is used as one column (k, 1). That size-1 axis is removed from the result.
    The contracted length k must match and does not appear in the result. Batch axes are broadcast: aligned from the right, a missing or size-1 axis stretched to the other operand's length.

    Entry formula: new_tensor[b, i, j] = sum over k of tensor_a[b, i, k] * tensor_b[b, k, j], with b read through the broadcast for each operand.

    Parameters:
    ---- 
    tensor_a: A torch tensor with 1, 2 or 3 axes and a floating dtype. The left operand of our matrix multiplication.
    tensor_b: A torch tensor with 1, 2 or 3 axes, the same dtype and the same device as tensor_a. The right operand of our matrix multiplication.
    
    Returns:
    ---- 
    new_tensor: A new torch tensor with the shape the shape rule gives and the dtype of the operands. The result of matrix multiplication on tensor_a and tensor_b 

    Raises:
    ---- 
    ValueError: If the contracted lengths differ, or if an operand breaks the axis count, dtype or device requirements.
    """
    shape_a, shape_b = tuple(tensor_a.shape), tuple(tensor_b.shape)
    if tensor_a.dim() not in (1, 2, 3) or tensor_b.dim() not in (1, 2, 3):
        raise ValueError(f"Operands must have 1, 2 or 3 axes. tensor_a shape: {shape_a}, tensor_b shape: {shape_b}")
    if not tensor_a.is_floating_point() or not tensor_b.is_floating_point():
        raise ValueError(f"Operands must have a floating dtype. tensor_a dtype: {tensor_a.dtype}, tensor_b dtype: {tensor_b.dtype}")
    if tensor_a.dtype != tensor_b.dtype:
        raise ValueError(f"Operands must have the same dtype. tensor_a dtype: {tensor_a.dtype}, tensor_b dtype: {tensor_b.dtype}")
    if tensor_a.device != tensor_b.device:
        raise ValueError(f"Operands must be on the same device. tensor_a device: {tensor_a.device}, tensor_b device: {tensor_b.device}")

    # Split each shape. Promote 1-D operands to a row or a column, and remember to drop that axis.
    a_is_vector = tensor_a.dim() == 1
    b_is_vector = tensor_b.dim() == 1
    if a_is_vector:
        tensor_a = tensor_a.unsqueeze(0)
    if b_is_vector:
        tensor_b = tensor_b.unsqueeze(1)

    batch_a, (n, k_a) = tuple(tensor_a.shape[:-2]), tuple(tensor_a.shape[-2:])
    batch_b, (k_b, m) = tuple(tensor_b.shape[:-2]), tuple(tensor_b.shape[-2:])

    # Check, then contract.
    if k_a != k_b:
        raise ValueError(
            f"Contracted lengths do not match. tensor_a shape: {shape_a} has k = {k_a}, "
            f"tensor_b shape: {shape_b} has k = {k_b}"
        )

    batch = broadcast_batch(batch_a, batch_b)

    new_tensor = torch.zeros(batch + (n, m), dtype=tensor_a.dtype, device=tensor_a.device)

    # Outer loops over the surviving axes, one output entry per iteration. Inner sum over the contracted axis.
    for b in itertools.product(*(range(size) for size in batch)):
        b_a = batch_index(b, batch_a)
        b_b = batch_index(b, batch_b)
        for i in range(n):
            for j in range(m):
                total = torch.zeros((), dtype=tensor_a.dtype, device=tensor_a.device)
                for k in range(k_a):
                    total = total + tensor_a[b_a + (i, k)] * tensor_b[b_b + (k, j)]
                new_tensor[b + (i, j)] = total

    # Remove the size-1 axes added for 1-D operands. Drop the row axis first, so axis -2 still exists when both are 1-D.
    if a_is_vector:
        new_tensor = new_tensor.squeeze(-2)
    if b_is_vector:
        new_tensor = new_tensor.squeeze(-1)

    return new_tensor


def broadcast_batch(batch_a: tuple[int, ...], batch_b: tuple[int, ...]) -> tuple[int, ...]:
    """
    Broadcast Batch: Combine two batch shapes, aligned from the right, stretching a missing or size-1 axis.

    Parameters:
    ----
    batch_a: The batch axes of the left operand.
    batch_b: The batch axes of the right operand.

    Returns:
    ----
    batch: The broadcast batch shape.
    """
    length = max(len(batch_a), len(batch_b))
    padded_a = (1,) * (length - len(batch_a)) + batch_a
    padded_b = (1,) * (length - len(batch_b)) + batch_b

    batch = []
    for size_a, size_b in zip(padded_a, padded_b):
        if size_a != size_b and size_a != 1 and size_b != 1:
            raise ValueError(f"Batch axes cannot be broadcast. tensor_a batch: {batch_a}, tensor_b batch: {batch_b}")
        batch.append(max(size_a, size_b))
    return tuple(batch)


def batch_index(b: tuple[int, ...], operand_batch: tuple[int, ...]) -> tuple[int, ...]:
    """
    Batch Index: Read an output batch index through the broadcast for one operand.

    Parameters:
    ----
    b: The batch index into the output.
    operand_batch: The batch axes of the operand.

    Returns:
    ----
    index: The batch index into the operand. Missing axes are dropped, and size-1 axes read index 0.
    """
    aligned = b[len(b) - len(operand_batch):]
    return tuple(0 if size == 1 else position for position, size in zip(aligned, operand_batch))
