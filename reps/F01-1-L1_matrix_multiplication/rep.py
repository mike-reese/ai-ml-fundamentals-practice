import evalkit.matrix_multiplication as matrix_multiplication
import torch
from pathlib import Path
import json

def state_0():
    X = torch.tensor([[0.5, 0.8, 1.0, 1.0], [0.9, 0.6, 0.0, 1.0], [0.2, 0.3, 1.0, 0.0]], dtype=torch.float64)
    X2 = torch.tensor([[0.1, 0.2, 1.0, 1.0], [0.7, 0.9, 1.0, 1.0], [0.4, 0.4, 0.0, 0.0]], dtype=torch.float64)
    XB = torch.stack([X, X2])
    w = torch.tensor([1.0, 2.0, 1.5, 0.5], dtype=torch.float64)
    w_alt = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    W2 = torch.stack([w, w_alt], dim=1)
    mm_v = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)

    return X, X2, XB, w, w_alt, W2, mm_v

def shape_check(tensor_a, tensor_b):
    """
    Shape Check: Step 0 of matmul. Split each shape, check the contracted lengths, then broadcast the batch axes.
    """
    shape_a, shape_b = tuple(tensor_a.shape), tuple(tensor_b.shape)
    # 1-D left operand is one row (1, k); 1-D right operand is one column (k, 1).
    rows_a = shape_a if len(shape_a) > 1 else (1,) + shape_a
    rows_b = shape_b if len(shape_b) > 1 else shape_b + (1,)
    k_a, k_b = rows_a[-1], rows_b[-2]
    check = {
        "Left Shape": list(shape_a),
        "Right Shape": list(shape_b),
        "Left Contracted Length": k_a,
        "Right Contracted Length": k_b,
    }
    if k_a != k_b:
        check["Mismatch"] = f"contracted lengths differ: {k_a} != {k_b}"
        return check

    batch = matrix_multiplication.broadcast_batch(rows_a[:-2], rows_b[:-2])
    result_shape = batch + (rows_a[-2],) * (len(shape_a) > 1) + (rows_b[-1],) * (len(shape_b) > 1)
    check["Batch Shape"] = list(batch)
    check["Result Shape"] = list(result_shape)
    return check

X, X2, XB, w, w_alt, W2, mm_v = state_0()

examples = [
    ("Example 1", "XB @ mm_v", XB, mm_v),
    ("Example 2", "XB @ w", XB, w),
    ("Example 3", "XB @ W2", XB, W2),
]

print("1. Step 0 - shape check")
shape_checks = {}
for name, call, tensor_a, tensor_b in examples:
    check = shape_check(tensor_a, tensor_b)
    shape_checks[name] = {"Call": call, **check}
    print(f"{name}: {call}")
    print(f"  Shapes: {tuple(check['Left Shape'])} @ {tuple(check['Right Shape'])}")
    print(f"  Contracted length: left {check['Left Contracted Length']}, right {check['Right Contracted Length']}")
    if "Mismatch" in check:
        print(f"  Mismatch: {check['Mismatch']}")
        try:
            matrix_multiplication.matmul(tensor_a, tensor_b)
        except ValueError as error:
            print(f"  matmul raises ValueError: {error}")
            shape_checks[name]["matmul Error"] = str(error)
    else:
        print(f"  Batch shape after broadcasting: {tuple(check['Batch Shape'])}")
        print(f"  Result shape: {tuple(check['Result Shape'])}")

print("")
print("2. Results against torch.matmul")
results = {}
for name, call, tensor_a, tensor_b in examples[1:]:
    ours = matrix_multiplication.matmul(tensor_a, tensor_b)
    library = torch.matmul(tensor_a, tensor_b)
    max_difference = (ours - library).abs().max().item()
    results[name] = {
        "Call": call,
        "Shape": list(ours.shape),
        "Result": ours.tolist(),
        "Max Difference From torch.matmul": max_difference,
    }

    print(f"{name}: {call}, shape {tuple(ours.shape)}")
    if ours.dim() == 2:
        # Rows are queries, columns are candidates.
        print(f"{'Query':<8} | " + " | ".join(f"{'c' + str(c + 1):>8}" for c in range(ours.shape[1])))
        for q in range(ours.shape[0]):
            print(f"{'q' + str(q + 1):<8} | " + " | ".join(f"{ours[q, c].item():>8.4f}" for c in range(ours.shape[1])))
    else:
        # One table per query: rows are candidates, columns are scorers (w, w_alt).
        for q in range(ours.shape[0]):
            print(f"Query q{q + 1}")
            print(f"{'Candidate':<10} | {'w':>8} | {'w_alt':>8}")
            for c in range(ours.shape[1]):
                print(f"{'c' + str(c + 1):<10} | {ours[q, c, 0].item():>8.4f} | {ours[q, c, 1].item():>8.4f}")
    print(f"Max difference from torch.matmul: {max_difference:.3e}")
    print("")

# Results file
json_schema = {
    "Shape Check": shape_checks,
    "Results": results,
}

path = Path(__file__).parent / "results" / "F01-1-L1.json"
with open(path, "w") as f:
    json.dump(json_schema, f, indent=2)
