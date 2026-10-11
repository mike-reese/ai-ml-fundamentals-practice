import evalkit.softmax as softmax
import torch
import json
from pathlib import Path

def logits():
    z_orient = torch.tensor([-1.0, 0.5], dtype=torch.float64)
    z = torch.tensor([3.1, 1.6, 1.3], dtype=torch.float64)
    Z = torch.tensor([[3.1, 1.6, 1.3], [1.5, 3.5, 0.2]], dtype=torch.float64)
    z_big = torch.tensor([1000.0, 999.0, 0.0], dtype=torch.float64)
    G = torch.tensor([[2.89, 1.93, 1.34], [1.93, 2.17, 0.36], [1.34, 0.36, 1.13]], dtype=torch.float64)
    Zc = Z[..., None]

    return z_orient, z, Z, z_big, G, Zc

def known_results():
    sm_z_orient = torch.tensor([0.1824255238, 0.8175744762], dtype=torch.float64)
    sm_z = torch.tensor([0.7202384603, 0.1607069230, 0.1190546167], dtype=torch.float64)
    m = torch.tensor(3.1, dtype=torch.float64)
    s = torch.tensor([0.0, -1.5, -1.8], dtype=torch.float64)
    e = torch.tensor([1.0, 0.2231301601, 0.1652988882], dtype=torch.float64)
    t = torch.tensor(1.3884290484, dtype=torch.float64)
    sm_Z_last = torch.tensor([[0.7202384603, 0.1607069230, 0.1190546167], [0.1154522719, 0.8530833135, 0.0314644147]], dtype=torch.float64)
    sm_Z_dim0 = torch.tensor([[0.8320183851, 0.1301084744, 0.7502601056], [0.1679816149, 0.8698915256, 0.2497398944]], dtype=torch.float64)
    sm_z_big = torch.tensor([0.7310585786, 0.2689414214, 0.0], dtype=torch.float64)
    sm_G = torch.tensor([[0.6269038837, 0.2400370372, 0.1330590791], [0.4033405743, 0.5127463624, 0.0839130633], [0.4574784435, 0.1716967373, 0.3708248192]], dtype=torch.float64)

    return sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G

def print_table(title, probabilities, row_label, col_label):
    """Print a 2-D table of probabilities with a Row Sum column and a Column Sum row."""
    n_rows, n_cols = probabilities.shape
    print(title)
    print(f"{row_label:<10} | " + " | ".join(f"{col_label + str(c + 1):>8}" for c in range(n_cols)) + f" | {'Row Sum':>8}")
    for r in range(n_rows):
        row = probabilities[r]
        print(f"{'r' + str(r + 1):<10} | " + " | ".join(f"{v.item():>8.4f}" for v in row) + f" | {row.sum().item():>8.4f}")
    col_sums = probabilities.sum(dim=0)
    print(f"{'Col Sum':<10} | " + " | ".join(f"{v.item():>8.4f}" for v in col_sums))
    print("")

z_orient, z, Z, z_big, G, Zc = logits()

print("1. The five sub-states for z")
print(f"z = {z.tolist()}")
m = torch.amax(z, dim=-1, keepdim=True)
print(f"  1. Max of the row: m = {m.item():.10f}")
s = z - m
print(f"  2. Shift by the max to prevent exp overflow: s = z - m = {[round(v, 10) for v in s.tolist()]}")
e = torch.exp(s)
print(f"  3. Exponentials: e = exp(s) = {[round(v, 10) for v in e.tolist()]}")
t = e.sum(dim=-1, keepdim=True)
print(f"  4. Sum along the axis: t = sum(e) = {t.item():.10f}")
p = e / t
print(f"  5. Divide by the total: p = e / t = {[round(v, 10) for v in p.tolist()]}")
print(f"p[0] = e[0] / t = {e[0].item():.10f} / {t.item():.10f} = {p[0].item():.10f}")

sub_states = {
    "z": z.tolist(),
    "m": m.item(),
    "s": s.tolist(),
    "e": e.tolist(),
    "t": t.item(),
    "p": p.tolist(),
    "p[0] Quotient": {"e[0]": e[0].item(), "t": t.item(), "p[0]": p[0].item()},
}

print("")
print("2. Softmax of Z over dim=-1 and dim=0")
Z_dim_last = softmax.softmax(Z, -1)
Z_dim_0 = softmax.softmax(Z, 0)
print_table("dim=-1: each row sums to 1", Z_dim_last, "Row", "c")
print_table("dim=0: each column sums to 1", Z_dim_0, "Row", "c")

axis_results = {
    "dim=-1": {
        "Probabilities": Z_dim_last.tolist(),
        "Row Sums": Z_dim_last.sum(dim=-1).tolist(),
        "Column Sums": Z_dim_last.sum(dim=0).tolist(),
    },
    "dim=0": {
        "Probabilities": Z_dim_0.tolist(),
        "Row Sums": Z_dim_0.sum(dim=-1).tolist(),
        "Column Sums": Z_dim_0.sum(dim=0).tolist(),
    },
}

print("3. float32: naive softmax against ours, and ours against torch.softmax")
z_100_float32 = (z + 100).float()
naive = torch.exp(z_100_float32) / torch.exp(z_100_float32).sum(dim=-1, keepdim=True)
ours = softmax.softmax(z_100_float32, -1)
print(f"z + 100 = {z_100_float32.tolist()}")
print(f"  Naive exp(z) / sum(exp(z)): {naive.tolist()}")
print(f"  Our softmax:                {ours.tolist()}")
print("")

library_inputs = [
    ("z_orient", z_orient, -1),
    ("z", z, -1),
    ("Z", Z, -1),
    ("Z", Z, 0),
    ("z + 100", z + 100, -1),
    ("z_big", z_big, -1),
    ("G", G, -1),
    ("Zc", Zc, -1),
]

library_results = {}
print(f"{'Input':<10} | {'dim':>4} | {'Shape':<10} | {'Max Diff From torch.softmax':>28}")
for name, logit, dim in library_inputs:
    logit_float32 = logit.float()
    ours_input = softmax.softmax(logit_float32, dim)
    library = torch.softmax(logit_float32, dim=dim)
    max_difference = (ours_input - library).abs().max().item()
    library_results[f"{name}, dim={dim}"] = {
        "Shape": list(ours_input.shape),
        "Max Difference From torch.softmax": max_difference,
    }
    print(f"{name:<10} | {dim:>4} | {str(tuple(ours_input.shape)):<10} | {max_difference:>28.3e}")

print("")
print("4. float32: a fully masked row")
z_masked = torch.tensor([float("-inf"), float("-inf")], dtype=torch.float32)
ours_masked = softmax.softmax(z_masked, -1)
library_masked = torch.softmax(z_masked, dim=-1)
print(f"z_masked = {z_masked.tolist()}")
print(f"  Our softmax:   {ours_masked.tolist()}")
print(f"  torch.softmax: {library_masked.tolist()}")

# Results file
json_schema = {
    "Sub-States": sub_states,
    "Softmax Of Z By Axis": axis_results,
    "float32": {
        "z + 100": z_100_float32.tolist(),
        # json writes NaN as the bare token NaN, which Python's json reads back.
        "Naive Softmax": naive.tolist(),
        "Our Softmax": ours.tolist(),
        "Against torch.softmax": library_results,
        "Fully Masked Row": {
            "z_masked": z_masked.tolist(),
            "Our Softmax": ours_masked.tolist(),
            "torch.softmax": library_masked.tolist(),
        },
    },
}

path = Path(__file__).parent / "results" / "F01-2-L1.json"
with open(path, "w") as f:
    json.dump(json_schema, f, indent=2)
