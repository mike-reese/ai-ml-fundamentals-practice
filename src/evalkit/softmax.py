import torch

def softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Softmax turns a tensor of logits into probabilities that are positive and sum to 1 across a chosen axis.

    Parameters:
    ----- 
    logits: a floating tensor (float32 or float64) of one or more axes.
    dim: integer. May be negative. Used to select the axis to normalize over in logits.
    
    Returns:
    ----- 
    probabilities: a tensor of the same shape, dtype, and device as logits which contains probabilities in which every slice along dim sums to 1. Formula: `p[i] = exp(z[i] − m) / Σ_k exp(z[k] − m)`, where `z` is logits, `m` is the max of logits along dim for each slice, and `Σ_k` runs over the entries of that slice along dim. The shift `m` cancels in the quotient, so it does not change `p`. Without it, `exp` overflows `float32` once a logit is above `ln(float32 max) ≈ 88.72`.
    """
    m = torch.amax(logits, dim=dim, keepdim=True)
    exp = torch.exp(logits - m)
    axis_sum = exp.sum(dim=dim, keepdim=True)

    probabilities = torch.divide(exp,axis_sum)    

    return probabilities 
