import pytest
import evalkit.softmax as softmax
import torch

@pytest.fixture
def logits():
    z_orient = torch.tensor([-1.0, 0.5], dtype=torch.float64)
    z = torch.tensor([3.1, 1.6, 1.3], dtype=torch.float64)
    Z = torch.tensor([[3.1, 1.6, 1.3], [1.5, 3.5, 0.2]], dtype=torch.float64)
    z_big = torch.tensor([1000.0, 999.0, 0.0], dtype=torch.float64)
    G = torch.tensor([[2.89, 1.93, 1.34], [1.93, 2.17, 0.36], [1.34, 0.36, 1.13]], dtype=torch.float64)
    Zc = Z[..., None]

    return z_orient, z, Z, z_big, G, Zc

@pytest.fixture
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


def test_known_values(logits, known_results):
    z_orient, z, Z, z_big, G, Zc = logits
    sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G = known_results

    check_zo = softmax.softmax(z_orient, 0)
    check_z = softmax.softmax(z,0)

    # Reason for tolerance: KNown results is rounded to ten decimals.
    torch.testing.assert_close(check_zo, sm_z_orient, atol=1e-10, rtol=0)
    torch.testing.assert_close(check_z, sm_z, atol=1e-10, rtol=0)

def test_sum_to_1(logits, known_results):
    z_orient, z, Z, z_big, G, Zc = logits
    sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G = known_results

    check_z_dim_neg_one = softmax.softmax(z,-1)
    check_Z_dim_neg_one = softmax.softmax(Z,-1)
    check_Z_dim_0 = softmax.softmax(Z,0)

    sum_z = check_z_dim_neg_one.sum(dim=-1)
    sum_Z_last = check_Z_dim_neg_one.sum(dim=-1)
    sum_Z_dim0 = check_Z_dim_0.sum(dim=0)

    # Reason for tolerance: Three float64 values summed, rounding error about 2e-16
    torch.testing.assert_close(sum_z, torch.ones_like(sum_z), atol=1e-12, rtol=0)
    torch.testing.assert_close(sum_Z_last, torch.ones_like(sum_Z_last), atol=1e-12, rtol=0)
    torch.testing.assert_close(sum_Z_dim0, torch.ones_like(sum_Z_dim0), atol=1e-12, rtol=0)

    assert (check_z_dim_neg_one > 0).all()
    assert (check_Z_dim_neg_one > 0).all()
    assert (check_Z_dim_0 > 0).all()

def test_axis_is_argument(logits, known_results):
    z_orient, z, Z, z_big, G, Zc = logits
    sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G = known_results

    check_Z_dim_neg_one = softmax.softmax(Z,-1)
    check_Z_dim_0 = softmax.softmax(Z,0)

    check_Z_dim_1 = softmax.softmax(Z, 1)
    torch.testing.assert_close(check_Z_dim_1, check_Z_dim_neg_one, atol=0, rtol=0)

    # Reason for tolerance: Known results are rounded to 10 decimals. 
    torch.testing.assert_close(check_Z_dim_neg_one, sm_Z_last, atol=1e-10, rtol=0)
    torch.testing.assert_close(check_Z_dim_0, sm_Z_dim0, atol=1e-10, rtol=0)

def test_square_input(logits, known_results):
    z_orient, z, Z, z_big, G, Zc = logits
    sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G = known_results

    check_G = softmax.softmax(G,-1)

    # Reason for 1e-10 tolerance: Known results are rounded to 10 decimals.
    torch.testing.assert_close(check_G, sm_G, atol=1e-10, rtol=0)

    sum_G = check_G.sum(dim=-1)
    # Reason for tolerance: 1e-12 is precise because 3 float64 values are summed, so the rounding error is 2e-16
    torch.testing.assert_close(sum_G, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64), atol=1e-12, rtol=0)

def test_shift_invariance(logits, known_results):
    z_orient, z, Z, z_big, G, Zc = logits
    sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G = known_results

    check_z_shift = softmax.softmax(z+100,-1)
    check_z = softmax.softmax(z,-1)

    # Reason for tolerance: in float64, rounding of z+100 moves each logit by at most 7.1e-15. Measured difference is 3e-16
    torch.testing.assert_close(check_z_shift, check_z, atol=1e-12, rtol=0)

def test_large_logits(logits, known_results):
    z_orient, z, Z, z_big, G, Zc = logits
    sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G = known_results

    check_z_big = softmax.softmax(z_big,-1)
    
    # Reason for tolerance: Known values are rounded to 10 decimals
    torch.testing.assert_close(check_z_big, sm_z_big, atol=1e-10, rtol=0)
    
    # Reason for exact 0.0: exp(-1000) underflows in float64 to 0. Stable softmax cannot represent that probability
    assert check_z_big[2].item() == 0.0 

def test_against_library(logits, known_results):
    z_orient, z, Z, z_big, G, Zc = logits
    sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G = known_results

    z_float32 = z.float()
    Z_float32 = Z.float()
    z_100_float32 = (z+100).float()
    z_big_float32 = z_big.float()
    G_float32 = G.float()
    Zc_float32 = Zc.float()

    # Our functions 
    check_z = softmax.softmax(z_float32,-1)
    check_Z = softmax.softmax(Z_float32,-1)
    check_Z_dim_0 = softmax.softmax(Z_float32,0)
    check_z_100 = softmax.softmax(z_100_float32,-1)
    check_z_big = softmax.softmax(z_big_float32,-1)
    check_G = softmax.softmax(G_float32,-1)
    check_Zc = softmax.softmax(Zc_float32,-1)

    # torch.softmax
    z_torch = torch.softmax(z_float32, dim=-1)
    Z_torch = torch.softmax(Z_float32, dim=-1)
    Z_dim_0_torch = torch.softmax(Z_float32, dim=0)
    z_100_torch = torch.softmax(z_100_float32, dim=-1)
    z_big_torch = torch.softmax(z_big_float32, dim=-1)
    G_torch = torch.softmax(G_float32,dim=-1)
    Zc_torch = torch.softmax(Zc_float32,dim=-1)
    
    # Reason for tolerance: float32 epsilon is 1.19e-7
    torch.testing.assert_close(check_z, z_torch, atol=1e-6, rtol=0)
    torch.testing.assert_close(check_Z, Z_torch, atol=1e-6, rtol=0)
    torch.testing.assert_close(check_Z_dim_0, Z_dim_0_torch, atol=1e-6, rtol=0)
    torch.testing.assert_close(check_z_100, z_100_torch, equal_nan=False, atol=1e-6, rtol=0)
    torch.testing.assert_close(check_z_big, z_big_torch, atol=1e-6, rtol=0)
    torch.testing.assert_close(check_G, G_torch, atol=1e-6, rtol=0)
    torch.testing.assert_close(check_Zc, Zc_torch, atol=1e-6, rtol=0)

def test_arg_unchanged(logits, known_results):
    z_orient, z, Z, z_big, G, Zc = logits
    sm_z_orient, sm_z, m, s, e, t, sm_Z_last, sm_Z_dim0, sm_z_big, sm_G = known_results

    Z_copy = Z.clone()
    check_Z = softmax.softmax(Z,-1)
    torch.testing.assert_close(Z, Z_copy, atol=0, rtol=0)
    assert Z.dtype == check_Z.dtype


    Z_float32 = Z.float()
    check_Z_float32 = softmax.softmax(Z_float32,-1)
    assert Z_float32.dtype == check_Z_float32.dtype
