import evalkit.matrix_multiplication as matrix_multiplication
import torch
import pytest


@pytest.fixture
def state_0():
    X = torch.tensor([[0.5, 0.8, 1.0, 1.0], [0.9, 0.6, 0.0, 1.0], [0.2, 0.3, 1.0, 0.0]], dtype=torch.float64)
    X2 = torch.tensor([[0.1, 0.2, 1.0, 1.0], [0.7, 0.9, 1.0, 1.0], [0.4, 0.4, 0.0, 0.0]], dtype=torch.float64)
    XB = torch.stack([X, X2])
    w = torch.tensor([1.0, 2.0, 1.5, 0.5], dtype=torch.float64)
    w_alt = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    W2 = torch.stack([w, w_alt], dim=1)
    mm_v = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)

    return X, X2, XB, w, w_alt, W2, mm_v

@pytest.fixture
def printed_results():
    X_w = torch.tensor([4.1, 2.6, 2.3], dtype=torch.float64)
    X_W2 = torch.tensor([[4.1, 1.8], [2.6, 1.6], [2.3, 0.3]], dtype=torch.float64)
    XB_w = torch.tensor([[4.1, 2.6, 2.3], [2.5, 4.5, 1.2]], dtype=torch.float64)
    XB_W2 = torch.tensor([[[4.1, 1.8], [2.6, 1.6], [2.3, 0.3]], [[2.5, 1.2], [4.5, 1.9], [1.2, 0.4]]], dtype=torch.float64)
    S_1_1 = torch.tensor(1.6, dtype=torch.float64)

    return X_w, X_W2, XB_w, XB_W2, S_1_1


def test_matrix_at_vector(state_0, printed_results):
    X, X2, XB, w, w_alt, W2, mm_v = state_0
    X_w, X_W2, XB_w, XB_W2, S_1_1 = printed_results

    new_tensor = matrix_multiplication.matmul(X,w)

    # Tolerance reason: 1e-12 is float64 precise to rounding errors.
    torch.testing.assert_close(new_tensor,X_w,atol=1e-12,rtol=0)

def test_metrix_at_matrix(state_0, printed_results):
    X, X2, XB, w, w_alt, W2, mm_v = state_0
    X_w, X_W2, XB_w, XB_W2, S_1_1 = printed_results
    
    new_tensor = matrix_multiplication.matmul(X,W2)

    # Tolerance reason: 1e-12 is float64 precise to rounding errors.
    torch.testing.assert_close(new_tensor,X_W2,atol=1e-12,rtol=0)
    torch.testing.assert_close(new_tensor[1,1],S_1_1,atol=1e-12,rtol=0)

def test_stack(state_0, printed_results):
    X, X2, XB, w, w_alt, W2, mm_v = state_0
    X_w, X_W2, XB_w, XB_W2, S_1_1 = printed_results
    
    new_tensor = matrix_multiplication.matmul(XB,w)

    # Tolerance reason: 1e-12 is float64 precise to rounding errors.
    torch.testing.assert_close(new_tensor,XB_w,atol=1e-12,rtol=0)

def test_batch_broadcast(state_0, printed_results):
    X, X2, XB, w, w_alt, W2, mm_v = state_0
    X_w, X_W2, XB_w, XB_W2, S_1_1 = printed_results
    
    new_tensor = matrix_multiplication.matmul(XB,W2)
    xb_at_w = matrix_multiplication.matmul(XB,w)

    # Tolerance reason: 1e-12 is float64 precise to rounding errors.
    torch.testing.assert_close(new_tensor,XB_W2,atol=1e-12,rtol=0)
    # Column 0 of W2 is w, so column 0 of the result is check 3's result.
    torch.testing.assert_close(new_tensor[...,0],xb_at_w,atol=1e-12,rtol=0)

def test_incompatible(state_0, printed_results):
    X, X2, XB, w, w_alt, W2, mm_v = state_0
    X_w, X_W2, XB_w, XB_W2, S_1_1 = printed_results
    
    with pytest.raises(ValueError):
        incompatible_1 = matrix_multiplication.matmul(XB,mm_v)

    with pytest.raises(ValueError):
        incompatible_2 = matrix_multiplication.matmul(w,X)

    with pytest.raises(ValueError):
        incompatible_3 = matrix_multiplication.matmul(X,X)

def test_library(state_0, printed_results):
    X, X2, XB, w, w_alt, W2, mm_v = state_0
    X_w, X_W2, XB_w, XB_W2, S_1_1 = printed_results
   
    # Convert to float32
    X = X.float()
    w = w.float()
    W2 = W2.float()
    XB = XB.float()
    # Our functions
    x_at_w = matrix_multiplication.matmul(X,w)
    x_at_w2 = matrix_multiplication.matmul(X,W2)
    w_at_w = matrix_multiplication.matmul(w,w)
    w_at_w2 = matrix_multiplication.matmul(w,W2)
    xb_at_w = matrix_multiplication.matmul(XB,w)
    xb_at_w2 = matrix_multiplication.matmul(XB,W2)
    x_at_w2_none = matrix_multiplication.matmul(X,W2[None])
    xb_at_w2_none = matrix_multiplication.matmul(XB,W2[None])
 
    # Library functions
    torch_x_at_w = torch.matmul(X,w)
    torch_x_at_w2 = torch.matmul(X,W2)
    torch_w_at_w = torch.matmul(w,w)
    torch_w_at_w2 = torch.matmul(w,W2)
    torch_xb_at_w = torch.matmul(XB,w)
    torch_xb_at_w2 = torch.matmul(XB,W2)
    torch_x_at_w2_none = torch.matmul(X,W2[None])
    torch_xb_at_w2_none = torch.matmul(XB,W2[None])
    
    # Tolerance reason: 1e-6 is float32 precise to rounding errors. float32 machine epsilon is 1.19e-7.
    torch.testing.assert_close(x_at_w,torch_x_at_w,atol=1e-6,rtol=0)
    torch.testing.assert_close(x_at_w2,torch_x_at_w2,atol=1e-6,rtol=0)
    torch.testing.assert_close(w_at_w,torch_w_at_w,atol=1e-6,rtol=0)
    torch.testing.assert_close(w_at_w2,torch_w_at_w2,atol=1e-6,rtol=0)
    torch.testing.assert_close(xb_at_w,torch_xb_at_w,atol=1e-6,rtol=0)
    torch.testing.assert_close(xb_at_w2,torch_xb_at_w2,atol=1e-6,rtol=0)
    torch.testing.assert_close(x_at_w2_none,torch_x_at_w2_none,atol=1e-6,rtol=0)
    # XB @ W2[None] stretches a size-1 batch axis on the right.
    torch.testing.assert_close(xb_at_w2_none,torch_xb_at_w2_none,atol=1e-6,rtol=0)

def test_unchanged(state_0, printed_results):
    X, X2, XB, w, w_alt, W2, mm_v = state_0
    X_w, X_W2, XB_w, XB_W2, S_1_1 = printed_results

    # Check 4 computation in float64. Copy the operands first, then compare after the call.
    XB_before = XB.clone()
    W2_before = W2.clone()
    xb_at_w2 = matrix_multiplication.matmul(XB,W2)

    # Tolerance reason: 0. The inputs must be bitwise unchanged.
    torch.testing.assert_close(XB,XB_before,atol=0,rtol=0)
    torch.testing.assert_close(W2,W2_before,atol=0,rtol=0)
    assert xb_at_w2.dtype == torch.float64

    # Check 6 computations in float32.
    X = X.float()
    w = w.float()
    W2 = W2.float()
    XB = XB.float()
    X_before = X.clone()
    w_before = w.clone()
    W2_before = W2.clone()
    XB_before = XB.clone()

    x_at_w = matrix_multiplication.matmul(X,w)
    w_at_w2 = matrix_multiplication.matmul(w,W2)
    xb_at_w2 = matrix_multiplication.matmul(XB,W2)
    x_at_w2_none = matrix_multiplication.matmul(X,W2[None])

    # Tolerance reason: 0. The inputs must be bitwise unchanged.
    torch.testing.assert_close(X,X_before,atol=0,rtol=0)
    torch.testing.assert_close(w,w_before,atol=0,rtol=0)
    torch.testing.assert_close(W2,W2_before,atol=0,rtol=0)
    torch.testing.assert_close(XB,XB_before,atol=0,rtol=0)

    assert x_at_w.dtype == torch.float32
    assert w_at_w2.dtype == torch.float32
    assert xb_at_w2.dtype == torch.float32
    assert x_at_w2_none.dtype == torch.float32
