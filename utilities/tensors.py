import torch
import contextlib

def list_to_tensor(l):
    '''Receives a 2d list of lists containing tensors (of any shape, as long as it's consistent), and stacks them along the last axis
    If l is (m,n), containing (b1,...bk)-shaped tensors, the output is (b1,...,bk,m,n)'''
    stacked_rows = [torch.stack(row, dim=-1) for row in l]
    stacked_matrix = torch.stack(stacked_rows, dim=-2)
    return stacked_matrix


@contextlib.contextmanager
def dtype_context(dtype):
    """Context manager to temporarily set the default torch dtype."""
    # Save the original dtype
    original_dtype = torch.get_default_dtype()
    
    # Set the new default
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        # Guarantee we revert back to the original dtype, even if exceptions occur
        torch.set_default_dtype(original_dtype)    


def format_Fourier_vector(y, basis):
    z = torch.zeros(size=y.shape[:-1]+(basis.N_cos, 2), dtype=y.dtype, device=y.device)
    z[..., 0] = y[..., :basis.N_cos]
    z[..., 1:1+basis.N_sin, 1] = y[..., basis.N_cos: basis.N_cos+basis.N_sin]
    return z


def unformat_Fourier_vector(z, basis):
    y = torch.zeros(size=z.shape[:-2] + (basis.M,), dtype=z.dtype, device=z.device)
    y[..., :basis.N_cos] = z[..., 0]
    y[..., basis.N_cos: basis.N_cos+basis.N_sin] = z[..., 1:1+basis.N_sin, 1]
    return y