from basis.sine1d import SineBasis1D
from basis.fourier1d import FourierBasis1D
import torch
from utilities.tensors import dtype_context

def rk4(u, dt, f):
    k1 = dt * f(u)
    k2 = dt * f(u + 0.5 * k1)
    k3 = dt * f(u + 0.5 * k2)
    k4 = dt * f(u + k3)

    u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    return u_next

def heat1d_groundtruth_generator(n_x,n_t,domain_x,domain_t,u0,kappa,device, dtype=None):

    with dtype_context(torch.float64):
        u0 = u0.to(device, dtype=torch.float64)
        u0 = u0.squeeze(0)
        
        kappa = kappa.to(device)

        ts_factor = 50
        n_t = ts_factor * (n_t-1) + 1

        ts = torch.linspace(*domain_t,n_t,device=device)

        basis = SineBasis1D(n_x,domain_x,device)
        D2 = basis.D2


        def f(u):
            u_hat = basis.dbt(u)
            D2u_hat = D2 @ u_hat
            D2u = basis.idbt(D2u_hat)

            return kappa * D2u


        T = [ts[0]]
        U = [u0]
        u = u0
        dt = ts[1]-ts[0]

        for t in ts[1:]:
            u = rk4(u,dt,f)
            T.append(t)
            U.append(u)

        U_tensor = torch.stack(U)

    output_dtype = dtype if dtype is not None else torch.get_default_dtype()    
    u_gt = U_tensor.detach().to(dtype=output_dtype, device="cpu").numpy()

    u_gt = u_gt[::ts_factor,:]

    return u_gt


def heat1d_groundtruth_generator_fourier(n_x,n_t,domain_x,domain_t,u0,kappa,device):

    u0 = u0.to(device)
    u0 = u0.squeeze(0)
    
    kappa = kappa.to(device)

    ts_factor = 50
    n_t = ts_factor * (n_t-1) + 1

    ts = torch.linspace(*domain_t,n_t,device=device)

    basis = FourierBasis1D(n_x,domain_x,device)
    D2 = basis.D2


    def f(u):
        u_hat = basis.dbt(u)
        D2u_hat = D2 @ u_hat
        D2u = basis.idbt(D2u_hat)

        return kappa * D2u


    T = [ts[0]]
    U = [u0]
    u = u0
    dt = ts[1]-ts[0]

    for t in ts[1:]:
        u = rk4(u,dt,f)
        T.append(t)
        U.append(u)

    U_tensor = torch.stack(U)

    u_gt = U_tensor.detach().cpu().numpy()

    u_gt = u_gt[::ts_factor,:]

    return u_gt