from basis.fourier1d import FourierBasis1D
import torch
from math import pi
from utilities.tensors import dtype_context


def rk4(u, dt, f):
    k1 = dt * f(u)
    k2 = dt * f(u + 0.5 * k1)
    k3 = dt * f(u + 0.5 * k2)
    k4 = dt * f(u + k3)

    u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    return u_next

def burgers1d_groundtruth_generator(nu,n_x,n_t,domain_x,domain_t, device="cuda:0", dtype=None):
    # output tensor has shape [n_t,n_x]
    with dtype_context(torch.float64):
        ts_factor = int(500 * nu)
        if ts_factor < 50:
            ts_factor = 50
        n_t = ts_factor * (n_t-1) + 1
        ts = torch.linspace(*domain_t,n_t,device=device)

        basis = FourierBasis1D(n_x,domain_x,device=device)
        D2 = basis.omega2.to(device)
        D = basis.D.to(device)

        # initial condition
        u0 = -torch.sin(pi*basis.x).squeeze(-1)

        def f(u):
            u_hat = basis.dbt(u)

            D2u_hat = u_hat * D2
            
            u2 = u * u 
            u2_hat = basis.dbt(u2)
            u2_hat_x = u2_hat @ D
            u2_x = basis.idbt(u2_hat_x)

            D2u = basis.idbt(D2u_hat)
            

            return nu * D2u - (1/2) * u2_x


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


