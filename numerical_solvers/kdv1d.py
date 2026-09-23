from basis.fourier1d import FourierBasis1D
import torch


def rk4(u, dt, f):
    k1 = dt * f(u)
    k2 = dt * f(u + 0.5 * k1)
    k3 = dt * f(u + 0.5 * k2)
    k4 = dt * f(u + k3)

    u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    return u_next

def kdv1d_groundtruth_generator(n_x,n_t,u0,domain_x,domain_t,nu=5,alpha=0.1,beta=1, ts_factor=1000):
    device = 'cuda:0'
    n_t = ts_factor * (n_t-1) + 1
    ts = torch.linspace(*domain_t,n_t,device=device)

    basis = FourierBasis1D(n_x,domain_x,device=device)
    D = basis.D.to(device)
    D3 = D @ D @ D
    u0 = u0.to(device)

    def f(u):
        u_hat = basis.dbt(u)
        D3u_hat = u_hat @ D3
        D3u = basis.idbt(D3u_hat)

        u2 = u * u 
        u2_hat = basis.dbt(u2)
        u2_hat_x = u2_hat @ D
        u2_x = basis.idbt(u2_hat_x)

        return beta * D3u + alpha * (1/2) * u2_x
    
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