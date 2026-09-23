from basis.sine2d import SineBasis2D
import torch


def rk4(u, dt, f):
    k1 = dt * f(u)
    k2 = dt * f(u + 0.5 * k1)
    k3 = dt * f(u + 0.5 * k2)
    k4 = dt * f(u + k3)

    u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    return u_next

def heat2d_groundtruth_generator(n_x,n_y,n_t,domain_x,domain_y,domain_t):
    device = 'cuda:0'
    basis = SineBasis2D(n_x,n_y,domain_x,domain_y,device=device)

    n_t = 25 * (n_t-1) + 1

    ts = torch.linspace(*domain_t,n_t)

    D2 = basis.D2.to(device)
    
    # basis and 2 dimensional spatial grid
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    
    grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]
    
    # initial condition
    # sigma = 0.25
    # u0 = torch.exp(-((xs)**2+(ys)**2)/(2*sigma**2)).squeeze(-1).to(device)

    # kappa_0 = torch.tensor([1e-1],device=device)
    # kappa = kappa_0*(1+0.5*(1 - 1/(1+torch.exp(-500*(abs(xs)-0.5)))) * (1 - 1/(1+torch.exp(-500*(abs(ys)-0.5))))).squeeze(-1).to(device)


    T_0 = torch.tensor(1)
    T_1 = torch.tensor(0)
    indicator_T = (1 - 1/(1+torch.exp(-50*(abs(X)-0.5)))) * (1 - 1/(1+torch.exp(-50*(abs(Y)-0.5))))
    u0 = (T_1+(T_0-T_1)*indicator_T).squeeze(-1).to(device)

    # velocity field
    kappa_0 = torch.tensor([1e-1])
    kappa_1 = torch.tensor([1e-3])
    indicator = (1 - 1/(1+torch.exp(-50*(abs(X)-1.0)))) * (1 - 1/(1+torch.exp(-50*(abs(Y)-1.0))))
    kappa = (kappa_1+(kappa_0-kappa_1)*indicator).squeeze(-1).to(device)

    def f(u):
        u_hat = basis.dbt(u)
        D2u_hat = D2 * u_hat
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

    u_gt = u_gt[::25,:,:]



    return u_gt

    