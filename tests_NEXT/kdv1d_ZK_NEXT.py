import torch
from torchdyn.core import NeuralODE
import numpy as np
import time
import argparse
import os
import json
from math import pi

from basis.fourier1d import FourierBasis1D
from models.vector_fields import GTNonLinearVectorFieldKDV, KdVDiagonalVectorField
from models.mlps import MLP
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid, create_time_evolution_gif, plot_1d_heatmap, plot_loss_track
from integrators.etd import ETD4FourierBasisKdV, ETDEulerFourierBasisKdV
from numerical_solvers.kdv1d import kdv1d_groundtruth_generator

def loss_function(hu0, ts, NODE, basis, F, N_true):
    hu = NODE(hu0,ts)
    Fhu_pred = F.factor * F.N(hu)
    Fhu_true = N_true(hu)

    loss_f = basis.idbt(Fhu_pred - Fhu_true)

    loss = torch.mean((loss_f)**2)

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser('KdV 1D NEXT')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=1000)  
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--solver', default='etd4', choices=('etd4', 'etd1', 'rk4'))
    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)

    save_dir = os.path.join("results", "kdv1d", dir_name,"NEXT",f"{args.solver}", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # domain, basis and grid
    domain_x = [0., 2.]
    n_x = 101
    n_x_basis = n_x - 1  # remove the endpoint for Fourier (periodic) basis
    basis = FourierBasis1D(n_x_basis, domain_x, device=device)
    xs = basis.x
    
    domain_t = [0.0,1.0]
    n_t = 501
    mul = 10000 // (n_t-1)
    ts = torch.linspace(*domain_t,n_t)
    dt = ts[1]-ts[0]
    # alpha = -1
    # beta = -0.0025
    alpha = -0.5
    beta = -0.022**2
    u0 = torch.cos(torch.pi*xs)
    hu0 = basis.dbt(u0).unsqueeze(0)

    # Linear and Non-Linear vector field

    L_matrix = beta * (basis.D @ basis.D @ basis.D).to(device)
    omega2 = basis.omega2[:basis.N_cos]  # array of frequencies for each mode in L
    omegas = torch.sqrt(torch.abs(omega2))
    A = torch.tensor([[0., 1.], [-1., 0.]], device=device)
    L = -beta * torch.einsum("i,jk->ijk", omegas**3, A)

    # ground-truth vector field and ETD
    N_true = GTNonLinearVectorFieldKDV(basis, alpha)

    # neural networks and NODE-ETD
    in_dim = len(xs)
    hidden_dim = 4 * in_dim
    out_dim = in_dim
    num_hidden_layers = 2

    N = MLP(in_dim,hidden_dim,out_dim,num_hidden_layers).to(device)

    param_count = sum([p.numel() for p in N.parameters()])


    match args.solver:
        case 'etd1':
            F = KdVDiagonalVectorField(L, N, basis, factor=0.1)
            solver = ETDEulerFourierBasisKdV(F, h=dt.to(device))
        case 'etd4':
            F = KdVDiagonalVectorField(L, N, basis, factor=0.1)
            solver = ETD4FourierBasisKdV(F, h=dt.to(device))
        case 'rk4':
            solver = 'rk4'
    NODE = NeuralODE(F,solver=solver, return_t_eval= False)


    F = F.to(device)
    NODE = NODE.to(device)

    # training
    basis = basis.to(device)
    hu0 = hu0.to(device)
    ts = ts.to(device)
    NODE = NODE.to(device)
    F = F.to(device)
    L = L.to(device)
    N_true = N_true.to(device)
    
    trainer = AdamTrainer(NODE.parameters(),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function= lambda: loss_function(hu0,ts,NODE,basis,F,N_true))

    start_time = time.time()
    trainer.train_neusa()
    end_time = time.time()

    training_time = end_time - start_time

    torch.save(NODE.state_dict(), os.path.join(save_dir, "model.pt"))

    # retrieve model solution     
    hu = NODE(hu0,ts)
    u_pred = basis.idbt(hu).squeeze().detach().to('cpu').numpy()
    

   # generate ground-truth
    u_gt = kdv1d_groundtruth_generator(n_x_basis,n_t,u0,domain_x,domain_t, alpha=alpha, beta=beta, ts_factor=mul)
    
    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)

    
    # compute metrics
    metrics= Metrics(u_pred,u_gt)

    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)


    metrics_dict = {
        "relative_l2_error": float(relative_l2_error_value),
        "relative_l1_error": float(relative_l1_error_value),
        "training_time": float(training_time),
        "steps": int(steps),
        "parameter_count": param_count
    }
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    # #### PLOTS ####
    offset = n_t // 3
    u_plot = [u_gt[0,:], u_gt[offset,:], u_gt[2 * offset,:], u_gt[n_t-1,:]]
    u_pred_plot = [u_pred[0,:], u_pred[offset,:], u_pred[2 * offset,:], u_pred[n_t-1,:]]
    x_plot = basis.x.cpu().numpy()
    t_plot = ts.cpu().numpy()
    titles = [f"t={t_plot[0].item()}", f"t={t_plot[offset].item()}", f"t={t_plot[2 * offset].item()}", f"t={t_plot[n_t-1].item()}"]

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)

    create_time_evolution_gif(x_plot, t_plot, u_gt, u_pred, figs_dir)
    plot_1d_heatmap(t_plot, x_plot, u_pred, figs_dir, cmap="jet")
    plot_loss_track(trainer.loss_tracks, os.path.join(figs_dir, "Loss.png"))