import argparse
import torch
import torch.nn as nn
import time
import os
import numpy as np
import json
import scipy.interpolate as spinterp

from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS

from utilities.grid import Regular2DGrid
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid, create_time_evolution_gif
from basis.sine1d import SineBasis1D

from numerical_solvers.wave import wave_groundtruth_generator

from wave1d_NEXT import gaussian_profile, bump_velocity_profile


def loss_function(model, training_grid, u0, v0, c, weights=(1,1,1)):
    ic_weight, bc_weight, res_weight = weights
    x = training_grid.x
    t0 = training_grid.t0.requires_grad_(True)

    #Initial conditions for function and derivative evaluated on the grid
    u0_pred = model(x, t0)
    v0_pred = torch.autograd.grad(u0_pred, t0, grad_outputs=torch.ones_like(u0_pred), create_graph=True)[0]
    loss_ic = torch.mean((u0_pred.flatten() - u0)**2 + (v0_pred.flatten() - v0)**2)

    t = training_grid.t
    x_lb = training_grid.x_lb
    x_ub = training_grid.x_ub

    # Homogeneous Dirichlet boundary conditions
    loss_bc = torch.mean(model(x_lb,t)**2) + torch.mean(model(x_ub,t)**2)

    x_res = training_grid.grid_column_x.requires_grad_()
    t_res = training_grid.grid_column_t.requires_grad_()

    u = model(x_res,t_res)

    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    loss_res = torch.mean((u_tt - c**2 * u_xx)**2)

    return (ic_weight * loss_ic, bc_weight * loss_bc, res_weight * loss_res)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('pinn_train_module')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes'
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=5000)  
    parser.add_argument('--lr', type=float, default=0.001)  
      
    args = parser.parse_args()
    model_name = args.model
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)

    save_dir = os.path.join("results", f"wave1d", dir_name, model_name, f"seed_{seed}")
    metrics_dir = os.path.join(save_dir, "metrics")
    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir,exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # initialize model
    if model_name == "PINN":
        model = PINN(in_dim=2, out_dim=1)
    elif model_name == "QRes":
        model = QRes(in_dim=2, out_dim=1)
    elif model_name == "FLS":
        model = FLS(in_dim=2, out_dim=1)
    model.apply(init_weights)
    model = model.to(device)

    param_count = sum([p.numel() for p in model.parameters()])

    # domain and grid
    domain_x = [-3.0, 3.0]
    domain_t = [0.0, 2.0]
    n_x = 301
    n_t = 201
    training_grid = Regular2DGrid(domain_x, domain_t, n_x,n_t)
    training_grid = training_grid.to(device)

    xs = torch.linspace(*domain_x, n_x, device=device)
    ts = torch.linspace(*domain_t, n_t, device=device)
    x0 = 0.0
    gaussian_sigma = 0.2
    u0 = gaussian_profile(xs, x0, sigma=gaussian_sigma).to(device)
    v0 = torch.zeros_like(u0)

    c0 = 1.0
    c1 = 0.5
    c = bump_velocity_profile(xs, c0, c1, half_width=1.0, center=0.0)

    ic_weight = 1000
    bc_weight = 100
    res_weight = 1
    weights = (ic_weight, bc_weight, res_weight)

    # training
    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model, training_grid, u0, v0, c, weights)
                    )


    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    
    # Building the ground truth solution with RK4 on a very fine time grid and a pseudo-spectral representation
    n_high = 2001
    ts_high = torch.linspace(*domain_t, n_high)
    n_basis = n_x - 2
    # the sine basis has 2 fewer elements than collocation points, because the edges are automatically zero
    basis = SineBasis1D(n_basis, domain_x, device=device)
    # We need to discard the endpoints for pseudo-spectral
    c_interior = c[1:-1]
    u0_interior = u0[1:-1]
    u_gt = wave_groundtruth_generator(c_interior, u0_interior, basis, ts_high, device)
    interpolator = spinterp.interp1d(ts_high.cpu().numpy(), u_gt, axis=0, kind='linear')
    u_gt = interpolator(ts.cpu().numpy())
    # Complete the edges of u_gt with zeros for the Dirichlet boundary condition
    u_gt = np.concatenate([np.zeros((n_t, 1)), u_gt, np.zeros((n_t, 1))], axis=1)

    # get predicted solution
    x = training_grid.grid_column_x.cpu()
    t = training_grid.grid_column_t.cpu()
    model = model.to('cpu')
    with torch.no_grad():
        u_pred = model(x,t).squeeze().numpy()
        u_pred = u_pred.reshape(n_t,n_x)

    metrics = Metrics(u_pred,u_gt)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)

    metrics_dict = {
        "relative_l2_error": relative_l2_error_value,
        "relative_l1_error": relative_l1_error_value,
        "training_time": training_time,
        "steps": steps,
        "parameter_count": param_count
    }
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # #### PLOTS ####
    offset = n_t // 3
    u_plot = [u_gt[0,:], u_gt[offset,:], u_gt[2 * offset,:], u_gt[n_t-1,:]]
    u_pred_plot = [u_pred[0,:], u_pred[offset,:], u_pred[2 * offset,:], u_pred[n_t-1,:]]

    x_plot = training_grid.x.squeeze(-1).cpu().numpy()
    t_plot = training_grid.t.squeeze(-1).cpu().numpy()

    titles = [f"t={t_plot[0].item()}", f"t={t_plot[offset].item()}", f"t={t_plot[2 * offset].item()}", f"t={t_plot[n_t-1].item()}"]

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)
    create_time_evolution_gif(x_plot,t_plot,u_gt,u_pred,figs_dir)