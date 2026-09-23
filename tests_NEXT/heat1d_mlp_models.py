import argparse
import torch
import time
import os
import json
import numpy as np
from math import pi, sqrt
from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS
from basis.sine1d import SineBasis1D

from utilities.network_initialization import init_weights
from utilities.grid import Regular2DGrid
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid, create_time_evolution_gif, plot_instant, plot_loss_track
from numerical_solvers.heat1d import heat1d_groundtruth_generator

def loss_function(model, training_grid,device):
    training_grid = training_grid.to(device)
    
    x = training_grid.x
    t = training_grid.t
    x_lb = training_grid.x_lb
    x_ub = training_grid.x_ub
    t0 = training_grid.t0

    # boundary condition loss
    loss_bc = torch.mean((model(t,x_lb))**2) + torch.mean((model(t,x_ub))**2)

    # initial condition loss
    sigma = 0.5
    u0=(torch.exp(-((x-0.4)**2)/(2*sigma**2))/(sigma*sqrt(2*pi)))
    u0_model = model(t0,x)
    loss_ic = torch.mean((u0_model-u0)**2)

    # equation residue
    t_res = training_grid.grid_column_t.requires_grad_()
    x_res = training_grid.grid_column_x.requires_grad_()

    #kappa = (nn.ELU()(x_res) / 6 + 0.3)
    kappa_0 = torch.tensor([1e-1],device=device)
    kappa_1 = torch.tensor([1e-3],device=device)
    kappa = kappa_1 + (kappa_0-kappa_1)*(1/(1+torch.exp(-50*(x_res+0.5)))+1/(1+torch.exp(-50*(x_res-0.5))))

    u = model(t_res,x_res)
    u_t = torch.autograd.grad(u,t_res,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_x = torch.autograd.grad(u,x_res,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x_res,grad_outputs=torch.ones_like(u_x),create_graph=True)[0]

    loss_res = torch.mean((u_t-kappa*u_xx)**2)

    return (loss_ic,loss_bc,loss_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Heat 1D baseline models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes', 'PINNsFormer
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

    save_dir = os.path.join("results", "heat1d", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

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
    print(f"{param_count} model parameters")

    # domain and grid
    domain_x = [-2,2]
    domain_t = [0,2]

    n_x = 201
    n_x_basis = n_x - 2  # The endpoints are zero in the sine basis, so they don't show up in the basis grid
    n_t = n_x

    training_grid = Regular2DGrid(domain_x,domain_t,n_x,n_t)

    # training
    trainer = AdamTrainer(params=list(model.parameters()),
                          adam_lr = lr,
                          adam_steps = steps,
                          learning_rate_scheduler=True,
                          loss_function=lambda: loss_function(model,training_grid,device)
                        )
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # retrieve predicted solution
    model = model.to('cpu')
    with torch.no_grad():

        basis = SineBasis1D(n_x_basis,domain_x,device)
        x = basis.x
        t = torch.linspace(*domain_t,n_t)
        T,X = torch.meshgrid(t,x,indexing='ij')
        
        t_inf = T.flatten().unsqueeze(1)
        x_inf = X.flatten().unsqueeze(1)

        u_pred = model(t_inf,x_inf).reshape(n_t,n_x_basis).cpu().numpy()



    #### GENERATE GROUND-TRUTH ####
    sigma = 0.5
    u0=(torch.exp(-((x-0.4)**2)/(2*sigma**2))/(sigma*sqrt(2*pi))).reshape(1,-1)
    kappa_0 = torch.tensor([1e-1],device=device)
    kappa_1 = torch.tensor([1e-3],device=device)
    xs = x.to(device)
    kappa = kappa_1 + (kappa_0-kappa_1)*(1/(1+torch.exp(-50*(xs+0.5)))+1/(1+torch.exp(-50*(xs-0.5))))

    u_gt = heat1d_groundtruth_generator(n_x_basis,n_t,domain_x,domain_t,u0,kappa,device)

    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)

    
    #### METRICS ####
    metrics = Metrics(u_pred,u_gt)

    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

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

    #### PLOTS ####

    x_plot = x.cpu().numpy()
    t_plot = np.linspace(*domain_t,n_t)

    offset = n_t // 3
    u_plot = [u_gt[0,:], u_gt[offset,:], u_gt[2 * offset,:], u_gt[n_t-1,:]]
    u_pred_plot = [u_pred[0,:], u_pred[offset,:], u_pred[2 * offset,:], u_pred[n_t-1,:]]

    titles = [f"t={t_plot[0].item()}", f"t={t_plot[offset].item()}", f"t={t_plot[2 * offset].item()}", f"t={t_plot[n_t-1].item()}"]

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)

    create_time_evolution_gif(x_plot,t_plot,u_gt,u_pred,figs_dir)

    fig = plot_instant(t_plot, x_plot, u_pred, cmap="coolwarm", x_name="t", y_name="x")
    fig.savefig(os.path.join(figs_dir, "u_heatmap.png"), dpi=300, bbox_inches="tight")
    
    plot_loss_track(trainer.loss_tracks, os.path.join(figs_dir, "Loss.png"))
    