import torch
from torch import nn
import numpy as np
import time
import argparse
import os
import json
from math import pi

from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS
from utilities.trainer import AdamTrainer
from utilities.grid import Regular2DGrid
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid, plot_loss_track, plot_parameter, plot_extrap_1d
from numerical_solvers.burgers1d import burgers1d_groundtruth_generator

parser = argparse.ArgumentParser('Burgers 1D inverse problem')
parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
parser.add_argument('--model', choices=('PINN', 'QRes', 'FLS'), default='PINN')
parser.add_argument('--seed', type=int,default= 42)    # random seed
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--steps', type=int, default=10000)  
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--grid_factor', type=int, default=4, help='How many times to increase the time-grid relative to the data')
parser.add_argument('--physics_weight', type=float, default=1.)
parser.add_argument('--data_weight', type=float, default=1.)
parser.add_argument('--ic_weight', type=float, default=1.)
parser.add_argument('--bc_weight', type=float, default=1.)
parser.add_argument('--data_points', type=int, default=-1)
parser.add_argument('--nu_0', type=float, default=0.01, help='Starting guess for the viscosity nu')
parser.add_argument('--nu',type=float,default=0.1)
parser.add_argument('--lambda_0', type=float, default=0.1, help='Starting value of the PDE parameter lambda_1')
parser.add_argument('--gamma', type=float, default=0.9995, help='Decay constant for the learning rate scheduler')
parser.add_argument('--num_layers', type=int, default=4, help='Number of hidden layers in the model')
parser.add_argument('--hidden_dim', type=int, default=512, help='Number of neurons per hidden layer')
parser.add_argument('--noise_level', type=float, default=0, help='Fractional standard deviation for the added noise')
parser.add_argument('--data_regime', type=str, choices=['uniform', 'endpoint'], required=True,
                    help='Format of the data: uniformly distributed or just the last frame')

args = parser.parse_args()
model_name = args.model
dir_name = args.dir
device = args.device
seed=args.seed
steps = args.steps
lr = args.lr

def loss_function(model, training_grid, u0_func, u_data, availability, eq_params, loss_weights):
    # NOTE - Raissi trains using only the data and physics losses, not IC or BC. 
    # Presumably the model can infer those from the data, but we should still enforce them
    x = training_grid.grid_column_x.clone().detach().requires_grad_()
    t = training_grid.grid_column_t.clone().detach().requires_grad_()
    n_x, n_t = len(training_grid.x), len(training_grid.t)
    u = model(x, t)
    physics_loss = physics_loss_function(u, x, t, eq_params())
    data_loss = data_driven_loss_function(u.reshape(n_t, n_x), u_data, availability)
    ic_loss = initial_loss_function(model, training_grid, u0_func)
    bc_loss = boundary_loss_function(model, training_grid)
    loss = loss_weights[0] * physics_loss + loss_weights[1] * data_loss + loss_weights[2] * ic_loss + loss_weights[3] * bc_loss

    return loss, (physics_loss, data_loss, ic_loss, bc_loss)

def initial_loss_function(model, training_grid, u0_func):
    x = training_grid.x
    t0 = training_grid.t0
    u0 = u0_func(x)
    loss_ic = torch.mean((model(x,t0)-u0)**2)
    return loss_ic

def boundary_loss_function(model, training_grid):
    x_lb = training_grid.x_lb
    x_ub = training_grid.x_ub
    t = training_grid.t
    loss_bc = torch.mean((model(x_lb,t)-model(x_ub,t))**2)
    return loss_bc

def physics_loss_function(u, x, t, param):
    lambda_1, lambda_2 = param
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    res = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
    loss = torch.mean((res)**2)

    return loss

def data_driven_loss_function(u, u_data, availability):
    '''NOTE this function is hardcoded to assume the data time-grid is covered by the model's grid, 
    and that the space-grid is the same for model and data'''
    u_downsampled = u[::args.grid_factor, :-1]
    loss = torch.mean(((u_downsampled - u_data)**2)[availability])

    return loss

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class EquationParameters(nn.Module):
    '''Keeps the learned value of the equation parameters.
    Forces the parameters to be positive'''
    def __init__(self, initial_parameters):
        super().__init__()
        initial_value = torch.log(initial_parameters)
        initial_value = torch.sqrt(initial_parameters)  ##DEBUG
        self.param = nn.Parameter(initial_value, requires_grad=True)
    
    def forward(self):
        return self.param**2  ##DEBUG
        return torch.exp(self.param)


if __name__ == "__main__":

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed*10)
    # torch.set_default_dtype(torch.float64) ##DEBUG

    save_dir = os.path.join("results", "inverse_burgers1d", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


    # domain, basis and grid
    domain_x = [-1.0,1.0]
    n_x_grid = 200
    n_x_gt = n_x_grid - 1
    x_grid = torch.linspace(*domain_x, n_x_grid, device=device)
    domain_t = [0.0, 1.0]
    domain_t_full = [0.0, 2.0]
    n_t_data = 101
    n_t = (n_t_data - 1) * args.grid_factor + 1  # In order to align with the points at every 0.01s in the data
    n_t_full = (n_t_data - 1) * args.grid_factor * 2 + 1
    ts = torch.linspace(*domain_t,n_t)
    dt = ts[1]-ts[0]
    t_grid = torch.linspace(*domain_t, n_t_data, device=device)
    t_grid_full = torch.linspace(*domain_t_full, n_t_full, device=device)
    u_gt_full = burgers1d_groundtruth_generator(args.nu,n_x_gt,n_t_full,domain_x,domain_t_full, device=device)[::args.grid_factor, :]
    u_gt = u_gt_full[:n_t_data, :]
    u_data = u_gt + np.random.randn(*u_gt.shape) * args.noise_level * np.std(u_gt)  # Adding noise in the same way as Raissi
    u_data = torch.tensor(u_data, device=device)

    if args.data_regime == 'uniform':
        if args.data_points == -1:
            availability = torch.ones(u_data.shape, dtype=bool, device=device)
        else:
            # Select N data points to be made available, while the rest will be hidden
            N = args.data_points
            flat_indices = torch.arange(u_data.shape[0]*u_data.shape[1])
            available_indices = np.random.choice(flat_indices, N, replace=False)
            availability = np.zeros(len(flat_indices), dtype=bool)
            availability[available_indices] = True
            availability = torch.tensor(availability.reshape(u_data.shape), device=device)
    elif args.data_regime == 'endpoint':
        availability = np.zeros(u_data.shape, dtype=bool)
        # Pass only the last frame
        availability[-1, :] = True
        availability = torch.tensor(availability, device=device)

    training_grid = Regular2DGrid(domain_x, domain_t, n_x_grid, n_t)
    training_grid = training_grid.to(device)

    u0_func = lambda x: -torch.sin(pi * x)

    # Starting guess for the viscosity parameter nu / lambda_2
    nu_true = args.nu
    nu_0 = args.nu_0

    # initialize model
    if model_name == "PINN":
        model = PINN(in_dim=2, out_dim=1, num_layer=args.num_layers, hidden_dim=args.hidden_dim)
    elif model_name == "QRes":
        model = QRes(in_dim=2, out_dim=1, num_layer=args.num_layers, hidden_dim=args.hidden_dim // 2)
    elif model_name == "FLS":
        model = FLS(in_dim=2, out_dim=1, num_layer=args.num_layers, hidden_dim=args.hidden_dim)
    model.apply(init_weights)
    model = model.to(device)
    param_count = sum([p.numel() for p in model.parameters()])
    print(f"{param_count} model parameters")

    # Learnable parameters
    lambda_1 = args.lambda_0
    lambda_2 = nu_0
    initial_parameters = torch.tensor([lambda_1, lambda_2], device=device)
    param = EquationParameters(initial_parameters)
    learnable_parameters = list(model.parameters()) + [param.param]

    loss_weights = [args.physics_weight, args.data_weight, args.ic_weight, args.bc_weight]

    trainer = AdamTrainer(learnable_parameters,
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    gamma=args.gamma,
                    loss_function= lambda: loss_function(model, training_grid, u0_func, u_data, availability, param, loss_weights),
                    loss_names=("Physics", "Data", "Initial", "Boundary"))

    with torch.autocast(device_type=device, dtype=torch.float32):
        start_time = time.time()
        trainer.train_inverse(param)
        end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.1f} seconds")

    # retrieve model solution
    n_t_data_full = u_gt_full.shape[0]
    ts_full = torch.linspace(*domain_t_full, n_t_data_full, device=device)
    t_inf, x_inf = torch.meshgrid(ts_full, x_grid[:-1],  indexing="ij")
    u_pred_full = model(x_inf.reshape([-1,1]), t_inf.reshape([-1,1])).detach().cpu().numpy()
    u_pred_full = u_pred_full.reshape((n_t_data_full, n_x_gt))
    u_pred = u_pred_full[:n_t_data, :]
    # compute metrics
    metrics= Metrics(u_pred,u_gt)

    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    u_pred_extrap = u_pred_full[n_t_data:, :]
    metrics_extrap = Metrics(u_pred_extrap, u_gt_full[n_t_data:, :])
    extrap_l1 = metrics_extrap.relative_l1_error()
    extrap_l2 = metrics_extrap.relative_l2_error()

    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)

    eq_params = param().detach().cpu().numpy()
    print(f"Learned parameters:\t {eq_params[0].item():.5f}, \t {eq_params[1].item():.5f}")
    print(f"True parameters: \t {1.:.5f}, \t {nu_true:.5f}")

    metrics_dict = {
        "relative_l2_error": float(relative_l2_error_value),
        "relative_l1_error": float(relative_l1_error_value),
        "extrapolation_relative_l1_error": float(extrap_l1),
        "extrapolation_relative_l2_error": float(extrap_l2),
        "training_time": float(training_time),
        "steps": int(steps),
        "learned_lambda_1": float(eq_params[0]),
        "learned_lambda_2": float(eq_params[1]),
        "parameter_count": param_count,
    }
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    # #### PLOTS ####
    offset = n_t_data // 3
    u_plot = [u_gt[0,:], u_gt[offset,:], u_gt[2 * offset,:], u_gt[-1,:]]
    u_pred_plot = [u_pred[0,:], u_pred[offset,:], u_pred[2 * offset,:], u_pred[-1,:]]
    x_plot = np.linspace(*domain_x, n_x_gt)
    t_plot = t_grid.cpu().numpy()
    titles = [f"t={t_plot[0].item()}", f"t={t_plot[offset].item()}", f"t={t_plot[2 * offset].item()}", f"t={t_plot[-1].item()}"]

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)
    n_t_data_full = u_gt_full.shape[0]
    ts_data_full = np.linspace(*domain_t_full, n_t_data_full)

    fig_u = plot_extrap_1d(u_pred_full, u_gt_full, x_grid[:-1].detach().cpu().numpy(), time=ts_data_full, training_limit=domain_t[1])
    fig_u.savefig(os.path.join(figs_dir, "extrapolation.png"), dpi=300, bbox_inches="tight", format="png")

    loss_track_dict = {name: trainer.loss_tracks[name] for name in ("Physics", "Data", "Initial", "Boundary")}
    plot_loss_track(loss_track_dict, os.path.join(figs_dir, "Loss.png"))
    plot_parameter(trainer.param_track, true_values=(1, nu_true), save_path=os.path.join(figs_dir, "Parameters.png"))
