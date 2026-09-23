import torch
from torch import nn
from torchdyn.core import NeuralODE
import numpy as np
import time
import argparse
import os
import json
from math import pi

from basis.fourier1d import FourierBasis1D
from models.vector_fields import HadamardVectorField
from models.mlps import MLP
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid, plot_loss_track, plot_parameter, plot_extrap_1d, create_time_evolution_gif
from integrators.etd import ETD4HadamardNonInvertibleL, BaseETDSolver
from numerical_solvers.burgers1d import burgers1d_groundtruth_generator

parser = argparse.ArgumentParser('Burgers 1D inverse problem')
parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
parser.add_argument('--seed', type=int,default= 42)    # random seed
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--steps', type=int, default=1000)  
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--grid_factor', type=int, default=4, help='How many times to increase the integration time-grid relative to the data')
parser.add_argument('--physics_weight', type=float, default=1.)
parser.add_argument('--data_weight', type=float, default=1.)
parser.add_argument('--data_points', type=int, default=-1)
parser.add_argument('--nu_0', type=float, default=0.05, help='Starting guess for the viscosity nu')
parser.add_argument('--solver', default='etd', choices=('rk4', 'etd'), help='Which numerical solver to use with the Neural ODE')
parser.add_argument('--adapt_frequency', type=int, default=-1, help='After this many epochs, NeuSA will adapt the linear field to fit the current value of the PDE parameters')
parser.add_argument('--nu',type=float,default=0.1)
parser.add_argument('--gamma', type=float, default=0.999, help='Decay constant for the learning rate scheduler')
parser.add_argument('--trainable_L', action='store_true')
parser.add_argument('--lambda_0', type=float, default=0.1, help='Starting value of the PDE parameter lambda_1')
parser.add_argument('--noise_level', type=float, default=0, help='Fractional standard deviation for the added noise')
parser.add_argument('--data_regime', type=str, choices=['uniform', 'endpoint'], required=True,
                    help='Format of the data: uniformly distributed or just the first and last frames')

args = parser.parse_args()
dir_name = args.dir
device = args.device
seed=args.seed
steps = args.steps
lr = args.lr

def loss_function(hu0, u_data, ts, NODE, basis, F, eq_params, loss_weights, availability, L_func):
    param_values = eq_params()
    if args.trainable_L:
        new_L = L_func(param_values)
        F.L = new_L
        if isinstance(NODE.solver, BaseETDSolver):
            NODE.solver.set_L(new_L)
    hu = NODE(hu0,ts).squeeze(1)
    u = basis.idbt(hu)
    physics_loss = physics_loss_function(hu, u, basis, F, param_values)
    data_loss = data_driven_loss_function(u, u_data, availability)
    loss = loss_weights[0] * physics_loss + loss_weights[1] * data_loss

    return loss, (physics_loss, data_loss)


def physics_loss_function(hu, u, basis, F, param):

    lambda_1, lambda_2 = param
    u_t = basis.idbt(F(hu))
    hu_x = hu @ basis.D
    hu_xx = hu_x @ basis.D
    u_x = basis.idbt(hu_x)
    u_xx = basis.idbt(hu_xx)
    res = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
    loss = torch.mean((res)**2)

    return loss

def data_driven_loss_function(u, u_data, availability):
    '''NOTE this function is hardcoded to assume the data time-grid is covered by the model's grid, 
    and that the space-grid is the same for model and data'''
    u_downsampled = u[::args.grid_factor, :]
    loss = torch.mean(((u_downsampled - u_data)**2)[availability])

    return loss

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
    torch.set_default_dtype(torch.float64)

    save_dir = os.path.join("results", "inverse_burgers1d", dir_name,"NEXT",f"{args.solver}", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    ## domain, basis and grid
    domain_x = [-1.0,1.0]
    n_x_grid = 200
    n_x = n_x_grid - 1  # remove the last point, since we use the Fourier basis
    # n_x = 199
    basis = FourierBasis1D(n_x, domain_x)
    xs = basis.x
    domain_t = [0.0, 1.0]
    domain_t_full = [0.0, 2.0]
    n_t_data = 101
    n_t = (n_t_data - 1) * args.grid_factor + 1  # In order to align with the points at every 0.01s in the data
    n_t_full = (n_t_data - 1) * args.grid_factor * 2 + 1
    ts = torch.linspace(*domain_t,n_t)
    dt = ts[1]-ts[0]
    u_gt_full = burgers1d_groundtruth_generator(args.nu,n_x,n_t_full,domain_x,domain_t_full, device=device)[::args.grid_factor, :]
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
    # initial condition
    u0 = -torch.sin(pi*xs)
    hu0 = basis.dbt(u0).unsqueeze(0)

    # Starting guess for the viscosity parameter nu / lambda_2
    nu_true = args.nu
    nu_0 = args.nu_0

    # Linear and Non-Linear vector field
    L = (nu_0*basis.omega2.to(device))
    L_func = lambda param_vector: param_vector[1] * basis.omega2.to(device)
    # ground-truth vector field and ETD
    # N_true = GTNonLinearVectorFieldBurgers(basis)

    # neural networks and NODE-ETD
    in_dim = len(xs)
    hidden_dim = 4 * in_dim
    out_dim = in_dim
    num_hidden_layers = 2

    N = MLP(in_dim,hidden_dim,out_dim,num_hidden_layers)
    param_count = sum([p.numel() for p in N.parameters()])
    print(f"{param_count} model parameters")

    F = HadamardVectorField(L, N, factor=0.1)

    basis = basis.to(device)
    hu0 = hu0.to(device)
    ts = ts.to(device)
    F = F.to(device)
    L = L.to(device)

    if args.solver == "etd":
        etd = ETD4HadamardNonInvertibleL(F,h=dt.to(device))
        solver = etd.to(device)
    elif args.solver == "rk4":
        solver = "rk4"

    NODE = NeuralODE(F,solver=solver, return_t_eval= False).to(device)

    ## Training setup
    # Learnable parameters
    lambda_1 = args.lambda_0
    lambda_2 = nu_0
    initial_parameters = torch.tensor([lambda_1, lambda_2], device=device) 
    param = EquationParameters(initial_parameters)
    learnable_parameters = list(NODE.parameters()) + [param.param]

    loss_weights = [args.physics_weight, args.data_weight]
    
    trainer = AdamTrainer(learnable_parameters,
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function= lambda: loss_function(hu0, u_data, ts, NODE, basis, F, param, loss_weights, availability, L_func),
                    loss_names=("Physics", "Data"),
                    gamma=args.gamma)

    start_time = time.time()
    trainer.train_inverse(param, adapt_frequency=args.adapt_frequency, neural_field=F, solver=NODE.solver, L_func=L_func)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time:.1f} seconds")

    # retrieve model solution    
    hu = NODE(hu0,ts)
    u_pred = basis.idbt(hu).squeeze().detach().to('cpu').numpy()
    u_pred = u_pred[::args.grid_factor, :]  #downsample to have the same shape as the ground truth data


    # compute metrics
    metrics= Metrics(u_pred,u_gt)

    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    ## Testing the extrapolation capability
    # NOTE this loss is computed only on the part of the time domain the model never saw during training
    ts_full = torch.linspace(*domain_t_full, n_t_full, device=device)
    hu_pred_full = NODE(hu0, ts_full)[::args.grid_factor, :]
    u_pred_full = basis.idbt(hu_pred_full).squeeze().detach().to('cpu').numpy()
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
    x_plot = basis.x.cpu().numpy()
    t_plot = ts.cpu().numpy()[::args.grid_factor]
    titles = [f"t={t_plot[0].item()}", f"t={t_plot[offset].item()}", f"t={t_plot[2 * offset].item()}", f"t={t_plot[-1].item()}"]

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)
    n_t_data_full = u_gt_full.shape[0]
    ts_data_full = np.linspace(*domain_t_full, n_t_data_full)

    fig_u = plot_extrap_1d(u_pred_full, u_gt_full, xs, time=ts_data_full, training_limit=domain_t[1])
    fig_u.savefig(os.path.join(figs_dir, "extrapolation.png"), dpi=300, bbox_inches="tight", format="png")

    loss_track_dict = {name: trainer.loss_tracks[name] for name in ("Physics", "Data")}
    plot_loss_track(loss_track_dict, os.path.join(figs_dir, "Loss.png"))
    plot_parameter(trainer.param_track, true_values=(1, nu_true), save_path=os.path.join(figs_dir, "Parameters.png"))
    np.savetxt(os.path.join(save_dir, "param_track.txt"), np.array(trainer.param_track))

    create_time_evolution_gif(x_plot,t_plot,u_gt,u_pred,figs_dir)