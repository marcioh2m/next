import torch
from torch import nn
from torchdyn.core import NeuralODE
import time
import argparse
import os
import json
import math
import scipy.interpolate as spinterp
import numpy as np
import matplotlib.pyplot as plt

from basis.sine1d import SineBasis1D
from models.mlps import MLP
from models.vector_fields import SecondOrderStackModel
from utilities.trainer import AdamTrainer
from utilities.plot import create_graphs_grid, create_time_evolution_gif, plot_1d_heatmap, plot_loss_track
from utilities.metrics import Metrics
from integrators.etd import ETDEulerSecondDerivative, ETD4SecondDerivative
from numerical_solvers.wave import wave_groundtruth_generator


def gaussian_profile(x, x0, sigma=0.2):
    sqrt_2pi = math.sqrt(2 * math.pi)
    return torch.exp(-((x - x0) ** 2) / (2 * sigma * sigma)) / (sqrt_2pi * sigma)

def bump_velocity_profile(x, c0, c1, half_width=1.0, center = 0.0):
    # 1D velocity profile with a bump of velocity c0 in the middle and lower velocity c1 outside
    gain = 100.0
    beginning = center - half_width
    end = center + half_width
    inside_indicator = torch.sigmoid(gain * (x - beginning)) - torch.sigmoid(gain * (x - end))
    outside_indicator = 1 - inside_indicator
    c = c0 * inside_indicator + c1 * outside_indicator
    return c

class Wave1DNonLinearVectorField(nn.Module):
    # Calculates the non linear part of the vector field for the 1d wave equation
    # Assumes the linear part is the vector field for a homogeneous domain with velocity c0
    def __init__(self, c, c0, basis):
        super().__init__()
        self.c2 = nn.Parameter(c**2, requires_grad=False)
        self.c02 = nn.Parameter(torch.tensor(c0)**2, requires_grad=False)
        self.basis = basis

    def forward(self, huv):
        hu, _ = torch.chunk(huv, 2, -1)
        hu = hu.squeeze(-1)
        hu_t = torch.zeros_like(hu)  # the derivative of u is v, but this is already calculated by the linear field
        laplacian_freq_domain = basis.omegas * hu
        laplacian_phys_domain = basis.idbt(laplacian_freq_domain)
        u_tt = (self.c2 - self.c02) * laplacian_phys_domain
        hv_t = basis.dbt(u_tt)

        return torch.stack([hu_t, hv_t], dim=-1)

class CombinedVectorField(nn.Module):
    # Vector field comprising a linear part given by a matrix L and a non-linear part N
    def __init__(self, L, N , factor=1.0):
        super().__init__()
        self.L = torch.nn.Parameter(L,requires_grad=False)
        self.N = N
        self.factor = factor
    
    def forward(self,y):
        return (self.L @ y.unsqueeze(-1)).squeeze(-1) + self.factor * self.N(y)

def loss_function_huv(huv,basis,F,F_true):
    # Calculates the L2 loss between a learnable nonlinear field and the real one, after having calculated huv
    # Assumes the linear part of the field is known and correct
    # The vector fields act on the frequency domain, but the loss is evaluated on the physical domain
    Fhuv_pred = F.factor * F.N(huv)
    Fhuv_true = F_true.N(huv)
    loss_f = Fhuv_pred - Fhuv_true
    loss_hu, loss_hv = torch.chunk(loss_f, 2, dim=-1)
    loss_hu, loss_hv = loss_hu.squeeze(-1), loss_hv.squeeze(-1)
    loss_u, loss_v = basis.idbt(loss_hu), basis.idbt(loss_hv)
    loss = torch.stack([loss_u, loss_v], dim=-1)
    
    return torch.mean(loss**2)

def loss_function(huv0, node, ts,basis, F, F_true):
    # Calculates the L2 loss between a learnable nonlinear field and the real one
    # Assumes the linear part of the field is known and correct
    # The vector fields act on the frequency domain, but the loss is evaluated on the physical domain
    huv = node(huv0, ts)

    return loss_function_huv(huv, basis, F, F_true)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 1D NEXT')
    parser.add_argument('--dir', type=str, default='free_experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=1000)  
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--solver', type=str, default='etd4')
    parser.add_argument('--timesteps', type=int, default=100, help='Number of timesteps for the solver/integrator')
    
    args = parser.parse_args()
    dir_name = args.dir
    seed = args.seed
    device = args.device
    seed = args.seed
    epochs = args.steps
    lr = args.lr
    solver_name = args.solver
    timesteps = args.timesteps

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join("results", "wave1d", dir_name, "NEXT",f"{solver_name}", f"seed_{seed}")
    snapshot_dir = os.path.join(save_dir, "model_snapshots")
    metrics_dir = os.path.join(save_dir, "metrics")
    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir,exist_ok=True)
    os.makedirs(snapshot_dir,exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # domain, basis and grid
    domain_x = [-3.0, 3.0]
    n_x = 301
    n_basis = n_x - 2
    # the sine basis has 2 fewer elements than collocation points, because the edges are automatically zero
    basis = SineBasis1D(n_basis, domain_x, device=device)
    xs = basis.x

    domain_t = [0.0,2.0]
    n_t = timesteps + 1
    ts = torch.linspace(*domain_t,n_t)
    dt = ts[1]-ts[0]

    x0 = 0.0
    gaussian_sigma = 0.2
    u0 = gaussian_profile(xs, x0, sigma=gaussian_sigma).to(device)
    v0 = torch.zeros_like(u0)

    hu0 = basis.dbt(u0)
    hv0 = basis.dbt(v0)

    huv0 = torch.stack([hu0, hv0], dim=-1)

    c0 = 1.0
    c1 = 0.5
    c = bump_velocity_profile(xs, c0, c1, half_width=1.0, center=0.0)

    # Defining the linear field matrix L
    L = torch.zeros([n_basis, 2, 2])
    L[..., 0, 1] = 1
    L[..., 1, 0] = c0**2 * basis.omegas
    L = L.to(device)

    N = Wave1DNonLinearVectorField(c, c0, basis)

    F_true = CombinedVectorField(L, N, factor=1.0).to(device)
    
    # Building the ground truth solution with RK4 on a very fine time grid
    n_high = 2001
    ts_high = torch.linspace(*domain_t, n_high)
    u_gt = wave_groundtruth_generator(c, u0, basis, ts_high, device)

    n_points = timesteps + 1  #  correct the fencepost error


    F_neural = MLP(in_dim=n_basis, hidden_dim=300, out_dim=n_basis, num_hidden_layers=2).to(device)
    # F_model takes hu and passes it through an MLP to return hv_t
    F_model = SecondOrderStackModel(F_neural)
    F_learnable = CombinedVectorField(L, F_model, factor=1.0)
    param_count = sum([p.numel() for p in F_learnable.parameters()])

    if solver_name in ("etd1",):
        solver = ETDEulerSecondDerivative(F_learnable, h=dt)
    elif solver_name == "etd4":
        solver = ETD4SecondDerivative(F_learnable, h=dt)
    elif solver_name in ("rk", "rk4"):
        solver = "rk4"
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
    node = NeuralODE(F_learnable, solver=solver, sensitivity="autograd", return_t_eval=False)

    optimizer = torch.optim.Adam(F_learnable.N.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    #Begin training

    trainer = AdamTrainer(params=node.parameters(),
                            adam_lr=lr,
                            adam_steps=epochs,
                            learning_rate_scheduler=True,
                            loss_function= lambda: loss_function(huv0, node, ts, basis, F_learnable, F_true),
                            snapshot_dir = snapshot_dir,
                            model = node
                            )
    start_time = time.time()
    trainer.train_neusa()
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    torch.save(node.state_dict(), os.path.join(save_dir, "model.pt"))

    loss_track = trainer.loss_track_total
    epochs = trainer.adam_steps

    torch.save(node.state_dict(), os.path.join(save_dir, "model.pt"))

    # Evaluation
    with torch.no_grad():
        huv = node(huv0, t_span=ts)
    hu, _ = torch.chunk(huv, 2, dim=-1)
    hu = hu.squeeze(-1)
    u_pred = basis.idbt(hu).cpu().numpy()
    u_gt = wave_groundtruth_generator(c, u0, basis, ts_high, device)
    interpolator = spinterp.interp1d(ts_high.cpu().numpy(), u_gt, axis=0, kind='linear')
    u_gt = interpolator(ts.cpu().numpy())

    metrics= Metrics(u_pred,u_gt)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    metrics_dict = {
        "relative_l2_error": relative_l2_error_value,
        "relative_l1_error": relative_l1_error_value,
        "training_time": training_time,
        "steps": epochs,
        "parameter_count": param_count,
    }
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    # Plotting results

    x_plot = basis.x.cpu().numpy()
    
    n_t = n_t
    t_plot = np.linspace(*domain_t,n_t)

    offset = n_t // 3
    u_plot = [u_gt[0], u_gt[offset], u_gt[2 * offset], u_gt[n_t-1]]
    u_pred_plot = [u_pred[0], u_pred[offset], u_pred[2 * offset], u_pred[n_t-1]]

    titles = [f"t={t_plot[0].item()}", f"t={t_plot[offset].item()}", f"t={t_plot[2 * offset].item()}", f"t={t_plot[n_t-1].item()}"]

    with plt.rc_context({"font.size":20}):
        fig, axes = plt.subplots(figsize=(10, 4))
        axes.grid(True)
        axes.set_xlabel("x")
        axes.set_ylabel("c")
        axes.plot(xs, c)
        plt.savefig(os.path.join(figs_dir, "velocity_profile.png"), dpi=300, bbox_inches='tight')

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)
    create_time_evolution_gif(x_plot,t_plot,u_gt,u_pred,figs_dir)
    plot_1d_heatmap(t_plot, x_plot, u_pred, figs_dir, cmap="seismic")
    plot_loss_track(trainer.loss_tracks, os.path.join(figs_dir, "Loss.png"))