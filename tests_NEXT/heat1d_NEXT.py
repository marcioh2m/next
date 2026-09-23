import torch
from torchdyn.core import NeuralODE
import time
import argparse
import os
import json
from math import pi, sqrt

from basis.sine1d import SineBasis1D
from models.mlps import MLP
from models.vector_fields import HadamardVectorField, GTNonLinearVectorFieldHeat
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid, create_time_evolution_gif, plot_instant, plot_loss_track
import numpy as np
from numerical_solvers.heat1d import heat1d_groundtruth_generator
from integrators.etd import ETD4Hadamard, ETDEulerHadamard

def loss_function(hu0,ts,NODE,basis,F,N_true):
    _,hu = NODE(hu0,ts)
    hu = hu.squeeze(1)
    Fhu_pred = F.factor * F.N(hu)
    Fhu_true = N_true(hu)
    
    loss = torch.mean((Fhu_pred - Fhu_true)**2)

    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Heat 1D NEXT')
    parser.add_argument('--dir', type=str, default='free_experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=500)  
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--solver', default='etd4', choices=('etd1', 'etd4', 'rk4'))
    
    args = parser.parse_args()
    dir_name = args.dir
    seed = args.seed
    device = args.device
    seed = args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)

    save_dir = os.path.join("results", "heat1d", dir_name,"NEXT", f"{args.solver}", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # domain, basis and grid
    domain_x = [-2.0,2.0]
    n_x = 201
    n_x_basis = n_x - 2  # The endpoints are zero in the sine basis, so they don't show up in the basis grid
    basis = SineBasis1D(n_x_basis,domain_x)
    xs = basis.x

    domain_t = [0.0,2.0]
    n_t = n_x
    ts = torch.linspace(*domain_t,n_t)
    dt = ts[1]-ts[0]

    # initial condition
    sigma=0.5
    u0=(torch.exp(-((xs-0.4)**2)/(2*sigma**2))/(sigma*sqrt(2*pi))).reshape(1,-1)
    #gamma = 8
    #u0 = (1/2) * (1-torch.tanh(gamma*(abs(xs)-1))).reshape(1,-1)
    hu0 = basis.dbt(u0)

    # conductivity profile
    kappa_0 = torch.tensor([1e-1],device=device)
    kappa_1 = torch.tensor([1e-3],device=device)
    xs = xs.to(device)
    kappa = kappa_1 + (kappa_0-kappa_1)*(1/(1+torch.exp(-50*(xs+0.5)))+1/(1+torch.exp(-50*(xs-0.5))))

    L = kappa_0*torch.diag(basis.D2).to(device)
    # check the order of magnitude of e^(hL) to understand which way of computing the ETD varphi's is better


    # ground-truth vector field and ETD
    N_true = GTNonLinearVectorFieldHeat(basis,kappa,kappa_0)
    #F_true = HeatLikeVectorFieldETD(L,N_true,factor=1.0)
    #etd_true = ETD4Heat(F_true,h=dt.to(device))
    #NODE_true = NeuralODE(F_true,solver=etd_true)
    
    # neural networks and NODE-ETD
    
    #N = MLPHeatETD(n_x)
    # neural networks and NODE-ETD
    in_dim = len(xs)
    hidden_dim = 4 * in_dim
    out_dim = in_dim
    num_hidden_layers = 2

    N = MLP(in_dim,hidden_dim,out_dim,num_hidden_layers)
    param_count = sum([p.numel() for p in N.parameters()])
    print(f"{param_count} model parameters")

    F = HadamardVectorField(L,N,factor=0.1)
    match args.solver:
        case 'etd1':
            solver = ETDEulerHadamard(F,h=dt.to(device))
        case 'etd4':
            solver = ETD4Hadamard(F,h=dt.to(device))
        case 'rk4':
            solver = 'rk4'
    NODE = NeuralODE(F,solver=solver)

    F = F.to(device)
    NODE = NODE.to(device)
    
    # training
    basis = basis.to(device)
    hu0 = hu0.to(device)
    ts = ts.to(device)
    N = N.to(device)
    kappa = kappa.to(device)
    N_true = N_true.to(device)
    N_true.kappa = N_true.kappa.to(device)

    trainer = AdamTrainer(N.parameters(),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function= lambda: loss_function(hu0,ts,NODE,basis,F,N_true)
                )
    
    start_time = time.time()

    trainer.train_neusa()

    end_time = time.time()
    training_time = end_time - start_time

    torch.save(NODE.state_dict(), os.path.join(save_dir, "model.pt"))

    # save trained model
    torch.save(F.state_dict(),os.path.join(save_dir, "model.pt"))

    # retrieve model solution
    _,hu = NODE(hu0,ts)
    hu = hu.squeeze(1)

    u_pred = basis.idbt(hu).detach().to('cpu').numpy()
    
    # #### GENERATE GROUND-TRUTH ####

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

    x_plot = basis.x.cpu().numpy()
    n_t = n_t
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