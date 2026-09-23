import torch
from torchdyn.core import NeuralODE
import numpy as np
import time
import argparse
import os
import json
from math import pi

from basis.fourier1d import FourierBasis1D
from models.vector_fields import GTNonLinearVectorFieldBurgers, HadamardVectorField
from models.mlps import MLP
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid, create_time_evolution_gif, plot_loss_track, plot_1d_heatmap
from integrators.etd import ETD4HadamardNonInvertibleL, ETDEulerHadamardNonInvertibleL

from numerical_solvers.burgers1d import burgers1d_groundtruth_generator


def loss_function(hu0,ts,NODE,basis,F,N_true):
    hu = NODE(hu0,ts)
    Fhu_pred = F.factor * F.N(hu)
    Fhu_true = N_true(hu)

    loss_f = basis.idbt(Fhu_pred - Fhu_true)

    loss = torch.mean((loss_f)**2)

    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Burgers 1D NEXT')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=500)  
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--nu',type=float,default=0.1)
    parser.add_argument('--solver', default='etd4', choices=('etd4', 'etd1', 'rk4'))
    
    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr
    nu = args.nu

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)

    save_dir = os.path.join("results", "burgers1d", dir_name,"NEXT",f"{args.solver}", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # domain, basis and grid
    domain_x = [-1.0,1.0]
    n_x = 201
    n_x_basis = n_x - 1
    basis = FourierBasis1D(n_x_basis, domain_x)
    xs = basis.x

    domain_t = [0.0,1.0]
    n_t = 401
    ts = torch.linspace(*domain_t,n_t)
    dt = ts[1]-ts[0]

    # initial condition
    u0 = -torch.sin(pi*xs).squeeze(-1)
    hu0 = basis.dbt(u0).unsqueeze(0)

    # Linear and Non-Linear vector field
    # L_matrix = nu*basis.D2.to(device)
    L = nu*basis.omega2.to(device)

    # ground-truth vector field and ETD
    N_true = GTNonLinearVectorFieldBurgers(basis)

    # neural networks and NODE-ETD
    in_dim = len(xs)
    hidden_dim = 4 * in_dim
    out_dim = in_dim
    num_hidden_layers = 2

    N = MLP(in_dim,hidden_dim,out_dim,num_hidden_layers)
    param_count = sum([p.numel() for p in N.parameters()])


    F = HadamardVectorField(L,N,factor=0.1)
    match args.solver:
        case 'etd4':
            solver = ETD4HadamardNonInvertibleL(F, h=dt.to(device)).to(device)
        case 'etd1':
            solver = ETDEulerHadamardNonInvertibleL(F,h=dt.to(device)).to(device)
        case 'rk4':
            solver = 'rk4'
    NODE = NeuralODE(F,solver=solver, return_t_eval= False)



    F_true = HadamardVectorField(L,N_true)
    etd_true = ETD4HadamardNonInvertibleL(F_true,h=dt.to(device))
    NODETRUE = NeuralODE(F_true,solver=etd_true, return_t_eval= False)

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

    # retrieve model solution (and copy the left boundary solution to the right boundary)    
    hu = NODE(hu0,ts)
    u_pred = basis.idbt(hu).squeeze().detach().to('cpu').numpy()

   # generate ground-truth
    u_gt = burgers1d_groundtruth_generator(nu,n_x_basis,n_t,domain_x,domain_t, device=device)

    F_true = F_true.to(device)
    NODETRUE = NODETRUE.to(device)

    hu_true = NODETRUE(hu0,ts)
    u_true_pred = basis.idbt(hu_true).squeeze().detach().to('cpu').numpy()
    metrics_true = Metrics(u_true_pred, u_gt)
    print(f"Error of the true vector field: {metrics_true.relative_l2_error()}")

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

    create_time_evolution_gif(x_plot,t_plot,u_gt,u_pred,figs_dir)
    
    fig = plot_1d_heatmap(t_plot, x_plot, u_pred, figs_dir, cmap="jet")

    plot_loss_track(trainer.loss_tracks, os.path.join(figs_dir, "Loss.png"))