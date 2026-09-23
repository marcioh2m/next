import torch
from torchdyn.core import NeuralODE
import time
import json
import argparse
import os
import numpy as np

from basis.sine2d import SineBasis2D
from models.vector_fields import ETDHeatLikeVectorField, NonLinearHeatEquationVectorField
from integrators.etd import ETD4Hadamard, ETDEulerHadamard
from models.mlps import MLPWithTranspose2D
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import plot_prediction_vs_groundtruth, plot_error, plot_loss_track
from numerical_solvers.heat2d import heat2d_groundtruth_generator


def loss_function(NODE, basis, hu0, ts, F_neural, F_nonlinear):
    hu = NODE(hu0, ts)
    F_pred = F_neural(hu)
    F_real = F_nonlinear(hu)
    loss_f = basis.idbt(F_pred-F_real)
    loss = torch.mean((loss_f)**2)
    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Heat 2D NEXT')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--steps', type=int, default=500)  
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--solver', type=str, default='etd4', choices=['etd4', 'etd1', 'rk4'])

    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr
    width = args.width
    depth = args.depth

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)

    save_dir = os.path.join("results", "heat2d", dir_name, "NEXT",f"{args.solver}", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # baseline domain
    domain_x = [-2.0,2.0]
    domain_y = [-2.0,2.0]
    domain_t = [0.0,2.0]
    n_x = 101
    n_y = 101
    n_x_basis = n_x - 2
    n_y_basis = n_y - 2
    n_t = 201
    
    ts = torch.linspace(*domain_t, n_t)

    
    # basis and 2 dimensional spatial grid
    basis = SineBasis2D(n_x_basis, n_y_basis, domain_x, domain_y, device=device)
    X,Y = torch.meshgrid(basis.x.to(device),basis.y.to(device),indexing='ij')
    
    
    grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]
    
    # initial condition
    sigma = 0.25
    # u0 = torch.exp(-((xs)**2+(ys)**2)/(2*sigma**2)).squeeze(-1).to(device)

    T_0 = torch.tensor(1)
    T_1 = torch.tensor(0)
    indicator_T = (1 - 1/(1+torch.exp(-50*(abs(X)-0.5)))) * (1 - 1/(1+torch.exp(-50*(abs(Y)-0.5))))
    u0 = (T_1+(T_0-T_1)*indicator_T).squeeze(-1).to(device)

    hu0 = basis.dbt(u0)     # shape = (M,N)
    hu0 = hu0.unsqueeze(0)


    # velocity field
    kappa_0 = torch.tensor([1e-1], device=device)
    kappa_1 = torch.tensor([1e-3], device=device)
    indicator = (1 - 1/(1+torch.exp(-50*(abs(X)-1.0)))) * (1 - 1/(1+torch.exp(-50*(abs(Y)-1.0))))
    kappa = (kappa_1+(kappa_0-kappa_1)*indicator).squeeze(-1)

    # Define the Linear and Non Linear Vector Field

    #F_linear = kappa_0*Heat2dTensorL(basis)
    F_linear = kappa_0 * basis.omegas
    F_nonlinear = NonLinearHeatEquationVectorField(basis, kappa, kappa_0, device)
    F_true = ETDHeatLikeVectorField(F_linear,F_nonlinear)

    # Define the Neural Vector Field and the Complete Vector Field.
    num_additional_hidden_layers = depth - 1
    widths = (width, width)
    F_neural = MLPWithTranspose2D(M=n_x_basis,N=n_y_basis, widths=widths,
                                  num_additional_hidden_layers=num_additional_hidden_layers,
                                  additional_hidden_layer_bias=False)
    # F_neural = LowRankMLP(n_x,n_x,n_x,0,1)
    F = ETDHeatLikeVectorField(F_linear, F_neural)
    param_count = sum([p.numel() for p in F_neural.parameters()])

    # Define the Neural ODE
    ts = ts.to(device)
    F_linear = F_linear.to(device)
    F = F.to(device)
    dt = ts[1] - ts[0]
    match args.solver:
        case 'etd4':
            solver = ETD4Hadamard(F, h=dt)
        case 'etd1':
            solver = ETDEulerHadamard(F, h=dt)
        case 'rk4':
            solver = "rk4"
    NODE = NeuralODE(F,solver=solver,sensitivity='autograd',return_t_eval=False)

    # training
    F = F.to(device)
    NODE = NODE.to(device)
    basis = basis.to(device)
    hu0 = hu0.to(device)
    ts = ts.to(device)

    trainer = AdamTrainer(NODE.parameters(),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    # loss_function= lambda: loss_function(NODE, basis, huv0, ts, F, F_true)
                    loss_function= lambda: loss_function(NODE, basis, hu0, ts, F_neural, F_nonlinear)
                )
    
    start_time = time.time()
    trainer.train_neusa()
    end_time = time.time()

    training_time = end_time - start_time

    # save trained model
    torch.save(F.state_dict(),os.path.join(save_dir, "model.pt"))

    # retrieve model solution
    start_time = time.time()
    hu = NODE(hu0, ts)
    hu = hu.squeeze(1)
    end_time = time.time()
    inference_time = end_time-start_time

    u_pred = basis.idbt(hu).detach()
    u_pred = u_pred.cpu().numpy()

    # Solution with ETD
    F_true = F_true.to(device)

    etd_true = ETD4Hadamard(F_true, h=dt)
    NODEETD = NeuralODE(F_true,solver=etd_true,sensitivity='autograd',return_t_eval=False)
    hu_true = NODEETD(hu0,ts)

    u_gt = basis.idbt(hu_true).squeeze().detach()
    u_gt = u_gt.cpu().numpy()
    u_gt = heat2d_groundtruth_generator(n_x_basis,n_y_basis,n_t,domain_x,domain_y,domain_t)

    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)

    
    # compute metrics
    metrics = Metrics(u_pred,u_gt)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    metrics_dict = {
        "relative_l2_error": float(relative_l2_error_value),
        "relative_l1_error": float(relative_l1_error_value),
        "training_time": float(training_time), 
        "steps": int(steps),
        "parameter count": param_count
    }
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
     ### PLOTS ###

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    save_path = os.path.join(figs_dir,"plots.png")
    plot_prediction_vs_groundtruth(ts=ts,u_pred=u_pred,u_gt=u_gt,extent=[*domain_x,*domain_y],save_path=save_path,cmap="jet", equation = 'heat')
    
    save_path = os.path.join(figs_dir,"error.png")
    plot_error(ts=ts,error=u_pred-u_gt,extent=[*domain_x,*domain_y],save_path=save_path,cmap="seismic")
    plot_loss_track(trainer.loss_tracks, os.path.join(figs_dir, "Loss.png"))


