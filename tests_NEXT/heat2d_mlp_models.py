import argparse
import torch
import time
import os
import numpy as np
import json

from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS
from basis.sine2d import SineBasis2D

from utilities.network_initialization import init_weights
from utilities.grid import Random3DGrid
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import plot_prediction_vs_groundtruth, plot_loss_track
from numerical_solvers.heat2d import heat2d_groundtruth_generator


def loss_function(model,training_grid,device):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    t_ic = training_grid.t_ic

    T_0 = torch.tensor(1)
    T_1 = torch.tensor(0)
    indicator_T = (1 - 1/(1+torch.exp(-50*(abs(x_ic)-0.5)))) * (1 - 1/(1+torch.exp(-50*(abs(y_ic)-0.5))))
    u0 = (T_1+(T_0-T_1)*indicator_T).to(device)

    #sigma = 0.25
    #u0 = torch.exp(-((x_ic)**2+(y_ic)**2)/(2*sigma**2)).to(device)
    
    u0_model = model(t_ic,x_ic,y_ic)
    
    loss_ic = 1e2*torch.mean((u0-u0_model)**2)

    t_bc = training_grid.t_bc
    x_bc = training_grid.x_bc
    x_lb_bc = training_grid.x_lb_bc
    x_ub_bc = training_grid.x_ub_bc
    y_bc = training_grid.y_bc
    y_lb_bc = training_grid.y_lb_bc
    y_ub_bc = training_grid.y_ub_bc


    # dirichlet on left boundary (x = x_min)
    u_lb = model(t_bc,x_lb_bc,y_bc)
    
    # dirichlet on right boundary (x = x_max)
    u_rb = model(t_bc,x_ub_bc,y_bc)
    
    # dirichlet on bottom boundary (y = y_min)
    u_bb = model(t_bc,x_bc,y_lb_bc)
    
    # dirichlet on top boundary (y = y_max)
    u_tb = model(t_bc,x_bc,y_ub_bc)
    
    loss_bc = (torch.mean((u_lb)**2) + torch.mean((u_rb)**2) + torch.mean((u_bb)**2) + torch.mean((u_tb)**2))

    x_res = training_grid.x_res.detach().requires_grad_()
    y_res = training_grid.y_res.detach().requires_grad_()
    t_res = training_grid.t_res.detach().requires_grad_()

    u = model(t_res,x_res,y_res)
    u_t = torch.autograd.grad(u,t_res,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_x = torch.autograd.grad(u,x_res,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x_res,grad_outputs=torch.ones_like(u_x),create_graph=True)[0]
    u_y = torch.autograd.grad(u,y_res,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y,y_res,grad_outputs=torch.ones_like(u_y),create_graph=True)[0]


    #kappa_0 = torch.tensor([1e-1],device=device)
    #kappa = kappa_0*(1+0.5*(1 - 1/(1+torch.exp(-500*(abs(x_res)-0.5)))) * (1 - 1/(1+torch.exp(-500*(abs(y_res)-0.5)))))

    kappa_0 = torch.tensor([1e-1]).to(device)
    kappa_1 = torch.tensor([1e-3]).to(device)
    indicator = (1 - 1/(1+torch.exp(-50*(abs(x_res)-1.0)))) * (1 - 1/(1+torch.exp(-50*(abs(y_res)-1.0))))
    kappa = (kappa_1+(kappa_0-kappa_1)*indicator).to(device)

    loss_res = torch.mean((u_t - kappa * (u_xx + u_yy))**2)

    return (loss_ic,loss_bc,loss_res)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Heat 2d baseline models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes', 'PINNsFormer
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')     
    parser.add_argument('--steps', type=int, default=20000)  
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    model_name = args.model
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    
   
    save_dir = os.path.join("results", "heat2d", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # initialize model
    if model_name == "PINN":
        model = PINN(in_dim=3, out_dim=1)
    elif model_name == "QRes":
        model = QRes(in_dim=3, out_dim=1)
    elif model_name == "FLS":
        model = FLS(in_dim=3, out_dim=1)
    model.apply(init_weights)
    model = model.to(device)
    param_count = sum([p.numel() for p in model.parameters()])

    # domain and grid
    domain_x = [-2.0,2.0]
    domain_y = [-2.0,2.0]
    domain_t = [0.0,2.0]
    
    n_ic = 1000
    n_bc = 500
    n_res = 10000
    training_grid = Random3DGrid(domain_x, domain_y, domain_t, n_ic, n_bc, n_res)

    # training
    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model,training_grid,device)
                )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    # save trained model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    
    # retrieve model predicted solution
    n_x = 101
    n_y = 101
    n_x_basis = n_x - 2
    n_y_basis = n_y - 2
    n_t = 201

    basis = SineBasis2D(n_x_basis,n_y_basis,domain_x,domain_y,device=device)
    
    ts = torch.linspace(*domain_t,n_t)
    xs = torch.linspace(*domain_x,n_x)[1:-1]
    ys = torch.linspace(*domain_y,n_y)[1:-1]

    T,X,Y = torch.meshgrid(ts,xs,ys,indexing='ij')
    t_inf = T.flatten().unsqueeze(1)
    x_inf = X.flatten().unsqueeze(1)
    y_inf = Y.flatten().unsqueeze(1)

    model = model.to('cpu')
    u_pred = model(t_inf,x_inf,y_inf).reshape(n_t,n_x_basis,n_y_basis)
    u_pred = u_pred.detach().cpu().numpy()

    u_gt = heat2d_groundtruth_generator(n_x_basis,n_y_basis,n_t,domain_x,domain_y,domain_t)
    
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
        "parameter count": param_count,
    }
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    #### PLOTS ####

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    save_path = os.path.join(figs_dir,"plots.png")
    plot_prediction_vs_groundtruth(ts,u_pred,u_gt,extent=[*domain_x,*domain_y],save_path=save_path,cmap="jet", equation = 'heat')
    plot_loss_track(trainer.loss_tracks, os.path.join(figs_dir, "Loss.png"))
