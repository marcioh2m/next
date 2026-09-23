import torch
import torch.optim as optim
import os

import torch.distributed as dist

class AdamTrainer:
    def __init__(self,params,adam_lr, adam_steps, learning_rate_scheduler, loss_function,
                 adam_eps=1e-8, adam_amsgrad=False, gamma=None, snapshot_dir=None, model=None, loss_names=(), muon = False):

        self.params = params
        self.loss_function = loss_function

        self.adam_steps = adam_steps
        self.adam_lr = adam_lr
        self.adam_eps = adam_eps
        self.adam_amsgrad = adam_amsgrad
        self.muon = muon

        if muon == False:
            self.optimizer = optim.Adam(self.params, lr=self.adam_lr, eps=self.adam_eps, amsgrad=adam_amsgrad)
        else:

            # Initialize process group for single GPU/CPU (local mode)
            if not dist.is_initialized():
                dist.init_process_group(backend="gloo", init_method="tcp://localhost:23456", 
                                    world_size=1, rank=0)

            from Muon.muon import MuonWithAuxAdam
            hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
            hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]
            param_groups = [
                dict(params=hidden_weights, use_muon=True,
                    lr=self.adam_lr, weight_decay=0.01),
                dict(params=hidden_gains_biases, use_muon=False,
                    lr=self.adam_lr, betas=(0.9, 0.95), weight_decay=0.01),
            ]
            self.optimizer = MuonWithAuxAdam(param_groups)

        self.learning_rate_scheduler = learning_rate_scheduler
        if gamma is None:  # default values for gamma
            self.lr_scheduler_mlp = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
            self.lr_scheduler_neusa = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999, last_epoch=-1)
        else:
            self.lr_scheduler_mlp = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
            self.lr_scheduler_neusa = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma, last_epoch=-1)
        # self.lr_scheduler_neusa = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100, factor=0.5)

        self.snapshot_dir = snapshot_dir
        self.model = model
        self.loss_names = loss_names

        self.lr_track = []
        self.loss_track_total = []
        self.loss_track_ic = []
        self.loss_track_bc = []
        self.loss_track_res = []

    def train(self):
        try:
            for step in range(self.adam_steps):
                self.optimizer.zero_grad()

                losses = self.loss_function()
                loss_ic, loss_bc, loss_res = losses
                total_loss = sum(losses)
                
                total_loss.backward()
                self.optimizer.step()

                if self.learning_rate_scheduler and step % 100 == 0 and step != 0:
                    self.lr_scheduler_mlp.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.lr_track.append(current_lr)

                self.loss_track_total.append(total_loss.item())
                self.loss_track_ic.append(loss_ic.item())
                self.loss_track_bc.append(loss_bc.item())
                self.loss_track_res.append(loss_res.item())
                if step % 10 == 0 or step == self.adam_steps - 1:
                    print(f"Adam step {step} | Loss: {total_loss.item():.6f}")
                # if step % 100 and self.snapshot_dir is not None:
                #     snapshot_path = os.path.join(self.snapshot_dir, f"model_epoch_{step}.pt")
                #     torch.save(self.model.state_dict(), snapshot_path)
        except KeyboardInterrupt:
            self.adam_steps = step
        print(f"Adam Optimization Completed. ")
        print(f'Adam training loss: {total_loss.item():.6f}')
        self.loss_tracks = {"Physics": self.loss_track_res, "IC": self.loss_track_ic, "BC": self.loss_track_bc}

    def train_neusa(self):
        try:
            for step in range(self.adam_steps):
                self.optimizer.zero_grad()

                loss = self.loss_function()

                loss.backward()
                self.optimizer.step()

                self.lr_scheduler_neusa.step()
                # self.lr_scheduler_neusa.step(loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_track.append(current_lr)

                self.loss_track_total.append(loss.item())
                if step % 10 == 0 or step == self.adam_steps - 1:
                    print(f"Adam step {step} | Loss: {loss.item():.6f}")
                if step % 100 == 0 and self.snapshot_dir is not None:
                    snapshot_path = os.path.join(self.snapshot_dir, f"model_epoch_{step}.pt")
                    torch.save(self.model.state_dict(), snapshot_path)
                if torch.isnan(loss):
                    print("Training has diverged. Stopping now.")
                    break
        except KeyboardInterrupt:
            self.adam_steps = step
        print(f"Adam Optimization Completed. ")
        print(f'Adam training loss: {loss.item():.6f}')
        self.loss_tracks = {"Total": self.loss_track_total}
    
    def train_inverse(self, param, adapt_frequency=-1, neural_field=None, solver=None, L_func=None):
        # Similar to train_neusa, but tracks different loss terms and the value of the inversion parameter
        self.loss_tracks = {name:[] for name in self.loss_names}
        self.param_track = []
        if adapt_frequency != -1:
            assert isinstance(neural_field, torch.nn.Module) and not None in (solver, L_func)
        try:
            for step in range(self.adam_steps):
                self.optimizer.zero_grad()

                loss, individual_losses = self.loss_function()
                loss_dict = {self.loss_names[i]: individual_losses[i] for i in range(len(self.loss_names))}
                for name in self.loss_names:
                    self.loss_tracks[name].append(loss_dict[name].item())

                loss.backward()
                self.optimizer.step()

                self.lr_scheduler_neusa.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_track.append(current_lr)
                self.loss_track_total.append(loss.item())
                self.param_track.append(param().tolist())

                if step % 10 == 0 or step == self.adam_steps - 1:
                    print(f"Adam step {step} | Loss: {loss.item():.6f}", end=" | ")
                    for name in loss_dict.keys():
                        print(f"{name} Loss: {loss_dict[name].item():.6f}", end=" | ")
                    print("")
                if step % 100 == 0 and self.snapshot_dir is not None:
                    snapshot_path = os.path.join(self.snapshot_dir, f"model_epoch_{step}.pt")
                    torch.save(self.model.state_dict(), snapshot_path)
                if adapt_frequency != -1 and step % adapt_frequency == 0 and step > 0:
                        with torch.no_grad():
                            print(f"Current value of the parameters: \t{param()}\nAdjusting the linear field of NeuSA")
                            new_L = L_func(param())
                            neural_field.L.data = new_L
                            try: solver.adapt_L(new_L)
                            except AttributeError: pass
                if torch.isnan(loss):
                    print("Training has diverged. Stopping now.")
                    break
        except KeyboardInterrupt:
            self.adam_steps = step
        print(f"Adam Optimization Completed. ")
        print(f'Adam training loss: {loss.item():.6f}')        
