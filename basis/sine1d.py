import torch

class SineBasis1D:
    def __init__(self,M,domain,device="cpu"):
        self.device = device
        
        self.M = M
        self.a, self.b = domain

        self.idx_freqs = torch.arange(1,self.M+1)

        self.z = torch.pi / (self.M+1) *  self.idx_freqs
        self.x = self.z * ((self.b-self.a)/torch.pi) + self.a        
        self.full_x = torch.cat([torch.tensor([self.a]), self.x, torch.tensor([self.b])])


        self.Cx = torch.zeros((self.M,self.M), device=self.device)
        for i in range(self.M):
            for j in range(self.M):
                self.Cx[i,j] = torch.sin((i+1)*self.z[j])

        self.Cx_inv = torch.inverse(self.Cx)

        # To compute second spatial derivative in the domain frequency, multiply hu @ D2.
        # The reason is that hu is implemented as a [t,x] tensor. Then, multiplying 
        # D2 on the right multiplies each hu[t_i,x_j] by the corresponding omega_j.

        self.D2 = torch.zeros((self.M,self.M),device=self.device)
        for i in range(self.M):
            self.D2[i,i] = -(torch.pi / (self.b-self.a))**2 * (i+1)**2
        self.omegas = torch.diagonal(self.D2)

        self.to(device)

    
    def dbt(self,u):
        u_hat = u @ self.Cx_inv
        return u_hat

    def idbt(self,u_hat):
        u = u_hat @ self.Cx
        return u

    def to(self,device):
        self.Cx = self.Cx.to(device)
        self.Cx_inv = self.Cx_inv.to(device)
        self.D2 = self.D2.to(device)
        self.device = device

        return self

    def spectral_interpolation(self, u_hat, x):
        u = torch.zeros(list(u_hat.shape[:-1]) + [len(x)], device=self.device)
        x_scaled = (x-self.a)/(self.b-self.a) * torch.pi
        for k in range(self.M):
            u += u_hat[...,k:k+1] * torch.sin((k + 1) * x_scaled)
        return u