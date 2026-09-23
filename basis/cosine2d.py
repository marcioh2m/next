import torch

class CosineBasis2D:
    def __init__(self,M,N,domain_x,domain_y,device='cpu'):
        self.device = device
        
        self.M = M
        self.N = N
        self.a, self.b = domain_x
        self.c, self.d = domain_y

        self.jacobian_x = torch.pi/(self.b - self.a)
        self.jacobian_y = torch.pi/(self.d - self.c)

        self.idx_freqs_x = torch.arange(0,self.M)
        self.idx_freqs_y = torch.arange(0,self.N)

        self.z = (torch.pi/(self.M-1)) * self.idx_freqs_x
        self.x = ((self.b-self.a)/torch.pi) * self.z + self.a
        self.full_x = self.x

        self.w = (torch.pi/(self.N-1)) * self.idx_freqs_y
        self.y = ((self.d-self.c)/torch.pi) * self.w + self.c
        self.full_y = self.y

        self.Cx = torch.zeros((self.M,self.M))
        for i in range(self.M):
            for j in range(self.M):
                self.Cx[i,j] = torch.cos(j * self.z[i]) 

        self.Cy = torch.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.Cy[i,j] = torch.cos(j * self.w[i])
        
        self.Cx_inv = torch.inverse(self.Cx)
        self.Cy_inv = torch.inverse(self.Cy)

        # D2 is a matrix whose Hadamard product with u_hat gives the Laplacian in the frequency domain
        self.D2 = torch.zeros((self.M,self.N))
        for k in range(self.M):
            for l in range(self.N):
                self.D2[k,l] = -(k*self.jacobian_x)**2 - (l*self.jacobian_y)**2
        self.omegas = self.D2

        self.Delta = self.hadamard_differential_matrix()
        
        self.to(device)
        
    def dbt(self,u):
        u_hat = torch.einsum('in,...nm,jm->...ij',self.Cx_inv,u,self.Cy_inv)
        return u_hat

    def idbt(self,u_hat):
        u = torch.einsum('in,...nm,jm->...ij',self.Cx,u_hat,self.Cy)
        return u

    def to(self,device):
        self.Cx = self.Cx.to(device)
        self.Cx_inv = self.Cx_inv.to(device)
        self.Cy = self.Cy.to(device)
        self.Cy_inv = self.Cy_inv.to(device)
        self.D2 = self.D2.to(device)
        self.device = device

        return self
    
    def hadamard_differential_matrix(self):
        # ASSUMES THAT dims=2

        num_basis_x = self.M
        num_basis_y = self.N

        x_range = torch.arange(1, num_basis_x + 1, dtype=torch.float32) * self.jacobian_x
        y_range = torch.arange(1, num_basis_y + 1, dtype=torch.float32) * self.jacobian_y

        weights = torch.meshgrid(x_range, y_range)
        weights = torch.stack(weights, dim=-1)
        multipliers = -torch.square(weights).sum(dim=-1, keepdim=False)
        return multipliers

    def spectral_interpolation(self, u_hat, xy):
        x_scaled = (xy[..., 0] - self.a) * torch.pi / (self.b - self.a)
        y_scaled = (xy[..., 1] - self.c) * torch.pi / (self.d - self.c)
        u = torch.zeros_like(xy[..., 0])
        for i in range(self.M):
            psi_x = torch.cos(i * x_scaled)
            for j in range(self.N):
                psi_y = torch.cos(j * y_scaled)
                u += u_hat[..., i:i+1, j:j+1] * psi_x * psi_y

        return u
