from models.hadamard_layer import HadamardLayer
import torch
import torch.nn as nn

from utilities.tensors import format_Fourier_vector
from utilities.tensors import unformat_Fourier_vector

class GTNonLinearVectorFieldHeat(nn.Module):
        def __init__(self, basis,kappa,kappa_0):
            super().__init__()
            self.basis = basis
            self.x = basis.x
            self.kappa = kappa
            self.kappa_0 = kappa_0


        def forward(self, x):
            x_xx = x @ self.basis.D2

            return self.basis.dbt((self.kappa - self.kappa_0) * self.basis.idbt(x_xx))
        
class ETDHeatLikeVectorField(nn.Module):
    def __init__(self, L, N, factor=1.0):
        super().__init__()
        self.L = torch.nn.Parameter(L, requires_grad=False)
        self.N = N
        self.factor = factor

    def forward(self, y):
        # linear_part = torch.einsum('ijk,pkj->pij', self.L, y)
        linear_part = self.L * y
        nonlinear_part = self.factor * self.N(y)
        return linear_part + nonlinear_part  

class HadamardVectorField(nn.Module):
    '''General vector field for first-order in time integration in which the linear field is an Hadamard product with L'''
    def __init__(self, L, N, factor=1.0):
        super().__init__()
        self.L = L
        self.N = N
        self.factor = factor

    def forward(self, y):
        # linear_part = torch.einsum('ijk,pkj->pij', self.L, y)
        linear_part = self.L * y
        nonlinear_part = self.factor * self.N(y)
        return linear_part + nonlinear_part  

class NonLinearHeatEquationVectorField(nn.Module):
    def __init__(self, basis, kappa, kappa_0, device):
        super().__init__()
        self.basis = basis.to(device)
        self.Delta = basis.Delta.to(device)
        self.kappa = kappa.to(device)
        self.kappa_0 = kappa_0.to(device)
    
    def forward(self, hu):
        # Get uhat
        # Compute laplacian
        hu_xx = self.Delta * hu
        u_xx = self.basis.idbt(hu_xx)
        hku_xx = self.basis.dbt((self.kappa-self.kappa_0)*u_xx)

        return hku_xx

class SecondOrderStackModel(nn.Module):
    '''Stacks hu and huv in a new dimension'''
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        hu, _ = x.chunk(2, dim=-1)
        hu = hu.squeeze(-1)
        return torch.stack((torch.zeros_like(hu), self.model(hu)), dim=-1)
    
class GTNonLinearVectorFieldBurgers(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.basis = basis
        self.x = basis.x

    def forward(self, hu):
        
        hu_x = hu @ self.basis.D
        u = self.basis.idbt(hu)
        u_x = self.basis.idbt(hu_x)

        return -self.basis.dbt(u * u_x)

class GTNonLinearVectorFieldKDV(nn.Module):
    def __init__(self, basis, alpha):
        super().__init__()
        self.basis = basis
        self.x = basis.x
        self.alpha = alpha

    def forward(self, hu):
        
        hu_x = hu @ self.basis.D
        u = self.basis.idbt(hu)
        u_x = self.basis.idbt(hu_x)

        return self.alpha*self.basis.dbt(u * u_x)

class KdVDiagonalVectorField(nn.Module):
    ''' Works by writing the Fourier basis vector as a stack of 2-vectors (cosine and sine), and applying L as a matrix'''
    def __init__(self, L ,N, basis, factor=1.0):
        super().__init__()
        self.L = torch.nn.Parameter(L,requires_grad=False)
        self.N = N
        self.factor = factor
        self.basis = basis

    def forward(self, y):
        '''Formats the (...,M) input vector hu as (...,N_cos,2), then applies L (N_cos,2,2) on it and formats back to the original shape'''
        y_formatted = format_Fourier_vector(y, self.basis).unsqueeze(-1)
        Ly_formatted = (self.L @ y_formatted).squeeze(-1)
        Ly = unformat_Fourier_vector(Ly_formatted, self.basis)

        Ny = self.N(y)

        return Ly + self.factor * Ny
