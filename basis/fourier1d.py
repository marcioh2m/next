import torch

class FourierBasis1D:
    def __init__(self, M, domain, device="cpu"):

        self.M = M  # number of points in the discretization of the physical domain
        self.K = self.M // 2
        ## Number of frequencies for sine and cosine
        self.N_cos = self.K + 1
        self.N_sin = self.M - self.N_cos
        self.device = device

        # # cosine frequencies are 0 to N_cos-1, and sine frequencies are 1 to N_sin
        self.idx_freqs_cos = torch.arange(0, self.N_cos)
        self.idx_freqs_sin = torch.arange(1, self.N_sin + 1)
        self.idx_freqs = torch.arange(0, self.M)

        self.z = torch.linspace(0, 2 * torch.pi, self.M + 1, device=device)[:-1]
        # (0,2pi/M,4pi/M,...,2pi(M-1)/M)

        self.a, self.b = domain

        self.x = (self.b - self.a) * self.z / (2*torch.pi) + self.a
        self.jacobian = 2 * torch.pi / (self.b - self.a)

        self.full_x = torch.cat([self.x, torch.tensor([self.b],device=self.x.device)])

        self.Cx = torch.zeros((self.M,self.M),device=self.device)
        for i in range(self.N_cos):
            for j in range(self.M):
                self.Cx[i,j] = torch.cos(i * self.z[j])
        
        for i in range(self.N_cos,self.M):
            for j in range(self.M):
                self.Cx[i,j] = torch.sin((i + 1 - self.N_cos) * self.z[j])

        
        self.Cx_inv = torch.inverse(self.Cx)

        # To derivate, multiply D ON THE RIGHT of u_hat.
        self.D = self.jacobian * self.differentiation_matrix()
        
        # Second derivative can be obtained by multiplying D2 below
        # ON THE RIGHT of u_hat, but in practice we use a Hadamard product
        # with the diagonal elements of D2
        self.D2 = self.D @ self.D
        self.omega2 = torch.diagonal(self.D2)
        # REVIEW We really shouldn't have a variable called omegas when it actually represents omega^2

        self.to(device)


    def dbt(self,u):
        u_hat = u @ self.Cx_inv
        return u_hat

    def idbt(self,u_hat):
        u = u_hat @ self.Cx
        return u

    def differentiation_matrix(self):

        diff_matrix = torch.zeros(self.M, self.M)
        omega_matrix = torch.zeros(self.N_sin, self.N_sin)
        for i in range(1, self.N_sin + 1):
            omega_matrix[i-1, i-1] = -i
        diff_matrix[1:self.N_sin+1, self.N_cos:self.N_cos+self.N_sin] = omega_matrix
        diff_matrix[self.N_cos:self.N_cos+self.N_sin, 1:self.N_sin+1] = -omega_matrix

        # If M is even, the last cosine term has no matching sine, because sin(Kz)=0 on the entire grid z
        # without a matching sine term. Its derivative is deliberately left as 0.

        return diff_matrix

    def to(self,device):
        self.Cx = self.Cx.to(device)
        self.Cx_inv = self.Cx_inv.to(device)
        self.D = self.D.to(device)
        self.D2 = self.D2.to(device)
        self.omega2 = self.omega2.to(device)
        self.device = device
        
        return self

    def spectral_interpolation(self, u_hat, x):
        u = torch.zeros_like(x)
        x_scaled = (x-self.a) * 2 * torch.pi / (self.b - self.a)
        for k in range(self.M):
            if k < self.N:
                u += u_hat[k] * torch.cos(k * x_scaled)
            else:
                freq = k - self.N + 1
                u += u_hat[k] * torch.sin(freq * x_scaled)
        return u
