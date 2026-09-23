import torch
from torchdyn.numerics.solvers.templates import DiffEqSolver
from utilities.tensors import list_to_tensor, format_Fourier_vector, unformat_Fourier_vector


class BaseETDSolver(DiffEqSolver):
    '''Superclass for all exponential numerical solvers'''
    pass

class ETD4Hadamard(BaseETDSolver):
    """This is for the case where the linear operator L acts by Hadamard product"""
    """L CANNOT have a zero entry"""

    def __init__(self, model, h):
        
        super().__init__(order=1)
        self.stepping_class = 'fixed'
        self.L_shape = model.L.shape
        self.N = model.N
        self.h = h
        self.factor = model.factor
        self.set_L(model.L)

    # ---- φ functions ----
    def varphi1(self, z):
        return (torch.exp(z)-1)/z

    def varphi2(self, z):
        return (torch.exp(z)-1-z)/z**2

    def varphi3(self, z):
        return (torch.exp(z)-1-z-0.5*z**2)/z**3

    def set_L(self, L):
        assert self.L_shape == L.shape, "The new L must have the same shape as the original L"
        self.L = L
        self.z = self.L * self.h
        self.Eh2 = torch.exp(self.L * (0.5 * self.h))
        self.Eh  = torch.exp(self.L * self.h)

        # Precompute φ-matrices
        self.phi1 = self.varphi1(self.z)
        self.phi2 = self.varphi2(self.z)
        self.phi3 = self.varphi3(self.z)

        # Coefficients (matrix versions)
        self.b1 = ( self.phi1 - 3*self.phi2 + 4*self.phi3 )
        self.b2 = 2*( self.phi2 - 2*self.phi3 )
        self.b3 = 2*( self.phi2 - 2*self.phi3 )
        self.b4 = ( -self.phi2 + 4*self.phi3 )

        self.z_h2 = self.L * (0.5 * self.h)


        self.varphi1_h = self.varphi1(self.z)
        self.varphi1_h2 = self.varphi1(self.z_h2)

    def step(self, f, x, t, dt, k1=None, args=None):
        Nn = self.factor*self.N(x)
        a = self.Eh2 * x + 0.5*self.h*self.varphi1_h2 * Nn
        Na = self.factor*self.N(a)
        b = self.Eh2 * x + 0.5*self.h*self.varphi1_h2 * Na
        Nb = self.factor*self.N(b)
        c = self.Eh2 * a + 0.5*self.h*self.varphi1_h2 * (2*Nb - Nn)  #version by Cox, Matthews
        Nc = self.factor*self.N(c)
        

        x_sol = self.Eh * x + self.h * (
                self.b1 * Nn +
                self.b2 * Na +
                self.b3 * Nb +
                self.b4 * Nc
                )

        return None, x_sol, None

class ETD4HadamardNonInvertibleL(BaseETDSolver):
    """This is for the case where the linear operator L acts by Hadamard product.
    L can be of any dimension, but L[0...0] must be equal to zero.
    In our experiments, this is the case with the cosine and Fourier bases (for any dimension)"""

    def __init__(self, model, h):
        
        super().__init__(order=1)
        self.stepping_class = 'fixed'
        self.dim = len(model.L.shape)
        self.L_shape = model.L.shape
        self.N = model.N
        self.factor = model.factor
        self.h = h
        # # Setting self.zero_elem_mask to be 1 (True) only on the entries where L is zero
        self.zero_elem_mask = (model.L == 0).to(dtype=bool, device=model.L.device)
        self.set_L(model.L)

    # ---- φ functions ----
    def varphi1(self, z):
        z = z + self.zero_elem_mask
        var = (torch.exp(z)-1)/z
        var = var * (~self.zero_elem_mask) + self.zero_elem_mask*1.  # Forces the zero elements to be equal to 1.
        return var

    def varphi2(self, z):
        z = z + self.zero_elem_mask
        var = (torch.exp(z)-1-z)/z**2
        var = var * (~self.zero_elem_mask) + self.zero_elem_mask*1.  # Forces the zero elements to be equal to 1.
        return var

    def varphi3(self, z):
        z = z + self.zero_elem_mask
        var = (torch.exp(z)-1-z-0.5*z**2)/z**3
        var = var * (~self.zero_elem_mask) + self.zero_elem_mask*1.  # Forces the zero elements to be equal to 1.
        return var

    def step(self, f, x, t, dt, k1=None, args=None):
        Nn = self.factor*self.N(x)
        a = self.Eh2 * x + 0.5*self.h*self.varphi1_h2 * Nn
        Na = self.factor*self.N(a)
        b = self.Eh2 * x + 0.5*self.h*self.varphi1_h2 * Na
        Nb = self.factor*self.N(b)
        c = self.Eh2 * a + 0.5*self.h*self.varphi1_h2 * (2*Nb - Nn)  #version by Cox, Matthews
        Nc = self.factor*self.N(c)

        x_sol = self.Eh * x + self.h * (
                self.b1 * Nn +
                self.b2 * Na +
                self.b3 * Nb +
                self.b4 * Nc
                )

        return None, x_sol, None

    def set_L(self, L):
        assert L.shape == self.L_shape
        assert torch.all(L[self.zero_elem_mask] == 0)
        self.L = L
        self.z = self.L * self.h
        self.Eh2 = torch.exp(self.L * (0.5 * self.h))
        self.Eh  = torch.exp(self.L * self.h)

        # Precompute φ-vectors (or matrices, if 2D)
        self.phi1 = self.varphi1(self.z)
        self.phi2 = self.varphi2(self.z)
        self.phi3 = self.varphi3(self.z)

        # Coefficients
        self.b1 = ( self.phi1 - 3*self.phi2 + 4*self.phi3 )
        self.b2 = 2*( self.phi2 - 2*self.phi3 )
        self.b3 = self.b2
        self.b4 = ( -self.phi2 + 4*self.phi3 )

        self.z_h2 = self.L * (0.5 * self.h)
        self.varphi1_h = self.varphi1(self.z)
        self.varphi1_h2 = self.varphi1(self.z_h2)

class ETD4HadamardInvertibleL(BaseETDSolver):
    """This is for the case where the linear operator L acts by Hadamard product.
    L can be of any dimension, but L[0...0] must be equal to zero.
    In our experiments, this is the case with the cosine and Fourier bases (for any dimension)"""

    def __init__(self, model, h):
        
        super().__init__(order=1)
        self.stepping_class = 'fixed'
        self.dim = len(model.L.shape)
        self.L_shape = model.L.shape
        self.N = model.N
        self.factor = model.factor
        self.h = h
        # Setting self.first_elem_mask to be a tensor that is all zeros, except for the very first entry, which is 1/True
        self.first_elem_mask = torch.zeros(self.L_shape, dtype=bool, device=model.L.device)
        zero_slice = tuple([0 for _ in range(self.dim)])
        self.first_elem_mask[zero_slice] = True
        self.set_L(model.L)

    # ---- φ functions ----
    def varphi1(self, z):
        var = (torch.exp(z)-1)/z
        return var

    def varphi2(self, z):
        var = (torch.exp(z)-1-z)/z**2
        return var

    def varphi3(self, z):
        var = (torch.exp(z)-1-z-0.5*z**2)/z**3
        return var

    def step(self, f, x, t, dt, k1=None, args=None):
        Nn = self.factor*self.N(x)
        a = self.Eh2 * x + 0.5*self.h*self.varphi1_h2 * Nn
        Na = self.factor*self.N(a)
        b = self.Eh2 * x + 0.5*self.h*self.varphi1_h2 * Na
        Nb = self.factor*self.N(b)
        c = self.Eh2 * a + 0.5*self.h*self.varphi1_h2 * (2*Nb - Nn)  #version by Cox, Matthews
        Nc = self.factor*self.N(c)

        x_sol = self.Eh * x + self.h * (
                self.b1 * Nn +
                self.b2 * Na +
                self.b3 * Nb +
                self.b4 * Nc
                )

        return None, x_sol, None

    def set_L(self, L):
        assert L.shape == self.L_shape
        assert L[self.first_elem_mask].item() == 0
        self.L = L
        self.z = self.L * self.h
        self.Eh2 = torch.exp(self.L * (0.5 * self.h))
        self.Eh  = torch.exp(self.L * self.h)

        # Precompute φ-vectors (or matrices, if 2D)
        self.phi1 = self.varphi1(self.z)
        self.phi2 = self.varphi2(self.z)
        self.phi3 = self.varphi3(self.z)

        # Coefficients
        self.b1 = ( self.phi1 - 3*self.phi2 + 4*self.phi3 )
        self.b2 = 2*( self.phi2 - 2*self.phi3 )
        self.b3 = self.b2
        self.b4 = ( -self.phi2 + 4*self.phi3 )

        self.z_h2 = self.L * (0.5 * self.h)
        self.varphi1_h = self.varphi1(self.z)
        self.varphi1_h2 = self.varphi1(self.z_h2)

class ETDEulerHadamard(BaseETDSolver):
    """ETD-Euler for Hadamard (elementwise) linear operator"""

    def __init__(self, model, h):
        super().__init__(order=1)

        self.stepping_class = 'fixed'
        self.N = model.N
        self.h = h
        self.L_shape = model.L.shape
        self.factor = model.factor

        self.set_L(model.L)

    # ---- φ functions ----
    def varphi1(self, z):
        return (torch.exp(z) - 1) / z

    def set_L(self, L):
        assert L.shape == self.L_shape
        self.L = L
        self.z = self.L * self.h
        self.Eh = torch.exp(self.z)
        self.phi1 = self.varphi1(self.z)

    def step(self, f, x, t, dt, k1=None, args=None):
        Nn = self.factor * self.N(x)
        x_next = self.Eh * x + self.h * (self.phi1 * Nn)
        return None, x_next, None

class ETDEulerHadamardNonInvertibleL(BaseETDSolver):
    """ETD-Euler for Hadamard (elementwise) linear operator"""

    def __init__(self, model, h):
        super().__init__(order=1)

        self.stepping_class = 'fixed'
        self.N = model.N
        self.h = h
        self.L_shape = model.L.shape
        self.dim = len(self.L_shape)
        self.factor = model.factor
        # # Setting self.zero_elem_mask to be 1 (True) only on the entries where L is zero
        self.zero_elem_mask = (model.L == 0).to(dtype=bool, device=model.L.device)
        self.set_L(model.L)

    # ---- φ functions ----
    def varphi1(self, z):
        z = z + self.zero_elem_mask
        var = (torch.exp(z)-1)/z
        var = var * (~self.zero_elem_mask) + self.zero_elem_mask*1.  # Forces the zeroth element to be equal to 1.
        return var

    def set_L(self, L):
        assert L.shape == self.L_shape
        assert torch.all(L[self.zero_elem_mask] == 0)
        self.L = L
        self.z = self.L * self.h
        self.Eh = torch.exp(self.z)
        self.phi1 = self.varphi1(self.z)

    def step(self, f, x, t, dt, k1=None, args=None):
        Nn = self.factor * self.N(x)
        x_next = self.Eh * x + self.h * (self.phi1 * Nn)
        return None, x_next, None

class ETDEulerSecondDerivative(BaseETDSolver):
    '''ETD integrator for problems where the second derivative of u is written as L*u + N(u), 
    where L is linear and thus becomes an Hadamard product in phase space
    We define the state vector as [hu, hv]^T, whose derivative in the linear case is [hv, c*hu]^T
    c is assumed to be a vector of same shape as hu and hv. The integrator will receive (hu,hv) stacked in (n,2) shape
    L should be an (n,2,2) tensor, where each slice is of the form [[0,1], [c_i², 0]]'''
    def __init__(self, model, h):
        super().__init__(order=1)
        self.stepping_class = 'fixed'
        self.dim = len(model.L.shape)
        self.L_shape = model.L.shape
        self.N = model.N
        self.factor = model.factor
        self.h = h
        self.set_L(model.L)

    def exponential(self, L, h):
        c2 = - L[..., 1, 0]
        c = torch.sqrt(c2)
        Eh_00 = torch.cos(h * c)
        Eh_01 = torch.sin(h * c) / c
        Eh_10 = - c * torch.sin(h * c)
        Eh_11 = Eh_00
        Eh = list_to_tensor([[Eh_00, Eh_01],
                             [Eh_10, Eh_11]])
        return Eh

    def varphi1(self, L, h):
        c2 = - L[..., 1, 0]
        c = torch.sqrt(c2)
        phi_00 = torch.sin(h * c) / (h * c)
        phi_01 = (1 - torch.cos(h * c)) / (h * c2)
        phi_10 = (torch.cos(h * c) - 1) / h
        phi_11 = phi_00
        phi = list_to_tensor([[phi_00, phi_01],
                              [phi_10, phi_11]])        
        return phi

    def set_L(self, L):
        assert L.shape == self.L_shape
        assert torch.all(L[..., 0, 0] == 0)
        assert torch.all(L[..., 0, 1] == 1)
        assert torch.all(L[..., 1, 0] <= 0)
        assert torch.all(L[..., 1, 1] == 0)
        self.L = L
        self.Eh = self.exponential(L, self.h)
        self.phi1 = self.varphi1(L, self.h)

    def step(self, f, x, t, dt, k1=None, args=None):
        Nn = self.factor * self.N(x)
        x_next = self.Eh @ x.unsqueeze(-1) + self.h * (self.phi1 @ Nn.unsqueeze(-1))
        x_next = x_next.squeeze(-1)
        return None, x_next, None

class ETD4FourierBasisKdV(BaseETDSolver):
    '''Implements ETD for the Fourier bais, where the sine and cosine components for each frequency are a 2-vector,
    and L acts by matrix multiplication on each of these 2-vectors'''
    def __init__(self, model, h):
        super().__init__(order=1)
        self.stepping_class = 'fixed'
        self.dim = len(model.L.shape)
        self.L_shape = model.L.shape
        self.N = model.N
        self.factor = model.factor
        self.h = h
        self.zero_elem_mask = torch.all(model.L == 0, dim=(-2,-1), keepdim=True).to(dtype=bool, device=model.L.device)
        self.zero_elem_ones = self.zero_elem_mask * torch.ones_like(model.L)
        self.basis = model.basis
        self.set_L(model.L)

    def exponential(self, z):
        # Here, z is assumed to be z=hL
        gamma = z[..., 0, 1]
        Eh_00 = torch.cos(gamma)
        Eh_01 = torch.sin(gamma)
        Eh_10 = -Eh_01
        Eh_11 = Eh_00
        Eh = list_to_tensor([[Eh_00, Eh_01],
                             [Eh_10, Eh_11]])
        return Eh

    def varphi1(self, z):
        z = z + self.zero_elem_ones
        gamma = z[..., 0, 1]
        phi_00 = torch.sin(gamma) / (gamma)
        phi_01 = (1 - torch.cos(gamma)) / (gamma)
        phi_10 = -phi_01
        phi_11 = phi_00
        phi = list_to_tensor([[phi_00, phi_01],
                              [phi_10, phi_11]])
        zero_elem_phi = torch.tensor([[1., 0], [0, 1.]], device=phi.device)
        phi = phi - phi*self.zero_elem_mask + self.zero_elem_ones * zero_elem_phi
        return phi
    
    def varphi2(self, z):
        z = z + self.zero_elem_ones
        gamma = z[..., 0, 1]
        phi_00 = (1 - torch.cos(gamma)) / (gamma**2)
        phi_01 = (gamma - torch.sin(gamma)) / (gamma**2)
        phi_10 = -phi_01
        phi_11 = phi_00
        phi = list_to_tensor([[phi_00, phi_01],
                              [phi_10, phi_11]])
        zero_elem_phi = torch.tensor([[1./2, 0], [0,1/2.]], device=phi.device)
        phi = phi - phi*self.zero_elem_mask + self.zero_elem_ones * zero_elem_phi
        return phi

    def varphi3(self, z):
        z = z + self.zero_elem_ones
        gamma = z[..., 0, 1]
        phi_00 = (gamma - torch.sin(gamma)) / (gamma**3)
        phi_01 = (torch.cos(gamma) - 1 + gamma**2 / 2) / (gamma**3)
        phi_10 = -phi_01
        phi_11 = phi_00
        phi = list_to_tensor([[phi_00, phi_01],
                              [phi_10, phi_11]])
        zero_elem_phi = torch.tensor([[1./6, 0],[0, 1./6]], device=phi.device)
        phi = phi - phi*self.zero_elem_mask + self.zero_elem_ones * zero_elem_phi
        return phi

    def set_L(self, L):
        assert L.shape == self.L_shape
        assert torch.all(L[..., 0, 0] == 0)
        assert torch.all(L[..., 1, 1] == 0)
        assert torch.all(L[..., 0, 1] == -L[..., 1, 0])

        self.L = L
        z = L * self.h
        self.Eh = self.exponential(z)
        self.Eh2 = self.exponential(z / 2)
        self.phi1 = self.varphi1(z)
        self.phi2 = self.varphi2(z)
        self.phi3 = self.varphi3(z)
        self.varphi1_h = self.phi1
        self.varphi1_h2 = self.varphi1(z / 2)

        # Coefficients
        self.b1 = ( self.phi1 - 3*self.phi2 + 4*self.phi3 )
        self.b2 = 2*( self.phi2 - 2*self.phi3 )
        self.b3 = self.b2
        self.b4 = ( -self.phi2 + 4*self.phi3 )

    def step(self, f, x, t, dt, k1=None, args=None):
        x0 = format_Fourier_vector(x, self.basis).unsqueeze(-1)
        # Nn = self.factor*self.N(x).unsqueeze(-1)
        Nn = format_Fourier_vector(self.factor*self.N(x), self.basis).unsqueeze(-1)
        a = self.Eh2 @ x0 + 0.5*self.h*self.varphi1_h2 @ Nn
        a_unf = unformat_Fourier_vector(a.squeeze(-1), self.basis)
        Na = format_Fourier_vector(self.factor*self.N(a_unf), self.basis).unsqueeze(-1)
        b = self.Eh2 @ x0 + 0.5*self.h*self.varphi1_h2 @ Na
        b_unf = unformat_Fourier_vector(b.squeeze(-1), self.basis)
        Nb = format_Fourier_vector(self.factor*self.N(b_unf), self.basis).unsqueeze(-1)
        c = self.Eh2 @ a + 0.5*self.h*self.varphi1_h2 @ (2*Nb - Nn)
        c_unf = unformat_Fourier_vector(c.squeeze(-1), self.basis)
        Nc = format_Fourier_vector(self.factor*self.N(c_unf), self.basis).unsqueeze(-1)

        x_sol = self.Eh @ x0 + self.h * (
                self.b1 @ Nn +
                self.b2 @ Na +
                self.b3 @ Nb +
                self.b4 @ Nc
                )
        x_sol = unformat_Fourier_vector(x_sol.squeeze(-1), self.basis)
        return None, x_sol, None

class ETDEulerFourierBasisKdV(ETD4FourierBasisKdV):

    def set_L(self, L):
        assert L.shape == self.L_shape
        assert torch.all(L[..., 0, 0] == 0)
        assert torch.all(L[..., 1, 1] == 0)
        assert torch.all(L[..., 0, 1] == -L[..., 1, 0])

        self.L = L
        self.z = self.L * self.h
        self.Eh = self.exponential(self.z)
        self.phi1 = self.varphi1(self.z)

    def step(self, f, x, t, dt, k1=None, args=None):
        Nn = format_Fourier_vector(self.factor * self.N(x), self.basis).unsqueeze(-1)
        x0 = format_Fourier_vector(x, self.basis).unsqueeze(-1)
        x_next = self.Eh @ x0 + self.h * (self.phi1 @ Nn)
        x_next = unformat_Fourier_vector(x_next.squeeze(-1), self.basis)
        return None, x_next, None

class ETD4SecondDerivative(BaseETDSolver):
    '''Same as ETDEulerSecondDerivative, but uses ETDRK4 (Cox, Matthews) fourht-order stepping scheme'''
    def __init__(self, model, h):
        super().__init__(order=1)
        self.stepping_class = 'fixed'
        self.dim = len(model.L.shape)
        self.L_shape = model.L.shape
        self.N = model.N
        self.factor = model.factor
        self.h = h
        self.set_L(model.L)

    def exponential(self, L, h):
        c2 = - L[..., 1, 0]
        c = torch.sqrt(c2)
        Eh_00 = torch.cos(h * c)
        Eh_01 = torch.sin(h * c) / c
        Eh_10 = - c * torch.sin(h * c)
        Eh_11 = Eh_00
        Eh = list_to_tensor([[Eh_00, Eh_01],
                             [Eh_10, Eh_11]])
        return Eh

    def varphi1(self, L, h):
        c2 = - L[..., 1, 0]
        c = torch.sqrt(c2)
        phi_00 = torch.sin(h * c) / (h * c)
        phi_01 = (1 - torch.cos(h * c)) / (h * c2)
        phi_10 = (torch.cos(h * c) - 1) / h
        phi_11 = phi_00
        phi = list_to_tensor([[phi_00, phi_01],
                              [phi_10, phi_11]])        
        return phi
    
    def varphi2(self, L, h):
        c2 = - L[..., 1, 0]
        c = torch.sqrt(c2)
        phi_00 = (1 - torch.cos(h*c)) / (h**2 * c**2)
        phi_01 = (h*c - torch.sin(h*c)) / (h**2 * c**3)
        phi_10 = (torch.sin(h*c) - h*c) / (h**2 * c)
        phi_11 = phi_00
        phi = list_to_tensor([[phi_00, phi_01],
                              [phi_10, phi_11]])        
        return phi

    def varphi3(self, L, h):
        c2 = - L[..., 1, 0]
        c = torch.sqrt(c2)
        phi_00 = (h*c - torch.sin(h*c)) / (h**3 * c**3)
        phi_01 = (torch.cos(h*c) - 1 + h**2 * c**2 / 2) / (h**3 * c**4)
        phi_10 = (1 - h**2 * c**2 / 2 - torch.cos(h*c)) / (h**3 * c**2)
        phi_11 = phi_00
        phi = list_to_tensor([[phi_00, phi_01],
                              [phi_10, phi_11]])        
        return phi

    def set_L(self, L):
        assert L.shape == self.L_shape
        assert torch.all(L[..., 0, 0] == 0)
        assert torch.all(L[..., 0, 1] == 1)
        assert torch.all(L[..., 1, 0] <= 0)
        assert torch.all(L[..., 1, 1] == 0)

        self.L = L
        self.Eh = self.exponential(L, self.h)
        self.Eh2 = self.exponential(L, self.h / 2)
        self.phi1 = self.varphi1(L, self.h)
        self.phi2 = self.varphi2(L, self.h)
        self.phi3 = self.varphi3(L, self.h)
        self.varphi1_h = self.phi1
        self.varphi1_h2 = self.varphi1(L, self.h / 2)

        # Coefficients
        self.b1 = ( self.phi1 - 3*self.phi2 + 4*self.phi3 )
        self.b2 = 2*( self.phi2 - 2*self.phi3 )
        self.b3 = self.b2
        self.b4 = ( -self.phi2 + 4*self.phi3 )

    def step(self, f, x, t, dt, k1=None, args=None):
        x0 = x.unsqueeze(-1)  # (N,2,1)
        Nn = self.factor*self.N(x).unsqueeze(-1)  # (N,2,1)
        # self.Eh and others are (N,2,2)
        a = self.Eh2 @ x0 + 0.5*self.h*self.varphi1_h2 @ Nn
        Na = self.factor*self.N(a.squeeze(-1)).unsqueeze(-1)
        b = self.Eh2 @ x0 + 0.5*self.h*self.varphi1_h2 @ Na
        Nb = self.factor*self.N(b.squeeze(-1)).unsqueeze(-1)
        c = self.Eh2 @ a + 0.5*self.h*self.varphi1_h2 @ (2*Nb - Nn)
        Nc = self.factor*self.N(c.squeeze(-1)).unsqueeze(-1)

        x_sol = self.Eh @ x0 + self.h * (
                self.b1 @ Nn +
                self.b2 @ Na +
                self.b3 @ Nb +
                self.b4 @ Nc
                )
        x_sol = x_sol.squeeze(-1)
        return None, x_sol, None