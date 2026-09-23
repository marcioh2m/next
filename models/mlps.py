import torch.nn as nn

class TransposeLayer2D(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        # (a,b) ---> (b,a)
        x = x.transpose(-2,-1)
        return x

class MLP:
    def __new__(
        cls,
        in_dim,
        hidden_dim,
        out_dim,
        num_hidden_layers,
        activation=nn.LeakyReLU(),
        bias=True,
    ) -> nn.Sequential:

        layers = []

        layers += [nn.Linear(in_dim,hidden_dim,bias=bias), activation]

        for _ in range(num_hidden_layers-1):
            layers += [nn.Linear(hidden_dim,hidden_dim,bias=bias), activation]

        layers += [nn.Linear(hidden_dim,out_dim,bias=bias)]

        return nn.Sequential(*layers)

class MLPWithTranspose2D:
    # usually, the input of an MLP is a matrix (M,N), where each line
    # corresponds to, e.g., a grid point. Thus, the input lines are independent, in some sense.
    # However, when the input is a 2D grid (M,N), then the lines are not independent.
    # A usual MLP would not be fully connected, in this case. To see this, suppose that the first layer
    # is (N,P). Then, this layer performs a matrix multiplication (M,N)x(N,P), where the first line of the input
    # only "appears" in the first line of the output.
    # to overcome this problem, we insert layers which transpose the outputs of the respective previous layers.
    
    def __new__(
        cls,
        M,
        N,
        widen_factor=2,
        widths=None,
        activation=nn.LeakyReLU(),
        bias=False,
        num_additional_hidden_layers=0,
        additional_hidden_layer_bias=True,
    ) -> nn.Sequential:
        # widths and widen_factor should not be passed simultaneously. If widths is passed, then widen_factor will be ignored
        # widths should be a tuple specifying the width for each dimension in latent space

        # the input is an (M,N) matrix

        if widths is None:
            widths = (int(widen_factor * M), int(widen_factor * N))
        else:
            assert len(widths)==2 and isinstance(widths[0], int) and isinstance(widths[1], int)
        
        layers = []
                                                                            # c = widen_factor
        layers += [nn.Linear(N,widths[1],bias=bias),TransposeLayer2D()] # (M,N) x (N,cN) ---> (M,cN) ----> (cN,M)
        layers += [nn.Linear(M,widths[0],bias=bias),TransposeLayer2D(),activation] # (cN,M) x (M,cM) ---> (cN,cM) ---> (cM,cN)

        for _ in range(num_additional_hidden_layers):
            layers += [nn.Linear(widths[1],widths[1],bias=additional_hidden_layer_bias)] # (cM,cN) x (cN,cN) ---> (cM,cN)
            layers += [TransposeLayer2D()]  # (cM,cN) ---> (cN,cM)
            layers += [nn.Linear(widths[0],widths[0], bias=additional_hidden_layer_bias)]# (cN,cM) × (cM,cM) ---> (cN,cM) ---> (cM,cN)
            layers += [TransposeLayer2D()]
            layers += [activation]
        
        layers += [nn.Linear(widths[1],N,bias=bias),TransposeLayer2D()] # (cM,cN) x (cN,N) ---> (cM,N) ---> (N,cM)
        layers += [nn.Linear(widths[0],M,bias=bias),TransposeLayer2D()] #  (N,cM) x (cM,M) ---> (N,M) ---> (M,N)

        return nn.Sequential(*layers)
