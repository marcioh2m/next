import torch


def gaussian(xs,ys,x0,y0):
    # xs and ys have shape [n_x,n_y,1]
    # the output has shape [n_x,n_y]

    return torch.exp(-((xs-x0)**2+(ys-y0)**2)/(2*0.1**2)).squeeze(-1)