import torch
import torch.nn as nn

class Biaffine(nn.Module):
    '''the Biaffine Layer of Biaffine Parser'''

    def __init__(self, n_in, n_out=1):
        super(Biaffine, self).__init__()

        # TODO why need n_out?
        # stach weight and bias together
        self.weight = nn.Parameter(torch.Tensor(n_in + 1,
                                                n_in))
        
    def forward(self, x, y):

        # add 1 to the left of x
        x = torch.cat((x, torch.ones_like(x[..., :1])), -1)

        # bilinear
        # TODO handle rels
        s = torch.einsum('bxi,ij,byj->bxy', x, self.weight, y)

        return s
