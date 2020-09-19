import torch.nn as nn

class MLP(nn.Module):
    '''the MLP Layer of Biaffine Parser, in fact, it is a Linear Layer'''

    def __init__(self, n_in, n_hidden, dropout=0):
        super(MLP, self).__init__()

        # Liear Layer
        self.linear = nn.Linear(n_in, n_hidden)
        # Activation
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        # TODO dropout

    def forward(self, x):

        # (batch, sentence_len, n_in) -> (batch, sentence_len, n_out)
        x = self.linear(x) 
        x = self.activation(x)
        # TODO dropout

        return x
