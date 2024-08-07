'''
GNN model for graph matching
'''


import torch
from torch.nn import Sequential, Linear
from torch_geometric.nn.inits import reset
from scipy.optimize import linear_sum_assignment

try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None


def masked_softmax(source):
    source_max1 = source - torch.max(source, dim=1, keepdim=True)[0]
    output1 = torch.softmax(source_max1, dim=1)

    source_max2 = source - torch.max(source, dim=0, keepdim=True)[0]
    output2 = torch.softmax(source_max2, dim=0)
    
    return (output1 + output2) / 2





class GM_GNN(torch.nn.Module):

    def __init__(self, num_layers, hidden):
        super(GM_GNN, self).__init__()
        self.hidden = hidden
        self.num_layers = num_layers

        self.mlp = torch.nn.ModuleList([Sequential(Linear(1, hidden-1))])
        self.readout = torch.nn.ModuleList([Sequential(Linear(1, 1))])

        for i in range(1, num_layers):
            self.mlp.append(Sequential(Linear(hidden, hidden-1),))
            self.readout.append(Sequential(Linear(hidden, 1),))
    
    def reset_parameter(self):
        for i in range(self.num_layers):
            reset(self.mlp[i])
            reset(self.readout[i])
    
    def forward(self, G1, G2, seeds, noisy=False):
        Y_total = []
        n1 = G1.shape[0]
        n2 = G2.shape[0]

        Seeds = torch.zeros([n1, n2])
        Seeds[seeds[0], seeds[1]] = 1
        S = Seeds.unsqueeze(-1)

        for layer_i in range(self.num_layers):
            H = torch.einsum("abh,bc->ach", torch.einsum("ij,jkh->ikh", G1, S), G2)
            if layer_i < self.num_layers - 1:
                X = self.mlp[layer_i](H) / 1000

            Match = self.readout[layer_i](H).squeeze(-1)
            Match_norm = masked_softmax(Match)
            Match_norm[seeds[0],:] = 0
            Match_norm[:,seeds[1]] = 0
            Match_norm[seeds[0],seeds[1]] = 1
            Y_total.append(Match_norm)

            Match_n = Match_norm.detach().numpy()
            row, col = linear_sum_assignment(-Match_n)
            NewSeeds = torch.zeros([n1, n2])
            NewSeeds[row, col] = 10

            Z = (Match_norm * NewSeeds).unsqueeze(-1)

            S = torch.cat([X, Z], dim=2)
        
        return Y_total[-1], Y_total
    
    def loss(self, S, y):
        # Negative Log Likelihood Loss
        nll = 0
        epsilon = 1e-12
        k = 1
        for S_i in S:
            val = S_i[y[0], y[1]]
            nll += torch.sum(-torch.log(val + epsilon))
        return nll
    
    def accuracy(self, S, y):
        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        prediction = torch.tensor(col)

        correct_match = sum(prediction[y[0]] == y[1])
        return correct_match





