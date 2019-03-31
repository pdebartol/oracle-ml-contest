import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_mean
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch.nn.init import xavier_uniform_


class SAGEConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, normalize=True, bias=False):
        super(SAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.weight)

    def forward(self, x, edge_index):

        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        row, col = edge_index

        x = torch.matmul(x, self.weight)
        out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGE(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x,edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
