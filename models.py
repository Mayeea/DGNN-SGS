import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math


def adjust_chunk_size(hidden_channel, chunk_size):
    """
    Helper function to adjust chunk_size to ensure divisibility with hidden_channel.
    When hidden_channel is not divisible by chunk_size, automatically adjusts 
    chunk_size to the largest divisor that is <= original chunk_size.
    
    Args:
        hidden_channel (int): The hidden channel dimension (F)
        chunk_size (int): The original chunk size (P)
    
    Returns:
        int: Adjusted chunk_size that divides hidden_channel
    """
    if hidden_channel % chunk_size == 0:
        return chunk_size
    
    # Find the largest divisor of hidden_channel that is <= chunk_size
    for i in range(chunk_size, 0, -1):
        if hidden_channel % i == 0:
            return i
    
    # Fallback to 1 if no divisor found (should not happen in practice)
    return 1


from mp_deterministic import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag
from torch.nn import Module, ModuleList, Linear, LayerNorm
from torch_geometric.utils import dense_to_sparse
import scipy.sparse as sp
import numpy as np
class ONGNNConv(MessagePassing):
    def __init__(self, tm_net, tm_norm, hidden_channel, chunk_size,
                 add_self_loops=True, tm=True, simple_gating=True,
                 diff_or=True):
        super(ONGNNConv, self).__init__('mean')
        self.tm_net = tm_net
        self.tm_norm = tm_norm
        self.tm = tm
        self.diff_or = diff_or
        self.simple_gating = simple_gating
        self.hidden_channel = hidden_channel
        # Automatically adjust chunk_size to ensure divisibility
        self.chunk_size = adjust_chunk_size(hidden_channel, chunk_size)
        if self.chunk_size != chunk_size:
            print(f"Warning: chunk_size adjusted from {chunk_size} to {self.chunk_size} to ensure divisibility with hidden_channel={hidden_channel}")

    def forward(self, x, edge_index, last_tm_signal):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if add_self_loops==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if add_self_loops==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        if self.tm==True:
            if self.simple_gating==True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.diff_or==True:
                    tm_signal_raw = last_tm_signal+(1-last_tm_signal)*tm_signal_raw


            tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.hidden_channel/self.chunk_size), dim=1)

            out = x*tm_signal + m*(1-tm_signal)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)


        return out, tm_signal_raw

class DGNN_SGS_Conv(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers_input=1,
                 global_gating=True, num_layers=2, dropout_rate=0.4, dropout_rate2=0.4):
        super().__init__()
        self.linear_trans_in = ModuleList()
        self.linear_trans_out = Linear(hidden_channel, out_channel)
        self.norm_input = ModuleList()
        self.convs = ModuleList()
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate = dropout_rate
        self.tm_norm = ModuleList()
        self.tm_net = ModuleList()
        # Automatically adjust chunk_size to ensure divisibility
        self.chunk_size = adjust_chunk_size(hidden_channel, 8)
        if self.chunk_size != 8:
            print(f"Warning: chunk_size adjusted from 8 to {self.chunk_size} to ensure divisibility with hidden_channel={hidden_channel}")
        self.linear_trans_in.append(Linear(in_channel, hidden_channel))

        self.norm_input.append(LayerNorm(hidden_channel))

        for i in range(num_layers_input - 1):
            self.linear_trans_in.append(Linear(hidden_channel, hidden_channel))
            self.norm_input.append(LayerNorm(hidden_channel))

        if global_gating == True:
            tm_net = Linear(2 * hidden_channel, self.chunk_size)

        for i in range(num_layers):
            self.tm_norm.append(LayerNorm(hidden_channel))

            if global_gating == False:
                self.tm_net.append(Linear(2 *hidden_channel, self.chunk_size))
            else:
                self.tm_net.append(tm_net)
            self.convs.append(ONGNNConv(tm_net=self.tm_net[i], tm_norm=self.tm_norm[i], hidden_channel=hidden_channel, chunk_size=self.chunk_size))

        self.params_conv = list(set(list(self.convs.parameters()) + list(self.tm_net.parameters())))
        self.params_others = list(self.linear_trans_in.parameters()) + list(self.linear_trans_out.parameters())

    def forward(self, x, edge_index):
        check_signal = []


        for i in range(len(self.linear_trans_in)):
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.relu(self.linear_trans_in[i](x))
            x = self.norm_input[i](x)


        tm_signal = x.new_zeros(self.chunk_size)

        for j in range(len(self.convs)):
            if self.dropout_rate2 != 'None':
                x = F.dropout(x, p=self.dropout_rate2, training=self.training)
            else:
                x = F.dropout(x, p=self.dropout_rate2, training=self.training)
            x, tm_signal = self.convs[j](x, edge_index, last_tm_signal=tm_signal)

            check_signal.append(dict(zip(['tm_signal'], [tm_signal])))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear_trans_out(x)

        encode_values = dict(zip(['x', 'check_signal'], [x, check_signal]))
        return encode_values['x']



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)

        return x



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class DGNN_SGS(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout, chunk_size=64):
        super(DGNN_SGS, self).__init__()
        self.DGNN_SGS_Conv = DGNN_SGS_Conv(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj):
        if fadj.is_sparse:
            fadj_coo = fadj.coalesce()
            edge_index = fadj_coo.indices()
        else:
            edge_index = fadj.nonzero(as_tuple=False).t().contiguous()
    
        if sadj.is_sparse:
            sadj_coo = sadj.coalesce()
            sedge_index = sadj_coo.indices()
        else:
            sedge_index = sadj.nonzero(as_tuple=False).t().contiguous()

        com1 = self.DGNN_SGS_Conv(x, sedge_index)
        com2 = self.DGNN_SGS_Conv(x, edge_index)

        Xcom = (com1 + com2) / 2
        emb = Xcom
        output = self.MLP(emb)
        return output, com1,com2, emb