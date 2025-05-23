# this file is to build a Graph Convolution Network. We brought and used the code from https://github.com/Lifoof/MoGCN
import torch
from torch.nn.parameter import Parameter
import math

# define the graph convolutional layer.
class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self, in_feas, out_feas, bias = True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_feas = in_feas
        self.out_feas = out_feas
        self.weight = Parameter(torch.FloatTensor(in_feas, out_feas))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feas))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,x,adj):
        x1 = torch.mm(x,self.weight)
        output = torch.mm(adj,x1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# create a graph convolutional network. 
class GCN(torch.nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout = None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(n_in, n_hid)
        self.gc2 = GraphConvolutionLayer(n_hid, n_hid)
        self.dp1 = torch.nn.Dropout(dropout)
        self.dp2 = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        x = torch.nn.functional.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = torch.nn.functional.elu(x)
        x = self.dp2(x)
        x = self.fc(x)
        return x
    
    # we add forward_1 function there, to retrive the data from the 1st hidden layer as latent data. 
    def forward_1(self, input, adj):
        x = self.gc1(input, adj)
        x = torch.nn.functional.elu(x)
        return x
    
    # we add forward_2 function there, to retrive the data from the 2st hidden layer as latent data. 
    def forward_2(self, input, adj):
        x = self.gc1(input, adj)
        x = torch.nn.functional.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = torch.nn.functional.elu(x)
        return x
    

    
    
