# this file is to build a Graph Attention Network. we brought and used the code from https://github.com/compbiolabucf/omicsGAT
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the graph attention layer with 2 heads
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat 

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))   ## declaring the weights for linear transformation
        nn.init.xavier_uniform_(self.W.data, gain=1)                           ## initializing the linear transformation weights from the uniform distribution U(-a,a)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))           ## declaring weights for creating self attention coefficients
        nn.init.xavier_uniform_(self.a.data, gain=1)                           ## initializing the attention-coefficient weights
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)    ## multiplying inputs with the weights for linear transformation with dimension (#input X out_features)
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = torch.zeros_like(e)                       
        attention = torch.where(adj > 0, e, zero_vec)             ## assigning values of 'e' to those which has value>0 in adj matrix
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)                      ## multiplying attention co-efficients with the input  -- dimension (#input X out_features)

        if self.concat:
            xtra = F.elu(h_prime)
            return F.elu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, h):
        
        h1 = torch.matmul(h, self.a[:self.out_features, :])
        h2 = torch.matmul(h, self.a[self.out_features:, :])
        e = h1 + h2.T   # broadcast add
        return self.leakyrelu(e)
    
# Create a Graph Attention Network
class omicsGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(omicsGAT, self).__init__()
        self.dropout = dropout

        ## creating attention layers for given number of heads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] 
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)     ## adding the modules for each head
        
        in_features = nhid * nheads
        
        self.dnn = nn.Sequential(
                    nn.BatchNorm1d(in_features),
                    nn.ReLU(inplace = True),
                    nn.Linear(in_features,nclass))
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  ## concatanating all the attention heads dimension (#input X out_features*nb_heads);   out_features = nhid... each head contributing to (#input X out_features)
        x = F.dropout(x, self.dropout, training=self.training)        
        x = self.dnn(x)
        x = F.log_softmax(x, dim = 1)
        return x 

    # added for comparsion study for all_GATs
    def forward_1(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  ## concatanating all the attention heads dimension (#input X out_features*nb_heads);   out_features = nhid... each head contributing to (#input X out_features)
        x = F.dropout(x, self.dropout, training=self.training)        
        x = self.dnn(x)
        return x 
