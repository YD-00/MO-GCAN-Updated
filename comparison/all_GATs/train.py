# this file is used to train the GCN and GAT model. The code is built upon the code from https://github.com/Lifoof/MoGCN and https://github.com/compbiolabucf/omicsGAT
import torch
import torch.nn.functional as F
from gcn import GCN
from gat import omicsGAT
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import classification_report
import os

# load data to tensor
def load_to_tensor(cna_df,met_df,mrna_df,rppa_df,labels_df):
    cna_data = torch.tensor(cna_df.iloc[:, 1:].values, dtype=torch.float, device=torch.device('cpu'))
    met_data = torch.tensor(met_df.iloc[:, 1:].values, dtype=torch.float, device=torch.device('cpu'))
    mrna_data = torch.tensor(mrna_df.iloc[:, 1:].values, dtype=torch.float, device=torch.device('cpu'))
    rppa_data = torch.tensor(rppa_df.iloc[:, 1:].values, dtype=torch.float, device=torch.device('cpu'))

    labels = torch.tensor(labels_df['class'].values, dtype=torch.long, device=torch.device('cpu'))
    return cna_data,met_data,mrna_data,rppa_data,labels

# load laplacian adj to tensor
def load_adj_to_tensor(affinity):
    cna_lap = torch.tensor(make_laplacian(affinity[0]), dtype=torch.float, device=torch.device('cpu'))
    met_lap = torch.tensor(make_laplacian(affinity[1]), dtype=torch.float, device=torch.device('cpu'))
    mrna_lap = torch.tensor(make_laplacian(affinity[2]), dtype=torch.float, device=torch.device('cpu'))
    rppa_lap = torch.tensor(make_laplacian(affinity[3]), dtype=torch.float, device=torch.device('cpu'))
    return [cna_lap,met_lap,mrna_lap,rppa_lap]

# normalize data
def normalize(x):
    x_normed = (x - x.mean())/x.std()
    return x_normed

# train process
def train(self, optimizer, features, adj, labels):
    labels.to(torch.device('cpu'))
    self.train()
    optimizer.zero_grad()
    output = self(features, adj)
    loss_train = F.cross_entropy(output, labels)
    loss_train.backward()
    optimizer.step()
    return loss_train.data.item()

# train a model 
def train_model (model_type, n_hid_num, features_df,adj_array,labels):
    features = features_df.clone().detach()
    adj = adj_array.clone().detach()

    if model_type == 'GCN':
        model = GCN(n_in=features.shape[1], n_hid=n_hid_num, n_out=int(labels.max()) + 1, dropout=0.5)
        model.to(torch.device('cpu'))
    else:
        model = omicsGAT(nfeat=features.shape[1], 
                nhid= n_hid_num, 
                nclass=int(labels.max()) + 1, 
                dropout=0.5, 
                nheads=2, 
                alpha=0.2)
        model.to(torch.device('cpu'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_values = []   
    best_trained_model = model
    bad_counter = 0
    best = 999   #record the lowest loss value
    for epoch in range(150):
        loss_values.append(train(model, optimizer, features, adj, labels))
        if loss_values[-1] < best:
            best = loss_values[-1]
            bad_counter = 0
            best_trained_model=model
        else:
            bad_counter += 1     #In this epoch, the loss value didn't decrease
        if bad_counter == 20:
            break
    best_trained_model.eval()
    output = best_trained_model(features, adj)
    pred = output.max(1)[1].type_as(labels)
    train_acc = accuracy_score(labels,pred)
    return best_trained_model, train_acc

# make laplacian matrix
def make_laplacian(adj):
    per = 0.01
    laplacian_adj = adj.copy()
    values = laplacian_adj.flatten()
    values.sort( )
    sorted_values = np.flip(values)

    threshold = sorted_values[math.floor(sorted_values.shape[0] * per)]
    laplacian_adj[laplacian_adj<threshold] = 0
    exist = (laplacian_adj!=0) * 1.0

    factor = np.ones(laplacian_adj.shape[1])
    res = np.dot(exist, factor)
    diag_matrix = np.diag(res)

    #while(np.linalg.det(diag_matrix) == 0):
    while np.linalg.matrix_rank(diag_matrix) < diag_matrix.shape[0]:
        per+=0.005
        laplacian_adj = adj.copy()
        values = laplacian_adj.flatten()
        values.sort( )
        sorted_values = np.flip(values)
        threshold = sorted_values[math.floor(sorted_values.shape[0] * per)]
        laplacian_adj[laplacian_adj<threshold] = 0
        exist = (laplacian_adj!=0) * 1.0

        factor = np.ones(laplacian_adj.shape[1])
        res = np.dot(exist, factor)
        diag_matrix = np.diag(res)

    d_inv = np.linalg.inv(diag_matrix)
    laplacian_adj = d_inv.dot(exist)
    return laplacian_adj

def evaluation (model, features, adj,cancer_type,labels, file_name,log_f):
    model.eval()
    output = model(features, adj)
    pred = output.max(1)[1].type_as(labels)
    cm = confusion_matrix(labels, pred)
    sns.heatmap(cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=labels.unique(), yticklabels=labels.unique()) 
    plt.ylabel('Actual',fontsize=12)
    plt.xlabel('Prediction',fontsize=12)
    plt.title('Confusion Matrix',fontsize=16)
    # create folder
    new_path = os.getcwd() + '/result/'+cancer_type+'/confusion_matrix'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    plt.savefig('result/'+cancer_type + '/confusion_matrix/' + file_name + '.png')
    plt.close()
    print('prediction for ' + file_name + ' data: ')
    print('prediction for ' + file_name + ' data: ',file=log_f)
    print(classification_report(labels, pred,digits=4,zero_division=0))
    print(classification_report(labels, pred,digits=4,zero_division=0),file=log_f)
    #f = open('result/'+ cancer_type + '/' + file_name+'.csv','w')
    #f.write(classification_report(labels, pred,digits=4,zero_division=0))
    #f.close()
