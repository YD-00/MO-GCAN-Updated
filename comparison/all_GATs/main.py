import argparse
from sklearn.model_selection import train_test_split
from train import *
import pandas as pd
import numpy as np
import torch
import snf
import argparse
import os
import time

torch.manual_seed(2025)
np.random.seed(2025)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer_type', '-t', type = str, choices = ['lgg','ucec','stad','sarc','coadread','cesc','hnsc','brca'], default='lgg', help = 'The cancer type of the source data. The default is lgg' )
args = parser.parse_args()

# rename the files if they start with random numbers. 
files = os.listdir('../../data/' + args.cancer_type)
for file in files:
    if file[0].isdigit():
        os.rename('../../data/'+args.cancer_type+'/'+file,'data/'+args.cancer_type+'/'+file[file.find('_')+1:] )

# read data
print("loading cna, met, mrna, and rppa data...")
cna_df = pd.read_csv('../../data/'+args.cancer_type+'/cna_data.csv', header = 0, index_col = None)
met_df = pd.read_csv('../../data/'+args.cancer_type+'/met_data.csv', header = 0, index_col = None)
mrna_df = pd.read_csv('../../data/'+args.cancer_type+'/mrna_data.csv', header = 0, index_col = None)
rppa_df = pd.read_csv('../../data/'+args.cancer_type+'/rppa_data.csv', header = 0, index_col = None)
labels_df = pd.read_csv('../../data/'+args.cancer_type+'/subtype_data.csv', header = 0, index_col = None)
print("data loading is finished!\n")

# split data to 75% trainingn data and 25% testing data
x_train, x_test, y_train, y_test = train_test_split(cna_df, labels_df['class'],random_state=0,test_size = 0.25,stratify=labels_df['class'])

# create folder
new_path = os.getcwd() + '/result/'+args.cancer_type
if not os.path.exists(new_path):
    os.makedirs(new_path)
log_f = open('result/' + args.cancer_type + '/' + args.cancer_type +'.log', 'w')
# start to record training time
start_time = time.time()

# making the affinity networks for training data 
print("making the affinity networks for training data...")
train_affinity = snf.compute.make_affinity(cna_df.iloc[x_train.index].iloc[:,1:].values.astype(np.float64),met_df.iloc[x_train.index].iloc[:,1:].values.astype(np.float64),mrna_df.iloc[x_train.index].iloc[:,1:].values.astype(np.float64),rppa_df.iloc[x_train.index].iloc[:,1:].values.astype(np.float64),
                                         metric = 'sqeuclidean',K = 20,mu=0.5)
print("affinity network for training data is finished!\n.")

# load data to torch.tensor
cna_data,met_data,mrna_data,rppa_data,labels = load_to_tensor(cna_df,met_df,mrna_df,rppa_df,labels_df)
train_lap = load_adj_to_tensor(train_affinity)

# Train omics-specific models...
print("Start to train omics-specific models...")
cna_model, cna_train_acc = train_model('GAT',100,cna_data[x_train.index],train_lap[0],labels[x_train.index])
met_model, met_train_acc = train_model('GAT',100,met_data[x_train.index],train_lap[1],labels[x_train.index])
mrna_model, mrna_train_acc = train_model('GAT',100,mrna_data[x_train.index],train_lap[2],labels[x_train.index])
rppa_model, rppa_train_acc = train_model('GAT',100,rppa_data[x_train.index],train_lap[3],labels[x_train.index])
print("The training of omics-specific models is finished!\n")

# making snf networks...
print("Making fused networks by SNF...")
train_fused_net = torch.tensor(make_laplacian(snf.snf(train_affinity,K=20)), dtype=torch.float, device=torch.device('cpu'))
print("Fused networks are finished!\n")

# concatenate nomalized embeddings retrived from each GAT model
print("Retrived embeddings from omics-specifc models...")
cna_embedding =  normalize(cna_model.forward_1(cna_data[x_train.index], train_lap[0]))
met_embedding =  normalize(met_model.forward_1(met_data[x_train.index], train_lap[1]))
mrna_embedding =  normalize(mrna_model.forward_1(mrna_data[x_train.index], train_lap[2]))
rppa_embedding =  normalize(rppa_model.forward_1(rppa_data[x_train.index], train_lap[3]))
hidden_embeddings = torch.cat((cna_embedding, met_embedding,mrna_embedding,rppa_embedding), dim=1)
print("Embeddings from omics-specifc models are concatenated!\n")

# Train final model for all the omics
print("Start to train the final model for all the omics...")
final_model,final_train_acc = train_model('GAT',100,hidden_embeddings,train_fused_net,labels[x_train.index])
print('The training of final model for all the omics is finished!')

# make inference and evaluate on test data
test_affinity = snf.compute.make_affinity(cna_df.iloc[x_test.index].iloc[:,1:].values.astype(np.float64),met_df.iloc[x_test.index].iloc[:,1:].values.astype(np.float64),mrna_df.iloc[x_test.index].iloc[:,1:].values.astype(np.float64),rppa_df.iloc[x_test.index].iloc[:,1:].values.astype(np.float64),
                                         metric = 'sqeuclidean',K = 20,mu=0.5)
test_lap = load_adj_to_tensor(test_affinity)

test_cna_embedding =  normalize(cna_model.forward_1(cna_data[x_test.index], test_lap[0]))
test_met_embedding =  normalize(met_model.forward_1(met_data[x_test.index], test_lap[1]))
test_mrna_embedding =  normalize(mrna_model.forward_1(mrna_data[x_test.index], test_lap[2]))
test_rppa_embedding =  normalize(rppa_model.forward_1(rppa_data[x_test.index], test_lap[3]))
test_hidden_embeddings = torch.cat((test_cna_embedding, test_met_embedding,test_mrna_embedding,test_rppa_embedding), dim=1)

test_fused_net = torch.tensor(make_laplacian(snf.snf(test_affinity,K=20)), dtype=torch.float, device=torch.device('cpu'))

evaluation(cna_model,cna_data[x_test.index],test_lap[0],args.cancer_type,labels[x_test.index],'cna',log_f)
evaluation(met_model,met_data[x_test.index],test_lap[1],args.cancer_type,labels[x_test.index],'met',log_f )
evaluation(mrna_model,mrna_data[x_test.index],test_lap[2],args.cancer_type,labels[x_test.index],'mrna',log_f )
evaluation(rppa_model,rppa_data[x_test.index],test_lap[3],args.cancer_type,labels[x_test.index],'rppa',log_f )
evaluation(final_model,test_hidden_embeddings,test_fused_net,args.cancer_type,labels[x_test.index],'all the omics',log_f)

acc_list = [cna_train_acc,met_train_acc,mrna_train_acc,rppa_train_acc]
threshold = 0.8
indices = [i for i, acc in enumerate(acc_list) if acc > threshold]
while(len(indices) < 2):
    threshold -= 0.05
    indices = [i for i, acc in enumerate(acc_list) if acc > threshold]

selected_omics =[]
if (len(indices) < 4):
    selected_train_affinity = []
    selected_test_affinity = []
    selected_embeddings= torch.empty(0)
    selected_test_embeddings= torch.empty(0)
    train_embeddings = [cna_embedding,met_embedding,mrna_embedding,rppa_embedding]
    test_embeddings = [test_cna_embedding,test_met_embedding,test_mrna_embedding,test_rppa_embedding]
    for i in indices:
        if i==0:
            selected_omics.append('cna')
        elif i==1:
            selected_omics.append('met')
        elif i==2:
            selected_omics.append('mrna')
        elif i==3:
            selected_omics.append('rppa')
        selected_train_affinity.append(train_affinity[i])
        selected_test_affinity.append(test_affinity[i])
        selected_embeddings = torch.cat((selected_embeddings,train_embeddings[i]),dim=1)
        selected_test_embeddings = torch.cat((selected_test_embeddings,test_embeddings[i]),dim=1)

    selected_train_fused_net = torch.tensor(make_laplacian(snf.snf( selected_train_affinity,K=20)), dtype=torch.float, device=torch.device('cpu'))
    selected_test_fused_net = torch.tensor(make_laplacian(snf.snf( selected_test_affinity,K=20)), dtype=torch.float, device=torch.device('cpu'))

    # Train final model for the selected omics
    print(f"Start to train the final model for the selected omics: {selected_omics}")
    print(f"selected omics are : {selected_omics}", file = log_f)
    final_selected_model,final_selected_train_acc = train_model('GAT',100,selected_embeddings,selected_train_fused_net,labels[x_train.index])
    print('The training of final model for the selected omics is finished!')
    evaluation(final_selected_model,selected_test_embeddings,selected_test_fused_net,args.cancer_type,labels[x_test.index],'selected omics',log_f)
    torch.save(final_selected_model.eval().state_dict(), 'result/'+args.cancer_type+'/{}.pkl'.format('selected_omics'))


end_time =time.time()
training_time = end_time-start_time
print(f"Training time : {training_time} seconds")
print(f"Training time : {training_time} seconds",file = log_f)

# save models
torch.save(cna_model.state_dict(), 'result/'+args.cancer_type +'/{}.pkl'.format('cna'))
torch.save(met_model.state_dict(), 'result/'+args.cancer_type +'/{}.pkl'.format('met'))
torch.save(mrna_model.state_dict(), 'result/'+args.cancer_type +'/{}.pkl'.format('mrna'))
torch.save(rppa_model.state_dict(), 'result/'+args.cancer_type +'/{}.pkl'.format('rppa'))
torch.save(final_model.state_dict(), 'result/'+args.cancer_type+'/{}.pkl'.format('all_omics'))

log_f.close() 