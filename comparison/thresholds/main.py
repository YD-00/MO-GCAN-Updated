import argparse
from sklearn.model_selection import train_test_split
from train import *
import pandas as pd
import numpy as np
import torch
import snf
import argparse
import os
#import time
import matplotlib.pyplot as plt

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
cna_df = pd.read_csv('../../data/'+args.cancer_type+'/cna_data.csv', header = 0, index_col = None)
met_df = pd.read_csv('../../data/'+args.cancer_type+'/met_data.csv', header = 0, index_col = None)
mrna_df = pd.read_csv('../../data/'+args.cancer_type+'/mrna_data.csv', header = 0, index_col = None)
rppa_df = pd.read_csv('../../data/'+args.cancer_type+'/rppa_data.csv', header = 0, index_col = None)
labels_df = pd.read_csv('../../data/'+args.cancer_type+'/subtype_data.csv', header = 0, index_col = None)

# split data to 75% training data and 25% testing data
x_train, x_test, y_train, y_test = train_test_split(cna_df, labels_df['class'],random_state=0,test_size = 0.25,stratify=labels_df['class'])

# create folder
new_path = os.getcwd() + '/result/'+args.cancer_type
if not os.path.exists(new_path):
    os.makedirs(new_path)
log_f = open('result/' + args.cancer_type + '/' + args.cancer_type +'.log', 'w')

# start to record training time
#start_time = time.time()


iter = 0
step = 0.1
iterations = 5
cna_test_list, cna_train_list = [],[]
met_test_list, met_train_list = [],[]
mrna_test_list, mrna_train_list = [],[]
rppa_test_list, rppa_train_list = [],[]
final_test_list, final_train_list = [],[]

while (iter < iterations):
    # making the affinity networks for training data 
    train_affinity = snf.compute.make_affinity(cna_df.iloc[x_train.index].iloc[:,1:].values.astype(np.float64),met_df.iloc[x_train.index].iloc[:,1:].values.astype(np.float64),mrna_df.iloc[x_train.index].iloc[:,1:].values.astype(np.float64),rppa_df.iloc[x_train.index].iloc[:,1:].values.astype(np.float64),
                                         metric = 'sqeuclidean',K = 20,mu=0.5)

    # load data to torch.tensor
    cna_data,met_data,mrna_data,rppa_data,labels = load_to_tensor(cna_df,met_df,mrna_df,rppa_df,labels_df)
    cna_per = find_minimum_per(train_affinity[0]) + iter * step
    met_per = find_minimum_per(train_affinity[1]) + iter * step
    mrna_per = find_minimum_per(train_affinity[2]) + iter * step
    rppa_per = find_minimum_per(train_affinity[3]) + iter * step
    train_lap = load_adj_to_tensor(train_affinity,cna_per,met_per,mrna_per,rppa_per)

    # Train omics-specific models...
    cna_model, cna_train_acc = train_model('GCN',100,cna_data[x_train.index],train_lap[0],labels[x_train.index])
    met_model, met_train_acc = train_model('GCN',100,met_data[x_train.index],train_lap[1],labels[x_train.index])
    mrna_model, mrna_train_acc = train_model('GCN',100,mrna_data[x_train.index],train_lap[2],labels[x_train.index])
    rppa_model, rppa_train_acc = train_model('GCN',100,rppa_data[x_train.index],train_lap[3],labels[x_train.index])

    # making snf networks...
    snf_net = snf.snf(train_affinity,K=20)
    fused_per = find_minimum_per(snf_net) + iter * step 
    train_fused_net = torch.tensor(make_laplacian_given_per(snf_net,fused_per), dtype=torch.float, device=torch.device('cpu'))

    # concatenate nomalized embeddings retrived from each GCN model
    cna_embedding =  normalize(cna_model.forward_1(cna_data[x_train.index], train_lap[0]))
    met_embedding =  normalize(met_model.forward_1(met_data[x_train.index], train_lap[1]))
    mrna_embedding =  normalize(mrna_model.forward_1(mrna_data[x_train.index], train_lap[2]))
    rppa_embedding =  normalize(rppa_model.forward_1(rppa_data[x_train.index], train_lap[3]))
    hidden_embeddings = torch.cat((cna_embedding, met_embedding,mrna_embedding,rppa_embedding), dim=1)

    # Train final model for all the omics
    final_model,final_train_acc = train_model('GAT',100,hidden_embeddings,train_fused_net,labels[x_train.index])

    # make inference and evaluate on test data
    test_affinity = snf.compute.make_affinity(cna_df.iloc[x_test.index].iloc[:,1:].values.astype(np.float64),met_df.iloc[x_test.index].iloc[:,1:].values.astype(np.float64),mrna_df.iloc[x_test.index].iloc[:,1:].values.astype(np.float64),rppa_df.iloc[x_test.index].iloc[:,1:].values.astype(np.float64),
                                         metric = 'sqeuclidean',K = 20,mu=0.5)
    test_cna_per = find_minimum_per(test_affinity[0]) + iter * step
    test_met_per = find_minimum_per(test_affinity[1]) + iter * step
    test_mrna_per = find_minimum_per(test_affinity[2]) + iter * step
    test_rppa_per = find_minimum_per(test_affinity[3]) + iter * step
    test_lap = load_adj_to_tensor(test_affinity,test_cna_per,test_met_per,test_mrna_per,test_rppa_per)

    test_cna_embedding =  normalize(cna_model.forward_1(cna_data[x_test.index], test_lap[0]))
    test_met_embedding =  normalize(met_model.forward_1(met_data[x_test.index], test_lap[1]))
    test_mrna_embedding =  normalize(mrna_model.forward_1(mrna_data[x_test.index], test_lap[2]))
    test_rppa_embedding =  normalize(rppa_model.forward_1(rppa_data[x_test.index], test_lap[3]))
    test_hidden_embeddings = torch.cat((test_cna_embedding, test_met_embedding,test_mrna_embedding,test_rppa_embedding), dim=1)

    test_snf_net = snf.snf(test_affinity,K=20)
    test_fused_per = find_minimum_per(test_snf_net) + iter * step
    test_fused_net = torch.tensor(make_laplacian_given_per(test_snf_net,test_fused_per), dtype=torch.float, device=torch.device('cpu'))

    cna_test_acc, cna_train_acc = evaluation(cna_model,cna_data[x_test.index],test_lap[0],labels[x_test.index],cna_data[x_train.index],train_lap[0],labels[x_train.index])
    met_test_acc, met_train_acc  = evaluation(met_model,met_data[x_test.index],test_lap[1],labels[x_test.index],met_data[x_train.index],train_lap[1],labels[x_train.index])
    mrna_test_acc, mrna_train_acc  = evaluation(mrna_model,mrna_data[x_test.index],test_lap[2],labels[x_test.index],mrna_data[x_train.index],train_lap[2],labels[x_train.index])
    rppa_test_acc, rppa_train_acc  = evaluation(rppa_model,rppa_data[x_test.index],test_lap[3],labels[x_test.index],rppa_data[x_train.index],train_lap[3],labels[x_train.index])
    final_test_acc, final_train_acc  = evaluation(final_model,test_hidden_embeddings,test_fused_net,labels[x_test.index],hidden_embeddings,train_fused_net,labels[x_train.index])



    # Train final model for selected omics
    acc_list = [cna_train_acc,met_train_acc,mrna_train_acc,rppa_train_acc]
    threshold = 0.8
    indices = []
    indices = [i for i, acc in enumerate(acc_list) if acc > threshold]
    print(indices)
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

        selected_train_snf_net = snf.snf( selected_train_affinity,K=20)
        selected_train_fused_per = find_minimum_per(selected_train_snf_net) + iter * step
        selected_train_fused_net = torch.tensor(make_laplacian_given_per(selected_train_snf_net,selected_train_fused_per), dtype=torch.float, device=torch.device('cpu'))
    
        selected_test_snf_net = snf.snf( selected_test_affinity,K=20)
        selected_test_fused_per = find_minimum_per(selected_test_snf_net) + iter * step
        selected_test_fused_net = torch.tensor(make_laplacian_given_per(selected_test_snf_net,selected_test_fused_per), dtype=torch.float, device=torch.device('cpu'))
    
        # Train final model for the selected omics
        final_selected_model,final_selected_train_acc = train_model('GAT',100,selected_embeddings,selected_train_fused_net,labels[x_train.index])
        final_selected_test_acc, final_selected_train_acc  = evaluation(final_selected_model,selected_test_embeddings,selected_test_fused_net,labels[x_test.index],selected_embeddings,selected_train_fused_net,labels[x_train.index])
        final_test_list.append(final_selected_test_acc)
        final_train_list.append(final_selected_train_acc)

    print(f"{iter} iteration is done. ")
    iter += 1
    cna_test_list.append(cna_test_acc)
    met_test_list.append(met_test_acc)
    mrna_test_list.append(mrna_test_acc)
    rppa_test_list.append(rppa_test_acc)

    cna_train_list.append(cna_train_acc)
    met_train_list.append(met_train_acc)
    mrna_train_list.append(mrna_train_acc)
    rppa_train_list.append(rppa_train_acc)

    if (len(indices) == 4):
        final_test_list.append(final_test_acc)
        final_train_list.append(final_train_acc)

#end_time =time.time()
#training_time = end_time-start_time
#print(f"Training time : {training_time} seconds")
#print(f"final test accuracy list : {final_test_list} ")

# save the result log and the plots
per_list = ['mini', '+0.1','+0.2','+0.3','+0.4']
print(f"cna test accuracy list : {cna_test_list} ")
print(f"cna train accuracy list : {cna_train_list} ", file = log_f)
plt.plot(per_list,cna_test_list,label = 'acc_test')
plt.plot(per_list,cna_train_list,label = 'acc_train')
plt.xlabel('Percentahes for calculating the thresholds')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.title('Accuracy on various thresholds for cna data')
plt.savefig('result/' + args.cancer_type + '/cna.png')
plt.close()

print(f"met test accuracy list : {met_test_list} ")
print(f"met train accuracy list : {met_train_list} ", file = log_f)
plt.plot(per_list,met_test_list,label = 'acc_test')
plt.plot(per_list,met_train_list,label = 'acc_train')
plt.xlabel('Percentahes for calculating the thresholds')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.title('Accuracy on various thresholds for met data')
plt.savefig('result/' + args.cancer_type + '/met.png')
plt.close()

print(f"mrna test accuracy list : {mrna_test_list} ")
print(f"mrna train accuracy list : {mrna_train_list} ", file = log_f)
plt.plot(per_list,mrna_test_list,label = 'acc_test')
plt.plot(per_list,mrna_train_list,label = 'acc_train')
plt.xlabel('Percentahes for calculating the thresholds')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.title('Accuracy on various thresholds for mrna data')
plt.savefig('result/' + args.cancer_type + '/mrna.png')
plt.close()

print(f"rppa test accuracy list : {rppa_test_list} ")
print(f"rppa train accuracy list : {rppa_train_list} ", file = log_f)
plt.plot(per_list,rppa_test_list,label = 'acc_test')
plt.plot(per_list,rppa_train_list,label = 'acc_train')
plt.xlabel('Percentahes for calculating the thresholds')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.title('Accuracy on various thresholds for rppa data')
plt.savefig('result/' + args.cancer_type + '/rppa.png')
plt.close()

print(f"final test accuracy list : {final_test_list} ")
print(f"final train accuracy list : {final_train_list} ", file = log_f)
plt.plot(per_list,final_test_list,label = 'acc_test')
plt.plot(per_list,final_train_list,label = 'acc_train')
plt.xlabel('Percentahes for calculating the thresholds')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.title('Accuracy on various thresholds for multi-omics data')
plt.savefig('result/' + args.cancer_type + '/multi-omics.png')
plt.close()
