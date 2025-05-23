# this file is to process the original data, and save the processed data to the data folder.
import pandas
import numpy
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer_type', '-t', type = str, choices = ['lgg','ucec','stad','sarc','coadread','cesc','hnsc','brca'], default='lgg', help = 'The cancer type of the source data.' )
args = parser.parse_args()

# read the data
# for reading subtype information, we add the 'data_clinical_patient_modified.txt' file by duplicating the data_clinical_patient.txt file and remove the comment sentences following the # symbol. 
print('reading the data...')
subtype_data = pandas.read_csv('original_data/'+args.cancer_type+'_tcga_pan_can_atlas_2018/data_clinical_patient_modified.txt', delimiter='\t')[['PATIENT_ID', 'SUBTYPE']]
cna_data= pandas.read_csv('original_data/'+args.cancer_type+'_tcga_pan_can_atlas_2018/data_log2_cna.txt', delimiter='\t').transpose()
cna_data.columns = cna_data.iloc[0].values
cna_data = cna_data.iloc[2:,:].reset_index()

methylation_data= pandas.read_csv('original_data/'+args.cancer_type+'_tcga_pan_can_atlas_2018/data_methylation_hm27_hm450_merged.txt', delimiter='\t').transpose()
methylation_data.columns = methylation_data.iloc[1].values
methylation_data = methylation_data.iloc[4:,:].reset_index()

mrna_data= pandas.read_csv('original_data/'+args.cancer_type+'_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt', delimiter='\t').transpose()
mrna_data.columns = mrna_data.iloc[0].values
mrna_data = mrna_data.iloc[2:,:].reset_index()

rppa_data= pandas.read_csv('original_data/'+args.cancer_type+'_tcga_pan_can_atlas_2018/data_rppa_zscores.txt', delimiter='\t').transpose()
rppa_data.columns = rppa_data.iloc[0].values
rppa_data = rppa_data.iloc[1:].reset_index()
print('data reading is done!\n')

print('data preprocessing...')
# drop the data with 'na' as subtype
subtype_data = subtype_data[subtype_data['SUBTYPE'].notna()]

# drop the duplicate samples and keep the last one
subtype_data= subtype_data.drop_duplicates(subset = 'PATIENT_ID', keep = 'last')
cna_data= cna_data.drop_duplicates(subset = 'index', keep = 'last')
methylation_data = methylation_data.drop_duplicates(subset = 'index', keep = 'last')
mrna_data = mrna_data.drop_duplicates(subset = 'index', keep = 'last')
rppa_data = rppa_data.drop_duplicates(subset = 'index', keep = 'last')

# update the sample names (only keep the first 12 digits of the sample id so that they can match to the clinical file)
def update_sample_name(original_name):
    new_name = original_name[:12]
    return new_name
cna_data['index'] = cna_data['index'].apply(update_sample_name)
methylation_data['index'] = methylation_data['index'].apply(update_sample_name)
mrna_data['index'] = mrna_data['index'].apply(update_sample_name)
rppa_data['index'] = rppa_data['index'].apply(update_sample_name)

# add categorical data 
subtype_data.SUBTYPE = pandas.Categorical(subtype_data.SUBTYPE )
subtype_data['class'] = subtype_data.SUBTYPE.cat.codes
subtype_data.insert(1,'class', subtype_data.pop('class'))

#sort the samples according to the samples ids
subtype_data = subtype_data.sort_values(by=['PATIENT_ID'], ascending=True)
cna_data = cna_data.sort_values(by=['index'], ascending=True)
methylation_data = methylation_data.sort_values(by=['index'], ascending=True)
mrna_data = mrna_data.sort_values(by=['index'], ascending=True)
rppa_data = rppa_data.sort_values(by=['index'], ascending=True)

# keep the common samples among the 5 files
subtype_data= subtype_data.loc[subtype_data['PATIENT_ID'].isin(cna_data['index'])]
subtype_data= subtype_data.loc[subtype_data['PATIENT_ID'].isin(methylation_data['index'])]
subtype_data= subtype_data.loc[subtype_data['PATIENT_ID'].isin(mrna_data['index'])]
subtype_data= subtype_data.loc[subtype_data['PATIENT_ID'].isin(rppa_data['index'])]

cna_data= cna_data.loc[cna_data['index'].isin(subtype_data['PATIENT_ID'])]
methylation_data= methylation_data.loc[methylation_data['index'].isin(subtype_data['PATIENT_ID'])]
mrna_data= mrna_data.loc[mrna_data['index'].isin(subtype_data['PATIENT_ID'])]
rppa_data= rppa_data.loc[rppa_data['index'].isin(subtype_data['PATIENT_ID'])]

# check and print if all the files are with the same samples in the same order
print('the number of cna samples match to met samples?')
print(numpy.array_equal(cna_data['index'].values, methylation_data['index'].values) )
print('the number of met samples match to mrna samples?')
print(numpy.array_equal(methylation_data['index'].values, mrna_data['index'].values) )
print('the number of mrna samples match to rppa samples?')
print(numpy.array_equal(mrna_data['index'].values, rppa_data['index'].values))
print('the number of rppa samples match to subtype samples?')
print(numpy.array_equal(rppa_data['index'].values, subtype_data['PATIENT_ID'].values) )

# for the 4 omics data files: remove the columns with more than 10% na, and fill the na values with 0s
cna_data = cna_data.dropna(thresh = len(cna_data) * 0.9, axis = 1)
cna_data = cna_data.fillna(0)

methylation_data=methylation_data.dropna(thresh = len(methylation_data) * 0.9, axis = 1)
methylation_data = methylation_data.fillna(0)

mrna_data = mrna_data.dropna(thresh = len(mrna_data) * 0.9, axis = 1)
mrna_data = mrna_data.fillna(0)

rppa_data = rppa_data.dropna(thresh = len(mrna_data) * 0.9, axis = 1)
rppa_data = rppa_data.fillna(0)

# keep columns with less than 10% 0
cna_data = cna_data.loc[:, (cna_data==0).mean() <.1]
methylation_data = methylation_data.loc[:, (methylation_data==0).mean() <.1]
mrna_data = mrna_data.loc[:, (mrna_data==0).mean() <.1]
rppa_data = rppa_data.loc[:, (rppa_data==0).mean() <.1]
print('data preprocess is done!\n')

# write to files
print('saving as csv files...')
cna_data.to_csv('data/'+args.cancer_type+'/cna_data.csv', header = True, index = False)
methylation_data.to_csv('data/'+args.cancer_type+'/met_data.csv', header = True, index = False)
mrna_data.to_csv('data/'+args.cancer_type+'/mrna_data.csv', header = True, index = False)
rppa_data.to_csv('data/'+args.cancer_type+'/rppa_data.csv', header = True, index = False)
subtype_data.to_csv('data/'+args.cancer_type+'/subtype_data.csv', header = True, index = False)
print('files are saved in the data folder!')
print('the shape for cna data is: '+str(cna_data.shape))
print('the shape for met data is: '+str(methylation_data.shape))
print('the shape for mrna data is: '+str(mrna_data.shape))
print('the shape for rppa data is: '+str(rppa_data.shape))
print('the shape for subtype data is: '+str(subtype_data.shape))
print('the class distribution: \n' + str(subtype_data['class'].value_counts()))