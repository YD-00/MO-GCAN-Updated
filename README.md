# MO-GCAN: Multi Omics integration based on Graph Convolutional and Attention Networks 
## What is it?
MO-GCAN is a framework that leverages supervised feature learning and classification based on a graph-based learning approach with attention mechanism for subtype detection. The work is inspired by MOGONET[1], MoGCN [2] and omicsGAT[3] and is built upon the code from https://github.com/Lifoof/MoGCN and https://github.com/compbiolabucf/omicsGAT.

[1]Wang, Tongxin, Wei Shao, Zhi Huang, Haixu Tang, Jie Zhang, Zhengming Ding, and Kun Huang. “MOGONET Integrates Multi-Omics Data Using Graph Convolutional Networks Allowing Patient Classification and Biomarker Identification.” Nature Communications 12, no. 1 (June 8, 2021). https://doi.org/10.1038/s41467-021-23774-w.

[2] Li, Xiao, Jie Ma, Ling Leng, Mingfei Han, Mansheng Li, Fuchu He, and Yunping Zhu. “MOGCN: A Multi-Omics Integration Method Based on Graph Convolutional Network for Cancer Subtype Analysis.” Frontiers in Genetics 13 (February 2, 2022). https://doi.org/10.3389/fgene.2022.806842.

[3] Baul, Sudipto, Khandakar Tanvir Ahmed, Joseph Filipek, and Wei Zhang. “OMICSGAT: Graph Attention Network for Cancer Subtype Analyses.” International Journal of Molecular Sciences 23, no. 18 (September 6, 2022): 10220. https://doi.org/10.3390/ijms231810220.

## How to run?
1. First download the processed data from the website https://figshare.com/articles/dataset/MO-GCAN_data/25823950 and saved them in the 'data' folder. The data folder should contains 8 subfolders, named by the cancer type name (like brca, cesc...ucec), and each subfolder should contains 5 files (cna_data.csv, met_data.csv, mrna_data.csv, rppa_data.csv and subtype_data.csv).Then create an empty 'result' folder to save the outputs. If you wanted to experience the whole process involing data processing, you can start with the original data (see the instrument inside the 'original_data' folder), and run the data process.py on each cancer type. An example command to process the original data is: python3 data_process.py -t lgg

2. to run the main.py: 
for lgg cancer: python3 main.py<br>
for ucec cancer: python3 main.py -t ucec 
for stad cancer: python3 main.py -t stad 
for sarc cancer: python3 main.py -t sarc 
for coadread cancer: python3 main.py -t coadread 
for cesc cancer: python3 main.py -t cesc 
for hnsc cancer: python3 main.py -t hnsc 
for brca cancer: python3 main.py -t brca 
