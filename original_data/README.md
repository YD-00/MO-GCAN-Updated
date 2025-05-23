
1. We downloaded the TCGA PanCancer Atlas 2018 data from the cbioportal website: https://www.cbioportal.org/datasets, and saved the data in seperate folder named by cancer type under the 'original_data' folder. So inside the 'original_data', there're 'brca','cesc',coadread',...,'ucec' 8 different folders.

2. Inside the cancer type folder, we unzipped the zip files, duplicated the 'data_clinical_patient.txt' file as 'data_clinical_patient_modified.txt'. In the 'data_clinical_patient_modified.txt' file, the only change we've made is to remove the comment part starting with '#' to read the data with a desired format.

3. Implement the 'data_process.py' to process the data. Details see the README.md file inside the data folder. 
