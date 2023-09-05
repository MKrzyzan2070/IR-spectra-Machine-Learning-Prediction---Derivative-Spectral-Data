# IR-spectra-Machine-Learning-Prediction---Derivative-Spectral-Data

Dear User,

This text provides a concise overview of the code and the instructions 
for you to obtain the results presented in the article.

Firstly, let us address the files that users need to run.

The main folder, where the README file is located, 
stores all the primary code that the user needs to run. These include:

1. Main_Machine.py
This script runs everything, making it a reference point for anyone wishing to run the same calculations as us. 
The file calls various functions with explicit variable names to define factors such as the range of the IR data, 
resolution, data-handling methods, and machine learning models.

2. The_analysis.py
This script is essentially the Main_Machine script, designed to generate summaries that examine the effects 
of variables on performance, as discussed in the article. The variable names are explicit.

3. Results_Analysis.inpynb
This Jupyter Notebook script creates almost all the figures present in the article. 
However, it's important to note that the code itself is stored in the Results_Analysis_Functions.py file.


Now, let's delve into the individual folders and their contents:

1. Downloading the spectral data from NIST
Downloading the spectral data from NIST takes a while, so the All_the_IR is already provided to eliminate the waiting time. 
The species.txt file is essential and must be present in the folder as it contains the names of the compounds and their CAS numbers. 
To create a new All_the_IR file, users can run the download.py script.

2. CSV files to work on
This folder was created solely to generate additional plots for the article, specifically to prove that the derivative 
algorithm works and to produce categorical/denoising algorithms. The code for creating these plots is located in the Jupyter Notebook Addtional Plot Making.ipynb.

3. Mining
As the name suggests, this folder contains the code that mines data from the downloaded All_the_IR file. 
As the code runs, it creates the Only_Organics.txt file, which is essentially the All_the_IR file but only containing organic molecules. 
Typically, generating the SMILES string for each molecule takes a lot of time. Therefore, we did it once and stored the SMILES strings for each 
organic molecule in the SMILES_CSV.csv file, which must remain intact. Finally, the Maxi_Dictionary.json and Mini_Dictionary.json files 
are created with all the necessary information for future use.

4. The Machine Proper
This file contains all the code that takes the .json files as input data. Firstly, it creates the CSV files that the machine learning algorithm uses. 
The program creates up to four different CSV files, depending on the data-handling method used. However, only one is taken as input for 
the machine learning algorithm, named Final_csv.csv. This file contains x and y data points. The y data points are binary classification values 
for the presence of organic moieties in the given compound. To determine this, we need the SMILES/SMARTS strings of the organic moieties stored in 
the SMILES_SMARTS_string.json file, which must remain intact. Besides the scripts that run the machine learning model, the folder contains scripts for writing a summary.

5. Summaries
The default folder stores the singular summary. However, if users wish to perform a full analysis by running The_analysis.py script, the Summaries are 
stored in the Summaries folder. Some summaries have been left in the folder for reference.
