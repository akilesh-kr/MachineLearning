# MachineLearning
Course Project for CS675

In this project I implemented F-score feature selection method to select the 18 best features from the total of 29263 features. SVM was used to train the model on the dataset with the selected 18 features and 8000 data lines.

The dataset was a simulated dataset of single nucleotide polymorphism (SNP) genotype data containing 29623 SNPs (total features). Amongst all SNPs are 15 causal ones which means they and neighboring ones discriminate between case and controls while remainder are noise. In the training are 4000 cases and 4000 controls.

The program takes as input the training dataset, the trueclass label file for training points, and the test dataset. The output is the prediction of the labels of the test dataset. The total number of features and the feature column numbers that were used for final predicton is also presented in the output.

Achieved validation accuracy of **66%**.
