# SimNetRLDR

Drug repositioning, the identification of new therapeutic indications for existing drugs, offers a cost-effective alternative to traditional drug discovery. Computational approaches, particularly network-based and deep learning methods, have advanced the prediction of drug-disease associations. However, existing methods often rely on single disease similarity networks or heterogeneous networks incorporating drug-disease associations, which may limit prediction accuracy and cause data leakage. 
We propose SimNetRLDR (Similarity Network-based Representation Learning for Drug Repositioning), a novel method integrating drug and disease similarity networks with representation learning to overcome these limitations. Drug similarity networks were constructed using SMILES data, while disease similarity networks were built from MeSH and protein interaction data, integrated via a per-edge average method. Low-dimensional representations of drugs and diseases were learned using weighted graph attention networks, followed by XGBoost classification to predict drug-disease associations. 
Evaluated via 10-fold cross-validation, SimNetRLDR achieved superior performance (AUROC: 0.979 ± 0.006, AUPRC: 0.982 ± 0.006) compared to state-of-the-art methods like DDAGDL, RGLDR, TP-NRWRH, and MHDR across various network configurations. 
Validation of predicted associations revealed 23 novel drug-disease pairs supported by shared KEGG pathways, with 11 substantiated by clinical trials from ClinicalTrials.gov. Notable associations include sulindac and breast cancer (15 shared pathways, 3 trials), liothyronine and breast cancer (thyroid hormone signaling, 7 trials), and ergocalciferol and asthma susceptibility (tuberculosis pathway, 25 trials). 
These findings demonstrate SimNetRLDR’s ability to leverage molecular and clinical evidence for accurate drug repositioning, highlighting its potential to identify therapeutically relevant drug-disease associations.

![SimNetRLDR](https://github.com/hauldhut/SimNetRLDR/blob/main/Figure1.png)

## Repo structure
- **Data**: Contains all data 
- **Code**: Contains all source code to reproduce all the results
- **Results**: To store simulation results

## How to run
- Download the repo
- Follow instructions (README.md) in the folder **Code** to run
  
