# SimNetRLDR: Similarity Network-based Representation Learning for Drug Repositioning

- Drug repositioning, the identification of new therapeutic indications for existing drugs, offers a cost-effective alternative to traditional drug discovery. Computational approaches, particularly network-based and deep learning methods, have advanced the prediction of drug-disease associations. However, existing methods often rely on single disease similarity networks or heterogeneous networks incorporating drug-disease associations, which may limit prediction accuracy and cause data leakage. 
- We propose SimNetRLDR, a novel method integrating drug and disease similarity networks with representation learning to overcome these limitations. Drug similarity networks were constructed using SMILES data, while disease similarity networks were built from MeSH and protein interaction data, integrated via a per-edge average method. Low-dimensional representations of drugs and diseases were learned using weighted graph attention networks, followed by XGBoost classification to predict drug-disease associations. 

![SimNetRLDR](https://github.com/hauldhut/SimNetRLDR/blob/main/Figure1.png)

## Repo structure
- **Data**: Contains all data 
- **Code**: Contains all source code to reproduce all the results
- **Results**: To store simulation results

## How to run
- Download the repo
- Follow instructions (README.md) in the folder **Code** to run
  
