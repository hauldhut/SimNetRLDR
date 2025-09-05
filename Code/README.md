## Environment Setup
- **For Python-based Code**
  - *conda env create -f SimNetRLDR.yml*: To run SimNetRLDR
  - *conda env create -f DDAGDL_RGLDR.yml*: To run DDAGDL and RGLDR
- **For R-based Code**: To run MHDR, and visualize drug-disease associations
  - *Install R packages: RandomWalkRestartMH, igraph, foreach, doParallel, ROCR, ggplot2, Metrics, hash*

## Simulations
- **Generate Embeddings**
  - *generate_embeddings_for_DrNet.py*: Generate embeddings for drugs from the drug similarity network.
  - *generate_embeddings_for_DiNet.py*: Generate embeddings for diseases from disease similarity.
 
- **Evaluate**:
  - *evaluate.py*: For various combinations of drug and disease similarity networks, embedding sizes, and epochs

- **Predict**:
  - *predict.py*: For prediction of novel drug-disease associations

## Summary
  - *summarize.py*: To summarize and create heatmaps for various combinations of drug and disease similarity networks, embedding sizes, and epochs
  - *create_Fig3_Fig4.py*: To create Figure 3 and 4
  - *visualize_DrugDisease_from_Evidence_Final.R*: To visualize drug-disease associations via pathways

## Comparison
  - *MHDR_KFold_Final_Balanced_Final.R*: To compare with MHDR (https://github.com/hauldhut/MHDR)
  - *DDAGDL.zip*: All data and code used to compare with DDAGDL
  - *RGLDR.zip*: All data and code used to compare with RGLDR
  

