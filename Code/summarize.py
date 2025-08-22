import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Define filtering criteria
datasets = ["DrugSimNet_CHEM"]
drug_emb_methods = ["gat"]
embedding_sizes = [128, 256, 512]
epochs = [100, 200, 400]
disease_nets = ["GeneNet"]#"OMIM","HPO","GeneNet","OG","OHG"
emb_methods = ["gat"]

dict_disease_net = dict()
dict_disease_net = {"OMIM":r"DiSimNet$_{M}$",
                    "HPO":r"DiSimNet$_{H}$",
                    "GeneNet":r"DiSimNet$_{P}$",
                    "OHG":r"DiSimNet$_{MPH}$",
                    "OG":r"DiSimNet$_{MP}$"}

# Generate disease_emb_methods
disease_emb_methods = []
for disease_net in disease_nets:
    for emb_method in emb_methods:
        for emb_size in embedding_sizes:
            for epoch in epochs:
                disease_emb_methods.append(f"DiseaseSimNet_{disease_net}_{emb_method}_d_{emb_size}_e_{epoch}")

# Metrics to analyze
metrics = ["auroc", "auprc", "f1", "accuracy"]

# Read the merged CSV file
input_file = "../Results/Performance_summary_metrics_all.csv"
if not os.path.exists(input_file):
    print(f"Error: File {input_file} not found")
    exit()

df = pd.read_csv(input_file)

# Filter rows based on drug_emb_file and disease_emb_file
def filter_rows(row):
    drug_match = any(dataset in row["drug_emb_file"] for dataset in datasets) and \
                 any(method in row["drug_emb_file"] for method in drug_emb_methods) and \
                 any(str(size) in row["drug_emb_file"] for size in embedding_sizes) and \
                 any(str(epoch) in row["drug_emb_file"] for epoch in epochs)
    disease_match = any(method in row["disease_emb_file"] for method in disease_emb_methods)
    return drug_match and disease_match

filtered_df = df[df.apply(filter_rows, axis=1)].copy()

if filtered_df.empty:
    print("No data matches the filtering criteria")
    exit()

# Extract relevant components from drug_emb_file
def extract_drug_components(drug_emb_file):
    dataset = next((d for d in datasets if d in drug_emb_file), None)
    method = next((m for m in drug_emb_methods if m in drug_emb_file), None)
    size = next((str(s) for s in embedding_sizes if str(s) in drug_emb_file), None)
    epoch = next((str(e) for e in epochs if str(e) in drug_emb_file), None)
    return dataset, method, size, epoch

# Extract relevant components from disease_emb_file
def extract_disease_components(disease_emb_file):
    disease_net = next((net for net in disease_nets if net in disease_emb_file), None)
    emb_method = next((m for m in emb_methods if m in disease_emb_file), None)
    return disease_net, emb_method

filtered_df[["dataset", "drug_method", "embedding_size", "epochs"]] = filtered_df["drug_emb_file"].apply(
    lambda x: pd.Series(extract_drug_components(x))
)
filtered_df[["disease_net", "disease_emb_method"]] = filtered_df["disease_emb_file"].apply(
    lambda x: pd.Series(extract_disease_components(x))
)

# Compute average values for CSV output
grouped = filtered_df.groupby(["dataset", "drug_method", "disease_net", "disease_emb_method"]).agg({
    "auroc_mean": "mean",
    "auroc_std": "mean",
    "auprc_mean": "mean",
    "auprc_std": "mean",
    "f1_mean": "mean",
    "f1_std": "mean",
    "accuracy_mean": "mean",
    "accuracy_std": "mean"
}).reset_index()

# Save to CSV
output_csv = os.path.join("performance_heatmaps_size_epochs_filter", "average_metrics.csv")
grouped.to_csv(output_csv, index=False, float_format="%.3f")
print(f"Average metrics saved to {output_csv}")

# Set up plotting style
# sns.set(style="whitegrid")

# Set up plotting style with global font size
plt.rcParams.update({'font.size': 14})  # Set global font size to 12

# Create a directory for saving plots
output_dir = "performance_heatmaps_size_epochs_filter"
os.makedirs(output_dir, exist_ok=True)

# Plot heatmaps for each dataset and drug method
for dataset in datasets:
    for drug_method in drug_emb_methods:
        subset_df = filtered_df[
            (filtered_df["dataset"] == dataset) & 
            (filtered_df["drug_method"] == drug_method)
        ]
        if subset_df.empty:
            print(f"No data for {dataset} with {drug_method}")
            continue

        for metric in metrics:
            # Create pivot table for heatmap
            pivot = subset_df.pivot_table(
                values=f"{metric}_mean",
                index="embedding_size",
                columns="epochs",
                aggfunc="mean"
            )
            
            # Ensure all embedding sizes and epochs are present
            pivot = pivot.reindex(
                index=[str(size) for size in sorted(embedding_sizes, reverse=True)],  # 512 to 128
                columns=[str(epoch) for epoch in sorted(epochs)],
                fill_value=np.nan
            )
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                pivot, 
                annot=False,  # Disable default annotation
                cmap="viridis", 
                vmin=0.5, 
                vmax=1.0, 
                cbar_kws={"label": f"{metric.upper()} Mean"},
                annot_kws={"size": 14}  # Larger font size for annotations
            )
            
            # Customize annotations with only annot_text and larger font
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    mean_val = pivot.iloc[i, j]
                    if pd.notna(mean_val):
                        std_val = subset_df[
                            (subset_df["embedding_size"] == pivot.index[i]) & 
                            (subset_df["epochs"] == pivot.columns[j])
                        ][f"{metric}_std"].mean()
                        annot_text = f"{mean_val:.3f}\n(\u00B1{std_val:.3f})"
                        plt.text(j + 0.5, i + 0.5, annot_text, 
                                ha="center", va="center", color="black", fontsize=14)

            plt.title(f"{dict_disease_net[disease_net]}")
            plt.xlabel("Epochs")
            plt.ylabel("Embedding Size")
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"{disease_net}_{drug_method}_{metric}_heatmap.png"), dpi=600, bbox_inches="tight")
            plt.close()

print(f"Heatmaps saved in {output_dir} directory")