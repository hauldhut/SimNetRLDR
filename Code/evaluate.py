import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import os
import sys
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)



# Custom class to redirect print output to both console and file
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    
    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()

# Step 1: Load embeddings and drug-disease pairs, filter invalid pairs
def load_data(drug_embeddings_file, disease_embeddings_file, drug_disease_file):
    print("Loading embeddings...")
    drug_embeddings_df = pd.read_csv(drug_embeddings_file)
    disease_embeddings_df = pd.read_csv(disease_embeddings_file)
    
    valid_drugs = set(drug_embeddings_df[drug_embeddings_df['type'] == 'drug']['node_id'])
    valid_diseases = set(disease_embeddings_df[disease_embeddings_df['type'] == 'disease']['node_id'])
    
    drug_emb = drug_embeddings_df[drug_embeddings_df['type'] == 'drug'].set_index('node_id')
    disease_emb = disease_embeddings_df[disease_embeddings_df['type'] == 'disease'].set_index('node_id')
    
    drug_emb_cols = [col for col in drug_embeddings_df.columns if col.startswith('dim_')]
    embedding_size_drug = len(drug_emb_cols)
    disease_emb_cols = [col for col in disease_embeddings_df.columns if col.startswith('dim_')]
    embedding_size_disease = len(disease_emb_cols)
    
    drug_emb = drug_emb[drug_emb_cols].to_numpy()
    disease_emb = disease_emb[disease_emb_cols].to_numpy()
    
    drugs = list(valid_drugs)
    diseases = list(valid_diseases)
    
    print(f"Number of drugs with embeddings: {len(drugs)}")
    print(f"Number of diseases with embeddings: {len(diseases)}")
    
    print("Loading positive drug-disease pairs...")
    drug_disease_df = pd.read_csv(drug_disease_file)
    
    positive_pairs = []
    total_pairs = len(drug_disease_df)
    for _, row in drug_disease_df.iterrows():
        drug, disease = row['drug'], row['disease']
        if drug in valid_drugs and disease in valid_diseases:
            positive_pairs.append((drug, disease))
    
    positive_pairs = set(positive_pairs)
    skipped_pairs = total_pairs - len(positive_pairs)
    print(f"Number of positive pairs loaded: {len(positive_pairs)}")
    print(f"Number of pairs skipped (missing embeddings): {skipped_pairs}")
    
    if not positive_pairs:
        raise ValueError("No valid positive pairs found after filtering. Check embeddings_file and drug_disease_file.")
    
    return drugs, diseases, drug_emb, disease_emb, positive_pairs, embedding_size_drug, embedding_size_disease

# Step 2: Generate feature vectors and labels with balanced negative sampling
def generate_features_labels(drugs, diseases, drug_emb, disease_emb, positive_pairs, embedding_size_drug, embedding_size_disease):
    print("Generating feature vectors and labels with balanced negative sampling...")
    features = []
    labels = []
    pairs = []
    
    drug_idx = {drug: i for i, drug in enumerate(drugs)}
    disease_idx = {disease: i for i, disease in enumerate(diseases)}
    
    positive_pairs_list = list(positive_pairs)
    num_positive = len(positive_pairs_list)
    print(f"Number of positive pairs: {num_positive}")
    
    all_pairs = list(itertools.product(drugs, diseases))
    negative_pairs = [pair for pair in all_pairs if pair not in positive_pairs]
    print(f"Total negative pairs available: {len(negative_pairs)}")
    
    if len(negative_pairs) < num_positive:
        print("Warning: Fewer negative pairs than positive pairs. Using all available negative pairs.")
        selected_negative_pairs = negative_pairs
    else:
        selected_negative_pairs = random.sample(negative_pairs, num_positive)
    
    print(f"Number of negative pairs sampled: {len(selected_negative_pairs)}")
    
    selected_pairs = positive_pairs_list + selected_negative_pairs
    random.shuffle(selected_pairs)
    
    for drug, disease in tqdm(selected_pairs, desc="Generating features for pairs"):
        pair = (drug, disease)
        pairs.append(pair)
        label = 1 if pair in positive_pairs else 0
        labels.append(label)
        
        drug_vec = drug_emb[drug_idx[drug]]
        disease_vec = disease_emb[disease_idx[disease]]
        feature_vec = np.concatenate([drug_vec, disease_vec])
        features.append(feature_vec)
    
    return np.array(features), np.array(labels), pairs

# Step 3: Train and evaluate XGBoost with 10-fold cross-validation
def evaluate_model(features, labels, base_name):
    print("Training and evaluating XGBoost model...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    auroc_scores = []
    auprc_scores = []
    f1_scores = []
    accuracy_scores = []
    
    roc_data = []
    pr_data = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    
    tprs = []
    precisions = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels), 1):
        print(f"Processing fold {fold}/10...")
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        xgb.fit(X_train, y_train)
        y_pred_proba = xgb.predict_proba(X_test)[:, 1]
        
        auroc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        roc_data.append(pd.DataFrame({'fold': fold, 'fpr': fpr, 'tpr': tpr}))
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(precision_interp)
        pr_data.append(pd.DataFrame({'fold': fold, 'recall': recall, 'precision': precision}))
        
        print(f"Fold {fold} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    auroc_mean, auroc_std = np.mean(auroc_scores), np.std(auroc_scores)
    auprc_mean, auprc_std = np.mean(auprc_scores), np.std(auprc_scores)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
    accuracy_mean, accuracy_std = np.mean(accuracy_scores), np.std(accuracy_scores)
    
    print("\nFinal Results:")
    print(f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"AUPRC: {auprc_mean:.4f} ± {auprc_std:.4f}")
    print(f"F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    
    # plot_curves(tprs, precisions, mean_fpr, mean_recall, base_name)
    # save_curve_data(roc_data, pr_data, tprs, precisions, mean_fpr, mean_recall, base_name)
    
    return auroc_mean, auroc_std, auprc_mean, auprc_std, f1_mean, f1_std, accuracy_mean, accuracy_std

# Step 4: Plot ROC and PR curves
def plot_curves(tprs, precisions, mean_fpr, mean_recall, base_name):
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUROC = {np.mean(tprs):.4f})', color='blue')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve with ±1 Std Dev')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'{base_name}_roc_curve.png')
    plt.close()
    
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_recall, mean_precision, label=f'Mean PR (AUPRC = {np.mean(precisions):.4f})', color='red')
    plt.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color='red', alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Mean Precision-Recall Curve with ±1 Std Dev')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(f'{base_name}_pr_curve.png')
    plt.close()

# Step 5: Save curve data to files
def save_curve_data(roc_data, pr_data, tprs, precisions, mean_fpr, mean_recall, base_name):
    roc_df = pd.concat(roc_data, ignore_index=True)
    roc_df.to_csv(f'{base_name}_roc_data.csv', index=False)
    print(f"Saved per-fold ROC data to {base_name}_roc_data.csv")
    
    pr_df = pd.concat(pr_data, ignore_index=True)
    pr_df.to_csv(f'{base_name}_pr_data.csv', index=False)
    print(f"Saved per-fold PR data to {base_name}_pr_data.csv")
    
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_roc_df = pd.DataFrame({
        'fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr
    })
    mean_roc_df.to_csv(f'{base_name}_mean_roc_data.csv', index=False)
    print(f"Saved mean ROC curve data to {base_name}_mean_roc_data.csv")
    
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    mean_pr_df = pd.DataFrame({
        'recall': mean_recall,
        'mean_precision': mean_precision,
        'std_precision': std_precision
    })
    mean_pr_df.to_csv(f'{base_name}_mean_pr_data.csv', index=False)
    print(f"Saved mean PR curve data to {base_name}_mean_pr_data.csv")

# Main function
# emb_method = "gtn"
def main():
    # embedding_size_drug = 128
    # epochs = 100

    # embedding_size_disease = 384
    

    # print(f"\nEmbedding info:")
    # print(f"embedding_size_drug: {embedding_size_drug} and epochs: {epochs}")
    # print(f"embedding_size_disease: {embedding_size_disease}")

    # phases = [1, 3]
    # chromosomes = list(range(1, 23))  # 1 to 22
    # ld_thresholds = ["r2def", "r208"]
    
    # phases = [1]
    # chromosomes = list(range(21, 23))  # 1 to 22
    # ld_thresholds = ["r208"]

    # Collect results for summary
    
    drug_emb_files = []
    
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_128_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_128_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_128_e_400")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_256_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_256_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_256_e_400")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_512_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_512_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_512_e_400")

    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_128_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_128_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_128_e_400")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_256_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_256_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_256_e_400")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_512_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_512_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_512_e_400")
    
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_128_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_128_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_128_e_400")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_256_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_256_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_256_e_400")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_512_e_100")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_512_e_200")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_512_e_400")
    
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_64_e_50")    
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_64_e_50")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_64_e_50")
    # drug_emb_files.append("DrugSimNet_PREDICT_gat_d_256_e_800")
    # drug_emb_files.append("DrugSimNet_PREDICT_gtn_d_256_e_800")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_128_e_800")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_256_e_800")
    # drug_emb_files.append("DrugSimNet_PREDICT_mp2v_d_512_e_800")
    

    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_128_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_256_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_512_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_128_e_200")
    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_256_e_200")
    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_512_e_200")
    drug_emb_files.append("DrugSimNet_CHEM_gtn_d_128_e_400")
    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_256_e_400")
    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_512_e_400")

    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_128_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_256_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_512_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_128_e_200")
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_256_e_200")
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_512_e_200")
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_128_e_400")
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_256_e_400")
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_512_e_400")
    
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_128_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_128_e_200")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_128_e_400")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_256_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_256_e_200")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_256_e_400")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_512_e_100")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_512_e_200")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_512_e_400")
    
    # drug_emb_files.append("DrugSimNet_CHEM_gat_d_64_e_50")
    # drug_emb_files.append("DrugSimNet_CHEM_gtn_d_64_e_50")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_128_e_50")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_128_e_800")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_256_e_800")
    # drug_emb_files.append("DrugSimNet_CHEM_mp2v_d_512_e_800")

    # disease_emb_files = ["BlueBERT_768", "BlueBERT_1024","PubMedBERT_Timofey","PubMedBERT_ml4pubmed","PubMedBERT_microsoft"]
    # disease_emb_files = ["sbert_384", "w2v_128","w2v_256","w2v_512","w2v_768"]
    
    disease_nets = ["DiseaseSimNet_OMIM", "DiseaseSimNet_HPO", "DiseaseSimNet_GeneNet", "DiseaseSimNet_OHG", "DiseaseSimNet_OG"]
    # disease_nets = ["DiseaseSimNet_OG"]
    emb_methods = ["gat", "gtn", "mp2v"]
    # emb_methods = ["gat"]
    disease_emb_files = []
    for disease_net in disease_nets:
        for emb_method in emb_methods:
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_128_e_100")
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_128_e_200")
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_128_e_400")
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_256_e_100")
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_256_e_200")
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_256_e_400")
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_512_e_100")
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_512_e_200")
            disease_emb_files.append(f"{disease_net}_{emb_method}_d_512_e_400")

    results = []
    for drug_emb_file in drug_emb_files:
        for disease_emb_file in disease_emb_files:
            drug_embeddings_file = f"../../121GNN4DSE/Results/{drug_emb_file}.csv"
            disease_embeddings_file = f"../Results/{disease_emb_file}.csv"
            drug_disease_file = os.path.expanduser(f"../Data/Drug2Disease_Name_PREDICT_ID.txt_BinaryInteraction.csv")
            
            base_name_drug = os.path.splitext(os.path.basename(drug_embeddings_file))[0]
            base_name_disease = os.path.splitext(os.path.basename(disease_embeddings_file))[0]
            
            base_name = base_name_drug + "_" + base_name_disease + "_Balanced_XGB"

            print(f"\nProcessing pair:")
            print(f"drug_embeddings_file: {drug_embeddings_file}")
            print(f"disease_embeddings_file: {disease_embeddings_file}")
            print(f"Drug-disease file: {drug_disease_file}")
            
            output_file = f'../Results_Detail/{base_name}_output.txt'
            tee = Tee(output_file)
            sys.stdout = tee
            
            try:
                drugs, diseases, drug_emb, disease_emb, positive_pairs, drug_emb_size, disease_emb_size = load_data(
                    drug_embeddings_file, disease_embeddings_file, drug_disease_file
                )
                
                features, labels, pairs = generate_features_labels(
                    drugs, diseases, drug_emb, disease_emb, positive_pairs, drug_emb_size, disease_emb_size
                )
                
                # from sklearn.manifold import TSNE
                # import seaborn as sns
                # tsne = TSNE(n_components=2, random_state=42)
                # embeddings_2d = tsne.fit_transform(features)
                # sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels)
                # plt.savefig(f'{base_name}_tsne.png')
                # plt.close()
                
                auroc_mean, auroc_std, auprc_mean, auprc_std, f1_mean, f1_std, accuracy_mean, accuracy_std = evaluate_model(features, labels, base_name)
                
                # Collect results
                results.append({
                    'drug_emb_file': drug_emb_file,
                    'disease_emb_file': disease_emb_file,
                    'auroc_mean': auroc_mean,
                    'auroc_std': auroc_std,
                    'auprc_mean': auprc_mean,
                    'auprc_std': auprc_std,
                    'f1_mean': f1_mean,
                    'f1_std': f1_std,
                    'accuracy_mean': accuracy_mean,
                    'accuracy_std': accuracy_std
                })
            
            except Exception as e:
                print(f"Error processing: {str(e)}")
            
            finally:
                sys.stdout = tee.stdout
                tee.close()
            
    # Save summary results to a CSV file
    # emb_str = "_".join(emb_methods)
    summary_df = pd.DataFrame(results)
    summary_file = f"../Results/Performance_summary_metrics.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary results saved to {summary_file}")

# Execute
if __name__ == "__main__":
    main()