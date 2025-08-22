import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm
import itertools
import os
import sys
import random
import heapq

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

    drug_idx = {drug: i for i, drug in enumerate(drugs)}
    disease_idx = {disease: i for i, disease in enumerate(diseases)}

    positive_pairs_list = list(positive_pairs)
    num_positive = len(positive_pairs_list)
    print(f"Number of positive pairs: {num_positive}")

    all_pairs = itertools.product(drugs, diseases)
    negative_pairs = [(d, dis) for d, dis in all_pairs if (d, dis) not in positive_pairs]
    print(f"Total negative pairs available: {len(negative_pairs)}")

    if len(negative_pairs) < num_positive:
        print("Warning: Fewer negative pairs than positive pairs. Using all available negative pairs.")
        selected_negative_pairs = negative_pairs
    else:
        selected_negative_pairs = random.sample(negative_pairs, num_positive)

    selected_pairs = positive_pairs_list + selected_negative_pairs
    random.shuffle(selected_pairs)

    for drug, disease in tqdm(selected_pairs, desc="Generating features for training pairs"):
        label = 1 if (drug, disease) in positive_pairs else 0
        labels.append(label)

        drug_vec = drug_emb[drug_idx[drug]]
        disease_vec = disease_emb[disease_idx[disease]]
        feature_vec = np.concatenate([drug_vec, disease_vec])
        features.append(feature_vec)

    return np.array(features), np.array(labels)

# Step 3: Train XGBoost model and predict novel associations (memory safe)
def train_and_predict(drugs, diseases, drug_emb, disease_emb, positive_pairs, embedding_size_drug, embedding_size_disease,
                      drug_embeddings_file, disease_embeddings_file, base_name, k=100, batch_size=500_000):

    print("Training XGBoost model on all labeled data...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    # Generate training data
    features, labels = generate_features_labels(drugs, diseases, drug_emb, disease_emb,
                                                positive_pairs, embedding_size_drug, embedding_size_disease)

    # Train model
    xgb.fit(features, labels)

    # Evaluate model on training data
    y_pred_proba = xgb.predict_proba(features)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auroc = roc_auc_score(labels, y_pred_proba)
    auprc = average_precision_score(labels, y_pred_proba)
    f1 = f1_score(labels, y_pred)
    accuracy = accuracy_score(labels, y_pred)

    if auroc > 0.999:
        print("Warning: AUROC=1.0000 or very high, indicating potential overfitting or data leakage.")

    print("\nTraining Data Performance:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics_df = pd.DataFrame([{
        'drug_emb_file': drug_embeddings_file,
        'disease_emb_file': disease_embeddings_file,
        'auroc_mean': auroc,
        'auroc_std': 0.0,
        'auprc_mean': auprc,
        'auprc_std': 0.0,
        'f1_mean': f1,
        'f1_std': 0.0,
        'accuracy_mean': accuracy,
        'accuracy_std': 0.0
    }])
    metrics_csv = f'../Results_Detail/{base_name}_model_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False, float_format="%.4f")
    print(f"Saved model metrics to {metrics_csv}")

    # --- Predict for ALL novel pairs in batches ---
    print("Generating predictions for novel drug-disease associations (batched)...")
    drug_idx = {drug: i for i, drug in enumerate(drugs)}
    disease_idx = {disease: i for i, disease in enumerate(diseases)}

    all_pairs_iter = itertools.product(drugs, diseases)
    novel_pairs_iter = (pair for pair in all_pairs_iter if pair not in positive_pairs)

    top_k_heap = []  # min-heap for top k
    batch_pairs = []
    batch_features = []

    total_novel = len(drugs) * len(diseases) - len(positive_pairs)
    for pair in tqdm(novel_pairs_iter, total=total_novel, desc="Processing batches"):
        drug, disease = pair
        drug_vec = drug_emb[drug_idx[drug]]
        disease_vec = disease_emb[disease_idx[disease]]
        batch_features.append(np.concatenate([drug_vec, disease_vec]))
        batch_pairs.append(pair)

        if len(batch_features) >= batch_size:
            probs = xgb.predict_proba(np.array(batch_features))[:, 1]
            for p, prob in zip(batch_pairs, probs):
                if len(top_k_heap) < k:
                    heapq.heappush(top_k_heap, (prob, p))
                else:
                    heapq.heappushpop(top_k_heap, (prob, p))
            batch_pairs.clear()
            batch_features.clear()

    # Process last batch
    if batch_features:
        probs = xgb.predict_proba(np.array(batch_features))[:, 1]
        for p, prob in zip(batch_pairs, probs):
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (prob, p))
            else:
                heapq.heappushpop(top_k_heap, (prob, p))

    # Sort results
    top_k_heap.sort(reverse=True, key=lambda x: x[0])
    top_k_probs, top_k_pairs = zip(*top_k_heap)

    predictions_df = pd.DataFrame({
        'drug': [pair[0] for pair in top_k_pairs],
        'disease': [pair[1] for pair in top_k_pairs],
        'predicted_probability': top_k_probs
    })
    predictions_csv = f'../Results_Detail/{base_name}_top_{k}_predictions.csv'
    predictions_df.to_csv(predictions_csv, index=False, float_format="%.4f")
    print(f"Saved top {k} predictions to {predictions_csv}")

    return auroc, 0.0, auprc, 0.0, f1, 0.0, accuracy, 0.0

# Main function
def main():
    disease_net = "DiseaseSimNet_OG"
    emb_method = "gat"
    emb_size = 512
    epoch = 100

    drug_embeddings_file = f"../../121GNN4DSE/Results/DrugSimNet_CHEM_{emb_method}_d_{emb_size}_e_{epoch}.csv"
    disease_emb_file = f"{disease_net}_{emb_method}_d_{emb_size}_e_{epoch}"
    disease_embeddings_file = f"../Results/{disease_emb_file}.csv"
    drug_disease_file = os.path.expanduser("../Data/Drug2Disease_Name_PREDICT_ID.txt_BinaryInteraction.csv")

    base_name_drug = os.path.splitext(os.path.basename(drug_embeddings_file))[0]
    base_name_disease = os.path.splitext(os.path.basename(disease_embeddings_file))[0]
    base_name = base_name_drug + "_" + base_name_disease + "_Balanced_XGB"

    output_file = f'../Results_Detail/{base_name}_output.txt'
    tee = Tee(output_file)
    sys.stdout = tee

    try:
        drugs, diseases, drug_emb, disease_emb, positive_pairs, drug_emb_size, disease_emb_size = load_data(
            drug_embeddings_file, disease_embeddings_file, drug_disease_file
        )

        train_and_predict(
            drugs, diseases, drug_emb, disease_emb, positive_pairs, drug_emb_size, disease_emb_size,
            drug_embeddings_file, disease_embeddings_file, base_name, k=100000000, batch_size=500_000
        )

    except Exception as e:
        print(f"Error processing: {str(e)}")

    finally:
        sys.stdout = tee.stdout
        tee.close()

if __name__ == "__main__":
    main()
