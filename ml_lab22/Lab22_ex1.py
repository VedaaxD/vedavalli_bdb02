import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from ISLP import load_data

# Load NCI60 data
NCI60 = load_data('NCI60')
data = NCI60['data']
labels = NCI60['labels'].values.ravel()
print(labels)

# Standardize gene expression data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split data BEFORE PCA and clustering
X_train_raw, X_test_raw, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.3, random_state=42)
print(y_train)

# --------------------------------------------
# 1. Dimensionality Reduction using PCA (Proper way)
# --------------------------------------------
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_raw)
X_test_pca = pca.transform(X_test_raw)

# --------------------------------------------
# 2. Dimensionality Reduction using Hierarchical Clustering
# --------------------------------------------

# Cluster genes (i.e., columns) using ONLY training data
clustering = AgglomerativeClustering(n_clusters=50)
gene_clusters = clustering.fit_predict(X_train_raw.T)

# For each cluster, use mean of the genes in that cluster
def reduce_by_clusters(data_matrix, gene_clusters):
    reduced_data = []
    for cluster_id in np.unique(gene_clusters):
        cluster_genes = data_matrix[:, gene_clusters == cluster_id]
        reduced_data.append(cluster_genes.mean(axis=1))
    return np.array(reduced_data).T

X_train_hier = reduce_by_clusters(X_train_raw, gene_clusters)
X_test_hier = reduce_by_clusters(X_test_raw, gene_clusters)

# --------------------------------------------
# 3. Train models and evaluate
# --------------------------------------------

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_hier, y_train)
y_pred_hier = log_reg.predict(X_test_hier)

log_reg.fit(X_train_pca, y_train)
y_pred_pca = log_reg.predict(X_test_pca)

# Support Vector Classifier (SVC)
svc = SVC()
svc.fit(X_train_hier, y_train)
y_pred_hier_svc = svc.predict(X_test_hier)

svc.fit(X_train_pca, y_train)
y_pred_pca_svc = svc.predict(X_test_pca)

# Accuracy comparison
print(f"Logistic Regression (Hierarchical Clustering): {accuracy_score(y_test, y_pred_hier):.4f}")
print(f"Logistic Regression (PCA): {accuracy_score(y_test, y_pred_pca):.4f}")
print(f"SVC (Hierarchical Clustering): {accuracy_score(y_test, y_pred_hier_svc):.4f}")
print(f"SVC (PCA): {accuracy_score(y_test, y_pred_pca_svc):.4f}")

# Confusion Matrices
print("Confusion Matrix (Hierarchical Clustering, Logistic Regression):")
print(confusion_matrix(y_test, y_pred_hier))
print("Confusion Matrix (PCA, Logistic Regression):")
print(confusion_matrix(y_test, y_pred_pca))

# Classification Reports
print("Classification Report (Hierarchical Clustering, Logistic Regression):")
print(classification_report(y_test, y_pred_hier,zero_division=0))
print("Classification Report (PCA, Logistic Regression):")
print(classification_report(y_test, y_pred_pca,zero_division=0))
