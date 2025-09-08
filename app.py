import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title="Clustering Global Development", layout="wide")

st.title("🌍 Clustering Global Development Data")

# --- Upload Dataset ---
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Raw Data")
    st.dataframe(df)

    # --- Preprocessing ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    string_cols = df.select_dtypes(include=['object']).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[string_cols] = df[string_cols].fillna('Unknown')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    # --- Clustering ---
    st.sidebar.header("🔧 Clustering Settings")
    n_clusters = st.sidebar.slider("Number of Clusters (KMeans & Hierarchical)", min_value=2, max_value=10, value=4)
    eps = st.sidebar.slider("DBSCAN eps", min_value=0.1, max_value=5.0, value=1.5)
    min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=2, max_value=20, value=5)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    df['KMeans_Cluster'] = kmeans.labels_

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_scaled)
    df['DBSCAN_Cluster'] = dbscan.labels_

    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hierarchical.fit_predict(X_scaled)
    df['Hierarchical_Cluster'] = hier_labels

    # --- Evaluation ---
    def safe_score(func, X, labels):
        try:
            return round(func(X, labels), 3)
        except:
            return 'N/A'

    kmeans_scores = {
        'Silhouette': safe_score(silhouette_score, X_scaled, kmeans.labels_),
        'Davies-Bouldin': safe_score(davies_bouldin_score, X_scaled, kmeans.labels_),
        'Calinski-Harabasz': safe_score(calinski_harabasz_score, X_scaled, kmeans.labels_)
    }

    dbscan_labels = dbscan.labels_
    if len(set(dbscan_labels)) > 1 and -1 in dbscan_labels:
        filtered = dbscan_labels != -1
        dbscan_scores = {
            'Silhouette': safe_score(silhouette_score, X_scaled[filtered], dbscan_labels[filtered]),
            'Davies-Bouldin': safe_score(davies_bouldin_score, X_scaled[filtered], dbscan_labels[filtered]),
            'Calinski-Harabasz': safe_score(calinski_harabasz_score, X_scaled[filtered], dbscan_labels[filtered])
        }
    else:
        dbscan_scores = {'Silhouette': 'N/A', 'Davies-Bouldin': 'N/A', 'Calinski-Harabasz': 'N/A'}

    hier_scores = {
        'Silhouette': safe_score(silhouette_score, X_scaled, hier_labels),
        'Davies-Bouldin': safe_score(davies_bouldin_score, X_scaled, hier_labels),
        'Calinski-Harabasz': safe_score(calinski_harabasz_score, X_scaled, hier_labels)
    }

    # --- Comparison Table ---
    st.subheader("📈 Model Evaluation Comparison")
    comparison_df = pd.DataFrame({
        'Method': ['KMeans', 'DBSCAN', 'Hierarchical'],
        'Silhouette Score': [kmeans_scores['Silhouette'], dbscan_scores['Silhouette'], hier_scores['Silhouette']],
        'Davies-Bouldin Index': [kmeans_scores['Davies-Bouldin'], dbscan_scores['Davies-Bouldin'], hier_scores['Davies-Bouldin']],
        'Calinski-Harabasz Index': [kmeans_scores['Calinski-Harabasz'], dbscan_scores['Calinski-Harabasz'], hier_scores['Calinski-Harabasz']]
    })
    st.dataframe(comparison_df)

    # --- PCA Visualization ---
    st.subheader("🧬 Cluster Visualization (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    def plot_clusters(labels, title):
        fig, ax = plt.subplots()
        unique_labels = sorted(set(labels))
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            mask = (labels == k)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], label=f"Cluster {k}", edgecolors='k', s=60)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        st.pyplot(fig)

    plot_clusters(kmeans.labels_, "KMeans Clusters")
    plot_clusters(dbscan.labels_, "DBSCAN Clusters")
    plot_clusters(hier_labels, "Hierarchical Clusters")

    # --- Optional Dendrogram ---
    st.subheader("🌿 Hierarchical Dendrogram")
    linked = linkage(X_scaled, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, ax=ax)
    st.pyplot(fig)
