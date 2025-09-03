import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="Global Development Clustering", layout="wide")
st.title("🌍 Global Development Clustering App")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel('World_development_mesurement.xlsx')

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

    # Fill missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')

    return df

df = load_data()

# --- Preprocessing ---
numeric_df = df.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_scaled = np.nan_to_num(X_scaled)

# --- Sidebar Options ---
model_choice = st.sidebar.selectbox("Choose Clustering Model", ["KMeans", "DBSCAN"])
n_clusters = st.sidebar.slider("Number of Clusters (KMeans)", min_value=2, max_value=10, value=4)
eps = st.sidebar.slider("DBSCAN eps", min_value=0.5, max_value=5.0, value=1.5)
min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=3, max_value=10, value=5)

# --- Clustering ---
if model_choice == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
elif model_choice == "DBSCAN":
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

df['Cluster'] = labels

# --- Accuracy Button ---
if st.button("📈 Calculate Model Accuracy"):
    if model_choice == "KMeans":
        score = silhouette_score(X_scaled, labels)
        st.success(f"KMeans Silhouette Score: {score:.3f}")
    elif model_choice == "DBSCAN":
        if len(set(labels)) > 1 and -1 in labels:
            score = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
            st.success(f"DBSCAN Silhouette Score (excluding noise): {score:.3f}")
        elif len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            st.success(f"DBSCAN Silhouette Score: {score:.3f}")
        else:
            st.warning("DBSCAN clustering too sparse for silhouette score.")

# --- Display Cluster Assignments ---
st.subheader("🌍 Countries by Cluster")
if 'Country' in df.columns:
    st.dataframe(df[['Country', 'Cluster']].sort_values(by='Cluster'))
else:
    st.dataframe(df[['Cluster']].sort_values(by='Cluster'))

# --- Cluster Summary ---
st.subheader("📋 Cluster Summary")
numeric_cols = numeric_df.columns
summary_raw = df[df['Cluster'] != -1].groupby('Cluster')[numeric_cols].mean()
valid_cols = summary_raw.columns[~summary_raw.isnull().any()]
summary = summary_raw[valid_cols].round(2)
st.dataframe(summary)

# --- Cluster Profiling with Adaptive Thresholds ---
st.subheader("🧠 Cluster Profiles")

# Compute global percentiles for each feature
percentiles = df[numeric_cols].quantile([0.25, 0.5, 0.75])

for cluster_id, row in summary.iterrows():
    st.markdown(f"### Cluster {cluster_id}")
    description = []

    for col in valid_cols:
        val = row[col]
        if pd.isna(val):
            continue

        p25 = percentiles.loc[0.25, col]
        p75 = percentiles.loc[0.75, col]

        if val <= p25:
            label = f"Low {col}"
        elif val <= p75:
            label = f"Moderate {col}"
        else:
            label = f"High {col}"

        description.append(label)

    if description:
        st.write("• " + "\n• ".join(description))
    else:
        st.write("No interpretable indicators for this cluster.")

# --- PCA Visualization with Legend ---
st.subheader("📉 Cluster Visualization (PCA)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()

if model_choice == "DBSCAN":
    unique_labels = sorted(set(labels))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        label_name = "Noise" if k == -1 else f"Cluster {k}"
        class_mask = (labels == k)
        ax.scatter(X_pca[class_mask, 0], X_pca[class_mask, 1],
                   c=[col], label=label_name, edgecolors='k', s=80)

    ax.set_title("DBSCAN Clusters (PCA)")
else:
    unique_labels = sorted(set(labels))
    for k in unique_labels:
        class_mask = (labels == k)
        ax.scatter(X_pca[class_mask, 0], X_pca[class_mask, 1],
                   label=f"Cluster {k}", s=80)

    ax.set_title("KMeans Clusters (PCA)")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(title="Clusters", loc="best", fontsize='medium')
st.pyplot(fig)
