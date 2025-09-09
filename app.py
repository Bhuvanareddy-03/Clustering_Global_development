import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="Global Development Clustering", layout="wide")
st.title("🌍 Global Development Clustering App")

# --- Load and Clean Data ---
@st.cache_data
def load_data():
    df = pd.read_excel('World_development_mesurement.xlsx')

    # Drop unwanted column early
    df = df.drop(columns=["Number of Records"], errors="ignore")

    # Remove $ and % symbols and convert to float
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(r"[\$,%]", "", regex=True)
        try:
            df[col] = df[col].astype(float)
        except:
            pass  # Skip non-numeric columns

    # Encode 'Country' column
    if 'Country' in df.columns:
        le = LabelEncoder()
        df['Country_encoded'] = le.fit_transform(df['Country'])
        df['Country_encoded'] = df['Country_encoded'].astype(float)
        df.drop('Country', axis=1, inplace=True)

    # Convert other categorical columns to float
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    cat_cols = [col for col in cat_cols if col != "Country_encoded"]
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0].astype(float)

    # Impute missing values
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df

df = load_data()

# --- Preprocessing ---
# Re-select numeric columns AFTER cleaning
numeric_df = df.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_scaled = np.nan_to_num(X_scaled)

# --- Sidebar Options ---
model_choice = st.sidebar.selectbox("Choose Clustering Model", ["KMeans", "DBSCAN", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)
eps = st.sidebar.slider("DBSCAN eps", min_value=0.5, max_value=5.0, value=1.5)
min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=3, max_value=10, value=5)

# --- Clustering ---
if model_choice == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
elif model_choice == "DBSCAN":
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
elif model_choice == "Hierarchical":
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)

df['Cluster'] = labels

# --- Evaluation Metrics ---
def safe_score(func, X, labels):
    try:
        return round(func(X, labels), 3)
    except:
        return 'N/A'

if st.button("📈 Compare Clustering Accuracy"):
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    hc = AgglomerativeClustering(n_clusters=n_clusters).fit(X_scaled)

    db_valid = len(set(db.labels_)) > 1 and -1 in db.labels_
    db_filtered = db.labels_ != -1 if db_valid else None

    comparison_df = pd.DataFrame({
        'Method': ['KMeans', 'DBSCAN', 'Hierarchical'],
        'Silhouette Score': [
            safe_score(silhouette_score, X_scaled, km.labels_),
            safe_score(silhouette_score, X_scaled[db_filtered], db.labels_[db_filtered]) if db_valid else 'N/A',
            safe_score(silhouette_score, X_scaled, hc.labels_)
        ],
        'Davies-Bouldin Index': [
            safe_score(davies_bouldin_score, X_scaled, km.labels_),
            safe_score(davies_bouldin_score, X_scaled[db_filtered], db.labels_[db_filtered]) if db_valid else 'N/A',
            safe_score(davies_bouldin_score, X_scaled, hc.labels_)
        ],
        'Calinski-Harabasz Index': [
            safe_score(calinski_harabasz_score, X_scaled, km.labels_),
            safe_score(calinski_harabasz_score, X_scaled[db_filtered], db.labels_[db_filtered]) if db_valid else 'N/A',
            safe_score(calinski_harabasz_score, X_scaled, hc.labels_)
        ]
    })

    st.subheader("📊 Clustering Evaluation Metrics")
    st.dataframe(comparison_df)

# --- Display Cluster Assignments ---
st.subheader("🌍 Cluster Assignments")
if 'Country_encoded' in df.columns:
    st.dataframe(df[['Country_encoded', 'Cluster']].sort_values(by='Cluster'))
else:
    st.dataframe(df[['Cluster']].sort_values(by='Cluster'))

# --- Cluster Summary ---
st.subheader("📋 Cluster Summary")
summary_raw = df[df['Cluster'] != -1].groupby('Cluster')[numeric_df.columns].mean()
valid_cols = summary_raw.columns[~summary_raw.isnull().any()]
summary = summary_raw[valid_cols].round(2)
st.dataframe(summary)

# --- Cluster Profiling ---
st.subheader("🧠 Cluster Profiles")
percentiles = df[numeric_df.columns].quantile([0.25, 0.5, 0.75])
for cluster_id, row in summary.iterrows():
    st.markdown(f"### Cluster {cluster_id}")
    description = []
    for col in valid_cols:
        val = row[col]
        if pd.isna(val): continue
        p25 = percentiles.loc[0.25, col]
        p75 = percentiles.loc[0.75, col]
        if val <= p25:
            label = f"Low {col}"
        elif val <= p75:
            label = f"Moderate {col}"
        else:
            label = f"High {col}"
        description.append(label)
    st.write("• " + "\n• ".join(description) if description else "No interpretable indicators.")

# --- PCA Visualization ---
st.subheader("📉 Cluster Visualization (PCA)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
unique_labels = sorted(set(labels))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    label_name = "Noise" if k == -1 else f"Cluster {k}"
    class_mask = (labels == k)
    ax.scatter(X_pca[class_mask, 0], X_pca[class_mask, 1],
               c=[col], label=label_name, edgecolors='k', s=80)

ax.set_title(f"{model_choice} Clusters (PCA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(title="Clusters", loc="best", fontsize='medium')
st.pyplot(fig)

# --- Optional Dendrogram ---
if model_choice == "Hierarchical":
    st.subheader("🌿 Hierarchical Dendrogram")
    linked = linkage(X_scaled, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, ax=ax)
    st.pyplot(fig)
