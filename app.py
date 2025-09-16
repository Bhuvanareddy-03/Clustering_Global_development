# clustering_global_development_app_local.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import missingno as msno

st.set_page_config(layout="wide")
st.title("🌍 Global Development Clustering")

# --- Load Local Data ---
df = pd.read_excel("World_development_mesurement.xlsx")

# --- Drop unwanted column ---
df = df.drop(columns=['Number of Records'], errors="ignore")

# --- Encode Country ---
le = LabelEncoder()
df['Country_encoded'] = le.fit_transform(df['Country'])
df.drop(['Country'], axis=1, inplace=True)

# --- Clean symbols ---
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(r"[\$,%]", "", regex=True)
    try:
        df[col] = df[col].astype(float)
    except:
        pass

# --- Impute Missing Values ---
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

if num_cols:
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
if cat_cols:
    cat_cols = [col for col in cat_cols if not df[col].isnull().all()]
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

# --- Visualizations ---
st.subheader("📊 Histograms")
for col in num_cols:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, bins=30, color="skyblue", ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)

st.subheader("📈 Correlation Matrix")
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)

st.subheader("📦 Boxplots")
for col in num_cols:
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], color="lightgreen", ax=ax)
    ax.set_title(f"Boxplot of {col}")
    st.pyplot(fig)

# --- Feature Scaling and PCA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols])

pca = PCA()
data_pca = pca.fit_transform(X_scaled)

var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
st.subheader("🔍 PCA Cumulative Variance")
fig, ax = plt.subplots()
ax.plot(var, color='red')
ax.set_xlabel('Index')
ax.set_ylabel('Cumulative Percentage')
st.pyplot(fig)

# --- Clustering ---
st.subheader("🧠 Clustering Methods")
method = st.selectbox("Choose clustering method", ["KMeans", "Hierarchical", "DBSCAN"])

if method == "KMeans":
    k = st.slider("Number of clusters (K)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(data_pca)
elif method == "Hierarchical":
    k = st.slider("Number of clusters", 2, 10, 4)
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hc.fit_predict(data_pca)
else:
    eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5)
    min_samples = st.slider("DBSCAN min_samples", 2, 10, 5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_pca)

st.subheader("🖼️ Cluster Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=50)
ax.set_title("Cluster Plot")
st.pyplot(fig)

# --- Evaluation ---
st.subheader("📏 Silhouette Score")
try:
    score = silhouette_score(data_pca, labels)
    st.success(f"Silhouette Score: {score:.3f}")
except:
    st.warning("Silhouette score could not be calculated due to noise or single cluster.")

df['Cluster'] = labels
st.subheader("📄 Clustered Data Sample")
st.dataframe(df.head())
