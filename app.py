import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Global Development Clustering", layout="wide")
st.title("🌍 Global Development Clustering App")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel('World_development_mesurement.xlsx')
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
    
    return df

df = load_data()

# --- Preprocessing ---
numeric_df = df.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# --- Sidebar Options ---
model_choice = st.sidebar.selectbox("Choose Clustering Model", ["KMeans", "DBSCAN"])
n_clusters = st.sidebar.slider("Number of Clusters (KMeans)", min_value=2, max_value=10, value=4)
eps = st.sidebar.slider("DBSCAN eps", min_value=0.5, max_value=5.0, value=1.5)
min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=3, max_value=10, value=5)

# --- Clustering ---
if model_choice == "KMeans":
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = labels
    score = silhouette_score(X_scaled, labels)
    st.subheader(f"KMeans Silhouette Score: {score:.3f}")

elif model_choice == "DBSCAN":
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    df['Cluster'] = labels
    if len(set(labels)) > 1 and -1 in labels:
        score = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
        st.subheader(f"DBSCAN Silhouette Score (excluding noise): {score:.3f}")
    else:
        st.subheader("DBSCAN clustering too sparse for silhouette score.")

# --- Display Results ---
if 'Country' in df.columns:
    st.dataframe(df[['Country', 'Cluster']])
else:
    st.dataframe(df[['Cluster']])

# --- PCA Visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='tab10')
ax.set_title("Cluster Visualization (PCA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)
