import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Clustering Hiérarchique", layout="wide")

def load_data():
    st.sidebar.header("Chargement des données")
    upload_option = st.sidebar.radio("Option de chargement", 
                                   ("Fichier CSV", "Entrée manuelle"))
    
    if upload_option == "Fichier CSV":
        uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("Fichier chargé avec succès!")
            return data
        return None
    else:
        st.sidebar.warning("Option manuelle - Entrez vos données")
        num_points = st.sidebar.number_input("Nombre de points", min_value=2, value=5)
        num_features = st.sidebar.number_input("Nombre de features", min_value=2, value=2)
        
        data = {}
        for i in range(num_features):
            feature_name = st.sidebar.text_input(f"Nom de la feature {i+1}", value=f"Feature_{i+1}")
            values = []
            for j in range(num_points):
                val = st.sidebar.number_input(f"Point {j+1}, {feature_name}", value=0.0)
                values.append(val)
            data[feature_name] = values
        
        if data:
            return pd.DataFrame(data)
        return None

def plot_dendrogram(Z, labels, height=600):
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=labels, orientation='left')
    plt.title("Dendrogramme")
    plt.xlabel("Distance")
    plt.ylabel("Points")
    st.pyplot(plt)

def predict_cluster(new_point, X_scaled, labels, method='ward'):
    """Prédit le cluster d'un nouveau point"""
    # Calculer les centroïdes de chaque cluster
    clusters = np.unique(labels)
    centroids = []
    for cluster in clusters:
        centroids.append(np.mean(X_scaled[labels == cluster], axis=0))
    
    centroids = np.array(centroids)
    
    # Standardiser le nouveau point
    scaler = StandardScaler()
    scaler.fit(X_scaled)  # On utilise le scaler déjà ajusté sur les données d'entraînement
    new_point_scaled = scaler.transform([new_point])[0]
    
    # Calculer la distance aux centroïdes
    distances = cdist([new_point_scaled], centroids, 'euclidean')[0]
    
    # Retourner le cluster le plus proche
    return clusters[np.argmin(distances)], distances

def main():
    st.title("Clustering Hiérarchique (Hclust)")
    
    data = load_data()
    if data is None:
        return
    
    st.subheader("Aperçu des données")
    st.write(data)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Il faut au moins 2 colonnes numériques")
        return
    
    X = data[numeric_cols].values
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Paramètres du clustering
    st.subheader("Paramètres du clustering")
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Méthode de liaison",
            ['ward', 'complete', 'average', 'single'],
            help="Méthode pour calculer la distance entre clusters"
        )
    
    with col2:
        k = st.number_input(
            "Nombre de clusters (K)",
            min_value=2,
            max_value=10,
            value=3,
            help="Nombre final de clusters à former"
        )
    
    # Calcul du clustering hiérarchique
    Z = linkage(X_scaled, method=method)
    labels = fcluster(Z, t=k, criterion='maxclust')
    
    # Ajout des labels au dataframe
    data['Cluster'] = labels
    
    # Affichage du dendrogramme
    st.subheader("Dendrogramme")
    plot_dendrogram(Z, labels=[f"Point {i+1}" for i in range(len(X))])
    
    # Métriques de qualité
    st.subheader("Métriques de qualité")
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    st.write(f"Score de silhouette: {silhouette:.3f}")
    st.write(f"Indice de Davies-Bouldin: {davies_bouldin:.3f}")
    
    # Visualisation des clusters
    st.subheader("Visualisation des clusters")
    
    if len(numeric_cols) > 2:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)
        reduced_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
        reduced_df['Cluster'] = labels
        explained_var = pca.explained_variance_ratio_.sum()
        
        fig = px.scatter(
            reduced_df, x='PC1', y='PC2', color='Cluster',
            title=f"Clusters (PCA - Variance expliquée: {explained_var:.2f})"
        )
        st.plotly_chart(fig)
    else:
        fig = px.scatter(
            data, x=numeric_cols[0], y=numeric_cols[1], color='Cluster',
            title="Clusters"
        )
        st.plotly_chart(fig)
    
    # Analyse pour différents K
    st.subheader("Analyse pour différents K")
    max_k = st.slider("K maximum à tester", min_value=2, max_value=10, value=5)
    
    if st.button("Analyser les K"):
        silhouette_scores = []
        db_scores = []
        k_values = range(2, max_k+1)
        
        for k_val in k_values:
            labels_temp = fcluster(Z, t=k_val, criterion='maxclust')
            silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
            db_scores.append(davies_bouldin_score(X_scaled, labels_temp))
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_xlabel('Nombre de clusters (K)')
        ax1.set_ylabel('Score de silhouette', color='tab:blue')
        ax1.plot(k_values, silhouette_scores, 'o-', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Indice Davies-Bouldin', color='tab:red')
        ax2.plot(k_values, db_scores, 'o-', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('Analyse des métriques pour différents K')
        st.pyplot(fig)
        
        best_k_silhouette = k_values[np.argmax(silhouette_scores)]
        best_k_db = k_values[np.argmin(db_scores)]
        
        st.write(f"Meilleur K selon silhouette: {best_k_silhouette}")
        st.write(f"Meilleur K selon Davies-Bouldin: {best_k_db}")
    
    # Prédiction pour un nouveau point
    st.subheader("Prédiction de cluster pour un nouveau point")
    
    st.write("Entrez les valeurs pour chaque feature du nouveau point à prédire:")
    new_point = []
    cols = st.columns(len(numeric_cols))
    for i, col in enumerate(cols):
        val = col.number_input(f"{numeric_cols[i]}", value=0.0)
        new_point.append(val)
    
    if st.button("Prédire le cluster"):
        if len(new_point) != len(numeric_cols):
            st.error("Nombre de valeurs incorrect")
        else:
            cluster, distances = predict_cluster(new_point, X_scaled, labels, method)
            st.success(f"Le point appartient au cluster {cluster}")
            
            # Afficher les distances aux centroïdes
            st.write("Distances aux centroïdes des clusters:")
            for i, dist in enumerate(distances):
                st.write(f"Cluster {i+1}: {dist:.4f}")

if __name__ == "__main__":
    main()
