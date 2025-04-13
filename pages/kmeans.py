import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("TP2 : Clustering K-means")

# Fonction pour charger les données
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
        else:
            return None
    else:
        st.sidebar.warning("Option manuelle - Entrez vos données ci-dessous")
        num_points = st.sidebar.number_input("Nombre de points", min_value=2, value=5, key='num_points')
        num_features = st.sidebar.number_input("Nombre de features", min_value=2, value=2, key='num_features')
        
        data = {}
        for i in range(num_features):
            feature_name = st.sidebar.text_input(f"Nom de la feature {i+1}", value=f"Feature_{i+1}", key=f'feature_{i}')
            values = []
            for j in range(num_points):
                val = st.sidebar.number_input(f"Point {j+1}, {feature_name}", value=0.0, key=f'point_{i}_{j}')
                values.append(val)
            data[feature_name] = values
        
        if data:
            df = pd.DataFrame(data)
            st.success("Données manuelles créées!")
            return df
        return None

# Fonction principale
def main():
    data = load_data()
    
    if data is not None:
        st.subheader("Aperçu des données")
        st.write(data)
        
        # Sélection des colonnes numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Il faut au moins 2 colonnes numériques pour le clustering.")
            return
        
        X = data[numeric_cols].values
        
        # Normalisation des données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Sélection du nombre de clusters
        st.subheader("Paramètres du clustering")
        k = st.number_input("Nombre de clusters (K)", min_value=2, max_value=10, value=3, key='k_value')
        
        # Exécution du K-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_
        
        # Ajout des labels au dataframe
        data['Cluster'] = labels
        
        # Affichage des centroïdes et écarts-types
        st.subheader("Résultats du clustering")
        
        # Calcul des centroïdes et écarts-types
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids, columns=numeric_cols)
        centroids_df['Cluster'] = centroids_df.index
        
        st.write("Centroïdes des clusters:")
        st.dataframe(centroids_df)
        
        # Calcul des écarts-types par cluster
        st.write("Écarts-types par cluster:")
        std_df = data.groupby('Cluster')[numeric_cols].std()
        st.dataframe(std_df)
        
        # Métriques de qualité
        st.subheader("Métriques de qualité du clustering")
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        st.write(f"Score de silhouette (Combine intra et inter): {silhouette:.3f}")
        st.write(f"Indice de Davies-Bouldin (Compare dispersion intra-cluster et distance inter-cluster): {davies_bouldin:.3f}")
        
        # Visualisation (Bonus)
        st.subheader("Visualisation des clusters")
        
        # Réduction de dimension si plus de 2 features
        if len(numeric_cols) > 2:
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_scaled)
            reduced_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
            reduced_df['Cluster'] = labels
            explained_var = pca.explained_variance_ratio_.sum()
            
            fig = px.scatter(reduced_df, x='PC1', y='PC2', color='Cluster',
                            title=f"Clusters (PCA - Variance expliquée: {explained_var:.2f})")
            st.plotly_chart(fig)
            
            # Affichage des centroïdes dans l'espace réduit
            centroids_reduced = pca.transform(kmeans.cluster_centers_)
            centroids_reduced_df = pd.DataFrame(centroids_reduced, columns=['PC1', 'PC2'])
            centroids_reduced_df['Cluster'] = centroids_reduced_df.index
            
            fig.add_scatter(x=centroids_reduced_df['PC1'], y=centroids_reduced_df['PC2'],
                          mode='markers', marker=dict(size=12, color='black', symbol='x'),
                          name='Centroïdes')
            st.plotly_chart(fig)
        else:
            # Visualisation directe pour 2D
            fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], color='Cluster',
                            title="Clusters")
            # Ajout des centroïdes
            fig.add_scatter(x=centroids_df[numeric_cols[0]], y=centroids_df[numeric_cols[1]],
                          mode='markers', marker=dict(size=12, color='black', symbol='x'),
                          name='Centroïdes')
            st.plotly_chart(fig)
        
        # Prédiction pour un nouveau point (Bonus)
        st.subheader("Prédiction pour un nouveau point")
        new_point = {}
        for col in numeric_cols:
            new_point[col] = st.number_input(f"Valeur pour {col}", value=0.0, key=f'new_{col}')
        
        if st.button("Prédire le cluster"):
            new_data = np.array([list(new_point.values())])
            new_data_scaled = scaler.transform(new_data)
            predicted_cluster = kmeans.predict(new_data_scaled)[0]
            st.success(f"Le point appartient au cluster {predicted_cluster}")
        
        # Analyse pour différents K (Bonus)
        st.subheader("Analyse pour différents K")
        max_k = st.slider("K maximum à tester", min_value=2, max_value=10, value=5, key='max_k')
        
        if st.button("Analyser les K"):
            silhouette_scores = []
            db_scores = []
            k_values = range(2, max_k+1)
            
            for k_val in k_values:
                kmeans_temp = KMeans(n_clusters=k_val, random_state=42)
                labels_temp = kmeans_temp.fit_predict(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
                db_scores.append(davies_bouldin_score(X_scaled, labels_temp))
            
            # Graphique des scores
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Nombre de clusters (K)')
            ax1.set_ylabel('Score de silhouette', color=color)
            ax1.plot(k_values, silhouette_scores, 'o-', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Indice Davies-Bouldin', color=color)
            ax2.plot(k_values, db_scores, 'o-', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Analyse des métriques pour différents K')
            st.pyplot(fig)
            
            # Suggestions pour le meilleur K
            best_k_silhouette = k_values[np.argmax(silhouette_scores)]
            best_k_db = k_values[np.argmin(db_scores)]
            
            st.write(f"Meilleur K selon silhouette: {best_k_silhouette}")
            st.write(f"Meilleur K selon Davies-Bouldin: {best_k_db}")

if __name__ == "__main__":
    main()