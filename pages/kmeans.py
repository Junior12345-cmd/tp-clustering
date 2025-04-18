import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO

st.title("TP2 : Clustering K-means")

# Fonction pour calculer la distance euclidienne
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Implémentation manuelle de K-means
class ManualKMeans:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        
    def fit(self, X):
        np.random.seed(self.random_state)
        
        # Initialisation aléatoire des centroïdes
        random_idx = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centroids = X[random_idx]
        
        for _ in range(self.max_iter):
            # Assigner chaque point au centroïde le plus proche
            distances = np.zeros((X.shape[0], self.n_clusters))
            for i in range(self.n_clusters):
                distances[:, i] = np.array([euclidean_distance(x, self.centroids[i]) for x in X])
            
            new_labels = np.argmin(distances, axis=1)
            
            # Vérifier la convergence
            if hasattr(self, 'labels_') and np.all(new_labels == self.labels_):
                break
                
            self.labels_ = new_labels
            
            # Mettre à jour les centroïdes
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                cluster_points = X[self.labels_ == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = X[np.random.randint(0, X.shape[0])]
            
            self.centroids = new_centroids
        
        # Calculer l'inertie (somme des distances au carré)
        self.inertia_ = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum((cluster_points - self.centroids[i])**2)
    
    def predict(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.array([euclidean_distance(x, self.centroids[i]) for x in X])
        return np.argmin(distances, axis=1)

# Fonction pour calculer le score de silhouette
def manual_silhouette_score(X, labels):
    n = len(X)
    silhouette_scores = np.zeros(n)
    
    for i in range(n):
        # Calculer a(i): distance moyenne aux autres points du même cluster
        cluster_i = labels[i]
        same_cluster = X[labels == cluster_i]
        a_i = np.mean([euclidean_distance(X[i], x) for x in same_cluster if not np.array_equal(X[i], x)])
        
        # Calculer b(i): distance moyenne aux points du cluster le plus proche
        other_clusters = [c for c in np.unique(labels) if c != cluster_i]
        b_i = np.inf
        
        for c in other_clusters:
            other_cluster_points = X[labels == c]
            mean_dist = np.mean([euclidean_distance(X[i], x) for x in other_cluster_points])
            if mean_dist < b_i:
                b_i = mean_dist
        
        # Calculer le score de silhouette pour ce point
        if a_i == 0 and b_i == 0:
            silhouette_scores[i] = 0
        else:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    return np.mean(silhouette_scores)

# Fonction pour calculer l'indice de Davies-Bouldin
def manual_davies_bouldin_score(X, labels):
    n_clusters = len(np.unique(labels))
    cluster_centers = []
    cluster_sizes = []
    cluster_dispersions = []
    
    # Calculer les centroïdes et dispersions pour chaque cluster
    for c in np.unique(labels):
        cluster_points = X[labels == c]
        centroid = np.mean(cluster_points, axis=0)
        cluster_centers.append(centroid)
        cluster_sizes.append(len(cluster_points))
        
        # Dispersion: distance moyenne des points au centroïde
        dispersion = np.mean([euclidean_distance(x, centroid) for x in cluster_points])
        cluster_dispersions.append(dispersion)
    
    # Calculer l'indice DB
    db_index = 0
    for i in range(n_clusters):
        max_ratio = -np.inf
        for j in range(n_clusters):
            if i != j:
                # Distance entre centroïdes
                centroid_dist = euclidean_distance(cluster_centers[i], cluster_centers[j])
                # Ratio (dispersion_i + dispersion_j) / distance_centroïdes
                ratio = (cluster_dispersions[i] + cluster_dispersions[j]) / centroid_dist
                if ratio > max_ratio:
                    max_ratio = ratio
        db_index += max_ratio
    
    return db_index / n_clusters

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

# Fonction pour normaliser les données
def manual_standard_scaler(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1  # Éviter la division par zéro
    return (X - means) / stds, means, stds

# Fonction pour appliquer la normalisation
def manual_transform(X, means, stds):
    return (X - means) / stds

# Fonction pour l'ACP manuelle (optionnel)
def manual_pca(X, n_components=2):
    # Centrer les données
    X_centered = X - np.mean(X, axis=0)
    
    # Calculer la matrice de covariance
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Calculer les valeurs et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Trier par valeur propre décroissante
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    eigenvalues = eigenvalues[sorted_idx]
    
    # Sélectionner les n premiers composants
    components = eigenvectors[:, :n_components]
    
    # Projeter les données
    X_pca = np.dot(X_centered, components)
    
    # Variance expliquée
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, explained_variance

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
        X_scaled, means, stds = manual_standard_scaler(X)
        
        # Sélection du nombre de clusters
        st.subheader("Paramètres du clustering")
        k = st.number_input("Nombre de clusters (K)", min_value=2, max_value=10, value=3, key='k_value')
        
        # Exécution du K-means manuel
        kmeans = ManualKMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_  # Maintenant utilisant labels_ au lieu de labels
        
        # Ajout des labels au dataframe
        data['Cluster'] = labels
        
        # Affichage des centroïdes et écarts-types
        st.subheader("Résultats du clustering")
        
        # Calcul des centroïdes (dans l'espace original)
        centroids_original = kmeans.centroids * stds + means
        centroids_df = pd.DataFrame(centroids_original, columns=numeric_cols)
        centroids_df['Cluster'] = centroids_df.index
        
        st.write("Centroïdes des clusters:")
        st.dataframe(centroids_df)
        
        # Calcul des écarts-types par cluster
        st.write("Écarts-types par cluster:")
        std_df = data.groupby('Cluster')[numeric_cols].std()
        st.dataframe(std_df)
        
        # Métriques de qualité
        st.subheader("Métriques de qualité du clustering")
        silhouette = manual_silhouette_score(X_scaled, labels)
        davies_bouldin = manual_davies_bouldin_score(X_scaled, labels)
        
        st.write(f"Score de silhouette (Combine intra et inter): {silhouette:.3f}")
        st.write(f"Indice de Davies-Bouldin (Compare dispersion intra-cluster et distance inter-cluster): {davies_bouldin:.3f}")
        
        # Visualisation
        st.subheader("Visualisation des clusters")
        
        # Réduction de dimension si plus de 2 features
        if len(numeric_cols) > 2:
            X_reduced, explained_var = manual_pca(X_scaled, n_components=2)
            reduced_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
            reduced_df['Cluster'] = labels
            explained_var_sum = np.sum(explained_var)
            
            fig = px.scatter(reduced_df, x='PC1', y='PC2', color='Cluster',
                            title=f"Clusters (PCA - Variance expliquée: {explained_var_sum:.2f})")
            st.plotly_chart(fig)
            
            # Affichage des centroïdes dans l'espace réduit
            centroids_reduced, _ = manual_pca(kmeans.centroids, n_components=2)
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
        
        # Prédiction pour un nouveau point
        st.subheader("Prédiction pour un nouveau point")
        new_point = {}
        for col in numeric_cols:
            new_point[col] = st.number_input(f"Valeur pour {col}", value=0.0, key=f'new_{col}')
        
        if st.button("Prédire le cluster"):
            new_data = np.array([list(new_point.values())])
            new_data_scaled = manual_transform(new_data, means, stds)
            predicted_cluster = kmeans.predict(new_data_scaled)[0]
            st.success(f"Le point appartient au cluster {predicted_cluster}")
        
        # Analyse pour différents K
        st.subheader("Analyse pour différents K")
        max_k = st.slider("K maximum à tester", min_value=2, max_value=10, value=5, key='max_k')
        
        if st.button("Analyser les K"):
            silhouette_scores = []
            db_scores = []
            k_values = range(2, max_k+1)
            
            for k_val in k_values:
                kmeans_temp = ManualKMeans(n_clusters=k_val, random_state=42)
                kmeans_temp.fit(X_scaled)
                labels_temp = kmeans_temp.labels_
                silhouette_scores.append(manual_silhouette_score(X_scaled, labels_temp))
                db_scores.append(manual_davies_bouldin_score(X_scaled, labels_temp))
            
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
