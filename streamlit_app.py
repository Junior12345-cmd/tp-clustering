import streamlit as st

# Titre de l'application
st.title("TP2 : Clustering K-means/Hclust")
st.write("Bienvenue sur l'application de test.")

# Bouton qui simule la redirection vers la page K-means
if st.button("Accéder à la page K-means"):
    st.switch_page("kmeans")  # Assure-toi que le nom est exact et sans l'extension .py
