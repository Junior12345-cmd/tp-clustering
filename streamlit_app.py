import streamlit as st

st.set_page_config(
    page_title="TP2 Clustering",
    layout="wide",
    page_icon="🧪"
)

# Navigation principale
st.title("TP2 : Clustering")

col1, col2 = st.columns(2)
with col1:
    if st.button("K-means Clustering 🟢"):
        st.switch_page("pages/kmeans.py")
with col2:
    if st.button("Hierarchical Clustering 🔵"):
        st.switch_page("pages/hclust.py")