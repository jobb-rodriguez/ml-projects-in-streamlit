import streamlit as st

st.subheader("Sentiment Analysis with Hugging Face")
st.write("Analyze reviews and comments from your users with comment analyzer.")
st.page_link("pages/comment_analyzer.py", icon="🔎")

st.subheader("Customer Segmentation - K-Means Clustering")
st.write("Let's analyze the segmentation of a retail store.")
st.page_link("pages/customer_segmentation.py", icon="🔮")

st.subheader("Classifying Dogs and Cats - Transfer Learning")
st.write("Let's identify if an image is a dog or a cat. View the finals notebook I created when I was teaching Deep Learning in Ateneo de Naga University.")
st.page_link("pages/dogs_vs_cats.py", icon="🐶")
