import streamlit as st

st.subheader("Sentiment Analysis with Hugging Face")
st.write("Analyze reviews and comments from your users with comment analyzer.")
st.page_link("pages/comment_analyzer.py", icon="🔎")

st.subheader("Customer Segmentation - EDA and Prediction")
st.write("Let's analyze the segmentation of a retail store. You can input values to predict its segment.")
st.page_link("pages/customer_segmentation.py", icon="🔮")
