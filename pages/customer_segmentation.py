import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

st.write("[The Customer Segmentation Dataset](https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset/data) is a Kaggle Dataset uploaded by M Yasser H. It was last updated in 2022.")

@st.cache_data
def get_data_dictionary():
  data_dictionary = pd.read_csv("data/customer_segmentation_data_dictionary.csv")
  return data_dictionary

@st.cache_data
def get_data():
  df = pd.read_csv("data/online_retail.csv")
  return df 

st.subheader("Data Dictionary")
st.write("The data has 541,909 records.")

data_dictionary = get_data_dictionary()
st.table(data_dictionary)

st.subheader("Situation")
st.write("Abakada Inc. has a budget of $5,000.00 allocated to running ads. They want to maximize the ROI on ad spend. As a Data Scientist, help them strategize.")

st.subheader("Recommendation")
st.write("Placeholder")

st.subheader("Numerical Overview")
df = get_data()
st.table(df.describe().drop("CustomerID", axis=1))

st.subheader("Scatter Plots")

