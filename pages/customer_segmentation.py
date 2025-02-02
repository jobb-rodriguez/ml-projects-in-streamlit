import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
# Specific imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

st.info("Customer Segmentation | K-Means Clustering", icon="ℹ️")
st.write("[The Customer Segmentation Dataset](https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset/data) is a Kaggle Dataset uploaded by M Yasser H. It was last updated in 2022.")

@st.cache_data
def get_data_dictionary():
  data_dictionary = pd.read_csv("data/customer_segmentation_data_dictionary.csv")
  return data_dictionary

@st.cache_data
def get_data():
  df = pd.read_csv("data/online_retail.csv")
  return df 

df = get_data()
st.subheader("Data Dictionary")
st.write("The data has 541,909 records.")

data_dictionary = get_data_dictionary()
st.table(data_dictionary)

st.subheader("Situation")
st.write("Abakada Inc. has a budget of $5,000.00 allocated to running ads. They want to maximize the ROI on ad spend. As a Data Scientist, help them strategize.")

st.subheader("1. Clean Data")
st.write("There is no missing data. Group customers according to Customer ID, Country, and Description (Proudct).")

# Process
#1 Correct CustomerID Format
df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
df['CustomerID'] = df['CustomerID'].astype(str).replace('<NA>', 'NA')

#2 Drop unnecessary columns
df.drop(["InvoiceNo", "StockCode", "InvoiceDate"], axis=1, inplace=True)

df_country = df.groupby("Country").agg({"CustomerID": "size", "Quantity": "sum", "UnitPrice": "sum"}).reset_index()
df_country = df_country.rename(columns={"CustomerID": "CustomerID_Count"})
df_description = df.groupby("Description").agg({"CustomerID": "size", "Quantity": "sum", "UnitPrice": "sum"}).reset_index()
df_description = df_country.rename(columns={"CustomerID": "CustomerID_Count"})

st.subheader("2. Train-Validation-Test Split")
st.write("""Before EDA, split the data to avoid data leakage.
- 60%: Train
- 20%: Validation
- 20%: Test

Codes for the groups:
- ID: Customer ID
- C: Country
- D: Description""")

C_train, C_validation = train_test_split(df_country, test_size=0.4)
C_validation, C_test = train_test_split(C_validation, test_size=0.5)
print(len(df_country), len(C_train), len(C_validation), len(C_test))

D_train, D_validation = train_test_split(df_description, test_size=0.4)
D_validation, D_test = train_test_split(D_validation, test_size=0.5)
print(len(df_description), len(D_train), len(D_validation), len(D_test))

st.subheader("3. Selecting variables")
st.write("Clustering customers according to quantity and unit price makes the most sense.")

st.subheader("4. Scale Numerical Data")
st.write("Use ```MinMaxScaler```.")

def apply_scaler(X_train):
    columns_to_scale = ['CustomerID_Count', 'Quantity', 'UnitPrice']
    scaler = MinMaxScaler()
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    return X_train

C_train_scaled = apply_scaler(C_train)
D_train_scaled = apply_scaler(D_train)

st.subheader("5. Choose the number of clusters")
st.write("Visual Assessment for Country")

cluster_cols = ['Quantity', 'UnitPrice']
markers = ['x', '*', '.', '|', '_', '1', '2']

plt.figure(figsize=[12,8])
for n in range(2,8):
    model = KMeans(n_clusters=n, random_state=42)
    C_train_scaled['Cluster']= model.fit_predict(C_train_scaled[cluster_cols])

    plt.subplot(2,3, n-1)
    for clust in range(n):
        temp = C_train_scaled[C_train_scaled.Cluster == clust]
        plt.scatter(temp.Quantity, temp.UnitPrice,
        marker=markers[clust],
        label="Cluster "+str(clust))
        plt.title("N clusters: "+str(n))
        plt.xlabel('Quantity')
        plt.ylabel('Unit Price')
        plt.legend()
st.pyplot(plt)

st.write("Visual Assessment for Country")
cluster_cols = ['Quantity', 'UnitPrice']
markers = ['x', '*', '.', '|', '_', '1', '2']

plt.figure(figsize=[12,8])
for n in range(2,8):
    model = KMeans(n_clusters=n, random_state=42)
    D_train_scaled['Cluster']= model.fit_predict(D_train_scaled[cluster_cols])

    plt.subplot(2,3, n-1)
    for clust in range(n):
        temp = D_train_scaled[D_train_scaled.Cluster == clust]
        plt.scatter(temp.Quantity, temp.UnitPrice,
        marker=markers[clust],
        label="Cluster "+str(clust))
        plt.title("N clusters: "+str(n))
        plt.xlabel('Quantity')
        plt.ylabel('Unit Price')
        plt.legend()
st.pyplot(plt)

st.write("""Elbow Method
1. Country: Four (4) is the optimal cluster count.
2. Description: Six (6) is the optimal cluster count.""")

C = C_train_scaled[cluster_cols]

C_inertia_scores = []
for K in range(2,11):
  inertia = KMeans(n_clusters=K, random_state=42).fit(C).inertia_
  C_inertia_scores.append(inertia)

plt.figure(figsize=[7,5])
plt.plot(range(2,11), C_inertia_scores)
plt.title("SSE/Inertia vs. number of clusters")
plt.xlabel("Number of clusters: K")
plt.ylabel('SSE/Inertia')
st.pyplot(plt)

D = D_train_scaled[cluster_cols]

D_inertia_scores = []
for K in range(2,11):
  inertia = KMeans(n_clusters=K, random_state=42).fit(D).inertia_
  D_inertia_scores.append(inertia)

plt.figure(figsize=[7,5])
plt.plot(range(2,11), D_inertia_scores)
plt.title("SSE/Inertia vs. number of clusters")
plt.xlabel("Number of clusters: K")
plt.ylabel('SSE/Inertia')
st.pyplot(plt)

st.subheader("6. Conclusion")
st.write("""### Ad Strategy
When to use: Google Ads, Facebook Ads, Tiktok Ads would be most applicable due to targeting according to country.
- Allocate \$3,000.00 to Cluster 0 (Low Quantity, Low Unit Price)--aim for volume
- Allocate \$1,000.00 to Cluster 3 (Relatively Low Quantity, Fair Unit Price)
- Allocate \$500.00 to Cluster 1 (High Quantity, High Unit Price)
- Allocate \$500.00 to Cluster 2 (Relatively High Quantity, Relatively Low Unit Price)""")

cluster_cols = ['Quantity', 'UnitPrice']
markers = ['x', '*', '.', '|', '_', '1', '2']

plt.figure(figsize=[12,8])
model = KMeans(n_clusters=n, random_state=42)
C_train_scaled['Cluster']= model.fit_predict(C_train_scaled[cluster_cols])

plt.subplot(2,3, n-1)
for clust in range(4):
    temp = C_train_scaled[C_train_scaled.Cluster == clust]
    plt.scatter(temp.Quantity, temp.UnitPrice,
    marker=markers[clust],
    label="Cluster "+str(clust))
    plt.title("N clusters: "+str(4))
    plt.xlabel('Quantity')
    plt.ylabel('Unit Price')
    plt.legend()

st.pyplot(plt)

st.write("""### Email Strategy
Send a target email per cluster.""")
from sklearn.cluster import KMeans
cluster_cols = ['Quantity', 'UnitPrice']
markers = ['x', '*', '.', '|', '_', '1', '2']

plt.figure(figsize=[12,8])
model = KMeans(n_clusters=n, random_state=42)
D_train_scaled['Cluster']= model.fit_predict(D_train_scaled[cluster_cols])

plt.subplot(2,3, n-1)
for clust in range(6):
    temp = D_train_scaled[D_train_scaled.Cluster == clust]
    plt.scatter(temp.Quantity, temp.UnitPrice,
    marker=markers[clust],
    label="Cluster "+str(clust))
    plt.title("N clusters: "+str(6))
    plt.xlabel('Quantity')
    plt.ylabel('Unit Price')
    plt.legend()
st.pyplot(plt)
