import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the scaler and model from .pkl files
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('clustering_model.pkl', 'rb') as model_file:
    clustering_model = pickle.load(model_file)

# Load the dataset directly
df = pd.read_csv("AI_index_db.csv")

# Streamlit interface
st.title("DBSCAN Clustering Results")

# Display dataset preview
st.subheader("Dataset Preview")
st.write(df.head())

# Extract numerical features
numerical_features = [
    "Commercial",
    "Development",
    "Government Strategy",
    "Infrastructure",
    "Operating Environment",
    "Research",
    "Talent",
    "Total score",
]

if not all(feature in df.columns for feature in numerical_features):
    st.error("Dataset is missing required features!")
else:
    # Scale the features
    X = df[numerical_features]
    X_scaled = scaler.transform(X)

    # Display best model details
    st.subheader("Best Model Details:")
    st.write(f"Best eps: {clustering_model['best_eps']}")
    st.write(f"Best min_samples: {clustering_model['best_min_samples']}")
    st.write(f"Best Silhouette Score: {clustering_model['best_score']:.4f}")

    # Add labels to the dataset
    df['Cluster'] = clustering_model['labels']

    # Display results
    st.subheader("Clustering Results:")
    st.write(df)
    st.bar_chart(df['Cluster'].value_counts())
