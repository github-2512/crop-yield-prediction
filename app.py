import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

# Load Models
lstm_model = tf.keras.models.load_model('lstm_model.h5')
transformer_model = tf.keras.models.load_model('transformer_model.h5')

# Functions
def simulate_iot_data(n_samples, n_features=3):
    return np.random.rand(n_samples, n_features)

def discretize(y):
    if y < 20:
        return 0
    elif y < 50:
        return 1
    else:
        return 2

def evaluate_classification(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return accuracy, precision, recall, f1

# Streamlit App
st.title("🌾 Crop Yield Prediction Website")
st.write("Upload your Time-Series + IoT Data CSV file to predict Crop Yields!")

uploaded_file = st.file_uploader("Upload your CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.values)

    iot_data = simulate_iot_data(n_samples=X_scaled.shape[0])
    X_combined = np.concatenate([X_scaled, iot_data], axis=1)

    # Reshape for models
    X_lstm = X_combined.reshape((X_combined.shape[0], X_combined.shape[1], 1))
    X_transformer = X_combined.reshape((X_combined.shape[0], X_combined.shape[1], 1))

    # Predict
    lstm_preds = lstm_model.predict(X_lstm)
    transformer_preds = transformer_model.predict(X_transformer)

    # Discretize Predictions
    lstm_preds_discretized = np.array([discretize(val) for val in lstm_preds.flatten()])
    transformer_preds_discretized = np.array([discretize(val) for val in transformer_preds.flatten()])

    # Plot Predictions
    st.subheader("📊 Prediction Comparison Plot")
    plt.figure(figsize=(10, 6))
    plt.plot(lstm_preds.flatten(), label="LSTM Predictions", linestyle='--')
    plt.plot(transformer_preds.flatten(), label="Transformer Predictions", linestyle='-.')
    plt.xlabel("Sample")
    plt.ylabel("Predicted Yield")
    plt.title("Predicted Crop Yields")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Show Metrics
    st.subheader("✅ Evaluation Metrics (Assuming true labels not available)")
    st.write("We are showing predictions only. Real evaluation needs real 'y_test'.")

    # Scatter Plot
    st.subheader("📈 Scatter Plot (LSTM vs Transformer)")
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(lstm_preds)), lstm_preds, label="LSTM", alpha=0.7)
    plt.scatter(range(len(transformer_preds)), transformer_preds, label="Transformer", alpha=0.7)
    plt.legend()
    plt.title("Scatter Plot of Predicted Yields")
    plt.grid(True)
    st.pyplot(plt)

st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This website predicts crop yield based on Time-Series + IoT data using LSTM and Transformer models.\n\n"
    "Created by [Your Name] 🚀"
)
