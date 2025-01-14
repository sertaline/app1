# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Create the dataset
data = {
    "E": [111, 209, 114, 114, 191, 209, 45, 71, 106, 209],
    "sigma_b": [1380, 688, 1044, 1132, 1289, 1743, 305, 597, 1009, 582],
    "sigma_0.2": [1215, 564, 934, 971, 992, 1573, 235, 538, 891, 378],
    "rho": [4.54, 7.99, 6.79, 4.48, 8.69, 7.75, 1.82, 2.79, 4.44, 7.85],
    "A": [14.8, 28, 11, 16.5, 15.3, 12, 11, 9.9, 10, 21],
    "R1": [1.5, 1.5, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    "R2": [5, 5, 5, 6, 5, 5, 3.5, 5, 5, 5],
    "L1": [20, 20, 13, 15, 12, 20, 15, 14.3, 14, 14],
    "fatigue_life": [1e7, 1.5e7, 2e7, 1.2e7, 3e7, 2.5e7, 0.8e7, 0.6e7, 1.1e7, 1.3e7]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Load dataset
X = df[['E', 'sigma_b', 'sigma_0.2', 'rho', 'A', 'R1', 'R2', 'L1']]
y = np.log(df['fatigue_life'])  # Logarithmic normalization for output

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Train Random Forest Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

gb_mse, gb_r2 = evaluate_model(gb_model, X_test, y_test)
rf_mse, rf_r2 = evaluate_model(rf_model, X_test, y_test)

# Save models
with open("gb_model.pkl", "wb") as f:
    pickle.dump(gb_model, f)
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Build Streamlit App
st.title("Fatigue Life Prediction App")
st.write("Predict the ultra-high-cycle fatigue life of metallic materials using Gradient Boosting and Random Forest models.")

# Input features with default values
E = st.number_input("Modulus of Elasticity (MPa)", value=200, min_value=0)
sigma_b = st.number_input("Tensile Strength (MPa)", value=1000, min_value=0)
sigma_0_2 = st.number_input("Yield Strength (MPa)", value=900, min_value=0)
rho = st.number_input("Density (g/cm3)", value=7.85, min_value=0.0, format="%.2f")
A = st.number_input("Elongation after Fracture (%)", value=15.0, min_value=0.0, format="%.1f")
R1 = st.number_input("Geometry R1 (mm)", value=1.5, min_value=0.0, format="%.1f")
R2 = st.number_input("Geometry R2 (mm)", value=5.0, min_value=0.0, format="%.1f")
L1 = st.number_input("Geometry L1 (mm)", value=20.0, min_value=0.0, format="%.1f")

# Predict and visualize
if st.button("Predict and Export Results"):

    input_data = np.array([[E, sigma_b, sigma_0_2, rho, A, R1, R2, L1]])
    try:
        gb_prediction = np.exp(gb_model.predict(input_data))[0]  # Inverse transform
        rf_prediction = np.exp(rf_model.predict(input_data))[0]  # Inverse transform

        st.write(f"Gradient Boosting Prediction: {gb_prediction:.2f} cycles")
        st.write(f"Random Forest Prediction: {rf_prediction:.2f} cycles")

        # Generate the graph
        fig, ax = plt.subplots(figsize=(8, 5))
        models = ["Gradient Boosting", "Random Forest"]
        predictions = [gb_prediction, rf_prediction]
        ax.bar(models, predictions, color=['blue', 'green'])
        ax.set_xlabel("Model")
        ax.set_ylabel("Predicted Fatigue Life (cycles)")
        ax.set_title("Fatigue Life Predictions by Model")

        # Display and save the graph
        st.pyplot(fig)
        fig.savefig("fatigue_life_predictions.png")
        st.write("Graph exported as 'fatigue_life_predictions.png' in the current directory.")
    except ValueError:
        st.write("Error: Please ensure all inputs are valid and non-empty.")

# Display model performance
st.subheader("Model Performance")
st.write(f"Gradient Boosting - MSE: {gb_mse:.4f}, R^2: {gb_r2:.4f}")
st.write(f"Random Forest - MSE: {rf_mse:.4f}, R^2: {rf_r2:.4f}")
