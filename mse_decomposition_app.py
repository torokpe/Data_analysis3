import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate MSE, Bias², and Variance
def calculate_bias_variance(df, actual_col, predicted_cols):
    results = []
    actual = df[actual_col]
    for col in predicted_cols:
        predictions = df[col]
        mse = np.mean((predictions - actual) ** 2)
        bias_squared = (np.mean(predictions - actual)) ** 2
        variance = np.var(predictions)
        results.append({'Model': col, 'MSE': mse, 'Bias²': bias_squared, 'Variance': variance})
    return pd.DataFrame(results)

# Streamlit App
st.title("MSE Decomposition: Bias and Variance Analysis")

# Sidebar to choose models
model_options = ['Predicted_price1', 'Predicted_price2', 'Predicted_price3']
selected_model = st.sidebar.selectbox("Select a model for detailed analysis:", model_options)

# Calculate metrics for all models
results_df = calculate_bias_variance(df, 'House_Price', model_options)

# Display results
st.subheader("Comparison Across Models")
st.dataframe(results_df)

# Show detailed decomposition for the selected model
selected_metrics = results_df[results_df['Model'] == selected_model].iloc[0]
st.subheader(f"Detailed Decomposition for {selected_model}")
st.write(f"**Mean Squared Error (MSE):** {selected_metrics['MSE']:.2f}")
st.write(f"**Bias²:** {selected_metrics['Bias²']:.2f}")
st.write(f"**Variance:** {selected_metrics['Variance']:.2f}")
st.write(f"**Irreducible Error:** {selected_metrics['MSE'] - selected_metrics['Bias²'] - selected_metrics['Variance']:.2f}")

# Plot decomposition for the selected model
fig, ax = plt.subplots()
labels = ['Bias²', 'Variance', 'Irreducible Error']
values = [selected_metrics['Bias²'], selected_metrics['Variance'], selected_metrics['MSE'] - selected_metrics['Bias²'] - selected_metrics['Variance']]
ax.bar(labels, values, color=['blue', 'orange', 'green'])
ax.set_title(f"MSE Decomposition for {selected_model}")
ax.set_ylabel("Error")
st.pyplot(fig)

# Additional visualization: Scatter Plot for predictions
st.subheader(f"Scatter Plot: Actual vs {selected_model}")
plt.figure(figsize=(8, 6))
plt.scatter(df['House_Price'], df[selected_model], alpha=0.7, label=selected_model, color='blue')
plt.plot([df['House_Price'].min(), df['House_Price'].max()],
         [df['House_Price'].min(), df['House_Price'].max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.title(f"Actual vs Predicted Prices ({selected_model})")
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
st.pyplot(plt)
