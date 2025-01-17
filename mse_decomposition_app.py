import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample DataFrame (replace with your actual data)
data = {
    'actual': [3, 5, 2, 8],
    'model1': [2.5, 5.5, 1.8, 7.8],
    'model2': [3.1, 4.8, 2.2, 8.1],
    'model3': [2.9, 5.0, 2.1, 7.9]
}
df = pd.DataFrame(data)

# Function to calculate MSE, Bias^2, and Variance
def calculate_bias_variance(df, actual_col, model_col):
    predictions = df[model_col]
    actual = df[actual_col]
    mse = np.mean((predictions - actual) ** 2)
    bias_squared = (np.mean(predictions) - np.mean(actual)) ** 2
    variance = np.var(predictions)
    return mse, bias_squared, variance

# Streamlit App
st.title("MSE Decomposition: Bias and Variance")

# Sidebar
st.sidebar.header("Choose a Model")
model_choice = st.sidebar.selectbox("Select a model:", ['model1', 'model2', 'model3'])

# Calculate metrics
mse, bias_squared, variance = calculate_bias_variance(df, 'actual', model_choice)

# Display results
st.subheader(f"Results for {model_choice}")
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"**Bias²:** {bias_squared:.4f}")
st.write(f"**Variance:** {variance:.4f}")
st.write(f"**Irreducible Error:** {mse - bias_squared - variance:.4f}")

# Bar chart of decomposition
fig, ax = plt.subplots()
labels = ['Bias²', 'Variance', 'Irreducible Error']
values = [bias_squared, variance, mse - bias_squared - variance]
ax.bar(labels, values, color=['blue', 'orange', 'green'])
ax.set_title(f"MSE Decomposition for {model_choice}")
ax.set_ylabel("Error")
st.pyplot(fig)

# Comparison table
st.subheader("Comparison Across Models")
comparison = []
for model in ['model1', 'model2', 'model3']:
    mse, bias_squared, variance = calculate_bias_variance(df, 'actual', model)
    comparison.append({'Model': model, 'MSE': mse, 'Bias²': bias_squared, 'Variance': variance})

comparison_df = pd.DataFrame(comparison)
st.write(comparison_df)
