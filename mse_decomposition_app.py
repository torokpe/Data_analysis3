import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/5656167/9334344/house_price_regression_dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250117%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250117T141127Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7355844dafbd6ff14cc019d15a8d613dfbc2b16a7c76fd7e494dd86efa5c77829dc432f2c1eee38004992da184da2996132be68e403b40912a232ce869f2ee446ec9f34f41270eeb5139c533e04ff032f8a30fd4d6dae7126114041ccd1d13eddae403e3ed91ce39d7ade7e5d3bf0b87a62c808764e054482a8cf9300cf743d0649d19396c4ef72bc801695157745251fa9d6f0b6a965c4261108052594e4e029284ef2f66d432cf72ece8f0be21f58ada29c5964e3f39113b7c94ff4936f3ec49c7902a6621dcdc8c38ddea74b9277f3f4ea9abdb9a8bafd4322b0d507d4fef0efe97e1a0f6f97b3b5b4c8b6bc53743fe54feb20db533543571610b054c0590')

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
