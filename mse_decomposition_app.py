import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load dataset from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/torokpe/Data_analysis3/refs/heads/main/house_prices.csv"
    return pd.read_csv(url)

df = load_data()

# Sidebar: Add a title and description
st.sidebar.title("Interactive Dashboard")
st.sidebar.write("Use the dropdown and filters below to analyze different models and subsets of the data.")

# Sidebar: Interactive Model Selection
model_formulas = {
    "Model 1: House_Price ~ Square_Footage": "House_Price ~ Square_Footage",
    "Model 2: House_Price ~ Square_Footage + Num_Bedrooms + Num_Bathrooms": "House_Price ~ Square_Footage + Num_Bedrooms + Num_Bathrooms",
    "Model 3: House_Price ~ Square_Footage + Num_Bedrooms + Year_Built + Neighborhood_Quality + Num_Bathrooms + Lot_Size + Garage_Size": "House_Price ~ Square_Footage + Num_Bedrooms + Year_Built + Neighborhood_Quality + Num_Bathrooms + Lot_Size + Garage_Size"
}
selected_model = st.sidebar.selectbox("Choose a model:", list(model_formulas.keys()))
selected_formula = model_formulas[selected_model]

# Sidebar: Filtering by Year_Built
min_year_built = st.sidebar.slider(
    "Minimum Year Built",
    int(df["Year_Built"].min()),
    int(df["Year_Built"].max()),
    int(df["Year_Built"].min())
)
filtered_df = df[df["Year_Built"] >= min_year_built]

# Sidebar: Checkbox to show filtered data
if st.sidebar.checkbox("Show Filtered Data"):
    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df)

# Fit the selected model using filtered data
model = smf.ols(formula=selected_formula, data=filtered_df).fit()
filtered_df["Predicted"] = model.fittedvalues

# Main Title and Dataset Preview
st.title("MSE Decomposition: Bias and Variance Analysis")
st.write("Dataset Preview")
st.dataframe(df.head())

# Show model summary
st.subheader(f"Model Summary: {selected_model}")
st.text(model.summary())

# Function to calculate MSE, Bias², and Variance
def calculate_bias_variance(df, actual_col, predicted_col):
    actual = df[actual_col]
    predicted = df[predicted_col]
    mse = np.mean((predicted - actual) ** 2)
    bias_squared = (np.mean(predicted - actual)) ** 2
    variance = np.var(predicted)
    return mse, bias_squared, variance

# Calculate MSE, Bias², and Variance
mse, bias_squared, variance = calculate_bias_variance(filtered_df, "House_Price", "Predicted")

# Display metrics
st.subheader("Decomposition of MSE")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Bias²:** {bias_squared:.2f}")
st.write(f"**Variance:** {variance:.2f}")
st.write(f"**Irreducible Error:** {mse - bias_squared - variance:.2f}")

# Visualization: Bar chart for MSE decomposition
st.subheader("Bar Chart: MSE Decomposition")
fig, ax = plt.subplots()
labels = ["Bias²", "Variance", "Irreducible Error"]
values = [bias_squared, variance, mse - bias_squared - variance]
ax.bar(labels, values, color=["blue", "orange", "green"])
ax.set_title("MSE Decomposition")
ax.set_ylabel("Error")
st.pyplot(fig)

# Visualization: Scatter plot for Actual vs Predicted
st.subheader(f"Scatter Plot: Actual vs Predicted ({selected_model})")
plt.figure(figsize=(8, 6))
plt.scatter(filtered_df["House_Price"], filtered_df["Predicted"], alpha=0.7, label="Predicted", color="blue")
plt.plot([filtered_df["House_Price"].min(), filtered_df["House_Price"].max()],
         [filtered_df["House_Price"].min(), filtered_df["House_Price"].max()],
         color="red", linestyle="--", label="Perfect Prediction")
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
st.pyplot(plt)

# Visualization: Residual plot
st.subheader("Residual Plot")
filtered_df["Residual"] = filtered_df["House_Price"] - filtered_df["Predicted"]
plt.figure(figsize=(8, 6))
plt.scatter(filtered_df["Predicted"], filtered_df["Residual"], alpha=0.7, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
st.pyplot(plt)
