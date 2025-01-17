import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load dataset from GitHub
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/<your-username>/<repo-name>/main/<file-name>.csv"
    return pd.read_csv(url)

df = load_data()

# Display dataset
st.title("MSE Decomposition: Bias and Variance Analysis")
st.write("Dataset Preview")
st.dataframe(df.head())

# Define model formulas
model_formulas = {
    "Model 1: House_Price ~ Square_Footage": "House_Price ~ Square_Footage",
    "Model 2: House_Price ~ Square_Footage + Num_Bedrooms + Num_Bathrooms": "House_Price ~ Square_Footage + Num_Bedrooms + Num_Bathrooms",
    "Model 3: House_Price ~ Square_Footage + Num_Bedrooms + Year_Built + Neighborhood_Quality + Num_Bathrooms + Lot_Size + Garage_Size": "House_Price ~ Square_Footage + Num_Bedrooms + Year_Built + Neighborhood_Quality + Num_Bathrooms + Lot_Size + Garage_Size"
}

# Sidebar model selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model:", list(model_formulas.keys()))
selected_formula = model_formulas[selected_model]

# Fit the selected model
model = smf.ols(formula=selected_formula, data=df).fit()
df["Predicted"] = model.fittedvalues

# Show model summary
st.subheader("Model Summary")
st.text(model.summary())

# Calculate MSE, Bias², and Variance
def calculate_bias_variance(df, actual_col, predicted_col):
    actual = df[actual_col]
    predicted = df[predicted_col]
    mse = np.mean((predicted - actual) ** 2)
    bias_squared = (np.mean(predicted - actual)) ** 2
    variance = np.var(predicted)
    return mse, bias_squared, variance

mse, bias_squared, variance = calculate_bias_variance(df, "House_Price", "Predicted")

# Display metrics
st.subheader("Decomposition of MSE")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Bias²:** {bias_squared:.2f}")
st.write(f"**Variance:** {variance:.2f}")
st.write(f"**Irreducible Error:** {mse - bias_squared - variance:.2f}")

# Bar chart for MSE decomposition
fig, ax = plt.subplots()
labels = ["Bias²", "Variance", "Irreducible Error"]
values = [bias_squared, variance, mse - bias_squared - variance]
ax.bar(labels, values, color=["blue", "orange", "green"])
ax.set_title("MSE Decomposition")
ax.set_ylabel("Error")
st.pyplot(fig)

# Scatter plot for actual vs. predicted values
st.subheader(f"Scatter Plot: Actual vs Predicted")
plt.figure(figsize=(8, 6))
plt.scatter(df["House_Price"], df["Predicted"], alpha=0.7, label="Predicted", color="blue")
plt.plot([df["House_Price"].min(), df["House_Price"].max()],
         [df["House_Price"].min(), df["House_Price"].max()],
         color="red", linestyle="--", label="Perfect Prediction")
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
st.pyplot(plt)

# Residual plot
st.subheader("Residual Plot")
df["Residual"] = df["House_Price"] - df["Predicted"]
plt.figure(figsize=(8, 6))
plt.scatter(df["Predicted"], df["Residual"], alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
st.pyplot(plt)

# Sidebar filters (optional)
st.sidebar.header("Filters")
min_year_built = st.sidebar.slider("Minimum Year Built", int(df["Year_Built"].min()), int(df["Year_Built"].max()), int(df["Year_Built"].min()))
filtered_df = df[df["Year_Built"] >= min_year_built]
st.write(f"Filtered Data: {len(filtered_df)} rows")
