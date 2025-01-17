import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

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

# Sidebar: Train-Test Split Ratio Slider
st.sidebar.subheader("Train-Test Split")
train_ratio = st.sidebar.slider("Training Set Ratio", 0.5, 0.9, 0.8)

# Sidebar: Model Evaluation Method
st.sidebar.subheader("Evaluation Method")
evaluation_method = st.sidebar.radio(
    "Choose evaluation method:",
    ("Direct Testing (Train-Test Split)", "Indirect Testing (BIC or CV)")
)

# Split data into train and test sets
train_data, test_data = train_test_split(df, train_size=train_ratio, random_state=42)

# Fit the selected model
model = smf.ols(formula=selected_formula, data=train_data).fit()
train_data["Predicted"] = model.fittedvalues
test_data["Predicted"] = model.predict(test_data)

# Evaluation Metrics
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return mse

# MSE Decomposition and Evaluation
if evaluation_method == "Direct Testing (Train-Test Split)":
    mse_train = calculate_metrics(train_data["House_Price"], train_data["Predicted"])
    mse_test = calculate_metrics(test_data["House_Price"], test_data["Predicted"])
    bias_squared = (test_data["Predicted"].mean() - test_data["House_Price"].mean()) ** 2
    variance = test_data["Predicted"].var()
else:  # Indirect Testing (BIC or Cross-Validation)
    bic = model.bic
    mse_cv = -np.mean(cross_val_score(
        smf.ols(formula=selected_formula, data=df).fit(),
        df.drop("House_Price", axis=1),
        df["House_Price"],
        scoring="neg_mean_squared_error",
        cv=5
    ))
    mse_train, mse_test, bias_squared, variance = mse_cv, mse_cv, None, None

# Display Metrics
st.subheader("Model Performance")
if evaluation_method == "Direct Testing (Train-Test Split)":
    st.write(f"**MSE (Train):** {mse_train:.2f}")
    st.write(f"**MSE (Test):** {mse_test:.2f}")
    st.write(f"**Bias² (Test):** {bias_squared:.2f}")
    st.write(f"**Variance (Test):** {variance:.2f}")
else:
    st.write(f"**BIC:** {bic:.2f}")
    st.write(f"**Cross-Validation MSE:** {mse_cv:.2f}")

# Visualization: Bar Chart for MSE Decomposition
if evaluation_method == "Direct Testing (Train-Test Split)":
    st.subheader("Bar Chart: MSE Decomposition (Test Data)")
    fig, ax = plt.subplots()
    labels = ["Bias²", "Variance", "Irreducible Error"]
    values = [bias_squared, variance, mse_test - bias_squared - variance]
    ax.bar(labels, values, color=["blue", "orange", "green"])
    ax.set_title("MSE Decomposition")
    ax.set_ylabel("Error")
    st.pyplot(fig)

# Visualization: Scatter Plot for Actual vs Predicted
st.subheader(f"Scatter Plot: Actual vs Predicted ({selected_model})")
plt.figure(figsize=(8, 6))
plt.scatter(test_data["House_Price"], test_data["Predicted"], alpha=0.7, label="Predicted", color="blue")
plt.plot([test_data["House_Price"].min(), test_data["House_Price"].max()],
         [test_data["House_Price"].min(), test_data["House_Price"].max()],
         color="red", linestyle="--", label="Perfect Prediction")
plt.title("Actual vs Predicted Prices (Test Data)")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
st.pyplot(plt)

# Visualization: Residual Plot
st.subheader("Residual Plot (Test Data)")
test_data["Residual"] = test_data["House_Price"] - test_data["Predicted"]
plt.figure(figsize=(8, 6))
plt.scatter(test_data["Predicted"], test_data["Residual"], alpha=0.7, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
st.pyplot(plt)
