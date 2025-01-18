import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/torokpe/Data_analysis3/refs/heads/main/house_prices.csv"
    return pd.read_csv(url)

df = load_data()

# Sidebar: Add a title and description
st.markdown("<h1 style='text-align: center;'>Model Performance & MSE Decomposition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>This is a centered body text.</p>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>  </h1>", unsafe_allow_html=True)

st.sidebar.write("Use the control panel to set model specifications.")

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
train_ratio = st.sidebar.slider("Set training-set ratio", 0.5, 0.9, 0.8)

# Sidebar: Model Evaluation Method
st.sidebar.subheader("Evaluation Method")
evaluation_method = st.sidebar.radio(
    "Choose evaluation method:",
    ("Direct Testing (Train-Test Split)", "Indirect Testing (BIC or CV)")
)

# Split data into train and test sets
train_data, test_data = train_test_split(df, train_size=train_ratio, random_state=42)

# Fit the selected model using statsmodels
model = smf.ols(formula=selected_formula, data=train_data).fit()
train_data["Predicted"] = model.fittedvalues
test_data["Predicted"] = model.predict(test_data)

# Function to calculate metrics
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return mse

# Function to calculate BIC and Cross-Validation MSE
def calculate_bic_and_cv(df, formula, target_col):
    # Calculate BIC using the full dataset
    model = smf.ols(formula=formula, data=df).fit()
    bic = model.bic

    # Prepare features (X) and target (y) for cross-validation
    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables if any
    y = df[target_col]

    # Perform Cross-Validation with sklearn
    sklearn_model = LinearRegression()
    mse_cv = -np.mean(cross_val_score(sklearn_model, X, y, scoring="neg_mean_squared_error", cv=5))

    return bic, mse_cv

# Calculate metrics based on the evaluation method
if evaluation_method == "Direct Testing (Train-Test Split)":
    mse_train = calculate_metrics(train_data["House_Price"], train_data["Predicted"])
    mse_test = calculate_metrics(test_data["House_Price"], test_data["Predicted"])
    bias_squared = (test_data["Predicted"].mean() - test_data["House_Price"].mean()) ** 2
    variance = test_data["Predicted"].var()
else:  # Indirect Testing (BIC or Cross-Validation)
    bic, mse_cv = calculate_bic_and_cv(df, selected_formula, "House_Price")
    mse_train, mse_test, bias_squared, variance = None, mse_cv, None, None

# Display Metrics as Dashboard Highlights
st.markdown("<h2 style='text-align: center;'>Key performance metrics</h2>", unsafe_allow_html=True)

if evaluation_method == "Direct Testing (Train-Test Split)":
    # Row layout for metrics with increased column width
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1.5])  # Wider columns

    with col1:
        st.metric(label="MSE (Train)", value=f"{mse_train/1e6:.2f}M")  # Abbreviate to millions

    with col2:
        st.metric(label="MSE (Test)", value=f"{mse_test/1e6:.2f}M")  # Abbreviate to millions

    with col3:
        st.metric(label="Bias² (Test)", value=f"{bias_squared/1e6:.2f}M")  # Abbreviate to millions

    with col4:
        st.metric(label="Variance (Test)", value=f"{variance/1e6:.2f}M")  # Abbreviate to millions

else:  # Indirect Testing (BIC or CV)
    col1, col2 = st.columns([1.5, 1.5])  # Wider columns for fewer metrics

    with col1:
        st.metric(label="BIC", value=f"{bic:.2f}")

    with col2:
        st.metric(label="Cross-Validation MSE", value=f"{mse_cv/1e6:.2f}M")  # Abbreviate to millions
st.markdown("<h3 style='text-align: center;'>  </h3>", unsafe_allow_html=True)

# Visualization: Bar Chart for MSE Decomposition
if evaluation_method == "Direct Testing (Train-Test Split)":
    st.markdown("<h2 style='text-align: center;'> MSE Decomposition (Test Data) </h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    labels = ["Bias²", "Variance", "Irreducible Error"]
    values = [bias_squared, variance, mse_test - bias_squared - variance]
    ax.bar(labels, values, color=["blue", "orange", "green"])
    ax.set_title("MSE Decomposition")
    ax.set_ylabel("Error")
    st.pyplot(fig)

# Visualization: Scatter Plot for Actual vs Predicted
st.markdown("<h3 style='text-align: center;'>Model Performance Metrics</h3>", unsafe_allow_html=True)
st.subheader(f"Actual vs Predicted")
plt.figure(figsize=(8, 6))
plt.scatter(test_data["House_Price"], test_data["Predicted"], alpha=0.7, label="Predicted", color="#156082")
plt.plot([test_data["House_Price"].min(), test_data["House_Price"].max()],
         [test_data["House_Price"].min(), test_data["House_Price"].max()],
         color="#FFC000", linestyle="--", label="Perfect Prediction")
plt.title("Actual vs Predicted Prices (Test Data)")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
st.pyplot(plt)

# Visualization: Residual Plot
st.subheader("Residual Plot (Test Data)")
test_data["Residual"] = test_data["House_Price"] - test_data["Predicted"]
plt.figure(figsize=(8, 6))
plt.scatter(test_data["Predicted"], test_data["Residual"], alpha=0.7, color="#156082")
plt.axhline(0, color="#FFC000", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
st.pyplot(plt)
