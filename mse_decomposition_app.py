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

# Add CSS for centering content
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }
    .stApp {
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Start centered content
st.markdown('<div class="center">', unsafe_allow_html=True)

# Title and description
st.markdown("<h1>Model Performance & MSE Decomposition</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='font-size:18px;'>
        The purpose of this dashboard is to provide a visualization of the decomposition of Mean Squared Error (MSE) for three different predictive models. 
        By breaking down MSE into its key components—Bias, Variance, and Irreducible Error—the dashboard allows users to understand how each model performs and where improvements might be made. 
        For the app an online dataset was utilized on house prices, where the predictor variables consist of various characteristics of the houses, such as size, location, and neighbourhood rating etc.
        The original dataset is sourced from 
        <a href='https://www.kaggle.com/datasets/prokshitha/home-value-insights' target='_blank' style='color:blue; text-decoration: none;'>this link</a>.
    </p>
    """,
    unsafe_allow_html=True,
)

# Data for the table
models_data = {
    "": ["Squarefootage", "# of bathrooms", "# of bedrooms", "Garage size", "Lot size", "Neighborhood quality", "Year built"],
    "Model I.": ["YES", "YES", "", "", "", "", ""],
    "Model II.": ["YES", "YES", "YES", "", "", "", ""],
    "Model III.": ["YES", "YES", "YES", "YES", "YES", "YES", "YES"],
}

# Create and display the table
df_models = pd.DataFrame(models_data)
st.table(df_models)

# Sidebar: Interactive Model Selection
model_formulas = {
    "Model 1: House_Price ~ Square_Footage": "House_Price ~ Square_Footage",
    "Model 2: House_Price ~ Square_Footage + Num_Bedrooms + Num_Bathrooms": "House_Price ~ Square_Footage + Num_Bedrooms + Num_Bathrooms",
    "Model 3: House_Price ~ Square_Footage + Num_Bedrooms + Year_Built + Neighborhood_Quality + Num_Bathrooms + Lot_Size + Garage_Size": "House_Price ~ Square_Footage + Num_Bedrooms + Year_Built + Neighborhood_Quality + Num_Bathrooms + Lot_Size + Garage_Size",
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
    ("Direct Testing (Train-Test Split)", "Indirect Testing (BIC or CV)"),
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
    model = smf.ols(formula=formula, data=df).fit()
    bic = model.bic
    X = df.drop(columns=[target_col])
    y = df[target_col]
    sklearn_model = LinearRegression()
    mse_cv = -np.mean(cross_val_score(sklearn_model, X, y, scoring="neg_mean_squared_error", cv=5))
    return bic, mse_cv

if evaluation_method == "Direct Testing (Train-Test Split)":
    mse_train = calculate_metrics(train_data["House_Price"], train_data["Predicted"])
    mse_test = calculate_metrics(test_data["House_Price"], test_data["Predicted"])
    bias_squared = (test_data["Predicted"].mean() - test_data["House_Price"].mean()) ** 2
    variance = test_data["Predicted"].var()
else:
    bic, mse_cv = calculate_bic_and_cv(df, selected_formula, "House_Price")
    mse_train, mse_test, bias_squared, variance = None, mse_cv, None, None

st.markdown("<h2>Key Performance Metrics</h2>", unsafe_allow_html=True)

if evaluation_method == "Direct Testing (Train-Test Split)":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE (Train)", f"{mse_train:.2f}")
    col2.metric("MSE (Test)", f"{mse_test:.2f}")
    col3.metric("Bias² (Test)", f"{bias_squared:.2f}")
    col4.metric("Variance (Test)", f"{variance:.2f}")
else:
    col1, col2 = st.columns(2)
    col1.metric("BIC", f"{bic:.2f}")
    col2.metric("CV MSE", f"{mse_cv:.2f}")

# Bar chart for MSE decomposition
if evaluation_method == "Direct Testing (Train-Test Split)":
    st.markdown("<h2>MSE Decomposition</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    labels = ["Bias²", "Variance", "Irreducible Error"]
    values = [bias_squared, variance, mse_test - bias_squared - variance]
    ax.bar(labels, values)
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)  # End centered content
