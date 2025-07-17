import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score
import matplotlib.gridspec as gridspec

# Load models
gb_model = joblib.load("gb_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Load encoders
le_item = joblib.load("le_item.pkl")
le_area = joblib.load("le_area.pkl")

# Load cleaned data
df = pd.read_csv("cleaned_data_final.csv")

# Sidebar input
st.sidebar.title("📥 Enter Input Features")
year = st.sidebar.slider("Year", int(df["Year"].min()), int(df["Year"].max()), step=1)
yield_val = st.sidebar.number_input("Yield (tonnes/ha)", min_value=0.0, step=0.01)
area_val = st.sidebar.number_input("Area Harvested (ha)", min_value=0.0, step=0.01)
item = st.sidebar.selectbox("Crop/Item", le_item.classes_)
area = st.sidebar.selectbox("Country/Area", le_area.classes_)

if st.sidebar.button("Predict Production"):
    item_encoded = le_item.transform([item])[0]
    area_encoded = le_area.transform([area])[0]

    input_df = pd.DataFrame([{
        "Year": year,
        "Yield (tonnes/ha)": yield_val,
        "Area Harvested (ha)": area_val,
        "Item": item_encoded,
        "Area": area_encoded
    }])
    input_log = input_df.apply(np.log1p)

    pred_gb = gb_model.predict(input_log)
    pred_xgb = xgb_model.predict(input_log)
    pred_avg = (pred_gb + pred_xgb) / 2

    pred_real = np.expm1(pred_avg[0])
    st.success(f"📊 Predicted Production: **{pred_real:,.2f} tonnes**")

# Main content
st.title("🌾 Agricultural Production Prediction - Ensemble Model")

# Evaluate ensemble on full dataset
features = ['Year', 'Yield (tonnes/ha)', 'Area Harvested (ha)', 'Item', 'Area']
target = ['Production (tonnes)']

X = df[features].astype(np.float32).apply(np.log1p)
y = np.expm1(np.log1p(df[target]))
y_test_array = y.values.flatten()

# Predict
y_pred_gb = np.expm1(gb_model.predict(X))
y_pred_xgb = np.expm1(xgb_model.predict(X))
y_pred_avg = (y_pred_gb + y_pred_xgb) / 2

# R2 Scores
r2_gb = r2_score(y_test_array, y_pred_gb)
r2_xgb = r2_score(y_test_array, y_pred_xgb)
r2_avg = r2_score(y_test_array, y_pred_avg)

st.markdown("### 📈 R² Scores")
st.write(f"✅ Average Model: `{r2_avg:.3f}`")
st.write(f"🔥 Gradient Boost: `{r2_gb:.3f}`")
st.write(f"🚀 XGB Regressor: `{r2_xgb:.3f}`")

# Plots
st.markdown("### 📊 Model Comparison Plots")

def add_plot(ax, y_pred, title, color):
    sns.scatterplot(x=y_test_array, y=y_pred, alpha=0.4, color=color, ax=ax)
    ax.plot(
        [min(y_test_array), max(y_test_array)],
        [min(y_test_array), max(y_test_array)],
        'k--', label='Perfect Prediction'
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Actual Production', fontsize=12)
    ax.set_ylabel('Predicted Production', fontsize=12)
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

# Increase figure size to make plots bigger
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Add all three plots
add_plot(axes[0], y_pred_avg, "Average of XGB + GB", 'green')
add_plot(axes[1], y_pred_gb, "Gradient Boost", 'red')
add_plot(axes[2], y_pred_xgb, "XGB Regressor", 'blue')

# Final touches and Streamlit rendering
plt.tight_layout()
st.pyplot(fig)
