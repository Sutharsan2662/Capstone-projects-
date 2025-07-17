# 🌾 Agricultural Production Prediction using Ensemble Models

This project predicts agricultural production (in tonnes) based on historical crop, yield, area, and region data from FAOSTAT. It uses ensemble machine learning techniques 
(XGBoost and Gradient Boosting) for high-accuracy prediction, with a user-friendly Streamlit interface.

---

## 📁 Project Structure
├── FAOSTAT_data.xlsx # Raw FAO dataset
├── cleaned_data_final.csv # Cleaned and processed data
├── gb_model.pkl # Trained GradientBoostingRegressor model
├── xgb_model.pkl # Trained XGBRegressor model
├── le_item.pkl # LabelEncoder for 'Item'
├── le_area.pkl # LabelEncoder for 'Area'
├── streamlit.py # Streamlit web app
├── README.md # Project documentation
└── requirements.txt # Dependencies list

## 🚀 Features

- Ensemble prediction using **XGBRegressor** and **GradientBoostingRegressor**
- Log-transformed feature engineering for enhanced model performance
- Missing value handling using domain-specific logic
- Fully interactive web app built with **Streamlit**
- Real-time input and visual result display

---

## install dependencies

pip install -r requirements.txt

## running the streamlit app:
Open the command prompt from the project directory and execute the below command

Command: streamlit run streamlit.py