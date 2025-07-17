import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read relevant columns
df = pd.read_excel("FAOSTAT_data.xlsx", usecols=[
    'Domain Code', 'Area', 'Element', 'Item', 'Year', 'Value'
])

# Step 2: Filter required elements
df = df[df['Element'].isin(['Area harvested', 'Yield', 'Production'])]

# Step 3: Pivot in the index
df_pivot = df.pivot_table(
    index=['Domain Code', 'Area', 'Item', 'Year'],
    columns='Element',
    values='Value'
).reset_index()

# Step 4: Rename columns
df_pivot.columns.name = None
df_pivot = df_pivot.rename(columns={
    'Area harvested': 'Area Harvested (ha)',
    'Yield': 'Yield (kg/ha)',
    'Production': 'Production (tonnes)'
})

#Dropping unwanted columns from the dataframe
df_pivot = df_pivot.drop(columns = ['Domain Code'], axis = 1)
#checking is there any null values 

#convert the unit 
df_pivot['Yield (tonnes/ha)'] = df_pivot['Yield (kg/ha)'] / 1000
df_pivot.drop(columns=['Yield (kg/ha)'], inplace=True)

# If Area is missing but Production and Yield are present
mask_area = df_pivot['Area Harvested (ha)'].isnull() & df_pivot['Production (tonnes)'].notnull() & df_pivot['Yield (tonnes/ha)'].notnull()
df_pivot.loc[mask_area, 'Area Harvested (ha)'] = df_pivot.loc[mask_area, 'Production (tonnes)'] / df_pivot.loc[mask_area, 'Yield (tonnes/ha)']

# If Yield is missing but Area and Production are present
mask_yield = df_pivot['Yield (tonnes/ha)'].isnull() & df_pivot['Area Harvested (ha)'].notnull() & df_pivot['Production (tonnes)'].notnull()
df_pivot.loc[mask_yield, 'Yield (tonnes/ha)'] = df_pivot.loc[mask_yield, 'Production (tonnes)'] / df_pivot.loc[mask_yield, 'Area Harvested (ha)']

# If Production is missing but Area and Yield are present
mask_prod = df_pivot['Production (tonnes)'].isnull() & df_pivot['Area Harvested (ha)'].notnull() & df_pivot['Yield (tonnes/ha)'].notnull()
df_pivot.loc[mask_prod, 'Production (tonnes)'] = df_pivot.loc[mask_prod, 'Area Harvested (ha)'] * df_pivot.loc[mask_prod, 'Yield (tonnes/ha)']

# Fill remaining missing values using overall median
df_pivot['Area Harvested (ha)'] = df_pivot['Area Harvested (ha)'].fillna(df_pivot['Area Harvested (ha)'].median())
df_pivot['Yield (tonnes/ha)'] = df_pivot['Yield (tonnes/ha)'].fillna(df_pivot['Yield (tonnes/ha)'].median())
df_pivot['Production (tonnes)'] = df_pivot['Production (tonnes)'].fillna(df_pivot['Production (tonnes)'].median())

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
le = LabelEncoder()
df_pivot['Item'] = le.fit_transform(df_pivot['Item'])
df_pivot['Area'] = le.fit_transform(df_pivot['Area'])
df_pivot['Yield (tonnes/ha)'] = df_pivot['Yield (tonnes/ha)'].replace([np.inf, -np.inf], df_pivot['Yield (tonnes/ha)'].median())
scaler = StandardScaler()

features = ['Year', 'Yield (tonnes/ha)', 'Area Harvested (ha)', 'Item', 'Area']
target = ['Production (tonnes)']
x = df_pivot[features]
y = df_pivot[target]
x = x.astype(np.float32)

# Apply log1p to features
x = x.apply(np.log1p)

# Apply log1p to target
y = y.apply(np.log1p)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Train each model
xgb_model = XGBRegressor(n_estimators=1000, random_state=42, objective='reg:squarederror', verbosity=0)
gb_model = GradientBoostingRegressor(n_estimators=1000, random_state=42)

xgb_model.fit(x_train, y_train)
gb_model.fit(x_train, y_train)

import joblib

# Save models
joblib.dump(gb_model, 'gb_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')

# Save encoders (trained on raw string values before encoding)
df_raw = pd.read_excel("FAOSTAT_data.xlsx", usecols=["Item", "Area"])
le_item = LabelEncoder().fit(df_raw["Item"])
le_area = LabelEncoder().fit(df_raw["Area"])

joblib.dump(le_item, 'le_item.pkl')
joblib.dump(le_area, 'le_area.pkl')

# Save cleaned data
df_pivot.to_csv("cleaned_data_final.csv", index=False)