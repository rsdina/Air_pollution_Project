# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

# Load the Dataset
df = pd.read_csv("src\\updated_pollution_dataset.csv")  # Make sure this file exists in the same directory
print("First few rows of the dataset:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Check for Missing Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop Missing Values
df.dropna(inplace=True)

# Summary statistics for numerical features
numerical_df = df.select_dtypes(include=['float64', 'int64'])
stats_summary = pd.DataFrame({
    'Mean': numerical_df.mean(),
    'Median': numerical_df.median(),
    'Variance': numerical_df.var(),
    'Standard Deviation': numerical_df.std()
})
print("\nSummary Statistics:\n", stats_summary)

# Convert 'Proximity_to_Industrial_Areas' to datetime
df['Proximity_to_Industrial_Areas'] = pd.to_datetime(df['Proximity_to_Industrial_Areas'], errors='coerce')
df.dropna(subset=['Proximity_to_Industrial_Areas'], inplace=True)  # Remove any invalid date rows

df['Year'] = df['Proximity_to_Industrial_Areas'].dt.year
df['Month'] = df['Proximity_to_Industrial_Areas'].dt.month
df['Day'] = df['Proximity_to_Industrial_Areas'].dt.day

# Encode Categorical Variables
le = LabelEncoder()
df['City_Encoded'] = le.fit_transform(df['Proximity_to_Industrial_Areas'].dt.strftime('%Y-%m-%d'))

# Feature Scaling
numerical_features = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO']  
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Correlation Matrix (only numerical features)
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Distribution Plots
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

# Define Features (X) and Target (y)
X = df[['PM10', 'NO2', 'SO2', 'CO', 'City_Encoded', 'Year', 'Month', 'Day']]
y = df['PM2.5']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluation Function
def evaluate_model(true, preds, name):
    mse = mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    r2 = r2_score(true, preds)
    print(f"\n{name} Evaluation Metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R² Score: {r2}")

# Evaluate Models
evaluate_model(y_test, lr_preds, "Linear Regression")
evaluate_model(y_test, rf_preds, "Random Forest Regressor")

# Plot Predictions
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=lr_preds)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=rf_preds)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_pollution_dataset.csv", index=False)
