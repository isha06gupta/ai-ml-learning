# House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
dataset = housing.frame

print("Dataset Preview:")
print(dataset.head())

# Step 2: Check dataset information
print("\nDataset Info:")
print(dataset.info())

# Step 3: Check missing values
print("\nMissing values in dataset:")
print(dataset.isnull().sum())

# (California housing has no missing values, so no filling required)

# Step 4: Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(),
            cmap='BrBG',
            annot=True,
            fmt='.2f',
            linewidths=1)
plt.title("Correlation Heatmap of California Housing Dataset")
plt.show()

# Step 5: Select feature & target
# Simple Linear Regression → one feature
X = dataset[['MedInc']]        # feature
y = dataset['MedHouseVal']       # target (house price)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 7: Train Linear Regression model

model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Residual Plot (Error Analysis)
residuals = y_test - y_pred

plt.scatter(y_test, residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual House Price")
plt.ylabel("Residual (Error)")
plt.title("Residual Plot for Linear Regression")
plt.show()

# Step 9: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Step 10: Visualization (Regression Line)
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel("Average Number of Rooms")
plt.ylabel("House Price")
plt.title("Simple Linear Regression on California Housing Data")
plt.legend()
plt.show()

# Support Vector Regression (SVM)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("\nSVM Regression Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_svm))
print("R2 Score:", r2_score(y_test, y_pred_svm))


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Regression Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))


print("\nModel Comparison Summary")
print("------------------------")
print("Linear Regression  -> R2:", r2)
print("SVM Regression     -> R2:", r2_score(y_test, y_pred_svm))
print("Random Forest      -> R2:", r2_score(y_test, y_pred_rf))