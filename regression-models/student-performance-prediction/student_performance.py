# =========================================
# Student Performance Prediction
# =========================================

# ---- Imports ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =========================================
# Step 1: Create Student Performance Dataset
# =========================================
dataset = pd.DataFrame({
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [55, 60, 65, 70, 75, 80, 85, 90, 92, 95],
    'Assignment_Score': [45, 50, 55, 60, 65, 70, 75, 85, 88, 90],
    'Final_Score': [50, 55, 60, 65, 70, 75, 80, 88, 90, 92]
})

print("Dataset Preview:")
print(dataset)

# =========================================
# Step 2: Dataset Information
# =========================================
print("\nDataset Info:")
print(dataset.info())

# =========================================
# Step 3: Check Missing Values
# =========================================
print("\nMissing values:")
print(dataset.isnull().sum())

# =========================================
# Step 4: Feature and Target Selection
# =========================================
X = dataset[['Hours_Studied', 'Attendance', 'Assignment_Score']]
y = dataset['Final_Score']

# =========================================
# Step 5: Train-Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# Step 6: Train Multiple Linear Regression Model
# =========================================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================================
# Step 7: Predictions
# =========================================
y_pred = model.predict(X_test)

# =========================================
# Step 8: Model Evaluation
# =========================================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# =========================================
# Step 9: Actual vs Predicted Scatter Plot
# =========================================
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Student Score")
plt.ylabel("Predicted Student Score")
plt.title("Actual vs Predicted Student Performance")
plt.show()

