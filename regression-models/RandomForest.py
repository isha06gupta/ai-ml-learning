import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Confusion matrix plot function
def plot_confusion_matrix(cm, title):
    plt.figure()
    plt.imshow(cm, cmap="Greens")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    # cell ke andar numbers likhne ke liye
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.show()


# Dataset load and cleaning
df = pd.read_csv("Telco-Customer-Churn.csv")

# customerID ka koi use nahi
df.drop("customerID", axis=1, inplace=True)

# TotalCharges ko numeric banaya string se
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Churn ko binary banaya
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# categorical columns ko numeric banaya
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Experiment 1: Base Random Forest

print("\nBase Random Forest Model")

model = RandomForestClassifier(
    n_estimators=100,  #100 decision tree banega
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# confusion matrix visualization (base model)
cm_base = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm_base, "Confusion Matrix - Base Random Forest")

# Experiment 2: n_estimators effect
print("\nEffect of n_estimators")

n_values = [50, 100, 500]
accuracies_n = []

for n in n_values:
    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies_n.append(acc)

    print(f"\nn_estimators = {n}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# graph: n_estimators vs accuracy
plt.figure()
plt.plot(n_values, accuracies_n, marker='o')
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs n_estimators")
plt.show()


# Experiment 3: max_depth effect

print("\nEffect of max_depth")

depth_values = [3, 4, 5]
accuracies_depth = []

for depth in depth_values:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies_depth.append(acc)

    print(f"\nmax_depth = {depth}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# graph: max_depth vs accuracy
plt.figure()
plt.plot(depth_values, accuracies_depth, marker='o')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs max_depth")
plt.show()

# Experiment 4: 5-Fold Cross Validation

print("\n5-Fold Cross Validation")

#kFolds- data ko parts(fold) mein divide karna
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []   #empty list to store fold accuracy
fold = 1  #for tracking

for train_index, test_index in kf.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train_cv, y_train_cv)
    y_pred_cv = model.predict(X_test_cv)

    acc = accuracy_score(y_test_cv, y_pred_cv)
    fold_accuracies.append(acc)

    print(f"\nFold {fold}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test_cv, y_pred_cv))

    fold += 1

# graph: fold vs accuracy
plt.figure()
plt.plot(range(1, 6), fold_accuracies, marker='o')
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross Validation Accuracy")
plt.show()

# -----------------------------
# Experiment 5: GridSearchCV
# -----------------------------

print("\nGridSearchCV for Hyperparameter Tuning")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)

# n_jobs=1 rakha hai taaki memory issue na aaye
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)
print("Test Accuracy:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))

# confusion matrix visualization (tuned model)
cm_best = confusion_matrix(y_test, y_pred_best)
plot_confusion_matrix(cm_best, "Confusion Matrix - Tuned Random Forest")
