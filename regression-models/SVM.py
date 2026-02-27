import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, classification_report
from scipy.special import softmax

# --------------------------------------------------
# 1. Load Dataset (OFFLINE replacement of MNIST)
# --------------------------------------------------
digits = load_digits()

X = digits.data
y = digits.target

# Normalize pixel values (0–16 → 0–1)
X = X / 16.0

# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# 3. Train + Validation Split
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

# --------------------------------------------------
# 4. Linear SVM Initialization
# --------------------------------------------------
svm = SGDClassifier(loss="hinge", max_iter=1, warm_start=True)

epochs = 10
train_acc = []
val_acc = []
train_loss = []
val_loss = []

classes = np.unique(y_train)

# --------------------------------------------------
# 5. Training Loop (10 Epochs)
# --------------------------------------------------
for epoch in range(epochs):

    svm.partial_fit(X_train, y_train, classes=classes)

    # Predictions
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)

    # Accuracy
    train_acc.append(accuracy_score(y_train, y_train_pred))
    val_acc.append(accuracy_score(y_val, y_val_pred))

    # Decision scores
    train_scores = svm.decision_function(X_train)
    val_scores = svm.decision_function(X_val)

    # Convert scores → probabilities
    train_probs = softmax(train_scores, axis=1)
    val_probs = softmax(val_scores, axis=1)

    # One-hot labels
    y_train_onehot = np.eye(10)[y_train]
    y_val_onehot = np.eye(10)[y_val]

    # Loss calculation
    train_loss.append(log_loss(y_train_onehot, train_probs))
    val_loss.append(log_loss(y_val_onehot, val_probs))

    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Acc: {train_acc[-1]:.4f}, "
        f"Val Acc: {val_acc[-1]:.4f}"
    )

# --------------------------------------------------
# 6. Final Evaluation
# --------------------------------------------------
y_test_pred = svm.predict(X_test)

print("\nFinal Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# --------------------------------------------------
# 7. Accuracy Plot
# --------------------------------------------------
plt.figure()
plt.plot(range(1, epochs+1), train_acc)
plt.plot(range(1, epochs+1), val_acc)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.show()

# --------------------------------------------------
# 8. Loss Plot
# --------------------------------------------------
plt.figure()
plt.plot(range(1, epochs+1), train_loss)
plt.plot(range(1, epochs+1), val_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()
