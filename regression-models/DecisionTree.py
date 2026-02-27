import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris =load_iris()
X,y=iris.data, iris.target

X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.3,random_state=42, stratify=y)

clf=DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)

y_pred =clf.predict(X_test)

accuracy =accuracy_score(y_test,y_pred)
print(f"Accuracy Score:{accuracy:.4f}")

print("\nClaasification Report")
print(classification_report(y_test,y_pred,target_names=iris.target_names))

cm= confusion_matrix(y_test,y_pred)

print("Confusion Matrix:")
print("Rows: Actual | Columns: Predicted")
print("                 Setosa Versicolor Virginica")
for i,species in enumerate(iris.target_names):
    print(f"{species:12} {cm[i][0]:7} {cm[i][1]:11} {cm[i][2]:10}")

print("\nManual Precison, Recall, and F1-score Calculation:")

for i,species in enumerate(iris.target_names):
    TP =cm[i,i]
    FP = cm[:, i].sum() - TP
    FN= cm[i, :].sum() - TP
    precision= TP/(TP+FP) if (TP +FP) != 0 else 0
    recall =TP /(TP+FN) if (TP+FN) != 0 else 0
    f1_score =(2*precision *recall) / (precision +recall) if (precision+recall) != 0 else 0

    print(f"\nSpecies :{species}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall  :{recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

plt.figure(figsize=(12,8))
plot_tree(
    clf,
    feature_names =iris.feature_names,
    class_names=iris.target_names,
    filled =True,
    rounded =True
)
plt.title("Decision Tree Iris Species Classification")
plt.show()

#Effect of different variables
print("\nEffect of Diffrent Variables(Feature Importance): ")
for feature, importance in zip(iris.feature_names, clf.feature_importances_):
    print(f"{feature:25}: {importance:.4f}")

#Effect of training–testing split percentage
print("\nEffect of Train-Test Split Percentage:")

for split in [0.2, 0.3, 0.4]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=split,
        stratify=y,
        random_state=42
    )
    
    temp_clf = DecisionTreeClassifier(random_state=42)
    temp_clf.fit(X_tr, y_tr)
    y_p = temp_clf.predict(X_te)
    
    acc = accuracy_score(y_te, y_p)
    print(f"Test size = {split} -> Accuracy = {acc:.4f}")

#Effect of stratify=y
print("\nEffect of Stratify Parameter:")

# Without stratify
X_tr_ns, X_te_ns, y_tr_ns, y_te_ns = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# With stratify
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Class distribution WITHOUT stratify:", np.bincount(y_te_ns))
print("Class distribution WITH stratify   :", np.bincount(y_te_s))

#Effect of random_state
print("\nEffect of random_state Parameter:")

X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
    X, y, test_size=0.3
)
clf_rs = DecisionTreeClassifier()
clf_rs.fit(X_tr1, y_tr1)
y_pred_rs = clf_rs.predict(X_te1)

print("Accuracy without random_state:",
      accuracy_score(y_te1, y_pred_rs))



# stratify =y nhi karne mein kya hoga usko check karna hai
#student dataset use karna, hot encoding <- categorical data 
    
