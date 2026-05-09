# =========================================
# CodeAlpha Internship Project
# Iris Flower Classification (Final Clean Version)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================
# Create Output Folder
# =========================================
os.makedirs("output", exist_ok=True)

# =========================================
# Load Dataset
# =========================================
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# =========================================
# Basic Info
# =========================================
print("First 5 Rows:\n")
print(X.head())

print("\nDataset Shape:", X.shape)

# =========================================
# Visualization Data
# =========================================
iris_df = X.copy()
iris_df["species"] = y

# =========================================
# Pairplot (SAVE FIXED)
# =========================================
plot = sns.pairplot(iris_df, hue="species")
plot.fig.savefig("output/pairplot.png")
plt.show()

# =========================================
# Heatmap (SAVE)
# =========================================
plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("output/heatmap.png")
plt.show()

# =========================================
# Train Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# Feature Scaling
# =========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================
# Model Training
# =========================================
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# =========================================
# Prediction
# =========================================
y_pred = model.predict(X_test)

# =========================================
# Accuracy
# =========================================
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc * 100, "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================================
# Confusion Matrix (SAVE)
# =========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("output/confusion_matrix.png")
plt.show()

print("\nProject Completed Successfully 🚀")