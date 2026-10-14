import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    f1_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

 #total no of rows and columns
print(df.shape)

#Clean the dataset

if "patient id" in df.columns:
    df = df.drop(columns=["patient id"])
    if "Name" in df.columns:
        df = df.drop(columns=["Name"])

#Find target Column
target_col = (
     "Heart Attack Risk" if "Heart Attack Risk" in df.columns else df.columns[-1]
 )

#convert text to numeric values
df = pd.get_dummies(df, drop_first=True)

# separate the dataset into features(x) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# split and scale the data 
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 , stratify=y
    )

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the model  

model = LogisticRegression(solver="lbfgs", C=1.0, random_state=42)
model.fit(x_train_scaled, y_train)
print("Model trained successfully.")

#make predictions 
y_prob = model.predict_proba(x_test_scaled)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

#Evaluate performance 

print ("\n---performance Results---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

#Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix = Heart Risk Prediction")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["No Risk (0)", "Risk (1)"])
plt.yticks(tick_marks, ["No Risk (0)", "Risk (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()