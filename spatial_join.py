import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

# --------------------------------------------
# STEP 1: Load your labeled data
# --------------------------------------------
print("Loading earthquakes_subduction_expanded.csv...")
df = pd.read_csv("earthquakes_subduction_expanded.csv")
print(df.head())

# --------------------------------------------
# STEP 2: Select features and label
# --------------------------------------------
# Define input columns (instant features only)
numeric_features = ['latitude', 'longitude', 'depth', 'mag', 'year', 'month', 'day_of_year']
categorical_features = ['magType']
label_column = 'subduction_flag'

# Drop rows with missing values in selected columns
df = df.dropna(subset=numeric_features + categorical_features + [label_column])

# One-hot encode categorical features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_features = ohe.fit_transform(df[categorical_features])
ohe_feature_names = ohe.get_feature_names_out(categorical_features)
ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names)

# Combine numeric and one-hot features
X = pd.concat([df[numeric_features].reset_index(drop=True), ohe_df], axis=1).values
y = df[label_column].values

# --------------------------------------------
# STEP 3: Train-test split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# --------------------------------------------
# STEP 4: Standardize features
# --------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print("✅ Data ready for PyTorch!")

# --------------------------------------------
# STEP 5: Define the neural network
# --------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # no sigmoid
        )

    def forward(self, x):
        return self.layers(x)

input_dim = X_train.shape[1]
model = SimpleNN(input_dim)
print(model)

# --------------------------------------------
# STEP 6: Use balanced loss function
# --------------------------------------------
n_positive = y_train.sum()
n_negative = len(y_train) - n_positive
pos_weight = torch.tensor([n_negative / n_positive])
print(f"Using pos_weight = {pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------
# STEP 7: Train the model
# --------------------------------------------
EPOCHS = 100
losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# Plot training loss
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# --------------------------------------------
# STEP 8: Evaluate the model and add visuals
# --------------------------------------------
model.eval()
with torch.no_grad():
    logits_test = model(X_test_tensor)
    y_pred_prob = torch.sigmoid(logits_test).numpy().flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC: {roc_auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], '--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
avg_prec = average_precision_score(y_test, y_pred_prob)
plt.figure()
plt.plot(recall, precision, label=f"AP = {avg_prec:.2f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.show()
