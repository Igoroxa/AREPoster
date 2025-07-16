import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# --------------------------------------------
# STEP 1: Load your labeled data
# --------------------------------------------
print("Loading earthquakes_subduction.csv...")
df = pd.read_csv("earthquakes_subduction.csv")
print(df.head())

# --------------------------------------------
# STEP 2: Prepare features and label
# --------------------------------------------
features = ['latitude', 'longitude', 'depth', 'mag', 'year', 'month', 'day_of_year']
X = df[features].values
y = df['subduction_flag'].values

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

# Convert to PyTorch tensors
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
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            # IMPORTANT: No Sigmoid here!
        )

    def forward(self, x):
        return self.layers(x)

input_dim = X_train_tensor.shape[1]
model = SimpleNN(input_dim)
print(model)

# --------------------------------------------
# STEP 6: Use balanced loss function
# --------------------------------------------
# Calculate positive class weight
n_positive = y_train.sum()
n_negative = len(y_train) - n_positive
pos_weight = torch.tensor([n_negative / n_positive])
print(f"Using pos_weight = {pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # balanced loss
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
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.show()

# --------------------------------------------
# STEP 8: Evaluate the model
# --------------------------------------------
model.eval()
with torch.no_grad():
    logits_test = model(X_test_tensor)
    y_pred_prob = torch.sigmoid(logits_test).numpy().flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
