import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

print("✅ Data ready for Keras!")

# --------------------------------------------
# STEP 5: Build the model
# --------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print(model.summary())

# --------------------------------------------
# STEP 6: Compute class weights to handle imbalance
# --------------------------------------------
# class_weight = {0: weight for non-subduction, 1: weight for subduction}
# Example: weight positives more if they are the minority
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Using class_weight:", class_weight_dict)

# --------------------------------------------
# STEP 7: Train the model with early stopping
# --------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# --------------------------------------------
# STEP 8: Plot training loss and AUC
# --------------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['auc'], label='train AUC')
plt.plot(history.history['val_auc'], label='val AUC')
plt.title('AUC')
plt.legend()

plt.show()

# --------------------------------------------
# STEP 9: Evaluate on test set
# --------------------------------------------
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
