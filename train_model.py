import pickle
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.preprocessing import StandardScaler
import numpy as np

# Simulating training data with only 4 features
X_train = np.random.rand(100, 4)  # ✅ 100 samples, 4 features
y_train = np.random.randint(2, size=100)  # Binary classification

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Train a scaler (with 4 features)
scaler = StandardScaler()
scaler.fit(X_train)

# Save the model
with open("your_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model & Scaler saved!")
