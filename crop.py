# ======================================================
# CROP RECOMMENDATION SYSTEM USING MACHINE LEARNING
# ======================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------
df = pd.read_csv("Crop_recommendation.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# ------------------------------------------------------
# 2. Feature & Target Split
# ------------------------------------------------------
X = df.drop("label", axis=1)
y = df["label"]

# ------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# 4. Train Model
# ------------------------------------------------------
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------------------
# 5. Model Accuracy
# ------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ------------------------------------------------------
# 6. USER INPUT LOOP
# ------------------------------------------------------
while True:
    print("\nENTER SOIL & CLIMATE DETAILS")

    try:
        N = float(input("Nitrogen (N): "))
        P = float(input("Phosphorus (P): "))
        K = float(input("Potassium (K): "))
        temperature = float(input("Temperature (C): "))
        humidity = float(input("Humidity (%): "))
        ph = float(input("pH value: "))
        rainfall = float(input("Rainfall (mm): "))

        user_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(user_data)

        print("\nRecommended Crop:", prediction[0])

    except ValueError:
        print("\nPlease enter numeric values only.")
        continue

    again = input("\nDo you want to predict again? (yes/no): ").lower()
    if again != "yes":
        print("\nExiting Crop Recommendation System")
        break
