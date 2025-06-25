import pandas as pd
import joblib

# Load saved model
model = joblib.load("best_model.pkl")

# Create a single synthetic sample (one row)
sample = pd.DataFrame([{
    "area": 300,
    "bedrooms": 1,
    "bathrooms": 1,
    "stories": 1,
    "mainroad": "no",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "no",
    "parking": 0,
    "prefarea": "no",
    "furnishingstatus": "semi-furnished"
}])


# sample = pd.DataFrame([{
#     "Age": 54,
#     "Sex": 1,
#     "Chest pain type": 3,
#     "BP": 130,
#     "Cholesterol": 250,
#     "FBS over 120": 0,
#     "EKG results": 2,
#     "Max HR": 150,
#     "Exercise angina": 0,
#     "ST depression": 1.5,
#     "Slope of ST": 2,
#     "Number of vessels fluro": 0,
#     "Thallium": 3
# }])

# Match training encoding (one-hot)
sample_encoded = pd.get_dummies(sample)

# Load original training columns to align (optional if you saved them)
# Here we just ensure no missing expected features
expected_columns = getattr(model, "feature_names_", getattr(model, "feature_names_in_", sample_encoded.columns))
for col in expected_columns:
    if col not in sample_encoded.columns:
        sample_encoded[col] = 0

# Ensure column order matches
sample_encoded = sample_encoded[expected_columns]

# Predict
prediction = model.predict(sample_encoded)

# print("Predicted:", "Presence" if prediction[0] == 1 else "Absence")

print("Predicted House Price (INR):", prediction[0])
