import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Print to confirm script is running
print("✅ Training model...")

# Example BMI data and labels
# Categories:
# 0 = Underweight, 1 = Normal, 2 = Overweight, 3 = Obese
X = np.array([
    [15], [16], [17],        # Underweight
    [18.5], [22], [24.9],    # Normal
    [25], [27], [29.9],      # Overweight
    [30], [33], [36]         # Obese
])
y = np.array([
    0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    3, 3, 3
])

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model
joblib.dump(model, "bmi_model.pkl")
print("✅ Model trained and saved as bmi_model.pkl")
