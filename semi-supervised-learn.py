# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Create Sample Data (same as before)
data = {
    'Study_Hours': [1, 2, 2.5, 3, 3.5, 4, 5, 6, 6.5, 7, 7.5, 8, 9],
    'Attendance':  [40, 50, 55, 60, 65, 70, 75, 80, 82, 85, 88, 90, 95],
    'Pass':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Prepare Features (X) and Labels (y)
X = df[['Study_Hours', 'Attendance']]  # Features
y = df['Pass']  # Target label (1 = Pass, 0 = Fail)

# Step 3: Split Data into Labeled and Unlabeled Data
# Let's assume we only have a few labeled data points, the rest are unlabeled
X_labeled = X[:5]  # First 5 samples are labeled
y_labeled = y[:5]  # Corresponding labels

X_unlabeled = X[5:]  # The remaining data is unlabeled
y_unlabeled = [-1] * len(X_unlabeled)  # Label -1 indicates unlabeled data

# Combine labeled and unlabeled data
X_combined = pd.concat([X_labeled, X_unlabeled], axis=0)
y_combined = pd.concat([pd.Series(y_labeled), pd.Series(y_unlabeled)], axis=0)

# Step 4: Train the Label Propagation Model (Semi-Supervised)
model = LabelPropagation()
model.fit(X_combined, y_combined)

# Step 5: Make Predictions on the Entire Dataset (including unlabeled data)
y_pred = model.predict(X_combined)

# Step 6: Evaluate the Model
# We can evaluate on the labeled portion of the data
accuracy = accuracy_score(y_labeled, y_pred[:len(y_labeled)])
print(f'Model Accuracy on Labeled Data: {accuracy * 100:.2f}%')

# Step 7: Predict for a New Student (using the full model)
new_student = np.array([[4.5, 72]])  # Example: 4.5 study hours, 72% attendance
prediction = model.predict(new_student)

print("Prediction for new student:", "Pass" if prediction[0] == 1 else "Fail")
