# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Step 1: Create Sample Data // Dataset generation
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

# Step 3: Split Data into Training and Testing Sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Logistic Regression Model
model = GaussianNB()
# Training the model 
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Step 7: Predict for a New Student
new_student = np.array([[2.5, 20]])  # Example: 4.5 study hours, 72% attendance
prediction = model.predict(new_student)

print("Prediction for new student:", "Pass" if prediction[0] == 1 else "Fail")
