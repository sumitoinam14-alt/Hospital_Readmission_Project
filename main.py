# Hospital Readmission Prediction Project
# Ready-to-run example

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load data
# Make sure you have your CSV dataset named "data.csv" in the same folder
try:
    data = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Please put 'data.csv' in the Hospital_Readmission_Project folder.")
    exit()

# Step 2: Quick look at the data
print("Data Preview:")
print(data.head())
print("\nColumns:")
print(data.columns)

# Step 3: Select some important features
# (You can adjust these columns depending on your dataset)
features = ["race", "age", "admission_type_id", "num_lab_procedures", "num_medications"]
target = "readmitted"  # This is what we want to predict

# Step 4: Encode categorical features
for col in features + [target]:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Step 5: Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
