# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
data = pd.read_csv('heart_disease.csv')

# Rename columns based on provided information
data.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# Convert 'ca' and 'thal' columns to numeric and replace missing values with the median
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
data['ca'].fillna(data['ca'].median(), inplace=True)
data['thal'].fillna(data['thal'].median(), inplace=True)

# Standardize numeric features
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Split data into features and target
X = data.drop("num", axis=1)
y = data["num"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=5, random_state=42)

# Create the Voting Classifier with soft voting
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('xgb', xgb_model)],
    voting='soft'
)

# Train the Voting Classifier
voting_model.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = voting_model.predict(X_test)

# Evaluate the model's performance
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nClassification Report on Test Set:\n", classification_report(y_test, y_pred))

import joblib

# Assuming your model is named `best_xgb_model`
joblib.dump(voting_model, "heart_disease_model.pkl")
