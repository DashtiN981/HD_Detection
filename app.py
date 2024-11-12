from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained model (replace with the actual path of the model file if needed)
model = joblib.load("heart_disease_model.pkl")
# Initialize the FastAPI app
app = FastAPI()

# Define input data model
class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    # Convert input data to a NumPy array
    data = np.array([[input_data.age, input_data.sex, input_data.cp, input_data.trestbps,
                      input_data.chol, input_data.fbs, input_data.restecg, input_data.thalach,
                      input_data.exang, input_data.oldpeak, input_data.slope, input_data.ca,
                      input_data.thal]])
    
    # Get model prediction
    prediction = model.predict(data)[0]
    
    # Map prediction to a readable result
    disease_severity = {0: "No Disease", 1: "Weak", 2: "Medium", 3: "Strong", 4: "Severe"}
    result = {"prediction": int(prediction), "severity": disease_severity[prediction]}
    
    return result