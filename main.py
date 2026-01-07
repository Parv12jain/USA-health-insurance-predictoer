from  fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi import HTTPException

with open("model_usa.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(
    title="USA HEALTH INSURANCE PREDICTOR API",
    description="predicts the insurance charges",
    version="1.0.0"
)

# input

class ChargeInput(BaseModel):
    age: int
    sex: str      
    bmi: float
    children: int
    smoker: str  
    region: str

# mapping

sex_map = {"female": 0, "male": 1}
smoker_map = {"yes": 1, "no": 0}
region_map = {
    "southeast": 1,
    "southwest": 2,
    "northeast": 3,
    "northwest": 4
}

# root 
@app.get("/")
def home():
    return {"message": "Insurance Charge Prediction API is running 🚀"}


# endpoint
@app.post("/predict")
def predict_charges(data: ChargeInput):

    # -------- VALIDATION --------
    if data.sex.lower() not in sex_map:
        raise HTTPException(status_code=400, detail="sex must be 'male' or 'female'")

    if data.smoker.lower() not in smoker_map:
        raise HTTPException(status_code=400, detail="smoker must be 'yes' or 'no'")

    if data.region.lower() not in region_map:
        raise HTTPException(status_code=400, detail="invalid region value")

    # -------- ENCODING --------
    sex = sex_map[data.sex.lower()]
    smoker = smoker_map[data.smoker.lower()]
    region = region_map[data.region.lower()]

    # -------- MODEL INPUT --------
    features = np.array([[ 
        data.age,
        sex,
        data.bmi,
        data.children,
        smoker,
        region
    ]])

    # -------- PREDICTION --------
    prediction = model.predict(features)[0]

    return {
        "predicted_insurance_charges": round(float(prediction), 2)
    }



