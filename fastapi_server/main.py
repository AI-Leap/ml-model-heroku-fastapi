from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from starter.starter.ml.data import process_data
from starter.starter.ml.model import load_model, inference
from typing import Optional


app = FastAPI()

model = load_model('./starter/model/model.pkl')
encoder = load_model('./starter/model/encoder.pkl')
lb = load_model('./starter/model/lb.pkl')


class InputExample(BaseModel):
    age: int
    workclass: str
    fnlgt: Optional[int] = 7000
    education: str
    education_num: Optional[int] = 10
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: Optional[int] = 0
    capital_loss: Optional[int] = 0
    hours_per_week: int
    native_country: str


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"project": "Census DVC Heroku"}


@app.post("/predict/")
async def predict(input: InputExample):
    input = input.dict()
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    x, _, _, _ = process_data(
        X=pd.DataFrame(input, index=[0]),
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    pred = inference(model, x)

    return {
        "prediction": pred.tolist()
    }
