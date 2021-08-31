import requests
import json

body = {
    "age": 60,
    "fnlgt": 337895,
    "education_num": 13,
    "workclass": "Federal-gov",
    "education": "Doctorate",
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Female",
    "hours_per_week": 40,
    "native_country": "Canada"
}

response = requests.post(
    'https://census-fastapi-server.herokuapp.com/predict',
    data=json.dumps(body)
)

print('status:', response.status_code)
print('response body:', response.json())
