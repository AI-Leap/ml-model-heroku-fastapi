from fastapi.testclient import TestClient

from fastapi_server.main import app

client = TestClient(app)


def test_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'project': 'Census DVC Heroku'}


def test_post():
    r = client.post(
        "/predict/",
        headers={"X-Token": "coneofsilence"},
        json={
            "age": 30,
            "workclass": "State-gov",
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    )
    assert r.status_code == 200
    assert r.json() == {
        'prediction': [0]
    }


def test_post2():
    r = client.post(
        "/predict/",
        headers={"X-Token": "coneofsilence"},
        json={
            "age": 29,
            "workclass": "Private",
            "fnlgt": 122127,
            "education_num": 13,
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "hours_per_week": 40,
            "native_country": "United-States",
            "capital_gain": 8614,
            "capital_loss": 0
        }
    )
    assert r.status_code == 200
    assert r.json() == {
        'prediction': [1]
    }
