from fastapi.testclient import TestClient
from predictive_maintenance.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_smoke():
    row = {
        "unit_id": 1, "cycle": 1,
        "op_1": 0.0, "op_2": 0.0, "op_3": 0.0,
        **{f"s_{i}": 0.0 for i in range(1, 22)},
    }
    r = client.post("/predict", json={"rows": [row]})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 1
    assert 0.0 <= body[0]["risk_proba"] <= 1.0
    assert body[0]["risk_label"] in (0, 1)
