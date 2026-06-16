from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

# --- Smoke tests TP0 (ne pas modifier) ---
def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_positive_text():
    response = client.post("/predict", json={"text": "Ce produit est excellent"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert 0.0 <= data["score"] <= 1.0
    assert data["text"] == "Ce produit est excellent"

# --- Tests fournis TP2 ---
def test_predict_retourne_structure_complete(client):
    response = client.post("/predict", json={"text": "produit excellent"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "text" in data
    assert data["label"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")
    assert 0.0 <= data["score"] <= 1.0

def test_stats_incremente_apres_predict(client):
    stats_avant = client.get("/stats").json()
    assert stats_avant["total_predictions"] == 0
    client.post("/predict", json={"text": "produit excellent"})
    stats_apres = client.get("/stats").json()
    assert stats_apres["total_predictions"] == 1
    assert stats_apres["positive_count"] == 1

# --- GET /health ---
def test_health_retourne_status_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# --- POST /predict : cas nominaux ---
def test_predict_avec_texte_positif(client, texte_positif):
    response = client.post("/predict", json={"text": texte_positif})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"

def test_predict_avec_texte_negatif(client, texte_negatif):
    response = client.post("/predict", json={"text": texte_negatif})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"

def test_predict_conserve_le_texte_original(client):
    texte = "produit vraiment excellent qualite"
    response = client.post("/predict", json={"text": texte})
    assert response.status_code == 200
    assert response.json()["text"] == texte

# --- GET /stats ---
def test_stats_contient_les_quatre_champs(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "positive_count" in data
    assert "negative_count" in data
    assert "neutral_count" in data

def test_stats_comptabilise_plusieurs_predictions(client):
    client.post("/predict", json={"text": "produit excellent"})
    client.post("/predict", json={"text": "service horrible"})
    client.post("/predict", json={"text": "produit excellent"})
    stats = client.get("/stats").json()
    assert stats["total_predictions"] == 3
    assert stats["positive_count"] == 2
    assert stats["negative_count"] == 1

# --- POST /reset ---
def test_reset_remet_les_compteurs_a_zero(client):
    client.post("/predict", json={"text": "produit excellent"})
    client.post("/predict", json={"text": "service horrible"})
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json() == {"status": "reset"}
    stats = client.get("/stats").json()
    assert stats["total_predictions"] == 0

# --- Cas d'erreur Pydantic ---
def test_predict_texte_vide_retourne_422(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_predict_texte_trop_long_retourne_422(client):
    response = client.post("/predict", json={"text": "a" * 6000})
    assert response.status_code == 422

def test_predict_champ_manquant_retourne_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_mauvais_type_retourne_422(client):
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422