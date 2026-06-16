import pytest
from unittest.mock import patch
from src.model import SentimentModel, SentimentError

# --- Fixture ---
@pytest.fixture
def model():
    """Instance de SentimentModel partagée par les tests."""
    return SentimentModel()

# --- Tests fournis ---
def test_predict_retourne_positive(model):
    """Un texte contenant un mot positif doit retourner POSITIVE."""
    texte = "Ce produit est excellent"
    result = model.predict(texte)
    assert result["label"] == "POSITIVE"
    assert 0.0 <= result["score"] <= 1.0
    assert result["text"] == texte

def test_predict_retourne_les_trois_champs(model):
    """La réponse doit toujours contenir label, score et text."""
    result = model.predict("produit horrible")
    assert "label" in result
    assert "score" in result
    assert "text" in result
    assert result["label"] == "NEGATIVE"

# --- Squelettes complétés ---
def test_predict_retourne_negative(model):
    """Un texte avec un mot négatif doit retourner NEGATIVE."""
    result = model.predict("Ce service est horrible")
    assert result["label"] == "NEGATIVE"
    assert 0.0 <= result["score"] <= 1.0

def test_predict_retourne_neutral(model):
    """Un texte sans mot-clé connu doit retourner NEUTRAL."""
    result = model.predict("Je regarde la pluie")
    assert result["label"] == "NEUTRAL"
    assert result["score"] == 0.5

def test_predict_leve_sentiment_error_sur_texte_sans_mots(model):
    """Un texte sans aucun mot doit lever SentimentError."""
    with pytest.raises(SentimentError):
        model.predict("1234 5678")

@pytest.mark.parametrize("texte, label_attendu", [
    ("produit super", "POSITIVE"),
    ("vraiment nul", "NEGATIVE"),
    ("bon et fiable", "POSITIVE"),
    ("lent et mauvais", "NEGATIVE"),
    ("recu hier", "NEUTRAL"),
])
def test_predict_labels_parametrises(model, texte, label_attendu):
    """Vérifier plusieurs cas de label en une seule fonction."""
    result = model.predict(texte)
    assert result["label"] == label_attendu

def test_predict_score_ne_depasse_pas_1(model):
    """Le score ne doit jamais dépasser 1.0, même avec beaucoup de mots."""
    texte = "super excellent parfait bon bien aime adore rapide fiable recommande"
    result = model.predict(texte)
    assert result["score"] <= 1.0
    assert result["label"] == "POSITIVE"

def test_predict_equilibre_retourne_neutral(model):
    """Autant de mots positifs que négatifs : le label doit être NEUTRAL."""
    result = model.predict("produit excellent mais service horrible")
    assert result["label"] == "NEUTRAL"

# --- Test avec mock ---
def test_predict_est_appele_avec_le_bon_argument():
    """Vérifier que predict() reçoit bien le texte transmis."""
    model = SentimentModel()
    texte = "produit excellent"
    with patch.object(
        model,
        "predict",
        return_value={"label": "POSITIVE", "score": 0.7, "text": texte}
    ) as mock_predict:
        result = model.predict(texte)
        mock_predict.assert_called_once_with(texte)
        assert result["label"] == "POSITIVE"

# --- Test libre ---
def test_predict_conserve_texte_original(model):
    """Le texte original doit être conservé même après analyse en minuscules."""
    texte = "Ce Produit Est EXCELLENT"
    result = model.predict(texte)
    assert result["text"] == texte