import logging
import re

logger = logging.getLogger("sentimentai")

class SentimentError(Exception):
    """Levée quand le texte ne contient aucun mot analysable."""
    pass

class SentimentModel:
    POSITIVE_WORDS = [
        "bien", "super", "excellent", "parfait", "bon",
        "aime", "adore", "rapide", "fiable", "recommande"
    ]
    NEGATIVE_WORDS = [
        "mal", "nul", "horrible", "mauvais", "deteste",
        "pire", "lent", "casse", "decu", "probleme"
    ]

    def __init__(self):
        logger.info("SentimentModel initialisé.")

    def predict(self, text: str) -> dict:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        if not tokens:
            raise SentimentError(f"Aucun mot détecté dans le texte : '{text}'")

        text_lower = text.lower()
        pos = sum(1 for w in self.POSITIVE_WORDS if re.search(r'\b' + w + r'\b', text_lower))
        neg = sum(1 for w in self.NEGATIVE_WORDS if re.search(r'\b' + w + r'\b', text_lower))

        if pos > neg:
             score = round(min(0.6 + 0.1 * pos, 1.0), 2)
             return {"label": "POSITIVE", "score": score, "text": text}
        elif neg > pos:
             score = round(min(0.6 + 0.1 * neg, 1.0), 2)
             return {"label": "NEGATIVE", "score": score, "text": text}
        return {"label": "NEUTRAL", "score": 0.5, "text": text}