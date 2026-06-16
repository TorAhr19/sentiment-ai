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

    NEGATION_WORDS = {"pas", "plus", "jamais", "sans", "aucun"}

    def __init__(self):
        logger.info("SentimentModel initialisé.")

    def predict(self, text: str) -> dict:
        """
        Analyse le sentiment d'un texte.
        Gère les négations simples : un mot de négation inverse
        le sentiment du mot positif ou négatif qui le suit directement.
        """
        tokens = re.findall(r"[a-zA-Z]+", text.lower())

        if not tokens:
            raise SentimentError(f"Aucun mot détecté dans le texte : '{text}'")

        pos = 0
        neg = 0
        negate = False

        for token in tokens:
            if token in self.NEGATION_WORDS:
                negate = True
                continue

            if token in self.POSITIVE_WORDS:
                if negate:
                    neg += 1
                else:
                    pos += 1
                negate = False

            elif token in self.NEGATIVE_WORDS:
                if negate:
                    pos += 1
                else:
                    neg += 1
                negate = False

            else:
                negate = False

        if pos > neg:
            score = round(min(0.6 + 0.1 * pos, 1.0), 2)
            return {"label": "POSITIVE", "score": score, "text": text}

        if neg > pos:
            score = round(min(0.6 + 0.1 * neg, 1.0), 2)
            return {"label": "NEGATIVE", "score": score, "text": text}

        return {"label": "NEUTRAL", "score": 0.5, "text": text}