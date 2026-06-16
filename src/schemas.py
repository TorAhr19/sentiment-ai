from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
        description="Texte à analyser (1 à 5000 caractères)"
    )

class PredictionResponse(BaseModel):
    label: str
    score: float
    text: str

class StatsResponse(BaseModel):
    total_predictions: int
    positive_count: int
    negative_count: int
    neutral_count: int