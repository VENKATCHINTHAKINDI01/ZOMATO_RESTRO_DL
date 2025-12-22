from pydantic import BaseModel
from typing import List


class RecommendationRequest(BaseModel):
    cuisine: str
    top_n: int = 5


class RestaurantResponse(BaseModel):
    name: str
    cuisines: str
    cost: float
    predicted_score: float
    similarity_score: float
    final_score: float
