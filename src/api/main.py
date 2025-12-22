from fastapi import FastAPI, HTTPException
from src.api.schemas import RecommendationRequest
from src.services.recommender import RestaurantRecommender

app = FastAPI(
    title="Zomato Restaurant Recommendation API",
    version="1.0"
)

recommender = RestaurantRecommender()


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/recommend")
def recommend_restaurants(request: RecommendationRequest):
    results = recommender.recommend(
        cuisine=request.cuisine,
        top_n=request.top_n
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No restaurants found for this cuisine"
        )

    return {
        "cuisine": request.cuisine,
        "top_n": request.top_n,
        "recommendations": results
    }
