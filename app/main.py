from fastapi import FastAPI
from app.routers import ml_router

app = FastAPI(title="CE Capstone - Intelligent Ticket Assignment")

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }

app.include_router(ml_router.router)