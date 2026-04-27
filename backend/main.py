from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.agents import run_agent
from backend.tools import get_vector_collection, check_tmdb_health
import uvicorn

app = FastAPI(title="Movie Recommender")

# Allow the HTML file to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    mood: str
    history: List[Dict[str, Any]] = []


@app.on_event("startup")
def preload_dependencies() -> None:
    try:
        get_vector_collection()
    except Exception:
        pass

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.mood.strip():
        raise HTTPException(status_code=400, detail="Mood input is required")

    try:
        result = run_agent(req.mood, req.history)
        return {"response": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health/tmdb")
async def tmdb_health_endpoint():
    return check_tmdb_health()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)