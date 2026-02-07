
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference.inference import Inference

app = FastAPI(title="Transformer Summarizer API")

# Global model instance
inference_model = None

class ArticleRequest(BaseModel):
    article: str
    
class SummaryResponse(BaseModel):
    summary: str

@app.on_event("startup")
async def load_model():
    global inference_model
    try:
        print("Loading model...")
        # Check if tokenizer exists, if not, wait or warn
        # Inference init handles config loading
        inference_model = Inference()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # In a real deployment, we might want to crash here
        
@app.post("/summarize", response_model=SummaryResponse)
async def summarize(request: ArticleRequest):
    if inference_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        summary = inference_model.predict(request.article)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": inference_model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
