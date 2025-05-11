from fastapi import FastAPI
from routes.summarization import router as summarization_router
import uvicorn

app = FastAPI(
    title="Bookbot AI Backend",
    description="API for book summarization using PEGASUS, BART, BERTSum",
    version="1.0.0"
)

# Mount the summarization router
app.include_router(summarization_router, prefix="/api/summarize", tags=["Summarization"])

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)