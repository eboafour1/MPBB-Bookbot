from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.summarization import router as summarization_router
import uvicorn

app = FastAPI(
    title="Bookbot AI Backend",
    description="API for book summarization using PEGASUS, BART, BERTSum",
    version="1.0.0"
)

# 1) Configure CORS to allow your React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React’s URL
    allow_credentials=True,
    allow_methods=["*"],      # GET, POST, OPTIONS, etc
    allow_headers=["*"],      # any headers you send
)


# Mount the summarization router
app.include_router(summarization_router, prefix="/api/summarize", tags=["Summarization"])

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)