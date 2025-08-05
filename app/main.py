from fastapi import FastAPI

from app.api.classification import router as classification_router
from app.utils.config import load_config

app = FastAPI(title="Payment Classifier")

app.include_router(classification_router)


@app.get("/")
def root():
    return {"message": "Payment Classifier API root"}


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(app, host="0.0.0.0", port=8000)
