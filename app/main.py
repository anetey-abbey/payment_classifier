from fastapi import FastAPI

from app.utils.config import get_args, load_config

app = FastAPI(title="Payment Classifier")


@app.get("/")
def root():
    return {"message": "Payment Classifier API root"}


if __name__ == "__main__":
    import uvicorn

    args = get_args()
    config = load_config(args.config)
    uvicorn.run(app, host="0.0.0.0", port=8000)
