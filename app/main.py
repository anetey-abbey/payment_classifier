import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.classification import router as classification_router
from app.utils.config import load_config
from app.utils.logging import log_api_request, log_shutdown, log_startup, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log_startup()
    app.state.llm_clients = {}
    yield
    log_shutdown()
    app.state.llm_clients.clear()


app = FastAPI(title="Payment Classifier", lifespan=lifespan)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    log_api_request(
        method=request.method,
        path=str(request.url.path),
        status_code=response.status_code,
        duration_ms=duration_ms,
    )

    return response


app.include_router(classification_router)


@app.get("/")
def root():
    return {"message": "Payment Classifier API root"}


if __name__ == "__main__":
    import uvicorn

    setup_logging()
    config = load_config()
    uvicorn.run(app, host="0.0.0.0", port=8000)
