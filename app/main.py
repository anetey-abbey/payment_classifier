from fastapi import FastAPI

from app.api.router import get_api_router
from app.core.config import load_config
from app.core.event_handlers import get_lifespan_handler


def get_app() -> FastAPI:
    config = load_config()
    app = FastAPI(
        title=config["app_name"],
        version=config["app_version"],
        description=config["app_description"],
        lifespan=get_lifespan_handler(),
    )

    app.include_router(get_api_router())

    @app.get("/")
    def root():
        return {
            "message": f"{config['app_name']} API root",
            "version": config["app_version"],
            "status": "healthy",
        }

    return app


app = get_app()


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(app, host="0.0.0.0", port=8000)
