import json
import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO"):
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "app.log")],
    )


def log_startup():
    logger = logging.getLogger("app.startup")
    environment = os.getenv("ENVIRONMENT", "dev")
    logger.info(f"Payment Classifier API starting up (env: {environment})")


def log_shutdown():
    logger = logging.getLogger("app.shutdown")
    logger.info("Payment Classifier API shutting down")


def log_classification(
    model_name: str,
    result_category: Optional[str],
    duration_ms: float,
    confidence: Optional[float] = None,
    error: Optional[str] = None,
):
    logger = logging.getLogger("classification")

    log_data = {
        "type": "classification",
        "model": model_name,
        "result": result_category if not error else None,
        "duration_ms": round(duration_ms, 2),
        "confidence": confidence,
        "error": error,
        "success": error is None,
    }

    level = logging.ERROR if error else logging.INFO
    logger.log(level, json.dumps(log_data))


def log_llm_call(
    model_name: str, duration_ms: float, success: bool, error_msg: Optional[str] = None
):
    logger = logging.getLogger("llm")

    log_data = {
        "type": "llm_call",
        "model": model_name,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "error": error_msg,
    }

    level = logging.ERROR if not success else logging.INFO
    logger.log(level, json.dumps(log_data))


def log_api_request(method: str, path: str, status_code: int, duration_ms: float):
    logger = logging.getLogger("api")

    log_data = {
        "type": "api_request",
        "method": method,
        "path": path,
        "status": status_code,
        "duration_ms": round(duration_ms, 2),
    }

    level = logging.ERROR if status_code >= 400 else logging.INFO
    logger.log(level, json.dumps(log_data))
