from typing import Optional


class LLMClientError(Exception):

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(f"[{model or 'Unknown Model'}] {message}")
        self.correlation_id = correlation_id
        self.model = model


class LLMTimeoutError(LLMClientError):
    pass


class LLMParseError(LLMClientError):
    pass


class LLMRateLimitError(LLMClientError):
    pass


class LLMValidationError(LLMClientError):
    pass
