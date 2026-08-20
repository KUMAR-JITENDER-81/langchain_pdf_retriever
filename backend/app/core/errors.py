from __future__ import annotations


class AppError(Exception):
    """A safe, structured error that can be returned by the API."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "application_error",
        status_code: int = 400,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class DocumentNotFoundError(AppError):
    def __init__(self, message: str = "Document not found") -> None:
        super().__init__(message, code="document_not_found", status_code=404)


class InvalidPDFError(AppError):
    def __init__(self, message: str = "The uploaded file is not a readable PDF") -> None:
        super().__init__(message, code="invalid_pdf", status_code=422)


class UploadTooLargeError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="upload_too_large", status_code=413)


class DocumentLimitError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="document_limit_exceeded", status_code=413)


class ConfigurationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="configuration_error", status_code=503)


class ProviderUnavailableError(AppError):
    def __init__(self, message: str, *, code: str = "provider_unavailable") -> None:
        super().__init__(message, code=code, status_code=503, retryable=True)


class ProviderQuotaError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="provider_quota_exceeded", status_code=503)
