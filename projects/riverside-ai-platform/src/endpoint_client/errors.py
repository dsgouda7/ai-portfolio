from __future__ import annotations

from email.utils import parsedate_to_datetime
from math import ceil
from time import time

from .models import AppError, DeploymentMetadata, ErrorDetail, TraceMetadata


_SAFE_MESSAGES = {
    "invalid_request": "The request was rejected by the endpoint.",
    "unauthorized": "Endpoint authentication failed.",
    "forbidden": "The caller is not authorized for this operation.",
    "policy_violation": "The request was rejected by policy.",
    "overloaded": "The service is temporarily overloaded.",
    "timeout": "The request deadline was exceeded.",
    "backend_failure": "The model backend could not complete the request.",
    "release_unavailable": "The requested model release is unavailable.",
    "internal_error": "The service could not complete the request.",
}


class EndpointClientError(Exception):
    def __init__(self, envelope: AppError, *, status_code: int | None = None) -> None:
        super().__init__(envelope.error.message)
        self.envelope = envelope
        self.status_code = status_code


def parse_retry_after(value: str | None, *, now: float | None = None) -> int | None:
    if not value:
        return None
    try:
        seconds = ceil(float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
        seconds = ceil(retry_at - (time() if now is None else now))
    return min(300, max(1, seconds))


def error_from_status(
    status_code: int,
    *,
    trace_id: str,
    retry_count: int,
    retry_after_seconds: int | None = None,
    deployment: DeploymentMetadata | None = None,
) -> EndpointClientError:
    if status_code == 400:
        code, error_type, retryable = "invalid_request", "request_error", False
    elif status_code == 401:
        code, error_type, retryable = "unauthorized", "authentication_error", False
    elif status_code == 403:
        code, error_type, retryable = "forbidden", "authorization_error", False
    elif status_code in {404, 410}:
        code, error_type, retryable = "release_unavailable", "service_error", False
    elif status_code in {408, 504}:
        code, error_type, retryable = "timeout", "deadline_error", True
    elif status_code == 429:
        code, error_type, retryable = "overloaded", "capacity_error", True
        retry_after_seconds = retry_after_seconds or 1
    elif status_code >= 500:
        code, error_type, retryable = "backend_failure", "backend_error", True
    else:
        code, error_type, retryable = "backend_failure", "backend_error", False

    detail = ErrorDetail(
        code=code,
        message=_SAFE_MESSAGES[code],
        type=error_type,
        retryable=retryable,
        retry_after_seconds=retry_after_seconds if code == "overloaded" else None,
    )
    return EndpointClientError(
        AppError(
            error=detail,
            trace=TraceMetadata(trace_id=trace_id, retry_count=retry_count),
            deployment=deployment,
        ),
        status_code=status_code,
    )


def transport_error(
    code: str,
    *,
    trace_id: str,
    retry_count: int,
    deployment: DeploymentMetadata | None = None,
) -> EndpointClientError:
    if code == "timeout":
        error_type, retryable = "deadline_error", True
    elif code == "release_unavailable":
        error_type, retryable = "service_error", False
    else:
        error_type, retryable = "backend_error", True
    return EndpointClientError(
        AppError(
            error=ErrorDetail(
                code=code,
                message=_SAFE_MESSAGES[code],
                type=error_type,
                retryable=retryable,
            ),
            trace=TraceMetadata(trace_id=trace_id, retry_count=retry_count),
            deployment=deployment,
        )
    )
