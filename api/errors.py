"""
Custom exception hierarchy for the Medical AI Agent API.

Provides consistent error responses across all endpoints.
Each exception type maps to a specific HTTP status code and error format.
"""

from __future__ import annotations
from typing import Optional


class AppError(Exception):
    """Base application error."""
    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    """Resource not found."""
    status_code = 404
    error_code = "not_found"


class ValidationError(AppError):
    """Input validation failed."""
    status_code = 400
    error_code = "validation_error"


class ConflictError(AppError):
    """Resource already exists or conflicts with current state."""
    status_code = 409
    error_code = "conflict"


class ServiceUnavailableError(AppError):
    """External service or dependency is unavailable."""
    status_code = 503
    error_code = "service_unauthorized"


class AuthenticationError(AppError):
    """Authentication failed."""
    status_code = 401
    error_code = "authentication_failed"


class AuthorizationError(AppError):
    """Insufficient permissions."""
    status_code = 403
    error_code = "forbidden"
