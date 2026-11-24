"""Validation errors that remain JSON serializable for invalid float tokens."""

import math

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    # JSON parsers may accept nonstandard NaN/Infinity tokens, but the default
    # validation response would try to serialize them as nonfinite JSON numbers.
    details = jsonable_encoder(
        exc.errors(),
        custom_encoder={
            float: lambda value: str(value) if not math.isfinite(value) else value
        },
    )
    return JSONResponse(status_code=422, content={"detail": details})
