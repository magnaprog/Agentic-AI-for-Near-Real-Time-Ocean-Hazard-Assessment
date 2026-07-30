"""Shared upstream-error translation for the Mission Control BFF routers.

Every router that proxies the core hazard API needs the same policy for an
error from that upstream call. Keeping it in one place also keeps the safety
distinction correct in one place: a transient transport failure must never be
answered with a fabricated demo payload.
"""

from __future__ import annotations

from typing import NoReturn

import httpx
from fastapi import HTTPException


class UpstreamKeyNotConfiguredError(RuntimeError):
    """MISSION_CONTROL_HAZARD_API_KEY is unset, so there is no upstream to poll.

    This is the only condition that may be answered with the built-in demo
    snapshot. It is a distinct type rather than a bare ``RuntimeError``
    because ``RuntimeError`` is also what httpx raises for a client that has
    been closed ("Cannot send a request, as the client has been closed.") and
    what ``HazardClient._get_client`` raises when startup did not run.
    Catching the base class meant either of those transport faults served
    fabricated demo evidence with HTTP 200 in a deployment that does have a
    key configured, which is the substitution this module exists to prevent.
    """


def raise_upstream_error(exc: httpx.HTTPError) -> NoReturn:
    """Translate an httpx error from the core API into an HTTPException.

    A transient transport failure (connection refused, timeout, DNS) means the
    core API is unreachable right now, so surface 503. This is deliberately
    distinct from the missing-key case (``UpstreamKeyNotConfiguredError``, raised
    before any request is made), which the routers handle as genuine demo
    mode: a duty
    scientist must not be shown fabricated demo evidence for a live event just
    because a request transiently failed. An upstream 4xx is passed through
    with its detail; any other upstream status is reported as 502. Never
    returns.

    Upstream 401 and 403 are the exception to that passthrough. They report
    that the BFF's own MISSION_CONTROL_HAZARD_API_KEY was rejected, which is a
    server misconfiguration, not a problem with the operator's credentials.
    Forwarding them verbatim made the console read its own access key as
    rejected and re-lock, so a duty scientist was told to re-enter a key that
    was already correct and could not get back in. They are reported as 502,
    which keeps a probe-confirmed 401 meaning exactly one thing: the key the
    operator supplied is bad.
    """
    if isinstance(exc, httpx.RequestError):
        raise HTTPException(status_code=503, detail="Core hazard API is unreachable")
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="Core hazard API rejected the service credentials",
            )
        if 400 <= status < 500:
            detail = "Upstream request failed"
            try:
                detail = exc.response.json().get("detail", detail)
            except Exception:
                pass
            raise HTTPException(status_code=status, detail=detail)
    raise HTTPException(status_code=502, detail="Upstream service error")
