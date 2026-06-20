"""Per-request trace context (Phase 49).

Import this module anywhere to get/set the current request_id.
Set in app.py middleware, read in agent nodes for log correlation.
"""
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="no-req")
