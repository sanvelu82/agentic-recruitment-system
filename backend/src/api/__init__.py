"""
FastAPI REST API for the Agentic Recruitment System.

Provides endpoints for:
- Job management
- Candidate management
- Pipeline execution
- Test management
- Audit and reporting
"""

from .main import app

__all__ = ["app"]
