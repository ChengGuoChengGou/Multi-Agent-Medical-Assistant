"""
Prompt Management System for Medical Assistant.

Provides centralized prompt template management with caching and versioning.
"""

from .manager import get_prompt_manager, PromptManager

__all__ = ["get_prompt_manager", "PromptManager"]
