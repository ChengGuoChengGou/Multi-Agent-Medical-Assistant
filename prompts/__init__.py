"""
Prompt Management System for Medical Assistant.

Provides centralized prompt template management with caching and versioning.
"""

from .manager import PromptManager, get_prompt_manager

__all__ = ["PromptManager", "get_prompt_manager"]
