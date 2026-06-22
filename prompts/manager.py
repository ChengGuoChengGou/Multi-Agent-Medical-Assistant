"""
Prompt Manager - Centralized prompt template management.

Features:
- Template loading from markdown files
- Variable substitution with {{variable}} syntax
- Caching for performance
- Version tracking
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Prompt templates directory
PROMPTS_DIR = Path(__file__).parent


class PromptManager:
    """Manages prompt templates with caching and variable substitution."""
    
    def __init__(self, prompts_dir: Path = None):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self._cache: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all .md templates from prompts directory."""
        if not self.prompts_dir.exists():
            logger.warning(f"[PROMPT_MGR] Prompts directory not found: {self.prompts_dir}")
            return
        
        for md_file in self.prompts_dir.glob("*.md"):
            try:
                template_name = md_file.stem
                with open(md_file, 'r', encoding='utf-8') as f:
                    self._cache[template_name] = f.read()
                logger.debug(f"[PROMPT_MGR] Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"[PROMPT_MGR] Failed to load {md_file}: {e}")
    
    def get(self, template_name: str, **kwargs) -> str:
        """
        Get prompt template with variable substitution.
        
        Args:
            template_name: Name of template (without .md extension)
            **kwargs: Variables to substitute in template
        
        Returns:
            Formatted prompt string
        """
        if template_name not in self._cache:
            raise KeyError(f"Template '{template_name}' not found. Available: {list(self._cache.keys())}")
        
        template = self._cache[template_name]
        
        # Substitute {{variable}} patterns
        if kwargs:
            def replace_var(match):
                var_name = match.group(1).strip()
                if var_name in kwargs:
                    return str(kwargs[var_name])
                return match.group(0)  # Leave unchanged if no value
            
            template = re.sub(r'\{\{(\w+)\}\}', replace_var, template)
        
        return template
    
    def list_templates(self) -> list:
        """List all available template names."""
        return list(self._cache.keys())
    
    def reload(self):
        """Reload all templates from disk."""
        self._cache.clear()
        self._load_templates()
        logger.info(f"[PROMPT_MGR] Reloaded {len(self._cache)} templates")


# Singleton instance
_manager_instance: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get singleton PromptManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = PromptManager()
    return _manager_instance
