"""
Tool Registry - Unified tool registration and discovery mechanism.

Wraps medical_tool.MedicalToolRegistry to provide a centralized
tool management interface for the entire application.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Centralized tool registry that wraps MedicalToolRegistry.
    
    Provides:
    - Tool registration and discovery
    - Category-based filtering
    - Tool summary for LLM context
    - Lazy initialization
    """
    
    _instance: Optional['ToolRegistry'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ToolRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._medical_registry = None
            self._external_tools: Dict[str, Any] = {}
            self._initialized = True
            logger.info("[TOOL_REGISTRY] ToolRegistry initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the registry by loading MedicalToolRegistry.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            from agents.medical_tool import get_tool_registry, init_tool_registry
            
            # Initialize medical tools
            init_tool_registry()
            self._medical_registry = get_tool_registry()
            
            tool_count = len(self._medical_registry) if self._medical_registry else 0
            logger.info(f"[TOOL_REGISTRY] Medical tools loaded: {tool_count}")
            return True
            
        except ImportError as e:
            logger.warning(f"[TOOL_REGISTRY] Medical tools not available: {e}")
            return False
        except Exception as e:
            logger.error(f"[TOOL_REGISTRY] Initialization failed: {e}")
            return False
    
    @property
    def medical_registry(self):
        """Get the underlying MedicalToolRegistry."""
        return self._medical_registry
    
    def get_tool(self, name: str) -> Optional[Any]:
        """
        Get a tool by name (checks medical registry first, then external).
        
        Args:
            name: Tool name to look up
            
        Returns:
            Tool instance or None if not found
        """
        # Check medical registry first
        if self._medical_registry:
            tool = self._medical_registry.get(name)
            if tool:
                return tool
        
        # Check external tools
        return self._external_tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[Any]:
        """
        Get all tools in a category.
        
        Args:
            category: Category name (e.g., 'drug', 'coding', 'literature')
            
        Returns:
            List of tools in the specified category
        """
        if self._medical_registry:
            return self._medical_registry.get_by_category(category)
        return []
    
    def register_external_tool(self, name: str, tool: Any) -> None:
        """
        Register an external tool (not from MCP).
        
        Args:
            name: Tool name
            tool: Tool instance with execute() method
        """
        self._external_tools[name] = tool
        logger.info(f"[TOOL_REGISTRY] External tool registered: {name}")
    
    def get_summary(self, max_desc_len: int = 100) -> str:
        """
        Get a summary of all available tools for LLM context.
        
        Args:
            max_desc_len: Maximum description length per tool
            
        Returns:
            Formatted string summarizing available tools
        """
        summaries = []
        
        # Medical tools summary
        if self._medical_registry:
            medical_summary = self._medical_registry.get_summary(max_desc_len=max_desc_len)
            if medical_summary:
                summaries.append(medical_summary)
        
        # External tools summary
        if self._external_tools:
            ext_lines = ["External Tools:"]
            for name, tool in self._external_tools.items():
                desc = getattr(tool, 'description', 'No description')
                if len(desc) > max_desc_len:
                    desc = desc[:max_desc_len] + "..."
                ext_lines.append(f"- {name}: {desc}")
            summaries.append("\n".join(ext_lines))
        
        return "\n\n".join(summaries) if summaries else "No tools available"
    
    def get_all_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        names = []
        
        if self._medical_registry:
            for tool in self._medical_registry._tools.values():
                names.append(tool.name)
        
        names.extend(self._external_tools.keys())
        return names
    
    def is_available(self) -> bool:
        """Check if the registry has any tools available."""
        return (self._medical_registry is not None and len(self._medical_registry) > 0) or \
               len(self._external_tools) > 0
    
    def reload(self) -> None:
        """Reload all tools from scratch."""
        self._medical_registry = None
        self._external_tools.clear()
        self._initialized = False
        self.initialize()
        logger.info("[TOOL_REGISTRY] Registry reloaded")


# Singleton accessor
_registry_instance: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    Get the singleton ToolRegistry instance.
    
    Returns:
        ToolRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance
