import inspect
from typing import Dict, List, Optional, Callable
import json

class ToolRegistry:
    """
    A registry to hold and discover tools dynamically.
    Can scale to thousands of tools.
    """
    def __init__(self, tools: Optional[List[Callable]] = None):
        self.registry: Dict[str, Callable] = {}
        self._discovery_tool_cache = None
        if tools:
            self.register_tools(tools)
        
    def register_tool(self, tool: Callable):
        """Register a tool in the registry."""
        if hasattr(tool, 'name'):
            self.registry[tool.name] = tool
        elif hasattr(tool, '__name__'):
            self.registry[tool.__name__] = tool
            
    def register_tools(self, tools: List[Callable]):
        """Register multiple tools."""
        for tool in tools:
            self.register_tool(tool)
            
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Fetch a specific tool by name."""
        return self.registry.get(tool_name)
        
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists."""
        return tool_name in self.registry
    

    def search_tools(self, query: str, top_k: int = 5) -> List[Callable]:
        """
        Discover tools based on a query. 
        Tries to match keywords. If nothing matches, returns top_k generic tools.
        """
        query_words = set(query.lower().split())
        scored_tools = []
        
        for name, tool_obj in self.registry.items():
            description = tool_obj.description.lower() if hasattr(tool_obj, 'description') else ""
            score = 0
            name_lower = name.lower()
            
            # Simple scoring based on word matching
            for word in query_words:
                if len(word) > 2:
                    if word in name_lower:
                        score += 2
                    if word in description:
                        score += 1
                        
            scored_tools.append((score, tool_obj))
            
        # Sort by score descending
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        
        # If no score > 0, just return the first few tools as fallback
        if scored_tools and scored_tools[0][0] == 0:
            return [t[1] for t in scored_tools[:top_k]]
            
        # Filter out zero scores if we have matches, and take top_k
        results = [t[1] for t in scored_tools if t[0] > 0][:top_k]
        return results

    def get_discovery_tool(self) -> Callable:
        """
        Returns a MASAI Tool that the LLM can use to search the registry.
        The output is formatted exactly like the AnswerModel Pydantic descriptions.
        Cached after first creation.
        """
        if self._discovery_tool_cache is not None:
            return self._discovery_tool_cache

        # Local import to avoid circular dependency if needed, but it's okay since Tool is in same module
        from .Tool import tool
        registry_ref = self
        
        @tool("discover_tools")
        def discover_tools(query: str) -> str:
            """Searches the tool registry for tools matching the query and returns their schemas."""
            results = registry_ref.search_tools(query)
            
            if not results:
                return "No tools found in registry."
                
            tool_descriptions = [(t.name, t.args_schema.model_json_schema().get('description', '')) for t in results if hasattr(t, 'args_schema')]
            tool_inputs = [(t.name, t.args_schema.model_json_schema().get('properties', {})) for t in results if hasattr(t, 'args_schema')]
            
            output_lines = ["AVAILABLE TOOLS:\n"]
            for (name, desc), (_, props) in zip(tool_descriptions, tool_inputs):
                output_lines.append(f"- Tool Name: {name}\n  Description: {desc}\n  Input Schema: {props}\n\n")
            return "\n".join(output_lines)
        
        self._discovery_tool_cache = discover_tools
        return discover_tools
