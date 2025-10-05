"""
Core Graph Implementation for Custom LangGraph

This module implements the StateGraph class and related constants
that are used throughout the MASAI framework for building agent workflows.
"""

import asyncio
from typing import Dict, Any, Callable, Union, List, Optional, TypedDict

# Special node constants
END = "__end__"
START = "__start__"


class StateGraph:
    """
    A graph-based workflow builder for state machines.
    
    This class allows building complex workflows by adding nodes and edges,
    then compiling them into an executable graph.
    """
    
    def __init__(self, state_schema: type):
        """
        Initialize a new StateGraph.
        
        Args:
            state_schema: The TypedDict class defining the state structure
        """
        self.state_schema = state_schema
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Dict[str, Any]] = {}
        self.entry_point: Optional[str] = None
        self._compiled = False
    
    def add_node(self, name: str, func: Callable) -> None:
        """
        Add a node to the graph.
        
        Args:
            name: Unique name for the node
            func: Function to execute for this node (should accept and return state)
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists in the graph")
        
        self.nodes[name] = func
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Add a direct edge between two nodes.
        
        Args:
            from_node: Source node name
            to_node: Destination node name (can be END)
        """
        if from_node not in self.nodes and from_node != START:
            raise ValueError(f"Source node '{from_node}' does not exist")
        
        if to_node not in self.nodes and to_node != END:
            raise ValueError(f"Destination node '{to_node}' does not exist")
        
        self.edges[from_node] = to_node
    
    def add_conditional_edges(
        self, 
        from_node: str, 
        condition_func: Callable,
        condition_map: Dict[str, str]
    ) -> None:
        """
        Add conditional edges from a node based on a condition function.
        
        Args:
            from_node: Source node name
            condition_func: Function that takes state and returns a condition key
            condition_map: Mapping from condition keys to destination nodes
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' does not exist")
        
        # Validate all destination nodes exist
        for condition_key, dest_node in condition_map.items():
            if dest_node not in self.nodes and dest_node != END:
                raise ValueError(f"Destination node '{dest_node}' does not exist")
        
        self.conditional_edges[from_node] = {
            'condition_func': condition_func,
            'condition_map': condition_map
        }
    
    def set_entry_point(self, node_name: str) -> None:
        """
        Set the entry point for the graph.
        
        Args:
            node_name: Name of the node to start execution from
        """
        if node_name not in self.nodes:
            raise ValueError(f"Entry point node '{node_name}' does not exist")
        
        self.entry_point = node_name
    
    def compile(self):
        """
        Compile the graph into an executable form.

        Returns:
            CompiledStateGraph: Executable graph instance
        """
        if not self.entry_point:
            raise ValueError("Entry point must be set before compiling")

        if not self.nodes:
            raise ValueError("Graph must have at least one node before compiling")

        # Import here to avoid circular imports
        from .state import CompiledStateGraph

        # Create and return compiled graph
        compiled_graph = CompiledStateGraph(
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            conditional_edges=self.conditional_edges.copy(),
            entry_point=self.entry_point,
            state_schema=self.state_schema
        )
        
        self._compiled = True
        return compiled_graph
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get information about the current graph structure.
        
        Returns:
            Dict containing graph structure information
        """
        return {
            'nodes': list(self.nodes.keys()),
            'edges': self.edges.copy(),
            'conditional_edges': {
                node: {
                    'condition_map': data['condition_map']
                }
                for node, data in self.conditional_edges.items()
            },
            'entry_point': self.entry_point,
            'compiled': self._compiled
        }
