"""
Compiled State Graph Implementation

This module implements the CompiledStateGraph class which executes
the workflows defined by StateGraph after compilation.
"""

import asyncio
import copy
import json
import os
from typing import Dict, Any, Callable, Union, List, Optional, AsyncGenerator, TypedDict, Tuple
# Define constants here to avoid circular imports
END = "__end__"
START = "__start__"


class GraphVisualization:
    """
    Graph visualization class that provides methods for drawing and exporting graphs.
    """

    def __init__(self, nodes: Dict[str, Callable], edges: Dict[str, str],
                 conditional_edges: Dict[str, Dict[str, Any]], entry_point: str):
        """
        Initialize graph visualization.

        Args:
            nodes: Dictionary of node names to functions
            edges: Dictionary of direct edges
            conditional_edges: Dictionary of conditional edge configurations
            entry_point: Starting node name
        """
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point

    def draw_mermaid_png(self) -> bytes:
        """
        Generate a Mermaid diagram PNG representation of the graph.

        Returns:
            bytes: PNG image data
        """
        # Generate Mermaid diagram text
        mermaid_text = self._generate_mermaid_diagram()

        # For now, return a simple placeholder PNG
        # In a full implementation, you would use a library like mermaid-cli
        # or integrate with a service to convert Mermaid text to PNG

        # Create a simple text-based representation as bytes
        diagram_text = f"Graph Diagram:\n{mermaid_text}"
        return diagram_text.encode('utf-8')

    def _generate_mermaid_diagram(self) -> str:
        """
        Generate Mermaid diagram text representation.

        Returns:
            str: Mermaid diagram text
        """
        lines = ["graph TD"]

        # Add entry point
        lines.append(f"    START --> {self.entry_point}")

        # Add direct edges
        for from_node, to_node in self.edges.items():
            if to_node == END:
                lines.append(f"    {from_node} --> END")
            else:
                lines.append(f"    {from_node} --> {to_node}")

        # Add conditional edges
        for from_node, config in self.conditional_edges.items():
            condition_func = config.get('condition_func')
            mapping = config.get('condition_map', {})

            for condition_result, target_node in mapping.items():
                if target_node == END:
                    lines.append(f"    {from_node} -->|{condition_result}| END")
                else:
                    lines.append(f"    {from_node} -->|{condition_result}| {target_node}")

        # Add node styling
        for node_name in self.nodes.keys():
            lines.append(f"    {node_name}[{node_name}]")

        lines.append("    START([START])")
        lines.append("    END([END])")

        return "\n".join(lines)

    def get_mermaid_text(self) -> str:
        """
        Get the Mermaid diagram as text.

        Returns:
            str: Mermaid diagram text
        """
        return self._generate_mermaid_diagram()

    def save_mermaid_text(self, filepath: str) -> None:
        """
        Save the Mermaid diagram text to a file.

        Args:
            filepath: Path to save the diagram text
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._generate_mermaid_diagram())


class CompiledStateGraph:
    """
    A compiled and executable state graph.
    
    This class takes the graph structure from StateGraph and provides
    execution methods (ainvoke, astream) to run the workflow.
    """
    
    def __init__(
        self,
        nodes: Dict[str, Callable],
        edges: Dict[str, str],
        conditional_edges: Dict[str, Dict[str, Any]],
        entry_point: str,
        state_schema: type
    ):
        """
        Initialize a compiled state graph.
        
        Args:
            nodes: Dictionary of node names to functions
            edges: Dictionary of direct edges
            conditional_edges: Dictionary of conditional edge configurations
            entry_point: Starting node name
            state_schema: TypedDict class defining state structure
        """
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.state_schema = state_schema
        self._execution_count = 0
        self._graph_visualization = None
    
    async def ainvoke(self, initial_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the graph asynchronously and return the final state.
        
        Args:
            initial_state: Initial state dictionary
            config: Optional configuration (e.g., recursion_limit)
            
        Returns:
            Final state dictionary after execution
        """
        config = config or {}
        recursion_limit = config.get('recursion_limit', 100)
        
        # Create a deep copy of the initial state to avoid mutations
        current_state = copy.deepcopy(initial_state)
        current_node = self.entry_point
        execution_steps = 0
        
        while current_node != END and execution_steps < recursion_limit:
            execution_steps += 1
            
            # Execute current node
            if current_node in self.nodes:
                node_func = self.nodes[current_node]
                
                # Check if the function is async
                if asyncio.iscoroutinefunction(node_func):
                    current_state = await node_func(current_state)
                else:
                    current_state = node_func(current_state)
                
                # Ensure state is a dictionary
                if not isinstance(current_state, dict):
                    raise ValueError(f"Node '{current_node}' must return a dictionary state")
            
            # Determine next node
            next_node = await self._get_next_node(current_node, current_state)
            current_node = next_node
        
        if execution_steps >= recursion_limit:
            raise RuntimeError(f"Graph execution exceeded recursion limit of {recursion_limit}")
        
        self._execution_count += 1
        return current_state
    
    async def astream(
        self,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        stream_mode: Optional[Union[str, List[str]]] = None
    ) -> AsyncGenerator[Union[Dict[str, Any], Tuple[str, Dict[str, Any]]], None]:
        """
        Execute the graph asynchronously and yield state updates.
        
        Args:
            initial_state: Initial state dictionary
            config: Optional configuration
            stream_mode: List of stream modes (e.g., ['updates'])
            
        Yields:
            State updates during execution
        """
        # Handle stream_mode parameter
        if stream_mode is None:
            stream_mode = ['updates']
        elif isinstance(stream_mode, str):
            stream_mode = [stream_mode]

        # Apply recursion limit if provided
        recursion_limit = 100  # Default
        if config and 'recursion_limit' in config:
            recursion_limit = config['recursion_limit']

        current_state = initial_state.copy()
        current_node = self.entry_point
        steps = 0

        while current_node != END and steps < recursion_limit:
            steps += 1

            # Update current node in state
            current_state['current_node'] = current_node

            # Execute the current node
            if current_node in self.nodes:
                node_func = self.nodes[current_node]
                try:
                    if asyncio.iscoroutinefunction(node_func):
                        result = await node_func(current_state)
                    else:
                        result = node_func(current_state)

                    if result:
                        # Update state with node result
                        current_state.update(result)

                        # Yield based on stream_mode - match original LangGraph format
                        for mode in stream_mode:
                            if mode == 'updates':
                                update_data = {current_node: result}
                            elif mode == 'values':
                                update_data = current_state.copy()
                            elif mode == 'debug':
                                update_data = {
                                    'node': current_node,
                                    'state': current_state.copy(),
                                    'update': result
                                }
                            else:
                                # For unsupported modes, just return the update
                                update_data = {current_node: result}

                            # Return tuple format: (mode, update_data) to match original LangGraph
                            if len(stream_mode) > 1:
                                yield (mode, update_data)
                            else:
                                # For single mode, return tuple (mode, data) to match test.py expectations
                                yield (mode, update_data)

                except Exception as e:
                    print(f"Error in node {current_node}: {e}")
                    break

            # Determine next node
            next_node = await self._get_next_node(current_node, current_state)
            current_state['previous_node'] = current_node
            current_node = next_node

            if current_node is None:
                break

        # Final state update
        current_state['current_node'] = END
        self._execution_count += 1
    
    async def _get_next_node(self, current_node: str, state: Dict[str, Any]) -> str:
        """
        Determine the next node based on edges and conditions.

        Args:
            current_node: Current node name
            state: Current state

        Returns:
            Next node name or END
        """
        # Check conditional edges first
        if current_node in self.conditional_edges:
            condition_config = self.conditional_edges[current_node]
            condition_func = condition_config['condition_func']
            condition_map = condition_config['condition_map']

            # Execute condition function (support both sync and async)
            if asyncio.iscoroutinefunction(condition_func):
                condition_result = await condition_func(state)
            else:
                condition_result = condition_func(state)

            # Map condition result to next node
            if condition_result in condition_map:
                return condition_map[condition_result]
            else:
                raise ValueError(f"Condition result '{condition_result}' not found in condition map for node '{current_node}'")

        # Check direct edges
        if current_node in self.edges:
            return self.edges[current_node]

        # Default to END if no edges defined
        return END
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for this compiled graph.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            'execution_count': self._execution_count,
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'conditional_edge_count': len(self.conditional_edges),
            'entry_point': self.entry_point
        }

    def get_graph(self) -> GraphVisualization:
        """
        Get a graph visualization object for drawing and exporting the graph.

        Returns:
            GraphVisualization: Object with methods for graph visualization
        """
        if not self._graph_visualization:
            self._graph_visualization = GraphVisualization(
                nodes=self.nodes,
                edges=self.edges,
                conditional_edges=self.conditional_edges,
                entry_point=self.entry_point
            )
        return self._graph_visualization
