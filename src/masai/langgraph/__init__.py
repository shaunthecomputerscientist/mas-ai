"""
Custom LangGraph Implementation for MASAI Framework

This module provides a custom implementation of LangGraph components
to replace the external langgraph dependency while maintaining
full compatibility with the MASAI framework.

Key Components:
- StateGraph: Core graph building class
- CompiledStateGraph: Compiled and executable graph
- END, START: Special node constants
- State management and execution utilities
"""

from .core import StateGraph, END, START
from .state import CompiledStateGraph

__all__ = [
    "StateGraph",
    "CompiledStateGraph",
    "END",
    "START"
]

__version__ = "1.0.0"
__author__ = "MASAI Framework"
__description__ = "Custom LangGraph implementation for MASAI multi-agent systems"
