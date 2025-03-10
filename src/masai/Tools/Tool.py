import inspect
import json
from typing import Any, Callable, Union, Dict, Optional, get_origin, get_args, List, Dict as TypingDict
from pydantic import BaseModel, create_model, ValidationError


class Tool:
    """Base class for tools, holding metadata and execution logic."""
    def __init__(self, name: str, description: str, func: Callable, return_direct: bool = False):
        self.name = name
        self.description = description
        self.func = func
        self.is_async = inspect.iscoroutinefunction(func)
        self.return_direct = return_direct

        # Generate args_schema as a BaseModel from function signature
        sig = inspect.signature(func)
        fields = {}
        for param_name, param in sig.parameters.items():
            # Use Any if no type annotation is provided
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            if param.default != inspect.Parameter.empty:
                # Parameter has a default value, so it's optional
                fields[param_name] = (param_type, param.default)
            else:
                # Parameter is required
                fields[param_name] = (param_type, ...)

        # Create args_schema as a BaseModel subclass
        self.args_schema = create_model(
            f"{name.title()}Schema",
            __doc__=description,  # Set docstring as description
            **fields
        )

    def _get_type_name(self, type_: Any) -> str:
        """Extract the original Python type name from a type annotation."""
        if type_ is Any:
            return "any"
        origin = get_origin(type_) or type_
        args = get_args(type_)

        if origin is Union and type(None) in args:
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return self._get_type_name(non_none_types[0])
            return "|".join(self._get_type_name(t) for t in non_none_types)

        if hasattr(origin, "__name__"):
            base_name = origin.__name__
            if args:
                arg_names = [self._get_type_name(arg) for arg in args]
                return f"{base_name}[{', '.join(arg_names)}]"
            return base_name
        return "str"  # Default fallback

    def __call__(self, *args, **kwargs):
        raise TypeError(f"Tool '{self.name}' cannot be called directly. Use .invoke(), .run(), or .arun() instead.")

    def invoke(self, input: Union[str, dict]) -> Any:
        """Invoke the tool with either a JSON string or a dict input."""
        if self.is_async:
            raise ValueError(f"Tool '{self.name}' is asynchronous; use an async context with await tool.ainvoke()")
        try:
            if isinstance(input, str):
                args_dict = json.loads(input)
            else:
                args_dict = input
            args = self.args_schema(**args_dict)
            result = self.func(**args.model_dump())
            return result if self.return_direct else str(result)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Error invoking tool '{self.name}': invalid input - {str(e)}")

    async def ainvoke(self, input: Union[str, dict]) -> Any:
        """Asynchronously invoke the tool with either a JSON string or a dict input."""
        if not self.is_async:
            raise ValueError(f"Tool '{self.name}' is synchronous; use invoke() instead")
        try:
            if isinstance(input, str):
                args_dict = json.loads(input)
            else:
                args_dict = input
            args = self.args_schema(**args_dict)
            result = await self.func(**args.model_dump())
            return result if self.return_direct else str(result)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Error invoking tool '{self.name}': invalid input - {str(e)}")

    def run(self, input_str: str) -> str:
        """Execute the tool synchronously (string input/output only)."""
        if self.is_async:
            raise ValueError(f"Tool '{self.name}' is asynchronous; use arun() instead")
        return str(self.invoke(input_str))

    async def arun(self, input_str: str) -> str:
        """Execute the tool asynchronously (string input/output only)."""
        if not self.is_async:
            raise ValueError(f"Tool '{self.name}' is synchronous; use run() instead")
        return str(await self.ainvoke(input_str))

    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata for agent systems."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema  # Return the BaseModel class
        }


def tool(name: str, return_direct: bool = False) -> Callable:
    """
    Decorator to create a tool, extracting args_schema and description from the function.

    Args:
        name (str): The unique name of the tool.
        return_direct (bool): If True, returns the raw output; if False, converts to string.

    Returns:
        Callable: A decorator that wraps the function in a Tool instance.
    """
    def decorator(func: Callable) -> Tool:
        description = func.__doc__.strip() if func.__doc__ else "No description provided"
        tool_instance = type(
            name.title(),
            (Tool,),
            {}
        )(name, description, func, return_direct)
        return tool_instance
    return decorator


# Example usage
# if __name__ == "__main__":
#     from typing import Optional, List, Dict
#     import asyncio

#     @tool("greet")
#     def greet(name) -> str:
#         """Greets a user by name."""
#         return f"Hello, {name}!"

#     @tool("add")
#     def add(x: int, y: Optional[int] = None) -> int:
#         """Adds two numbers, with y optional."""
#         return x + (y if y is not None else 0)

#     @tool("process", return_direct=True)
#     async def process(items: List[str], config: Optional[Dict[str, int]] = None) -> List[str]:
#         """Processes a list of items with an optional config."""
#         await asyncio.sleep(1)
#         return items if config is None else [f"{item}-{config.get('value', 0)}" for item in items]

#     tools = [greet, add, process]

#     # Test tools
#     print(greet.invoke('{"name": "Alice"}'))  # Output: "Hello, Alice!"
#     print(greet.args_schema.model_json_schema()['description'])  # Output: "Greets a user by name."
#     print(greet.args_schema.model_json_schema()['properties'])  # Output: {'name': {'title': 'Name'}}

#     print(add.invoke('{"x": 3}'))  # Output: "3"
#     print(add.args_schema.model_json_schema()['properties'])  # Output: {'x': {'title': 'X', 'type': 'integer'}, 'y': {'title': 'Y', 'type': 'integer'}}

#     async def test_async():
#         result = await process.ainvoke('{"items": ["a", "b"]}')
#         print(result)  # Output: ['a', 'b']
#         print(process.args_schema.model_json_schema()['properties'])  # Output: {'items': {...}, 'config': {...}}

#     asyncio.run(test_async())

#     # Example usage in AnswerModel-like context
#     print([(tool.name, tool.args_schema.model_json_schema()['description']) for tool in tools])
#     # Output: [('greet', 'Greets a user by name.'), ('add', 'Adds two numbers...'), ('process', 'Processes a list...')]