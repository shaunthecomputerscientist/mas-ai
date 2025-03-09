
from typing import List, Tuple, Type, Union, Literal, Dict, Optional
from pydantic import BaseModel, Field

def answermodel(tool_names: List[str], tools) -> Type[BaseModel]:
        """Define the AnswerFormat model dynamically based on tools."""
        class AnswerFormat(BaseModel):
            answer: Optional[Union[str,List[str],Dict]] = Field(..., description="Your final generated answer which will be seen by human.")
            satisfied: bool = Field(..., description="Set to True to return final answer and (tool,tool_input=None). Set to False for further work on task (reflection, tool usage, etc).")
            tool: Optional[str] = Field(
                None,
                description=f"""Select tools among: {[(tool.name, tool.args_schema.model_json_schema()['description']) for tool in tools]}.
                            To use tool, set satisfied=False and specify the tool name, tool_input.
                            To return answer field, set satisfied=True and tool,tool_input=None. 
                            On returning answer, you can not work on current problem anymore."""
            )
            tool_input: Optional[Union[Dict, str]] = Field(
                None,
                description=f"""Always provide tool input as valid JSON. TOOL INPUT SCHEMA: {[(tool.name, tool.args_schema.model_json_schema()['properties']) for tool in tools]}.
                            Set to None if no tool is needed.
                            If optional parameters are not needed in tool input, for a task, refrain from using them."""
            )
            reasoning: str = Field(..., description="Logical reasoning and context analysis that justifies your answer.")
            delegate_to_agent: Optional[str] = Field(
                None,
                description="If task delegation is needed, specify one target agent's name. Set satisfied=True and tool and tool_input to None when delegating. Defaults to None."
            )

            @staticmethod
            def validate_tool_name(value: str):
                if value not in tool_names:
                    raise ValueError(f"Invalid tool name: {value}. Must be one of {tool_names}.")
                return value

            @staticmethod
            def validate_tool_input(tool: str, tool_input: Union[Dict, str]):
                if tool != 'None' and not isinstance(tool_input, dict):
                    raise ValueError(f"Tool input must be a dictionary when tool is {tool}.")
                return tool_input

        return AnswerFormat