
from typing import List, Tuple, Type, Union, Literal, Dict, Optional
from pydantic import BaseModel, Field

def answermodel(tool_names: List[str], tools) -> Type[BaseModel]:
        """Define the AnswerFormat model dynamically based on tools."""
        class AnswerFormat(BaseModel):

            reasoning: str = Field(..., description="LOGICAL REASONING AND CONTEXT AANLYSIS THAT JUSTIFIES YOUR ANSWER.")
            answer: Optional[str] = Field(None, description="GENERATE FINAL ANSWER IN THIS FIELD.")
            satisfied: bool = Field(..., description="SET TO: True to return final answer and (tool,tool_input=None). SET TO: False for further work on task (reflection, tool usage, etc).")
            tool: Optional[str] = Field(
                None,
                description=f"""SELECT TOOLS FROM: {[(tool.name, tool.args_schema.model_json_schema()['description']) for tool in tools]}.
                            \n\nTo use tool, set satisfied=False and specify the tool name, tool_input.
                            TO RETURN answer field, set satisfied=True and tool,tool_input=None.
                            ON RETURNING answer, YOUR RESPONSE GOES TO USER."""
            )
            tool_input: Optional[str] = Field(
                None,
                description=f"""ALWAYS PROVIDE TOOL INPUT AS VALID JSON string. TOOL INPUT SCHEMA: {[(tool.name, tool.args_schema.model_json_schema()['properties']) for tool in tools]}.
                            \n\nSET TO: None if no tool is needed.
                            If optional parameters are NOT NEEDED in tool input, for a task, refrain from using them in tool input."""
            )

            delegate_to_agent: Optional[str] = Field(
                None,
                description="If task delegation is needed, specify one target agent's name. SET: satisfied=True and tool and tool_input to None when delegating. Defaults to None."
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