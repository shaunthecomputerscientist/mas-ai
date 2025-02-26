from typing import List, Dict, Any, Literal, TypedDict, Tuple, Union, Type, Optional
from pydantic import BaseModel, Field

def structure_supervisor(agents):
    class SupervisorAgent(BaseModel):
        answer: str = Field(None,description="generate answer here for the use. Fill this when satisfied=<True>,Even when directly returning answers.This field is what the human can see so do not keep it empty.")
        # satisfied: bool = Field(...,description="report if you are satisfied with the agent answer. True when you want to report final answer. set True to return answer.")
        reasoning: str = Field(...,description="report the reasoning behind the agent answer here. Do not write reason in answer.")
        delegate_to_agent: str = Field(None,description=f"Name of agent you want to delegate next. Choose one of {agents.keys()}. Set None to return answer.")
        agent_input: Optional[str] = Field(...,description="Can not be None.Pass question to the agent here.This field is for agent to see your question. Example: user needs this...here is all context, so solve this...explain the requirements")
        return_agent_response_directly: bool = Field(False,description="False by default.set True to directly return agent's generated answer, set False when you want your generated answer to be returned.set True to pass agent's answer directly.")

    return SupervisorAgent