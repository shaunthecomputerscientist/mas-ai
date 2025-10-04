import asyncio
from datetime import datetime
from typing import Optional, Dict, AsyncGenerator, List, Set

class PromptTemplate:
    """A custom prompt template class to replace LangChain's ChatPromptTemplate."""
    def __init__(
        self,
        system_template: str = "",
        human_template: str = "",
        input_variables: Optional[List[str]] = None
    ):
        """
        Initialize the prompt template.

        Args:
            system_template (str): The system message template.
            human_template (str): The human message template.
            input_variables (List[str]): List of variable names expected in the templates.
        """
        self.system_template = system_template
        self.human_template = human_template
        self.input_variables = set(input_variables) if input_variables else set()

        # Validate templates for missing variables
        self._validate_template(self.system_template)
        self._validate_template(self.human_template)

    def _validate_template(self, template: str):
        """Check if all variables in the template are in input_variables."""
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        for var in placeholders:
            if var not in self.input_variables:
                self.input_variables.add(var)

    def format(self, **kwargs) -> str:
        """
        Format the prompt with provided variables.

        Args:
            **kwargs: Dictionary of variable names and their values.

        Returns:
            str: Formatted prompt combining system and human messages.
        """
        # Provide default empty strings for missing variables
        formatted_inputs = {var: kwargs.get(var, "") for var in self.input_variables}
        
        # Format system and human templates
        system_message = self.system_template.format(**formatted_inputs) if self.system_template else ""
        human_message = self.human_template.format(**formatted_inputs) if self.human_template else ""
        
        # Combine messages, ensuring proper separation
        messages = [msg for msg in [system_message, human_message] if msg]
        return "\n\n".join(messages)