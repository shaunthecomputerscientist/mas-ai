import os
import warnings
# MASAI Vanilla SDK Wrappers - Drop-in replacements for LangChain
from ..vanilla_wrappers import ChatOpenAI, ChatGoogleGenerativeAI

# Import parameter configuration and mapping functions
from ..parameter_config import (
    extract_gemini_params,
    extract_openai_params,
    validate_parameters,
)


from pydantic import BaseModel
from typing import Dict, List, Literal, Type, AsyncGenerator, Union
from typing import Optional
from ...prompts import ChatPromptTemplate
from datetime import datetime, timezone
from ...prompts.Template.template import PromptTemplate
from ...Tools.logging_setup.logger import setup_logger
import logging

# Suppress LangChain parameter validation warnings for reasoning models
# These are false positives when using reasoning_effort as a direct parameter (which is correct)
# The warning appears because LangChain's internal validation checks parameter passing,
# but our implementation already passes reasoning_effort correctly as a direct parameter
warnings.filterwarnings(
    'ignore',
    message="Parameters .* should be specified explicitly",
    category=UserWarning,
    module='langchain_openai'
)

# Use shared logger instead of separate instance
logger = setup_logger()


class BaseGenerativeModel:
    def __init__(
        self,
        model_name: str,
        category: str,
        temperature: float,
        memory: bool = True,
        memory_order: int = 5,
        prompt_template: Optional[Union[PromptTemplate, ChatPromptTemplate]] = None,
        info: Optional[dict] = None,
        system_prompt: Optional[str]= None,
        input_variables: Optional[List[str]] = None,
        logging: Optional[bool] = False,
        **kwargs: Optional[dict]
    ):
        """Initialize the BaseGenerativeModel.

        Args:
            model_name (str): Name of the model to use.
            category (str): Category of the model to use one of the following: gemini, openai
            temperature (float): Temperature for the model
            memory (bool, optional): Whether to use memory. Defaults to True.
            memory_order (int, optional): Number of messages to keep in memory. Defaults to 5.
            prompt_template (Optional[ChatPromptTemplate], optional): Prompt template for the model. Defaults to None.
            info (Optional[dict], optional): Information to the model. Defaults to None.
            system_prompt(Optional[str]): Bypass Chatprompt Template by simply providing a system prompt. Useful when using astream_response for simple usecases where defining complicated chatpromptemplate is not necessary.
            input_variables (Optional[List[str]], optional): Input variables for the prompt template. Defaults to None.
            logging (Optional[bool], optional): Whether to log the responses. Defaults to False.
            **kwargs: Additional arguments for the model.

        Add api keys to the environment variables as follows for the categories we support:
        OPENAI_API_KEY
        GOOGLE_API_KEY

        Note: Support for Anthropic, Groq, HuggingFace, and Ollama is commented out but can be extended later.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.memory = memory
        self.memory_order = memory_order
        self.prompt = prompt_template
        self.info = info
        self.chat_history = []
        self.category = category
        if logging:
            self.logger = logger
        else: self.logger = None
        # Set verbose before initializing LLM (needed in _get_llm)
        self.verbose = kwargs.get('verbose', False) if kwargs else False
        # Store kwargs for passing to wrapper classes
        self.kwargs = kwargs

        # Register embedding_model if provided (for persistent memory)
        self.embedding_model = kwargs.get('embedding_model') if kwargs else None

        # Initialize the LLM
        self.model = self._get_llm()

        if system_prompt:
            self.prompt = self.prompt_formatter(system_prompt=system_prompt, input_variables=input_variables)

    def _get_llm(self):
            """
            Initialize the LLM with proper parameter filtering and mapping.

            This method:
            1. Validates parameters for the selected provider
            2. Extracts and maps parameters using standardized names
            3. Filters out incompatible parameters (prevents OpenAI 400 errors)
            4. Auto-configures reasoning/thinking parameters if not explicitly set

            Returns:
                Initialized LLM instance (ChatOpenAI or ChatGoogleGenerativeAI)
            """
            try:
                # Validate parameters and warn about potential issues
                if self.kwargs:
                    validate_parameters(self.kwargs, self.category)

                if "gemini" in self.category:
                    # Extract and map parameters for Gemini
                    # This automatically:
                    # - Adds shared parameters (temperature, top_p, etc.)
                    # - Maps standardized names to Gemini names (e.g., stop_sequences → stop_sequences)
                    # - Adds Gemini-specific parameters (removes gemini_ prefix)
                    # - Filters out OpenAI-only parameters
                    extracted_params = extract_gemini_params(self.kwargs or {}, verbose=self.verbose)

                    # Build final parameters for Gemini
                    gemini_params = {
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "api_key": os.environ.get('GOOGLE_API_KEY'),
                        "verbose": self.verbose,
                        **extracted_params  # Add all extracted parameters
                    }

                    # Auto-map temperature to thinking_budget if not explicitly set
                    # Only for thinking models (Gemini 2.5 series)
                    if 'thinking_budget' not in gemini_params and 'gemini_thinking_budget' not in (self.kwargs or {}):
                        is_thinking_model = (
                            self.model_name.startswith('gemini-2.5') or
                            'thinking' in self.model_name.lower()
                        )
                        if is_thinking_model:
                            # Use dynamic thinking budget (-1) for all temperatures
                            # This lets Gemini decide the optimal thinking budget
                            gemini_params['thinking_budget'] = -1
                            if self.verbose and self.logger:
                                self.logger.info(f"✅ Thinking model '{self.model_name}' initialized with thinking_budget=-1 (dynamic)")

                    # print(gemini_params)
                    llm = ChatGoogleGenerativeAI(**gemini_params)

                elif "openai" in self.category:
                    # Extract and map parameters for OpenAI
                    # This automatically:
                    # - Adds shared parameters (temperature, top_p, etc.)
                    # - Maps standardized names to OpenAI names (e.g., max_output_tokens → max_tokens)
                    # - Adds OpenAI-specific parameters (removes openai_ prefix)
                    # - FILTERS OUT Gemini-only parameters (CRITICAL: prevents 400 errors)
                    extracted_params = extract_openai_params(self.kwargs or {}, verbose=self.verbose)

                    # Build final parameters for OpenAI
                    openai_params = {
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "api_key": os.environ.get('OPENAI_API_KEY'),
                        "verbose": self.verbose,
                        **extracted_params  # Add all extracted parameters
                    }

                    # Auto-map temperature to reasoning_effort if not explicitly set
                    # Only for reasoning models (GPT-5, o-series, GPT-4.1)
                    if 'reasoning_effort' not in openai_params and 'openai_reasoning_effort' not in (self.kwargs or {}):
                        is_reasoning_model = (
                            self.model_name.startswith('gpt-5') or
                            self.model_name.startswith('o1') or
                            self.model_name.startswith('o3') or
                            self.model_name.startswith('o4') or
                            self.model_name.startswith('gpt-4.1')
                        )
                        if is_reasoning_model:
                            # Map temperature to reasoning_effort
                            if self.temperature <= 0.3:
                                openai_params['reasoning_effort'] = "low"
                            elif self.temperature <= 0.7:
                                openai_params['reasoning_effort'] = "medium"
                            else:
                                openai_params['reasoning_effort'] = "high"

                            # Log the reasoning effort being used
                            if self.verbose and self.logger:
                                self.logger.info(f"✅ Reasoning model '{self.model_name}' initialized with reasoning_effort='{openai_params['reasoning_effort']}' (mapped from temperature={self.temperature})")

                    llm = ChatOpenAI(**openai_params)

                else:
                    raise ValueError(f"Unsupported category: {self.category}")
                
                # print(llm)
                return llm

            except Exception as e:
                raise e
            
    def generate_response(self, prompt: str, output_structure: Optional[Type[BaseModel]] = None, custom_inputs: Optional[dict] = None):
        """
        Generates a response, optionally with a structured output.
        
        Args:
            prompt: The input prompt.
            output_structure: Optional Pydantic model for structured output.
            custom_inputs: Optional dictionary of custom template variables corresponding to the prompt template.
            
            Mandatory variables that should always be present:
            question: The input prompt.
            useful_info: Useful info for the model, default is None.
            current_time: The current time. Added for additional context.
            history: The chat history. If memory is True, the chat history is automatically added to the prompt.
            schema: The schema of the output structure.
            These variables are automatically added to the prompt.
        """
        if self.memory:
            self.chat_history.append({'role': 'user', 'content': prompt})

        if output_structure is None:
            # Unstructured output
            if self.memory:
                messages = self.chat_history[-self.memory_order:] if len(self.chat_history) > self.memory_order else self.chat_history
                response = self.model.invoke(messages).content
            else:
                response = self.model.invoke(prompt).content
            
            self.chat_history.append({'role': 'assistant', 'content': response})
            return response

        # Structured output
        # Use json_mode for OpenAI and Gemini for better reliability and compatibility
        structured_llm = self._return_structured_model(prompt, output_structure)
        
        # Mandatory variables that should always be present
        mandatory_vars = ['question', 'useful_info', 'current_time', 'history', 'schema']

        # Create base inputs with fallback values
        base_inputs = {
            'question': prompt,
            'schema': output_structure.model_json_schema() if output_structure else 'None',
            'current_time': datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            'history': self.chat_history[-self.memory_order:] if self.memory else [],
            'useful_info': str(self.info.items()) if self.info else 'None'
        }

        # Determine which mandatory vars are in the template
        template_vars = set(self.prompt.input_variables)
        present_in_template = {var for var in mandatory_vars if var in template_vars}
        missing_from_template = set(mandatory_vars) - present_in_template

        # Start building the prompt components
        prompt_parts = []

        # 1. Add mandatory vars NOT in template as raw strings
        for var in missing_from_template:
            prompt_parts.append(f"{var}: {base_inputs[var]}")

        # 2. Create template inputs from base + custom inputs that are in template
        template_inputs = {
            k: v for k, v in {**base_inputs, **(custom_inputs or {})}.items()
            if k in template_vars
        }

        # 3. Add formatted template if there are inputs for it
        if template_inputs:
            prompt_parts.append(self.prompt.format(**template_inputs))

        # 4. Add remaining custom inputs not in template or mandatory vars
        if custom_inputs:
            extra_custom = {
                k: v for k, v in custom_inputs.items()
                if k not in template_vars and k not in mandatory_vars
            }
            for k, v in extra_custom.items():
                prompt_parts.append(f"{k}: {v}")

        # Combine all components into final prompt
        full_prompt = "\n".join(prompt_parts)
        
        if self.logger:
            logger.info(full_prompt)

        # Generate response
        response = structured_llm.invoke(full_prompt).model_dump()

        if self.memory:
            self.chat_history.append({'role': 'assistant', 'content': response})
        return response
    
    def _return_structured_model(self, prompt: str, output_structure: Type[BaseModel]):
        # Use json_mode for OpenAI and Gemini for better reliability and compatibility
        if self.category.lower() in ["openai"]:
            structured_llm = self.model.with_structured_output(output_structure, method="json_mode")
        else:
            structured_llm = self.model.with_structured_output(output_structure)
        return structured_llm
    
    async def astream_response(self, prompt: str, custom_inputs: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """
        Streams a response from the model, with memory support but no structured output.
        
        Args:
            prompt: The input prompt.
            custom_inputs: Optional dictionary of custom template variables corresponding to the prompt template.
            
            Mandatory variables that should always be present:
            question: The input prompt.
            useful_info: Useful info for the model, default is None.
            current_time: The current time. Added for additional context.
            history: The chat history. If memory is True, the chat history is automatically added to the prompt.
            These variables are automatically added to the prompt.
        """
        if self.memory:
            self.chat_history.append({'role': 'user', 'content': prompt})

        # Mandatory variables
        mandatory_vars = ['question', 'useful_info', 'current_time', 'history']

        # Create base inputs with fallback values
        base_inputs = {
            'question': prompt,
            'current_time': datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            'history': self.chat_history[-self.memory_order:] if self.memory else [],
            'useful_info': str(self.info.items()) if self.info else 'None'
        }

        # Determine which mandatory vars are in the template
        template_vars = set(self.prompt.input_variables) if self.prompt else set()
        present_in_template = {var for var in mandatory_vars if var in template_vars}
        missing_from_template = set(mandatory_vars) - present_in_template

        # Build prompt components
        prompt_parts = []

        # Add mandatory vars NOT in template as raw strings
        for var in missing_from_template:
            prompt_parts.append(f"{var}: {base_inputs[var]}")

        # Create template inputs from base + custom inputs that are in template
        template_inputs = {
            k: v for k, v in {**base_inputs, **(custom_inputs or {})}.items()
            if k in template_vars
        }

        # Add formatted template if there are inputs for it
        if template_inputs and self.prompt:
            prompt_parts.append(self.prompt.format(**template_inputs))

        # Add remaining custom inputs not in template or mandatory vars
        if custom_inputs:
            extra_custom = {
                k: v for k, v in custom_inputs.items()
                if k not in template_vars and k not in mandatory_vars
            }
            for k, v in extra_custom.items():
                prompt_parts.append(f"{k}: {v}")

        # Combine all components into final prompt
        full_prompt = "\n".join(prompt_parts)

        
        if self.logger:
            logger.info(full_prompt)

        # Stream response
        response_content = ""
        if self.memory:
            # messages = self.chat_history[-self.memory_order:] if len(self.chat_history) > self.memory_order else self.chat_history
            async for chunk in self.model.astream(full_prompt):
                response_content += chunk.content
                yield chunk.content
        else:
            async for chunk in self.model.astream(full_prompt):
                response_content += chunk.content
                yield chunk.content

        # Update chat history with the complete response
        if self.memory:
            self.chat_history.append({'role': 'assistant', 'content': response_content})
            

            
    def prompt_formatter(
        self,
        system_prompt: str,
        input_variables: list = []
    ) -> PromptTemplate:
        """
        Format prompts into PromptTemplates with dynamic sections based on input variables.

        Args:
            system_prompt (str): Base system prompt.
            input_variables (list): List of input variables to include in the template.

        Returns:
            PromptTemplate: A formatted prompt template with all specified variables.
        """
        # Ensure required variables are included
        mandatory_vars=[]
        # print(input_variables)
        if isinstance(input_variables,list):
            if len(input_variables)<1:
                input_variables = ['useful_info','current_time','history','question']
            elif len(input_variables) >= 1:
                for element in ['useful_info','current_time','history','question']:
                    if element not in input_variables:
                        mandatory_vars.append(element)

        
        # Dynamically build template sections for each input variable
        template_sections = []
        input_variables=mandatory_vars+input_variables
        for var in input_variables:
            # Format the variable name for display (uppercase with spaces)
            display_name = var.replace('_', ' ').upper()
            
            # Create a section for this variable
            template_sections.append(f"<{display_name}>:{{{var}}}</{display_name}>")
        
        # Join all sections with newlines
        base_template = "\n\n".join(template_sections)
        
        # Create PromptTemplate instance
        prompt_template = PromptTemplate(
            system_template="SYSTEM: \n" + system_prompt,
            human_template='HUMAN: \n\n' + base_template,
            input_variables=input_variables
        )

        # print(prompt_template.human_template)
        return prompt_template
    
    def _format_info_for_llm(self,data: dict) -> str:
        """Format a dict into readable LLM-friendly text with proper structure.

        Args:
            data: Dictionary to format (can contain nested dicts, lists, or primitives)

        Returns:
            Formatted string with outer keys clearly visible and inner content preserved
        """
        if not data:
            return "None"

        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Format nested dict with indentation
                nested_lines = []
                for nested_key, nested_value in value.items():
                    nested_lines.append(f"    • {nested_key}: {nested_value}")
                lines.append(f"- {key}:\n" + "\n".join(nested_lines))
            elif isinstance(value, list):
                # Format list items
                list_items = []
                for item in value:
                    if isinstance(item, dict):
                        # Nested dict in list
                        for nested_key, nested_value in item.items():
                            list_items.append(f"    • {nested_key}: {nested_value}")
                    else:
                        list_items.append(f"    • {item}")
                lines.append(f"- {key}:\n" + "\n".join(list_items))
            else:
                # Simple key-value pair
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)