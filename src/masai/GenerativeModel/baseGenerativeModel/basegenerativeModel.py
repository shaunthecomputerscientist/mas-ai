import os
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.anthropic import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from pydantic import BaseModel
from typing import Dict, List, Literal, Type, AsyncGenerator, Union
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timezone
from ...prompts.Template.template import PromptTemplate
from ...Tools.logging_setup.logger import setup_logger
import logging

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
        logging: Optional[bool] = False
    ):
        """Initialize the BaseGenerativeModel.

        Args:
            model_name (str): Name of the model to use.
            category (str): Category of the model to use one of the following: gemini, huggingface, openai, anthropic, ollama, groq
            temperature (float): Temperature for the model
            memory (bool, optional): Whether to use memory. Defaults to True.
            memory_order (int, optional): Number of messages to keep in memory. Defaults to 5.
            prompt_template (Optional[ChatPromptTemplate], optional): Prompt template for the model. Defaults to None.
            info (Optional[dict], optional): Information to the model. Defaults to None.
            system_prompt(Optional[str]): Bypass Chatprompt Template by simply providing a system prompt. Useful when using astream_response for simple usecases where defining complicated chatpromptemplate is not necessary.
            input_variables (Optional[List[str]], optional): Input variables for the prompt template. Defaults to None.
        
        Add api keys to the environment variables as follows for the categories we support:
        OPENAI_API_KEY
        HUGGINGFACEHUB_API_TOKEN
        GOOGLE_API_KEY
        ANTHROPIC_API_KEY
        OLLAMA_BASE_URL
        GROQ_API_KEY
        Only for ollama, you need to install ollama and run it locally.
        Set OLLAMA_BASE_URI in env variable.
        Provide api keys only for selected categories.
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
        # Initialize the LLM
        self.model = self._get_llm()
        
        if system_prompt:
            self.prompt = self.prompt_formatter(system_prompt=system_prompt, input_variables=input_variables)

    def _get_llm(self):
            try:
                if "gemini" in self.category:
                    # Gemini 2.5 models have built-in thinking capability
                    # Check if this is a thinking-capable model
                    is_thinking_model = (
                        self.model_name.startswith('gemini-2.5') or
                        'thinking' in self.model_name.lower()         # Experimental thinking models
                    )

                    model_kwargs = {}
                    if is_thinking_model:
                        # Map temperature to thinking budget for reasoning models
                        # Low temp = focused thinking, high temp = exploratory thinking
                        if self.temperature <= 0.3:
                            model_kwargs["thinkingBudget"] = "low"
                        elif self.temperature <= 0.7:
                            model_kwargs["thinkingBudget"] = "medium"
                        else:
                            model_kwargs["thinkingBudget"] = "high"

                    llm = ChatGoogleGenerativeAI(
                        api_key=os.environ.get('GOOGLE_API_KEY'),
                        verbose=True,
                        model=self.model_name,
                        temperature=self.temperature,
                        model_kwargs=model_kwargs if model_kwargs else None
                    )
                elif "huggingface" in self.category:
                    llm = ChatHuggingFace(
                        llm=HuggingFaceEndpoint(repo_id=self.model_name,
                        huggingfacehub_api_token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
                        temperature=self.temperature),


                    )
                elif "openai" in self.category:
                    # Reasoning models don't support temperature parameter
                    # Check if this is a reasoning model (GPT-5, o-series, GPT-4.1)
                    is_reasoning_model = (
                        self.model_name.startswith('gpt-5') or      # GPT-5 series
                        self.model_name.startswith('o1') or         # o1, o1-mini, o1-preview
                        self.model_name.startswith('o3') or         # o3, o3-mini
                        self.model_name.startswith('o4') or         # o4-mini
                        self.model_name.startswith('gpt-4.1')       # GPT-4.1, gpt-4.1-nano
                    )

                    if is_reasoning_model:
                        # Reasoning models use reasoning_effort instead of temperature
                        # Map temperature to reasoning_effort: 0-0.3=low, 0.4-0.7=medium, 0.8+=high
                        if self.temperature <= 0.3:
                            reasoning_effort = "low"
                        elif self.temperature <= 0.7:
                            reasoning_effort = "medium"
                        else:
                            reasoning_effort = "high"

                        # Note: reasoning_effort should be passed via model_kwargs
                        # Some models may not support this parameter yet
                        llm = ChatOpenAI(
                            model=self.model_name,
                            model_kwargs={"reasoning_effort": reasoning_effort},
                            api_key=os.environ.get('OPENAI_API_KEY')
                        )
                    else:
                        # GPT-4o and earlier models support temperature
                        llm = ChatOpenAI(
                            model=self.model_name,
                            temperature=self.temperature,
                            api_key=os.environ.get('OPENAI_API_KEY')
                        )
                elif "antrophic" in self.category or "anthropic" in self.category:
                    # Check if this is a thinking/reasoning model
                    is_thinking_model = (
                        self.model_name.startswith('claude-4') or
                        self.model_name.startswith('claude-3.7')
                    )

                    model_kwargs = {}
                    if is_thinking_model:
                        # Enable extended thinking for Claude 4 and 3.7 models
                        # Map temperature to thinking budget tokens
                        if self.temperature <= 0.3:
                            budget_tokens = 5000  # Focused thinking
                        elif self.temperature <= 0.7:
                            budget_tokens = 10000  # Balanced thinking
                        else:
                            budget_tokens = 20000  # Deep thinking

                        model_kwargs["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": budget_tokens
                        }

                    llm = ChatAnthropic(
                        model_name=self.model_name,
                        temperature=self.temperature,
                        anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY'),
                        model_kwargs=model_kwargs if model_kwargs else None
                    )
                elif "ollama" in self.category:
                    llm = ChatOllama(
                        model=self.model_name,
                        temperature=self.temperature,
                        base_url=os.environ.get(
                            'OLLAMA_BASE_URL', 
                            'http://localhost:11434'  # Default local URL
                        ),
                    )
                elif "groq" in self.category:
                    llm = ChatGroq(model_name=self.model_name,
                                temperature=self.temperature,
                                api_key=os.environ.get('GROQ_API_KEY'))

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