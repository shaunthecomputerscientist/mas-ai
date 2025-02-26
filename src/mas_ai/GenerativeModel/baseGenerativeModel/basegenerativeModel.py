import os
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.anthropic import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from pydantic import BaseModel
from typing import Dict, List, Literal, Type
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timezone

class BaseGenerativeModel:
    def __init__(
        self,
        model_name: str,
        category: str,
        temperature: float,
        memory: bool = True,
        memory_order: int = 5,
        prompt_template: Optional[ChatPromptTemplate] = None,
        info: Optional[dict] = None
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
        
        Add api keys to the environment variables as follows for the categories we support:
        OPENAI_API_KEY
        HUGGINGFACEHUB_API_TOKEN
        GOOGLE_API_KEY
        ANTHROPIC_API_KEY
        OLLAMA_BASE_URL
        GROQ_API_KEY
        Only for ollama, you need to install ollama and run it locally.
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
        
        # Initialize the LLM
        self.model = self._get_llm()
    def _get_llm(self):
            try:
                if "gemini" in self.category:
                    llm = ChatGoogleGenerativeAI(
                        api_key=os.environ.get('GOOGLE_API_KEY'),
                        verbose=True,
                        model=self.model_name,
                        temperature=self.temperature
                    )
                elif "huggingface" in self.category:
                    llm = ChatHuggingFace(
                        llm=HuggingFaceEndpoint(repo_id=self.model_name,
                        huggingfacehub_api_token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
                        temperature=self.temperature),


                    )
                elif "openai" in self.category:
                    llm = ChatOpenAI(
                        model=self.model_name,
                        temperature=self.temperature,
                        api_key=os.environ.get('OPENAI_API_KEY')
                    )
                elif "antrophic" in self.category:
                    llm = ChatAnthropic(
                        model_name=self.model_name,
                        temperature=self.temperature,
                        anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
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
                    raise e
                
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
        structured_llm = self.model.with_structured_output(output_structure)
        
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

        # Generate response
        response = structured_llm.invoke(full_prompt).model_dump()

        if self.memory:
            self.chat_history.append({'role': 'assistant', 'content': response})
        return response