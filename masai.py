

#SAMPLE SCRIPT SHOWING EXAMPLE OF USING MAS-AI
#---------------------------------AgentManager---------------------------------

# Suppress Google gRPC ALTS warnings BEFORE any imports (harmless but noisy)
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

import sys
import logging
import warnings

# Additional suppression
logging.getLogger('absl').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='google')

from typing import Optional
from src.masai.AgentManager.AgentManager import AgentManager, AgentDetails
from src.masai.MultiAgents.MultiAgent import MultiAgentSystem, SupervisorConfig
from src.masai.Memory.InMemoryStore import InMemoryDocStore
from src.masai.GenerativeModel.baseGenerativeModel.basegenerativeModel import BaseGenerativeModel
#---------------------------------Tools---------------------------------
# Define your tools and import as needed
# from src.masai.Tools.tools.baseTools import human_in_loop_input
# from src.masai.Tools.tools.InputOutputTools import file_handler_IO, files_checker, save_long_term_details
# from src.masai.Tools.tools.visiontools import Vision_Model
from src.masai.Tools.utilities.tokenGenerationTool import token_stream, MarkupProcessor
from dotenv import load_dotenv
load_dotenv()
import asyncio
import time


# User provides path to their model config
model_config_path = os.path.join(os.getcwd(), 'model_config.json')


# Optional: Define async streaming callback for real-time output
async def async_streaming_callback(chunk):
    """
    An async streaming callback function that prints chunks to the terminal.

    Args:
        chunk: A chunk of the LLM response (could be a string, dict, or other data type)
    """
    if isinstance(chunk, dict):
        # For structured data with different formatting
        if "answer" in chunk:
            print(f"[STREAM ASYNC] Answer: {chunk['answer']}")
        else:
            print(f"[STREAM ASYNC] Chunk: {chunk}")
    else:
        # For string chunks
        print(chunk, end="", flush=True)


# Define Agent Manager
manager = AgentManager(
    context={"HUMAN NAME": "SHAUN"},
    logging=True,
    model_config_path=model_config_path,
    chat_log='MAS/WORKSPACE/chat_log.json',  # ensure it exists
    streaming=True,  # Set to True to enable streaming
    streaming_callback=async_streaming_callback,  # Required if streaming=True
)

# Define tools
tools_for_researcher = []
tools_for_personal = []
tools_for_productivity = []

# Define agent details for each agent
research_agent_details = AgentDetails(
    capabilities=[
        "reasoning",
        "coding",
        "science",
        "research",
        "mathematics",
        "searching capabilities"
    ],
    description="Make best use of the tools by analyzing entire context and history, question and deciding next best step.",
    style="You give very elaborate answers"
)

personal_agent_details = AgentDetails(
    capabilities=[
        "general answering",
        "personalized conversation",
        "complex reasoning",
        "creative tasks",
    ],
    description="""Have access to current human's personal data to give personalized answers.
    You can save interesting facts about user's life, personal details, etc in long term memory.
    Assign appropriate tasks to other agents (if available)""",
    style="acts as a personal assistant focusing on personalized interactions. Do not share details about other agents."
)

productivity_agent_details = AgentDetails(
    capabilities=[
        "email management",
        "calendar management",
        "meeting scheduling",
        "productivity optimization",
        "time management"
    ],
    description="Specializing in productivity work for user like send/read emails, read meetings, set calendar events/schedules",
    style="focuses on efficient task execution and organization"
)
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Create agents with their respective details
# Example 1: Research agent with planning and long context
manager.create_agent(
    agent_name="research_agent",
    tools=tools_for_researcher,
    agent_details=research_agent_details,
    plan=True,
    long_context_order=100,
    in_memory_store=InMemoryDocStore(embedding_model="all-MiniLM-L6-v2")  # Optional: requires sentence-transformers
)

# Example 2: Personal agent with per-component configuration
manager.create_agent(
    agent_name="general_personal_agent",
    tools=tools_for_personal,
    agent_details=personal_agent_details,
    plan=False,
    in_memory_store=InMemoryDocStore(embedding_model="all-MiniLM-L6-v2"),  # Optional: requires sentence-transformers
    config_dict={
        'evaluator_streaming': False,  # Disable streaming for evaluator
        'reflector_streaming': False,  # Disable streaming for reflector
        'router_temperature': 0.3,     # Custom temperature for router
        'evaluator_temperature': 0.1,  # Custom temperature for evaluator
    }
)

# Example 3: Productivity agent with planning
manager.create_agent(
    agent_name="productivity_agent",
    tools=tools_for_productivity,
    agent_details=productivity_agent_details,
    plan=True
)
#---------------------------------------------------------------------------------------------------------

# Create Supervisor Configuration for Hierarchical MAS

supervisor_config = SupervisorConfig(
    model_name="gemini-2.0-flash",
    temperature=0.7,  # Standard temperature for balanced creativity/consistency
    model_category="gemini",
    memory_order=20,  # Keep last 20 messages in context
    memory=True,      # Enable memory/context tracking
    extra_context={"user name": "shaun"},  # Additional context for supervisor role
    supervisor_system_prompt=None  # Uses default prompt if None
)

# Callback function to handle completed task results in hierarchical MAS
def handle_task_result(task):
    """Callback function to handle completed task results."""
    print("-" * 50, "\n")
    token_stream(task['answer'], delay=0.05, color='blue', token_type='word')
    print("-" * 50, "\n")


# Function to run hierarchical MAS
async def run_hierarchical_mas():
    """
    Hierarchical MAS with supervisor-based task management.
    Supervisor assigns tasks to agents and manages quality control.
    """
    mas_hierarchical = MultiAgentSystem(
        agentManager=manager,
        supervisor_config=supervisor_config,
        heirarchical_mas_result_callback=handle_task_result,
        agent_return_direct=True  # If False, supervisor reviews agent responses
    )

    try:
        while True:
            query = await asyncio.to_thread(input, "Enter a query for hierarchical MAS:\n")
            if query.lower() == "exit":
                break
            result = await mas_hierarchical.initiate_hierarchical_mas(query)
            if result["status"] == "completed":
                token_stream(result["answer"], delay=0.05, color="blue", token_type="word")
            elif result["status"] == "queued":
                print(f"Task {result['task_id']} queued for processing.")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await mas_hierarchical.task_manager.stop()


# Function to run decentralized MAS
async def run_decentralized_mas():
    """
    Decentralized MAS where agents can delegate tasks to each other.
    No central supervisor - peer-to-peer collaboration.
    """
    mas_decentralized = MultiAgentSystem(agentManager=manager)

    while True:
        try:
            query = input("Enter your query for decentralized MAS: ")
            if query.lower() == 'exit':
                break
            start = time.time()
            result = await mas_decentralized.initiate_decentralized_mas(
                query=query,
                set_entry_agent=manager.get_agent(agent_name="general_personal_agent")
            )
            # print(result['answer'], time.time()-start)
            token_stream(
                result['answer'],
                delay=0.05,
                color='blue',
                token_type='word'
            )
        except KeyboardInterrupt:
            print("\nExiting decentralized MAS...")
            break


def run_sequential_mas():
    """
    Sequential MAS where agents process tasks in a predefined sequence.
    Each agent builds on the previous agent's output.
    """
    agent_sequence = input("Enter agent sequence (comma-separated, e.g., research_agent,general_personal_agent):\n").split(',')
    agent_sequence = [agent.strip() for agent in agent_sequence]  # Clean up whitespace

    while True:
        try:
            query = input("Enter your query for sequential MAS: ")
            if query.lower() == 'exit':
                break

            # Initialize MultiAgentSystem (without supervisor config)
            mas_sequential = MultiAgentSystem(agentManager=manager)

            result = mas_sequential.initiate_sequential_mas(
                query=query,
                agent_sequence=agent_sequence,
                memory_order=3  # You can adjust the memory order
            )
            token_stream(
                result,  # The result is a string in sequential MAS
                delay=0.05,
                color='blue',
                token_type='word'
            )
        except KeyboardInterrupt:
            print("\nExiting sequential MAS...")
            break
        except Exception as e:
            print(f"Error in sequential MAS: {e}")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        execution_type = sys.argv[1].lower()
        if execution_type == 'hierarchical':
            asyncio.run(run_hierarchical_mas())
        elif execution_type == 'decentralized':
            asyncio.run(run_decentralized_mas())
        elif execution_type == 'sequential':
            run_sequential_mas()
        else:
            print("Invalid execution type. Please specify 'hierarchical', 'decentralized', or 'sequential'.")
    else:
        print("Please specify the execution type ('hierarchical', 'decentralized', or 'sequential') as a command-line argument.")