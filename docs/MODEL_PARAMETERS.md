# Model Parameters Guide

Complete reference for all supported models and their parameters in MASAI framework.

---

## Table of Contents

1. [Supported Models](#supported-models)
2. [Parameter Categories](#parameter-categories)
3. [Shared Parameters](#shared-parameters)
4. [Mapped Parameters](#mapped-parameters)
5. [Gemini-Specific Parameters](#gemini-specific-parameters)
6. [OpenAI-Specific Parameters](#openai-specific-parameters)
7. [Anthropic-Specific Parameters](#anthropic-specific-parameters)
8. [MASAI Framework Parameters](#masai-framework-parameters)
9. [Configuration Methods](#configuration-methods)
10. [Complete Examples](#complete-examples)

---

## Supported Models

### Google Gemini Models

| Model Name | Description | Thinking Support |
|------------|-------------|------------------|
| `gemini-2.5-pro` | Latest Gemini Pro with thinking | ✅ Yes |
| `gemini-2.5-flash` | Latest Gemini Flash with thinking | ✅ Yes |
| `gemini-2.0-flash-thinking-exp` | Experimental thinking model | ✅ Yes |
| `gemini-1.5-pro` | Previous generation Pro | ❌ No |
| `gemini-1.5-flash` | Previous generation Flash | ❌ No |

### OpenAI Models

| Model Name | Description | Reasoning Support |
|------------|-------------|-------------------|
| **GPT-5 Series** | | |
| `gpt-5` | Latest GPT-5 with reasoning | ✅ Yes |
| `gpt-5-mini` | Smaller GPT-5 variant | ✅ Yes |
| `gpt-5-nano` | Smallest GPT-5 variant | ✅ Yes |
| `gpt-5-thinking*` | GPT-5 thinking variants | ✅ Yes |
| **GPT-4.1 Series** | | |
| `gpt-4.1` | GPT-4.1 with reasoning | ✅ Yes |
| `gpt-4.1-nano` | Smaller GPT-4.1 variant | ✅ Yes |
| **GPT-4 Series** | | |
| `gpt-4o` | GPT-4 Optimized (standard) | ❌ No |
| `gpt-4o-mini` | Smaller GPT-4o variant | ❌ No |
| `gpt-4-turbo` | GPT-4 Turbo (standard) | ❌ No |
| `gpt-3.5-turbo` | GPT-3.5 Turbo (standard) | ❌ No |
| **Reasoning Models** | | |
| `o1` | OpenAI o1 reasoning model | ✅ Yes |
| `o1-mini` | Smaller o1 variant | ✅ Yes |
| `o1-preview` | o1 preview version | ✅ Yes |
| `o3` | OpenAI o3 reasoning model | ✅ Yes |
| `o3-mini` | Smaller o3 variant | ✅ Yes |
| `o4-mini` | OpenAI o4 mini reasoning | ✅ Yes |

### Anthropic Claude Models

| Model Name | Description | Extended Thinking |
|------------|-------------|-------------------|
| `claude-4-opus` | Claude 4 Opus with extended thinking | ✅ Yes |
| `claude-4-sonnet` | Claude 4 Sonnet with extended thinking | ✅ Yes |
| `claude-3.7-sonnet` | Claude 3.7 Sonnet hybrid reasoning | ✅ Yes |
| `claude-3.5-sonnet` | Claude 3.5 Sonnet (standard) | ❌ No |
| `claude-3-opus` | Claude 3 Opus (standard) | ❌ No |
| `claude-3-haiku` | Claude 3 Haiku (fast) | ❌ No |

---

## Parameter Categories

MASAI uses a sophisticated parameter system with 5 categories:

1. **Shared Parameters**: Work identically across all providers
2. **Mapped Parameters**: Same concept, different names per provider (auto-mapped)
3. **Provider-Specific Parameters**: Unique to each provider
4. **MASAI Framework Parameters**: Framework-level, filtered before API calls
5. **Unknown Parameters**: Passed through for forward compatibility

---

## Shared Parameters

These parameters work identically across **all providers** with the same name and type.

### `temperature`
- **Type**: `float`
- **Range**: `0.0` - `2.0`
- **Default**: `0.7`
- **Description**: Controls randomness in output
  - `0.0`: Deterministic, focused responses
  - `0.7`: Balanced creativity and coherence
  - `2.0`: Maximum randomness and creativity
- **⚠️ Note**: Ignored for reasoning models (GPT-5, o1, o3, o4)

### `top_p`
- **Type**: `float`
- **Range**: `0.0` - `1.0`
- **Default**: `0.95`
- **Description**: Nucleus sampling - cumulative probability threshold
  - `0.1`: Very focused, only top tokens
  - `0.95`: Balanced diversity
  - `1.0`: Consider all tokens

### `presence_penalty`
- **Type**: `float`
- **Range**: `-2.0` - `2.0`
- **Default**: `0.0`
- **Description**: Penalizes tokens that have already appeared
  - Positive values: Encourage new topics
  - Negative values: Allow repetition
  - `0.0`: No penalty

### `frequency_penalty`
- **Type**: `float`
- **Range**: `-2.0` - `2.0`
- **Default**: `0.0`
- **Description**: Penalizes tokens based on their frequency
  - Positive values: Reduce repetition
  - Negative values: Allow repetition
  - `0.0`: No penalty

### `seed`
- **Type**: `int`
- **Description**: Random seed for deterministic output
  - Same seed + same input = same output (when `temperature=0`)
  - Useful for reproducibility and testing

---

## Mapped Parameters

These parameters have **different names per provider** but represent the same concept. MASAI automatically maps them.

### `max_output_tokens`
**Standardized name in MASAI**

- **Type**: `int`
- **Description**: Maximum number of tokens in the response
- **Provider Mappings**:
  - **OpenAI (reasoning)**: `max_completion_tokens`
  - **OpenAI (standard)**: `max_tokens`
  - **Gemini**: `max_output_tokens`
  - **Anthropic**: `max_tokens`

**Usage**: Always use `max_output_tokens` in your config - MASAI handles the mapping.

### `stop_sequences`
**Standardized name in MASAI**

- **Type**: `List[str]` or `str`
- **Description**: Stop generation when these strings are encountered
- **Provider Mappings**:
  - **OpenAI**: `stop`
  - **Gemini**: `stop_sequences`
  - **Anthropic**: `stop_sequences`

**Example**:
```json
"stop_sequences": ["END", "STOP", "###"]
```

### `enable_logprobs`
**Standardized name in MASAI**

- **Type**: `bool`
- **Default**: `false`
- **Description**: Return log probabilities for tokens
- **Provider Mappings**:
  - **OpenAI**: `logprobs`
  - **Gemini**: `response_logprobs`

### `num_logprobs`
**Standardized name in MASAI**

- **Type**: `int`
- **Range**: `0` - `20`
- **Description**: Number of top candidate tokens to return logprobs for
- **Provider Mappings**:
  - **OpenAI**: `top_logprobs`
  - **Gemini**: `logprobs`

**Example**:
```json
"enable_logprobs": true,
"num_logprobs": 5
```

---

## Gemini-Specific Parameters

These parameters are **only available for Gemini models**.

### `top_k`
- **Type**: `int`
- **Range**: `1` - `100`
- **Default**: `40`
- **Description**: Top-k sampling - consider only top k tokens
  - Lower values: More focused output
  - Higher values: More diverse output

### `safety_settings`
- **Type**: `List[Dict[str, str]]`
- **Description**: Content safety filter configuration
- **Categories**:
  - `HARM_CATEGORY_HARASSMENT`
  - `HARM_CATEGORY_HATE_SPEECH`
  - `HARM_CATEGORY_SEXUALLY_EXPLICIT`
  - `HARM_CATEGORY_DANGEROUS_CONTENT`
- **Thresholds**:
  - `BLOCK_NONE`: No filtering (maximum reliability)
  - `BLOCK_ONLY_HIGH`: Block only high-risk content
  - `BLOCK_MEDIUM_AND_ABOVE`: Block medium and high-risk
  - `BLOCK_LOW_AND_ABOVE`: Block all potentially harmful content

**Example**:
```json
"safety_settings": [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
]
```

### `thinking_budget`
- **Type**: `int`
- **Range**: `-1`, `0`, `1` - `10000`
- **Description**: Token budget for thinking (Gemini 2.5+ only)
  - `-1`: Dynamic thinking (model decides)
  - `0`: Thinking disabled
  - `1-10000`: Fixed token budget
- **⚠️ Only for**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash-thinking-exp`

### `candidate_count`
- **Type**: `int`
- **Range**: `1` - `8`
- **Default**: `1`
- **Description**: Number of response variations to generate

---

## OpenAI-Specific Parameters

These parameters are **only available for OpenAI models**.

### `reasoning_effort`
- **Type**: `str`
- **Options**: `"low"`, `"medium"`, `"high"`
- **Default**: `"medium"`
- **Description**: Reasoning effort level for reasoning models
  - `"low"`: Fast, less thorough reasoning
  - `"medium"`: Balanced reasoning (recommended)
  - `"high"`: Slow, most thorough reasoning
- **⚠️ Only for**: GPT-5, o1, o3, o4 series (reasoning models)

**Example**:
```json
{
    "model_name": "gpt-5",
    "category": "openai",
    "reasoning_effort": "high",
    "max_output_tokens": 4096
}
```

---

## Anthropic-Specific Parameters

Anthropic models use standard parameters. Extended thinking is automatically enabled for Claude 4 and 3.7 models.

---

## MASAI Framework Parameters

These parameters are **framework-specific** and are automatically filtered out before passing to model APIs.

### Memory & Context Parameters

#### `memory_order`
- **Type**: `int`
- **Default**: `10`
- **Description**: Number of past interactions to keep in memory
- **Used in**: `AgentManager.create_agent()`

#### `long_context`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable long context handling with summarization
- **Used in**: `AgentManager.create_agent()`

#### `long_context_order`
- **Type**: `int`
- **Default**: `20`
- **Description**: Number of context summaries to keep before flushing to persistent storage
- **Used in**: `AgentManager.create_agent()`

### Persistent Memory Parameters

#### `user_id`
- **Type**: `str` or `int`
- **Description**: User identifier for memory isolation
- **Used in**: `AgentManager.__init__()`

#### `memory_config`
- **Type**: `QdrantConfig` or `RedisConfig` or `Dict`
- **Description**: Configuration for persistent long-term memory backend
- **Used in**: `AgentManager.__init__()`

#### `persist_memory`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable persistent memory for this agent/component
- **Used in**: `AgentManager.create_agent()`

#### `categories_resolver`
- **Type**: `Callable[[Document], List[str]]`
- **Description**: Function to extract categories from documents for filtering
- **Used in**: `AgentManager.__init__()`

### Context Callable Parameters

#### `context_callable`
- **Type**: `Callable`
- **Description**: Function that uses user input to provide additional context to LLM
- **Used in**: `AgentManager.create_agent()`

#### `callable_config`
- **Type**: `Dict[str, Callable]`
- **Description**: Dictionary mapping node names to context callables
- **Example**: `{"router": callable_1, "evaluator": callable_2}`
- **Used in**: `AgentManager.create_agent()`

---

## Configuration Methods

MASAI supports **3 ways** to configure model parameters:

### 1. model_config.json (Static Configuration)

Define parameters in `model_config.json` file:

```json
{
    "all": {
        "router": {
            "model_name": "gemini-2.5-pro",
            "category": "gemini",
            "temperature": 0.2,
            "max_output_tokens": 2048,
            "top_k": 40,
            "thinking_budget": -1,
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
            ]
        }
    }
}
```

### 2. config_dict (Runtime Override)

Override parameters at runtime using `config_dict` in `create_agent()`:

```python
config_dict = {
    "router_temperature": 0.5,
    "router_max_output_tokens": 4096,
    "evaluator_temperature": 0.3
}

agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=AgentDetails(...),
    config_dict=config_dict  # Runtime overrides
)
```

### 3. Direct kwargs in create_agent()

Pass parameters directly as kwargs:

```python
agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=AgentDetails(...),
    temperature=0.7,  # Applies to all nodes
    max_output_tokens=2048,  # Applies to all nodes
    memory_order=15,
    long_context=True
)
```

**Priority Order**: `config_dict` > `kwargs` > `model_config.json`

---

## Complete Examples

### Example 1: Gemini Thinking Model

```json
{
    "router": {
        "model_name": "gemini-2.5-pro",
        "category": "gemini",
        "temperature": 0.2,
        "max_output_tokens": 4096,
        "top_p": 0.95,
        "top_k": 40,
        "thinking_budget": -1,
        "safety_settings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 42
    }
}
```

### Example 2: OpenAI Reasoning Model (GPT-5)

```json
{
    "reflector": {
        "model_name": "gpt-5",
        "category": "openai",
        "max_output_tokens": 8192,
        "reasoning_effort": "high",
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "seed": 42
    }
}
```

**Note**: `temperature` is automatically ignored for reasoning models.

### Example 3: OpenAI Standard Model (GPT-4o)

```json
{
    "evaluator": {
        "model_name": "gpt-4o",
        "category": "openai",
        "temperature": 0.7,
        "max_output_tokens": 2048,
        "top_p": 0.95,
        "stop_sequences": ["END", "STOP"],
        "enable_logprobs": true,
        "num_logprobs": 5,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }
}
```

### Example 4: Anthropic Claude Model

```json
{
    "router": {
        "model_name": "claude-4-opus",
        "category": "anthropic",
        "temperature": 0.5,
        "max_output_tokens": 4096,
        "top_p": 0.95
    }
}
```

### Example 5: Complete Agent Configuration with All Parameters

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.Memory.LongTermMemory import QdrantConfig
from langchain_huggingface import HuggingFaceEmbeddings

# Setup persistent memory
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant_config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="masai_memories",
    vector_size=384,
    embedding_model=embedding_model,
    dedup_mode="similarity",
    dedup_similarity_threshold=0.75
)

# Create AgentManager with persistent memory
manager = AgentManager(
    user_id="user_123",
    model_config_path="model_config.json",
    memory_config=qdrant_config,
    logging=True,
    streaming=False
)

# Define context callables
def router_context(query):
    return {"additional_info": "Router-specific context"}

def evaluator_context(query):
    return {"evaluation_criteria": "Evaluator-specific context"}

# Create agent with all parameters
agent = manager.create_agent(
    agent_name="assistant",
    tools=[],  # Add your tools here
    agent_details=AgentDetails(
        capabilities=["analysis", "reasoning", "planning"],
        description="Comprehensive AI assistant",
        style="detailed and thoughtful"
    ),

    # Memory parameters
    memory_order=15,
    long_context=True,
    long_context_order=25,
    shared_memory_order=10,
    retain_messages_order=10,
    persist_memory=True,

    # Planning
    plan=True,

    # Context callables
    callable_config={
        "router": router_context,
        "evaluator": evaluator_context
    },

    # Tool output control
    max_tool_output_words=3000,

    # Model parameters (apply to all nodes unless overridden in config_dict)
    temperature=0.7,
    max_output_tokens=2048,

    # Runtime overrides for specific nodes
    config_dict={
        "router_temperature": 0.2,
        "router_max_output_tokens": 1024,
        "evaluator_temperature": 0.5,
        "reflector_reasoning_effort": "high"
    }
)
```

### Example 6: Multi-Node Configuration with Different Providers

```json
{
    "all": {
        "router": {
            "model_name": "gemini-2.5-flash",
            "category": "gemini",
            "temperature": 0.2,
            "max_output_tokens": 1024,
            "top_k": 20,
            "thinking_budget": -1,
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
            ]
        },
        "evaluator": {
            "model_name": "gpt-4o",
            "category": "openai",
            "temperature": 0.5,
            "max_output_tokens": 2048,
            "top_p": 0.95,
            "enable_logprobs": false
        },
        "reflector": {
            "model_name": "gpt-5",
            "category": "openai",
            "max_output_tokens": 4096,
            "reasoning_effort": "high"
        },
        "planner": {
            "model_name": "claude-3.5-sonnet",
            "category": "anthropic",
            "temperature": 0.3,
            "max_output_tokens": 2048
        }
    }
}
```

---

## Parameter Validation & Error Handling

### Automatic Filtering

MASAI automatically filters out:
1. **MASAI-specific parameters** before passing to model APIs
2. **Provider-specific parameters** for other providers (e.g., `thinking_budget` filtered for OpenAI)
3. **Unknown parameters** are passed through for forward compatibility

### Parameter Mapping

MASAI automatically maps standardized names to provider-specific names:
- `max_output_tokens` → `max_completion_tokens` (OpenAI reasoning) or `max_tokens` (OpenAI standard)
- `stop_sequences` → `stop` (OpenAI) or `stop_sequences` (Gemini)
- `enable_logprobs` → `logprobs` (OpenAI) or `response_logprobs` (Gemini)
- `num_logprobs` → `top_logprobs` (OpenAI) or `logprobs` (Gemini)

### Reasoning Model Handling

For reasoning models (GPT-5, o1, o3, o4), MASAI automatically:
1. Ignores `temperature` parameter
2. Uses `max_completion_tokens` instead of `max_tokens`
3. Validates `reasoning_effort` values

---

## Best Practices

### 1. Use Standardized Names

Always use MASAI's standardized parameter names (`max_output_tokens`, `stop_sequences`, etc.) instead of provider-specific names. MASAI handles the mapping automatically.

### 2. Configure Safety Settings for Gemini

For production use with Gemini, configure safety settings to avoid unexpected blocks:

```json
"safety_settings": [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
]
```

### 3. Use Dynamic Thinking for Gemini 2.5

For Gemini 2.5 models, use `thinking_budget: -1` for dynamic thinking:

```json
"thinking_budget": -1
```

### 4. Choose Appropriate Reasoning Effort

For OpenAI reasoning models:
- `"low"`: Fast responses, good for simple tasks
- `"medium"`: Balanced, recommended for most use cases
- `"high"`: Thorough reasoning, use for complex problems

### 5. Configure Memory Parameters

For long conversations, configure memory parameters appropriately:

```python
memory_order=15,           # Keep last 15 interactions
long_context=True,         # Enable summarization
long_context_order=25,     # Keep 25 summaries before flushing
persist_memory=True        # Enable persistent storage
```

### 6. Use config_dict for Runtime Overrides

Use `config_dict` to override specific node parameters at runtime without modifying `model_config.json`:

```python
config_dict = {
    "router_temperature": 0.2,
    "evaluator_max_output_tokens": 4096
}
```

---

## Troubleshooting

### Issue: "Unknown field for GenerationConfig" (Gemini)

**Cause**: Parameter not supported by Gemini or incorrectly nested.

**Solution**:
- Check parameter name spelling
- For `thinking_budget`, ensure it's at root level (not nested under `thinking_config`)
- Remove OpenAI-specific parameters like `reasoning_effort`

### Issue: Temperature ignored for reasoning models

**Expected Behavior**: OpenAI reasoning models (GPT-5, o1, o3, o4) automatically ignore `temperature`.

**Solution**: Remove `temperature` from config or accept that it will be ignored.

### Issue: Parameters not taking effect

**Check**:
1. Parameter priority: `config_dict` > `kwargs` > `model_config.json`
2. Spelling and case sensitivity
3. Parameter is supported by the model provider

---

## Related Documentation

- [Quick Start Guide](QUICK_START.md) - Get started with MASAI
- [Configuration Guide](CONFIGURATION.md) - General configuration
- [Memory System Guide](MEMORY_SYSTEM.md) - Persistent memory setup
- [Agent Manager Detailed](AGENTMANAGER_DETAILED.md) - AgentManager API reference
- [API Reference](API_REFERENCE.md) - Complete API documentation

---

## Summary

MASAI provides a unified parameter system that:
- ✅ Supports **Gemini, OpenAI, and Anthropic** models
- ✅ Uses **standardized parameter names** across providers
- ✅ Automatically **maps and filters** parameters
- ✅ Supports **reasoning/thinking models** with special handling
- ✅ Provides **3 configuration methods** (JSON, config_dict, kwargs)
- ✅ Includes **framework-level parameters** for memory and context management

Use this guide as a reference when configuring your agents for optimal performance.
