# Model Configuration Guide

## Overview

The `model_config.json` file defines LLM configurations for all agents in MASAI. It specifies which models to use for each component (Router, Evaluator, Reflector, Planner) and their parameters.

---

## File Structure

```json
{
    "all": {
        "router": { ... },
        "evaluator": { ... },
        "reflector": { ... },
        "planner": { ... }
    },
    "agent_name": {
        "router": { ... },
        "evaluator": { ... },
        "reflector": { ... },
        "planner": { ... }
    }
}
```

### Configuration Precedence

1. **Agent-specific config** (e.g., "research_agent")
2. **"all" config** (fallback)
3. **Runtime overrides** (config_dict in create_agent)

---

## Component Configuration

### Basic Structure

```json
{
    "model_name": "gemini-2.5-flash",
    "category": "gemini",
    "description": "Gemini 2.5 Flash for fast evaluation",
    "temperature": 0.3,
    "max_output_tokens": 1024,
    "top_p": 0.95,
    "top_k": 40
}
```

### Core Parameters (All Providers)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | Required | Model identifier |
| `category` | str | Required | Provider: "gemini", "openai", "anthropic" |
| `description` | str | Optional | Human-readable description |
| `temperature` | float | 0.7 | Randomness (0.0-2.0) |
| `max_output_tokens` | int | 2048 | Maximum response length |
| `top_p` | float | 1.0 | Nucleus sampling (0.0-1.0) |
| `top_k` | int | 40 | Top-k sampling |

---

## Provider-Specific Parameters

### Google Gemini

```json
{
    "model_name": "gemini-2.5-flash",
    "category": "gemini",
    "temperature": 0.3,
    "max_output_tokens": 1024,
    "top_p": 0.95,
    "top_k": 40,
    "thinking_budget": -1,
    "safety_settings": [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        }
    ]
}
```

**Gemini-Specific Parameters**:
- `thinking_budget`: Extended thinking tokens (-1 for dynamic)
- `safety_settings`: Content filtering rules

**Available Models**:
- `gemini-2.5-flash` - Fast, efficient
- `gemini-2.5-pro` - More capable
- `gemini-1.5-flash` - Older fast model
- `gemini-1.5-pro` - Older capable model

### OpenAI

```json
{
    "model_name": "gpt-4o",
    "category": "openai",
    "temperature": 0.3,
    "max_output_tokens": 2048,
    "top_p": 0.95,
    "reasoning_effort": "low",
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 42
}
```

**OpenAI-Specific Parameters**:
- `reasoning_effort`: "low", "medium", "high" (for o1, o3 models)
- `presence_penalty`: Penalize repeated tokens (-2.0 to 2.0)
- `frequency_penalty`: Penalize frequent tokens (-2.0 to 2.0)
- `seed`: Reproducibility seed

**Available Models**:
- `gpt-4o` - Latest, most capable
- `gpt-4-turbo` - Previous generation
- `gpt-4` - Older version
- `gpt-3.5-turbo` - Fast, cheap
- `o1` - Reasoning model
- `o3` - Advanced reasoning

### Anthropic Claude

```json
{
    "model_name": "claude-3.5-sonnet",
    "category": "anthropic",
    "temperature": 0.3,
    "max_output_tokens": 2048,
    "top_p": 0.95,
    "top_k": 40
}
```

**Available Models**:
- `claude-3.5-sonnet` - Latest, most capable
- `claude-3-opus` - Previous generation
- `claude-3-sonnet` - Older version
- `claude-3-haiku` - Fast, cheap

---

## Safety Settings (Gemini)

```json
"safety_settings": [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_LOW_AND_ABOVE"
    }
]
```

**Threshold Options**:
- `BLOCK_NONE` - Allow all content
- `BLOCK_ONLY_HIGH` - Block only high-risk
- `BLOCK_MEDIUM_AND_ABOVE` - Block medium and high
- `BLOCK_LOW_AND_ABOVE` - Block all risky content

---

## Real-World Examples

### Example 1: Research Agent

```json
{
    "research_agent": {
        "router": {
            "model_name": "gemini-1.5-flash",
            "category": "gemini",
            "temperature": 0.7,
            "max_output_tokens": 2048
        },
        "evaluator": {
            "model_name": "gemini-1.5-pro",
            "category": "gemini",
            "temperature": 0.5,
            "max_output_tokens": 2048
        },
        "reflector": {
            "model_name": "gpt-4o",
            "category": "openai",
            "temperature": 0.3,
            "max_output_tokens": 3072
        }
    }
}
```

### Example 2: Cost-Optimized

```json
{
    "budget_agent": {
        "router": {
            "model_name": "gpt-3.5-turbo",
            "category": "openai",
            "temperature": 0.3,
            "max_output_tokens": 512
        },
        "evaluator": {
            "model_name": "gpt-3.5-turbo",
            "category": "openai",
            "temperature": 0.3,
            "max_output_tokens": 512
        },
        "reflector": {
            "model_name": "gpt-3.5-turbo",
            "category": "openai",
            "temperature": 0.3,
            "max_output_tokens": 1024
        }
    }
}
```

### Example 3: High-Quality Reasoning

```json
{
    "reasoning_agent": {
        "router": {
            "model_name": "gpt-4o",
            "category": "openai",
            "temperature": 0.2,
            "max_output_tokens": 1024
        },
        "evaluator": {
            "model_name": "gpt-4o",
            "category": "openai",
            "temperature": 0.3,
            "max_output_tokens": 2048
        },
        "reflector": {
            "model_name": "o1",
            "category": "openai",
            "reasoning_effort": "high",
            "max_output_tokens": 4096
        }
    }
}
```

---

## Runtime Overrides

### Using config_dict

```python
agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=...,
    config_dict={
        "router_temperature": 0.5,
        "router_max_output_tokens": 512,
        "evaluator_temperature": 0.3,
        "reflector_temperature": 0.2,
        "reflector_max_output_tokens": 2048,
        "planner_temperature": 0.4
    }
)
```

### Precedence

1. config_dict (highest priority)
2. Agent-specific config
3. "all" config (lowest priority)

---

## Best Practices

### 1. Temperature Settings

- **Router** (0.2-0.5): Fast decisions, low randomness
- **Evaluator** (0.3-0.5): Balanced evaluation
- **Reflector** (0.2-0.4): Deep reasoning, low randomness
- **Planner** (0.3-0.5): Planning with some creativity

### 2. Token Limits

- **Router**: 512-1024 tokens (quick decisions)
- **Evaluator**: 1024-2048 tokens (evaluation)
- **Reflector**: 2048-4096 tokens (deep reasoning)
- **Planner**: 1024-2048 tokens (planning)

### 3. Model Selection

- **Fast**: Use flash/turbo models for Router
- **Quality**: Use pro/opus models for Reflector
- **Balance**: Use standard models for Evaluator

### 4. Safety Settings

- **Public-facing**: Use BLOCK_MEDIUM_AND_ABOVE
- **Internal**: Use BLOCK_ONLY_HIGH
- **Research**: Use BLOCK_NONE

---

## Troubleshooting

### Issue: Model Not Found

```
Error: Model 'gemini-2.5-flash' not found
```

**Solution**: Check model name spelling and availability in your region.

### Issue: Invalid Parameter

```
Error: Unknown parameter 'thinking_budget' for OpenAI
```

**Solution**: Use provider-specific parameters only.

### Issue: Rate Limiting

**Solution**: Reduce max_output_tokens or use cheaper models.

---

## See Also

- [AGENTMANAGER_DETAILED.md](AGENTMANAGER_DETAILED.md) - How to use configs
- [FRAMEWORK_OVERVIEW.md](FRAMEWORK_OVERVIEW.md) - Architecture overview
- [QUICK_START.md](QUICK_START.md) - Quick setup guide

