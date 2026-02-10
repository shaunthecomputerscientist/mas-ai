# 📋 Comprehensive Model Configuration Guide

This guide documents all available parameters for OpenAI and Google Gemini models in MASAI, with proper configuration examples.

## Quick Reference

| Parameter | OpenAI | Gemini | Type | Range | Notes |
|-----------|--------|--------|------|-------|-------|
| **temperature** | ✅ | ✅ | float | 0.0-2.0 | Controls randomness |
| **max_output_tokens** | ✅ | ✅ | int | 1-∞ | Max response length |
| **top_p** | ✅ | ✅ | float | 0.0-1.0 | Nucleus sampling |
| **stop_sequences** | ✅ | ✅ | list | - | Stop generation at these strings |
| **presence_penalty** | ✅ | ✅ | float | -2.0-2.0 | Penalize existing tokens |
| **frequency_penalty** | ✅ | ✅ | float | -2.0-2.0 | Penalize by frequency |
| **seed** | ✅ | ✅ | int | - | Deterministic output |
| **enable_logprobs** | ✅ | ✅ | bool | - | Return log probabilities |
| **num_logprobs** | ✅ | ✅ | int | 0-20 | Top candidate tokens *requires enable_logprobs* |
| **reasoning_effort** | ✅ | ❌ | str | low/medium/high | Reasoning models only |
| **top_k** | ❌ | ✅ | int | 1-100 | Top-k sampling (Gemini only) |
| **thinking_budget** | ❌ | ✅ | int | -1, 0-10000 | Thinking token budget (Gemini only) |
| **safety_settings** | ❌ | ✅ | list | - | Safety filters (Gemini only) |

---

## Shared Parameters (Work for both OpenAI & Gemini)

These parameters use the same name and behavior across providers:

### temperature
Controls the randomness of responses.
- **Range:** 0.0 (deterministic) to 2.0 (maximum randomness)
- **Default:** 0.7 (balanced)
- **Ignored for:** Reasoning models (GPT-5, o1, o3, o4)
```json
"temperature": 0.5
```

### top_p
Nucleus sampling: only consider tokens with cumulative probability ≤ top_p.
- **Range:** 0.0 to 1.0
- **Default:** 1.0 (all tokens)
- **Tip:** Recommended to use one of temperature or top_p, not both
```json
"top_p": 0.95
```

### presence_penalty & frequency_penalty
Control how much to penalize repeated tokens.
- **Range:** -2.0 to 2.0
- **Default:** 0.0 (no penalty)
- **Positive values:** Encourage new topics / reduce repetition
```json
"presence_penalty": 0.1,
"frequency_penalty": 0.1
```

### seed
Deterministic output seed (best effort).
- **Range:** any integer
- **Note:** Same seed + temperature=0 → same output
```json
"seed": 42
```

---

## Mapped Parameters (Standardized Names)

These parameters use standardized MASAI names that are automatically mapped to provider-specific names:

### max_output_tokens
Maximum length of the response. The framework automatically maps this:
- **OpenAI standard:** → `max_tokens`
- **OpenAI reasoning:** → `max_completion_tokens`
- **Gemini:** → `maxOutputTokens`

```json
"max_output_tokens": 2048
```

### stop_sequences
List of strings where generation stops. Automatically mapped:
- **OpenAI:** → `stop`
- **Gemini:** → `stopSequences`

```json
"stop_sequences": ["END", "STOP", "---"]
```

### enable_logprobs
Return log probabilities for output tokens. Automatically mapped:
- **OpenAI:** → `logprobs`
- **Gemini:** → `responseLogprobs`

```json
"enable_logprobs": true,
"num_logprobs": 5
```

⚠️ **Important Constraint:** `num_logprobs` is only valid when `enable_logprobs=true`. The framework automatically:
- Drops `num_logprobs` if `enable_logprobs` is explicitly `false`
- Auto-enables `enable_logprobs` if `num_logprobs` is set and `enable_logprobs` is not explicitly `false`

**Example (both params):**
```json
{
  "enable_logprobs": true,
  "num_logprobs": 3
}
```

**Example (logprobs disabled → num_logprobs dropped automatically):**
```json
{
  "enable_logprobs": false,
  "num_logprobs": 5  // <- This will be ignored
}
```

---

## OpenAI-Specific Parameters

### reasoning_effort
Effort level for reasoning models (GPT-5, o1, o3, o4).
- **Values:** `low`, `medium`, `high`, `xhigh` (if supported)
- **Default:** `medium`
- **Note:** Temperature is auto-mapped to reasoning_effort if not explicitly set
```json
{
  "model_name": "gpt-5",
  "temperature": 0.5,  // Auto-mapped to reasoning_effort="medium"
  "max_output_tokens": 4096
}
```

Or explicitly set:
```json
{
  "model_name": "gpt-5",
  "reasoning_effort": "high",
  "max_output_tokens": 8192
}
```

### response_format
Advanced: Structured output schema (JSON mode).
```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "ResponseSchema",
      "schema": { "type": "object", "properties": { ... } }
    }
  }
}
```

### logit_bias
Advanced: Adjust token probabilities by token ID.
```json
{
  "logit_bias": {
    "15043": 100,  // Token ID: bias (strongly promote)
    "2051": -100   // Token ID: bias (strongly demote)
  }
}
```

---

## Gemini-Specific Parameters

### top_k
Top-k sampling: only consider the k most likely tokens.
- **Range:** 1 to 100
- **Default:** 40
- **Note:** Different from top_p (nucleus sampling)
```json
"top_k": 40
```

### thinking_budget
Token budget for thinking/reasoning (Gemini 2.5+ only).
- **Value:** `-1` (dynamic, model decides), `0` (disable thinking), or fixed count (1-10000)
- **Default:** `-1` (dynamic)
- **Note:** Auto-set to -1 for Gemini 2.5 models if not specified
```json
{
  "model_name": "gemini-2.5-pro",
  "thinking_budget": -1  // Dynamic thinking
}
```

### safety_settings
Safety content filtering rules.
- **Structure:** Array of `{category, threshold}` objects
- **Categories:** `HARM_CATEGORY_HARASSMENT`, `HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_SEXUALLY_EXPLICIT`, `HARM_CATEGORY_DANGEROUS_CONTENT`
- **Thresholds:** `BLOCK_NONE`, `BLOCK_ONLY_HIGH`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_LOW_AND_ABOVE`

```json
{
  "safety_settings": [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
  ]
}
```

### candidate_count
Number of response variations to generate (advanced).
- **Range:** 1 to 8
- **Default:** 1
```json
"candidate_count": 1
```

---

## Complete Configuration Examples

### Example 1: OpenAI GPT-4o (Standard Model)
```json
{
  "planner": {
    "model_name": "gpt-4o",
    "category": "openai",
    "temperature": 0.3,
    "max_output_tokens": 1024,
    "top_p": 0.95,
    "stop_sequences": ["END", "STOP"],
    "enable_logprobs": false,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
  }
}
```

### Example 2: OpenAI GPT-5 (Reasoning Model)
```json
{
  "reflector": {
    "model_name": "gpt-5",
    "category": "openai",
    "max_output_tokens": 4096,
    "reasoning_effort": "high",
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 42
  }
}
```

**Note:** Temperature is ignored for reasoning models; use `reasoning_effort` instead.

### Example 3: Gemini 2.5 Pro (with Thinking)
```json
{
  "deep_reflector": {
    "model_name": "gemini-2.5-pro",
    "category": "gemini",
    "temperature": 0.5,
    "max_output_tokens": 3072,
    "top_p": 0.95,
    "top_k": 40,
    "thinking_budget": -1,
    "safety_settings": [
      {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
  }
}
```

### Example 4: Gemini 2.5 Flash (Fast Response)
```json
{
  "router": {
    "model_name": "gemini-2.5-flash",
    "category": "gemini",
    "temperature": 0.2,
    "max_output_tokens": 1024,
    "top_k": 20,
    "safety_settings": [
      {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
      {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
      {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
      {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]
  }
}
```

---

## Common Mistakes & How to Avoid Them

### ❌ Mistake 1: Setting both enable_logprobs=false and num_logprobs > 0
```json
// WRONG - will cause num_logprobs to be dropped
{
  "enable_logprobs": false,
  "num_logprobs": 5
}
```

✅ **Fix:**
```json
// CORRECT - either enable both or use neither
{
  "enable_logprobs": true,
  "num_logprobs": 5
}
```

### ❌ Mistake 2: Setting temperature on reasoning models
```json
// WRONG - temperature is ignored for GPT-5, o1, o3, o4
{
  "model_name": "gpt-5",
  "temperature": 0.7
}
```

✅ **Fix:**
```json
// CORRECT - use reasoning_effort instead
{
  "model_name": "gpt-5",
  "reasoning_effort": "high"
}
```

### ❌ Mistake 3: Using both max_tokens and max_completion_tokens
```json
// WRONG - let MASAI map max_output_tokens automatically
{
  "max_tokens": 1024,
  "max_completion_tokens": 2048
}
```

✅ **Fix:**
```json
// CORRECT - use standardized name
{
  "max_output_tokens": 2048
}
```

### ❌ Mistake 4: Using OpenAI params for Gemini models
```json
// WRONG - reasoning_effort is OpenAI-only
{
  "model_name": "gemini-2.5-pro",
  "reasoning_effort": "high"
}
```

✅ **Fix:**
```json
// CORRECT - use Gemini equivalent (thinking_budget)
{
  "model_name": "gemini-2.5-pro",
  "thinking_budget": -1
}
```

---

## How Parameters Flow Through MASAI

1. **Load from `model_config.json`** → Raw config dict
2. **Extract & Map** (`extract_openai_params()` / `extract_gemini_params()`)
   - Shared params passed through
   - Standardized names mapped to provider names
   - Incompatible params filtered out
   - Constraints enforced (logprobs, etc.)
3. **Initialize Model** (ChatOpenAI / ChatGoogleGenerativeAI)
   - Extracted params become wrapper init parameters
   - Wrapper stores them in instance variables
4. **Prepare Request** (`_prepare_request_params()`)
   - Model-specific logic applied (reasoning vs. standard)
   - Final API request built
5. **Send to API** (OpenAI / Google Generative AI SDK)

---

## Troubleshooting

### Error: "top_logprobs parameter is only allowed when logprobs is enabled"
- **Cause:** `enable_logprobs=false` but `num_logprobs` was set
- **Fix:** Either remove `num_logprobs` or set `enable_logprobs=true`

### Error: "Setting max_tokens and max_completion_tokens at the same time is not supported"
- **Cause:** Both explicitly set in config
- **Fix:** Use only `max_output_tokens` (MASAI handles the mapping)

### Error: "temperature parameter not supported for this model"
- **Cause:** Reasoning model with explicit `temperature` set
- **Fix:** Use `reasoning_effort` for reasoning models instead

### Error: "safety_settings only works with Gemini models"
- **Cause:** Trying to use `safety_settings` with OpenAI
- **Fix:** Remove `safety_settings`; OpenAI doesn't support this parameter

---

## API Compatibility

This guide is up-to-date with:
- **OpenAI API:** Latest (2025)
  - Latest models: gpt-5, gpt-5.1, o1, o3, o4
  - Chat Completions API
- **Google Generative AI:** Latest (2025)
  - Latest models: gemini-2.5-pro, gemini-2.5-flash
  - Models API (REST v1beta)

Last updated: January 2026
