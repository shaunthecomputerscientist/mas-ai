## Quick Summary: Direct Provider-Specific Names

**TL;DR**: ❌ **Don't use provider-specific names directly in JSON** — they have inconsistent behavior across providers.

---

## What Happens When You Use Provider-Specific Names

### ❌ OpenAI (BAD)
```json
{
  "planner": {
    "category": "openai",
    "max_tokens": 1024,        ← ❌ Gets passed through (unintended)
    "stop": ["END"],           ← ❌ SILENTLY FILTERED OUT
    "logprobs": true,          ← ❌ SILENTLY FILTERED OUT
    "top_logprobs": 3          ← ❌ SILENTLY FILTERED OUT
  }
}
```

**Result**: Some params work, some silently disappear. 😞

---

### ✅ Gemini (WORKS - but not recommended)
```json
{
  "evaluator": {
    "category": "gemini",
    "maxOutputTokens": 2048,   ← ✅ Works (passed through)
    "stopSequences": ["END"],  ← ✅ Works (passed through)
    "responseLogprobs": true   ← ✅ Works (passed through)
  }
}
```

**Result**: All work, but inconsistent with OpenAI. 😐

---

## ✅ The RIGHT Way (Always Use Standardized Names)

```json
{
  "planner": {
    "category": "openai",
    "temperature": 0.3,              ← ✅ Shared param
    "max_output_tokens": 1024,       ← ✅ Standardized (maps to max_tokens)
    "stop_sequences": ["END"],       ← ✅ Standardized (maps to stop)
    "enable_logprobs": false,        ← ✅ Standardized (maps to logprobs)
    "num_logprobs": 0                ← ✅ Standardized (maps to top_logprobs)
  }
}
```

**Result**: Works perfectly. ✨

```json
{
  "evaluator": {
    "category": "gemini",
    "temperature": 0.7,              ← ✅ Shared param
    "max_output_tokens": 2048,       ← ✅ Standardized (maps to maxOutputTokens)
    "stop_sequences": ["END"],       ← ✅ Standardized (maps to stopSequences)
    "enable_logprobs": true,         ← ✅ Standardized (maps to responseLogprobs)
    "num_logprobs": 2                ← ✅ Standardized (maps to logprobs)
  }
}
```

**Result**: Works perfectly. ✨

---

## Mapping Reference

| Your JSON Key | OpenAI → | Gemini → |
|---------------|----------|----------|
| `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `top_p` |
| `max_output_tokens` | `max_tokens` or `max_completion_tokens` | `max_output_tokens` |
| `stop_sequences` | `stop` | `stop_sequences` |
| `enable_logprobs` | `logprobs` | `response_logprobs` |
| `num_logprobs` | `top_logprobs` | `logprobs` |
| `presence_penalty` | `presence_penalty` | `presence_penalty` |
| `frequency_penalty` | `frequency_penalty` | `frequency_penalty` |
| `seed` | `seed` | `seed` |
| `reasoning_effort` | `reasoning_effort` | ❌ (OpenAI only) |
| `top_k` | ❌ (Gemini only) | `top_k` |

---

## The Problem With Direct Names

### Why You Shouldn't Use Them

1. **Silent Failures** (OpenAI)
   - `max_tokens` might get through
   - But `stop`, `logprobs` silently disappear
   - You don't get an error — it just doesn't work

2. **Inconsistent Behavior** (Across Providers)
   - Works for Gemini ✅
   - Fails for OpenAI ❌
   - Same config, different results

3. **Cross-Provider Contamination**
   - Use OpenAI names in Gemini config → Some leak through
   - No warning that they're wrong

4. **Future API Changes**
   - If OpenAI renames `max_tokens` → `max_length`, your config breaks
   - Standardized names are buffered against this

---

## Key Takeaway

```
Standardized Names (in your JSON)
         ↓
         ↓ (Auto-mapped by framework)
         ↓
Provider-Specific Names (sent to API)

✅ Always define standardized names
❌ Never define provider-specific names directly
✅ Let the framework do the mapping for you
```

For full details, see [AUDIT_PROVIDER_SPECIFIC_PARAMS.md](AUDIT_PROVIDER_SPECIFIC_PARAMS.md)
