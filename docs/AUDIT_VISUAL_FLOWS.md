# Visual Flow Diagrams: Parameter Handling

## Complete Parameter Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         model_config.json                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  {                                                                             │
│    "planner": {                                                               │
│      "model_name": "gpt-4o",                                                 │
│      "category": "openai",                                                   │
│      "temperature": 0.3,                    (1) Shared Param                 │
│      "max_output_tokens": 1024,             (2) Mapped Param               │
│      "stop_sequences": ["END"],             (2) Mapped Param                │
│      "reasoning_effort": "low",             (3) OpenAI-Specific             │
│      "custom_param": "value"                (4) Unknown Param                │
│    }                                                                          │
│  }                                                                            │
└────────────────┬─────────────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │  category: "openai"?           │
        │  Yes → Use extract_openai_params
        │  No → Use extract_gemini_params
        └────────┬───────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ FILTER 1: Shared Params       │
        │ temperature, top_p, seed...   │
        │ ✅ temperature: 0.3 → ADDED  │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ FILTER 2: Mapped Params       │
        │ max_output_tokens → ???       │
        │ stop_sequences → ???          │
        │ (Maps to provider-specific)   │
        │ ✅ max_tokens: 1024 → ADDED  │
        │ ✅ stop: ["END"] → ADDED     │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ FILTER 3: Provider-Specific   │
        │ reasoning_effort (OpenAI only)│
        │ ✅ reasoning_effort → ADDED   │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ FILTER 4: Pass-Through        │
        │ Unknown params + safety checks│
        │ ✅ custom_param → ADDED       │
        └────────┬──────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │ EXTRACTED PARAMS:              │
        │ {                              │
        │   temperature: 0.3,            │
        │   max_tokens: 1024,            │
        │   stop: ["END"],               │
        │   reasoning_effort: "low",     │
        │   custom_param: "value"        │
        │ }                              │
        └────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │ ChatOpenAI.__init__(**params)  │
        └────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │ OpenAI API Call                │
        │ (with correct param names)     │
        └────────────────────────────────┘
```

---

## Comparison: Standardized vs. Direct Names

### ✅ CORRECT: Using Standardized Names

```
User's JSON              Extractor              API Params              Result
────────────────────────────────────────────────────────────────────────────────
max_output_tokens:1024   ──map──→  max_tokens:1024   ──→  ✅ OpenAI API accepts
(standardized)           (MAPPED)   (provider-specific)

stop_sequences:["END"]   ──map──→  stop:["END"]      ──→  ✅ OpenAI API accepts
(standardized)           (MAPPED)   (provider-specific)

enable_logprobs:false    ──map──→  logprobs:false    ──→  ✅ OpenAI API accepts
(standardized)           (MAPPED)   (provider-specific)
```

### ❌ WRONG: Using Direct Provider Names (OpenAI)

```
User's JSON              Extractor              What Happens           Result
────────────────────────────────────────────────────────────────────────────────
max_tokens:1024          ──pass──→  max_tokens:1024   ──→  ✅ Works (leaked through)
(provider-specific)      (UNKNOWN)   (as-is)

stop:["END"]             ──✂️───→  FILTERED OUT       ──→  ❌ Silent failure!
(provider-specific)      (SAFETY)

logprobs:false           ──✂️───→  FILTERED OUT       ──→  ❌ Silent failure!
(provider-specific)      (SAFETY)
```

### ⚠️ RISKY: Using Direct Provider Names (Gemini)

```
User's JSON              Extractor              What Happens           Result
────────────────────────────────────────────────────────────────────────────────
maxOutputTokens:2048     ──pass──→  maxOutputTokens   ──→  ✅ Works (passed through)
(provider-specific)      (UNKNOWN)   (as-is)

stopSequences:["END"]    ──pass──→  stopSequences     ──→  ✅ Works (passed through)
(provider-specific)      (UNKNOWN)   (as-is)

responseLogprobs:true    ──pass──→  responseLogprobs  ──→  ✅ Works (passed through)
(provider-specific)      (UNKNOWN)   (as-is)
```

---

## Safety Filters Detail

### What Each Filter Does

```
┌─────────────────────────────────────────────────────────────────┐
│ FILTER 1: SHARED_PARAMS                                         │
├─────────────────────────────────────────────────────────────────┤
│ Check if key in {'temperature', 'top_p', 'presence_penalty',   │
│                  'frequency_penalty', 'seed'}                   │
│ If yes → Add to params AS-IS (same name for all providers)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ FILTER 2: MAPPED_PARAMS                                         │
├─────────────────────────────────────────────────────────────────┤
│ Check if key in standardized names like:                        │
│   'max_output_tokens', 'stop_sequences', 'enable_logprobs',     │
│   'num_logprobs'                                                │
│ If yes → Map to provider-specific name (max_tokens for OpenAI)  │
│ If no → Continue to next filter                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ FILTER 3: PROVIDER_SPECIFIC_PARAMS                              │
├─────────────────────────────────────────────────────────────────┤
│ OpenAI: reasoning_effort, response_format, logit_bias           │
│ Gemini: top_k, thinking_budget, safety_settings, etc.           │
│ If yes → Add to params AS-IS                                    │
│ If no → Continue to next filter                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ FILTER 4: PASS-THROUGH SAFETY                                   │
├─────────────────────────────────────────────────────────────────┤
│ Check if key is safe to pass through (not in any of above)      │
│ AND not in dangerous sets:                                       │
│   - mapped_provider_names (e.g., 'max_tokens', 'stop')          │
│   - MASAI_SPECIFIC_PARAMS (e.g., 'memory_store', 'k')           │
│   - Other provider's specific params                             │
│   - Keys starting with "openai_", "gemini_", "//"               │
│ If all checks pass → Add to params (forward compatibility)      │
│ If any check fails → Filter OUT (safety)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Extraction Flow: OpenAI vs Gemini

### OpenAI Extract Flow

```
Input kwargs
     │
     ▼
┌─────────────────────────┐
│ Shared params?          │
│ temperature, top_p, etc.│
└──────┬──────────────────┘
       │ YES → add to params
       │ NO  → continue
       ▼
┌─────────────────────────┐
│ Mapped params?          │
│ max_output_tokens, etc. │
└──────┬──────────────────┘
       │ YES → map to max_tokens
       │ NO  → continue
       ▼
┌─────────────────────────┐
│ OpenAI-specific?        │
│ reasoning_effort, etc.  │
└──────┬──────────────────┘
       │ YES → add to params
       │ NO  → continue
       ▼
┌─────────────────────────┐
│ Apply constraints:      │
│ Drop top_logprobs if    │
│ logprobs=false          │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Pass-through check:     │
│ NOT IN:                 │
│ - mapped_provider_names │
│ - MAPPED_PARAMS         │
│ - GEMINI_SPECIFIC       │
│ - MASAI_SPECIFIC        │
└──────┬──────────────────┘
       │ PASS → add to params
       │ FAIL → filter out
       ▼
Extracted params ready for API
```

### Gemini Extract Flow

```
Input kwargs
     │
     ▼
┌─────────────────────────┐
│ Shared params?          │
│ temperature, top_p, etc.│
└──────┬──────────────────┘
       │ YES → add to params
       │ NO  → continue
       ▼
┌─────────────────────────┐
│ Mapped params?          │
│ max_output_tokens, etc. │
└──────┬──────────────────┘
       │ YES → map to maxOutputTokens
       │ NO  → continue
       ▼
┌─────────────────────────┐
│ Gemini-specific?        │
│ top_k, thinking_budget  │
└──────┬──────────────────┘
       │ YES → add to params
       │ NO  → continue
       ▼
┌─────────────────────────┐
│ Apply constraints:      │
│ Drop logprobs if        │
│ response_logprobs=false │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Pass-through check:     │
│ NOT IN:                 │
│ - mapped_provider_names │
│ - MAPPED_PARAMS         │
│ - OPENAI_SPECIFIC*      │
│ - MASAI_SPECIFIC        │
│                         │
│ *Only checks explicit   │
│  OpenAI features, not   │
│  OpenAI mapped names!   │
└──────┬──────────────────┘
       │ PASS → add to params
       │ FAIL → filter out
       ▼
Extracted params ready for API
```

**KEY DIFFERENCE**: Gemini doesn't filter out OpenAI mapped names like `max_tokens`, `stop`, `logprobs` → causes contamination

---

## Risk Matrix: Parameter Type vs. Provider

```
                      OpenAI Provider      Gemini Provider
                      ─────────────────    ─────────────────
Standardized Names    ✅ SAFE              ✅ SAFE
                      All work              All work

Direct OpenAI Names   ⚠️ PARTIAL           ❌ RISKY
                      Some work             Leak through
                      Some filtered         Not filtered

Direct Gemini Names   ✅ WORKS             ✅ SAFE
                      Passed through        All work

Mixed Standardized    ⚠️ PARTIAL           ⚠️ PARTIAL
+ Provider Names      Std work             Std work
                      Direct fail          Direct work but mixed

Cross-Provider Names  ❌ CONTAMINATION     ❌ CONTAMINATION
(OpenAI in Gemini)    (some leak)          (more leak)
```

---

## The Bottom Line

```
  ALWAYS                NEVER
  ────────────────────────────────────────────
  ✅ Use standardized    ❌ Use provider-specific
     names only             names directly

  ✅ Use these names:     ❌ Don't use these:
     - max_output_tokens    - max_tokens
     - stop_sequences       - stop
     - enable_logprobs      - logprobs
     - num_logprobs         - top_logprobs
     - temperature          - maxOutputTokens
     - top_p               - stopSequences
     - etc.                - responseLogprobs

  ✅ Framework handles    ❌ You get silent
     mapping safely          failures or
                             contamination
```

