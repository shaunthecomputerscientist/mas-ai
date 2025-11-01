"""
Parameter Configuration and Mapping for MASAI Framework

This module provides centralized parameter management for AI model providers (Gemini, OpenAI).
It standardizes parameter naming, handles provider-specific mappings, and filters out
framework-specific parameters before passing to model APIs.

KEY FEATURES:
─────────────
1. **Standardized Parameter Names**: Users specify parameters using consistent names
   (e.g., 'max_output_tokens', 'stop_sequences') regardless of provider.

2. **Automatic Provider Mapping**: Parameters are automatically mapped to provider-specific
   names (e.g., 'max_output_tokens' → 'max_completion_tokens' for OpenAI reasoning models).

3. **Provider-Specific Parameters**: Each provider has exclusive parameters that are
   automatically filtered for other providers (e.g., 'thinking_budget' for Gemini only).

4. **Framework Parameter Filtering**: MASAI-specific parameters (memory_store, k, etc.)
   are automatically filtered out before passing to model APIs.

5. **Future-Ready Design**: Unknown parameters are passed through untouched for
   forward compatibility with new provider features.

PARAMETER CATEGORIES:
─────────────────────
- SHARED_PARAMS: Work identically across all providers
- MAPPED_PARAMS: Different names per provider, same concept
- GEMINI_SPECIFIC_PARAMS: Gemini-only features
- OPENAI_SPECIFIC_PARAMS: OpenAI-only features
- MASAI_SPECIFIC_PARAMS: Framework-specific, filtered before API calls

USAGE EXAMPLE:
──────────────
In model_config.json:
    {
        "router": {
            "model_name": "gemini-2.5-pro",
            "category": "gemini",
            "temperature": 0.7,
            "max_output_tokens": 2048,
            "thinking_budget": -1,
            "top_p": 0.95
        }
    }

The framework automatically:
1. Extracts these parameters
2. Maps them to provider-specific names
3. Filters out unsupported parameters
4. Passes them to the model wrapper
"""

from typing import Dict, Any, Set
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SHARED PARAMETERS (Same name, same type for both providers)
# ============================================================================
# These parameters work identically across all providers with the same name and type.
SHARED_PARAMS: Set[str] = {
    # Sampling temperature (0.0-2.0)
    # Controls randomness: 0.0 = deterministic, 2.0 = maximum randomness
    'temperature',

    # Top-p (nucleus) sampling (0.0-1.0)
    # Cumulative probability threshold for token selection
    'top_p',

    # Presence penalty (-2.0 to 2.0)
    # Penalizes tokens that have already appeared in the response
    'presence_penalty',

    # Frequency penalty (-2.0 to 2.0)
    # Penalizes tokens based on their frequency in the response
    'frequency_penalty',

    # Random seed for deterministic output
    # Same seed produces same output (when temperature=0)
    'seed',
}

# ============================================================================
# MAPPED PARAMETERS (Different key names across providers)
# ============================================================================
# These parameters have different names across providers but represent the same concept.
# Users specify them using the standardized name, and they're automatically mapped.
MAPPED_PARAMS: Dict[str, Dict[str, str]] = {
    # Maximum output tokens
    # - OpenAI: max_completion_tokens (for reasoning models) or max_tokens (standard)
    # - Gemini: maxOutputTokens
    'max_output_tokens': {
        'openai': 'max_completion_tokens',  # Reasoning models use this
        'gemini': 'max_output_tokens',
    },

    # Stop sequences for generation
    # - OpenAI: stop (string or list of strings)
    # - Gemini: stopSequences (list of strings)
    'stop_sequences': {
        'openai': 'stop',
        'gemini': 'stop_sequences',
    },

    # Enable log probabilities
    # - OpenAI: logprobs (boolean)
    # - Gemini: responseLogprobs (boolean)
    'enable_logprobs': {
        'openai': 'logprobs',
        'gemini': 'response_logprobs',
    },

    # Number of log probabilities to return
    # - OpenAI: top_logprobs (int, 0-20)
    # - Gemini: logprobs (int, 0-20)
    'num_logprobs': {
        'openai': 'top_logprobs',
        'gemini': 'logprobs',
    },
}

# ============================================================================
# PROVIDER-SPECIFIC PARAMETERS (Unprefixed and future-ready)
# ============================================================================

# Gemini-only supported params
# - top_k: int - Top-k sampling (Gemini only)
# - safety_settings: list - Safety filter settings
# - thinking_budget: int - Thinking token budget for thinking models (-1 for dynamic)
# - max_thinking_tokens: int - Maximum thinking tokens
# - min_thinking_tokens: int - Minimum thinking tokens
# - candidate_count: int - Number of response candidates
GEMINI_SPECIFIC_PARAMS: Set[str] = {
    'top_k',
    'safety_settings',
    'thinking_budget',
    'max_thinking_tokens',
    'min_thinking_tokens',
    'candidate_count',
}

# OpenAI-only supported params
# - reasoning_effort: str - Reasoning effort level (low/medium/high) for reasoning models
# - response_format: dict - Response format specification (json_schema, etc.)
# - logit_bias: dict - Token logit bias adjustments
OPENAI_SPECIFIC_PARAMS: Set[str] = {
    'reasoning_effort',
    'response_format',
    'logit_bias',
}

# ============================================================================
# MASAI-SPECIFIC PARAMETERS (Should be filtered out before passing to LLM)
# ============================================================================
# These parameters are MASAI framework-specific and should NOT be passed to model APIs.
# They are filtered out during parameter extraction.
MASAI_SPECIFIC_PARAMS: Set[str] = {
    # InMemoryDocStore instance for MASAI memory management (DEPRECATED: use qdrant_config)
    'memory_store',

    # Number of top results to retrieve from memory store (used for Qdrant retrieval)
    'k',

    # MASAI-specific content capping limit for tool outputs
    'content_capping_limit',

    # Model description from config (metadata only, not for API)
    'description',

    # Persistent memory / Long-term memory integration (framework-level only)
    'user_id',  # User ID for filtering memories in long-term memory
    'memory_config',  # Configuration for persistent long-term memory backend
    'backend_config',  # [DEPRECATED] Use memory_config instead. Backend configuration for persistent memory
    'qdrant_config',  # [DEPRECATED] Use memory_config instead. Legacy Qdrant configuration for persistent memory
    'long_term_memory',  # Shared LongTermMemory instance from AgentManager
    'persist_memory',  # Enable/disable persistent memory per component
    'categories_resolver',  # Function to extract categories from documents for filtering
    'callable_config',  # Configuration for context callables
}

# Parameters that start with these prefixes are comments/metadata and should be filtered
# These are typically found in JSON config files as documentation
COMMENT_PREFIXES = ('// ', '/*', '*/')

# ============================================================================
# EXTRACTOR FUNCTIONS
# ============================================================================

def extract_gemini_params(kwargs: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Extract and normalize parameters for Gemini models.

    This function processes raw kwargs and returns only parameters supported by Gemini:
    1. Adds shared parameters (temperature, top_p, etc.)
    2. Maps standardized names to Gemini-specific names
    3. Adds Gemini-specific parameters (top_k, thinking_budget, etc.)
    4. Filters out OpenAI-only parameters
    5. Filters out MASAI framework parameters
    6. Passes through unknown parameters for forward compatibility

    Args:
        kwargs: Raw parameters dict (typically from model_config.json)
        verbose: Enable debug logging

    Returns:
        Dict of parameters ready for ChatGoogleGenerativeAI initialization
    """
    params = {}

    # 1️⃣ Add shared parameters
    for key in SHARED_PARAMS:
        if key in kwargs:
            params[key] = kwargs[key]
            if verbose:
                logger.debug(f"✅ Gemini: Added shared '{key}' = {kwargs[key]}")

    # 2️⃣ Map standardized parameters
    for std_name, mapping in MAPPED_PARAMS.items():
        if std_name in kwargs:
            params[mapping['gemini']] = kwargs[std_name]
            if verbose:
                logger.debug(f"✅ Gemini: Mapped '{std_name}' → '{mapping['gemini']}'")

    # 3️⃣ Add Gemini-specific parameters
    for key in GEMINI_SPECIFIC_PARAMS:
        if key in kwargs:
            params[key] = kwargs[key]
            if verbose:
                logger.debug(f"✅ Gemini: Added specific '{key}' = {kwargs[key]}")

    # 4️⃣ Pass through unknown (future) parameters safely
    for key, value in kwargs.items():
        # Skip comment fields and MASAI-specific params
        if any(key.startswith(prefix) for prefix in COMMENT_PREFIXES):
            continue

        if (
            key not in params
            and key not in OPENAI_SPECIFIC_PARAMS
            and key not in MASAI_SPECIFIC_PARAMS
            and not key.startswith("openai_")
        ):  
            params[key] = value
            if verbose:
                logger.debug(f"🪶 Gemini: Passed through custom param '{key}'")

    # 5️⃣ Handle backward-compatible model_kwargs
    if 'model_kwargs' in kwargs:
        params['model_kwargs'] = kwargs['model_kwargs']

    return params


def extract_openai_params(kwargs: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Extract and normalize parameters for OpenAI models.

    This function processes raw kwargs and returns only parameters supported by OpenAI:
    1. Adds shared parameters (temperature, top_p, etc.)
    2. Maps standardized names to OpenAI-specific names
    3. Adds OpenAI-specific parameters (reasoning_effort, response_format, etc.)
    4. Filters out Gemini-only parameters
    5. Filters out MASAI framework parameters
    6. Passes through unknown parameters for forward compatibility

    Args:
        kwargs: Raw parameters dict (typically from model_config.json)
        verbose: Enable debug logging

    Returns:
        Dict of parameters ready for ChatOpenAI initialization
    """
    params = {}

    # 1️⃣ Add shared parameters
    for key in SHARED_PARAMS:
        if key in kwargs:
            params[key] = kwargs[key]
            if verbose:
                logger.debug(f"✅ OpenAI: Added shared '{key}' = {kwargs[key]}")

    # 2️⃣ Map standardized parameters
    for std_name, mapping in MAPPED_PARAMS.items():
        if std_name in kwargs:
            params[mapping['openai']] = kwargs[std_name]
            if verbose:
                logger.debug(f"✅ OpenAI: Mapped '{std_name}' → '{mapping['openai']}'")

    # 3️⃣ Add OpenAI-specific parameters
    for key in OPENAI_SPECIFIC_PARAMS:
        if key in kwargs:
            params[key] = kwargs[key]
            if verbose:
                logger.debug(f"✅ OpenAI: Added specific '{key}' = {kwargs[key]}")

    # 4️⃣ Pass through unknown (future) parameters safely
    for key, value in kwargs.items():
        # Skip comment fields and MASAI-specific params
        if any(key.startswith(prefix) for prefix in COMMENT_PREFIXES):
            continue

        if (
            key not in params
            and key not in GEMINI_SPECIFIC_PARAMS
            and key not in MASAI_SPECIFIC_PARAMS
            and not key.startswith("gemini_")
        ):
            params[key] = value
            if verbose:
                logger.debug(f"🪶 OpenAI: Passed through custom param '{key}'")

    # 5️⃣ Handle backward-compatible model_kwargs
    if 'model_kwargs' in kwargs:
        params['model_kwargs'] = kwargs['model_kwargs']

    return params


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_parameters(kwargs: Dict[str, Any], category: str) -> None:
    """Warn about incompatible parameters across providers."""
    if 'openai' in category:
        gemini_conflicts = [k for k in kwargs if k in GEMINI_SPECIFIC_PARAMS]
        if gemini_conflicts:
            logger.warning(f"⚠️ Found Gemini-only params {gemini_conflicts} for OpenAI model.")

    elif 'gemini' in category:
        openai_conflicts = [k for k in kwargs if k in OPENAI_SPECIFIC_PARAMS]
        if openai_conflicts:
            logger.info(f"ℹ️ Found OpenAI-only params {openai_conflicts}, ignored for Gemini.")


def get_all_supported_params() -> Dict[str, Set[str]]:
    """Return all known parameter categories."""
    return {
        'shared': SHARED_PARAMS,
        'mapped': set(MAPPED_PARAMS.keys()),
        'gemini_only': GEMINI_SPECIFIC_PARAMS,
        'openai_only': OPENAI_SPECIFIC_PARAMS,
        'all_standardized': SHARED_PARAMS | set(MAPPED_PARAMS.keys()) | GEMINI_SPECIFIC_PARAMS | OPENAI_SPECIFIC_PARAMS,
    }



# def print_parameter_guide():
#     """Print a user-friendly guide to parameter naming."""
#     print("\n" + "="*80)
#     print("MASAI PARAMETER NAMING GUIDE")
#     print("="*80)
    
#     print("\n📋 SHARED PARAMETERS (Work for both OpenAI and Gemini):")
#     for param in sorted(SHARED_PARAMS):
#         print(f"  • {param}")
    
#     print("\n🔄 MAPPED PARAMETERS (Standardized names, auto-mapped to provider-specific):")
#     for std_name, providers in MAPPED_PARAMS.items():
#         print(f"  • {std_name}")
#         print(f"    → OpenAI: {providers['openai']}")
#         print(f"    → Gemini: {providers['gemini']}")
    
#     print("\n🔵 GEMINI-ONLY PARAMETERS (Prefix with 'gemini_'):")
#     for param in sorted(GEMINI_ONLY_PARAMS):
#         actual_name = param.replace('gemini_', '')
#         print(f"  • {param} → {actual_name}")
    
#     print("\n🟢 OPENAI-ONLY PARAMETERS (Prefix with 'openai_'):")
#     for param in sorted(OPENAI_ONLY_PARAMS):
#         actual_name = param.replace('openai_', '')
#         print(f"  • {param} → {actual_name}")
    
#     print("\n" + "="*80)
#     print("✅ Use standardized names in model_config.json")
#     print("✅ Prefix provider-specific parameters (e.g., gemini_top_k)")
#     print("✅ MASAI automatically filters and maps parameters")
#     print("="*80 + "\n")


# # ============================================================================
# # EXAMPLE USAGE
# # ============================================================================
# if __name__ == "__main__":
#     # Print parameter guide
#     print_parameter_guide()
    
#     # Example: Extract parameters for Gemini
#     example_config = {
#         'temperature': 0.7,
#         'max_output_tokens': 2048,
#         'top_p': 0.95,
#         'stop_sequences': ['STOP!', 'END'],
#         'enable_logprobs': True,
#         'num_logprobs': 5,
#         'gemini_top_k': 20,
#         'gemini_safety_settings': [
#             {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'}
#         ],
#         'gemini_thinking_budget': -1,
#         'openai_reasoning_effort': 'medium',  # Will be filtered out for Gemini
#     }
    
#     print("\n📦 Example Config:")
#     for k, v in example_config.items():
#         print(f"  {k}: {v}")
    
#     print("\n🔵 Extracted Gemini Parameters:")
#     gemini_params = extract_gemini_params(example_config, verbose=True)
#     for k, v in gemini_params.items():
#         print(f"  {k}: {v}")
    
#     print("\n🟢 Extracted OpenAI Parameters:")
#     openai_params = extract_openai_params(example_config, verbose=True)
#     for k, v in openai_params.items():
#         print(f"  {k}: {v}")

