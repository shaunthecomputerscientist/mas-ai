"""
Parameter Configuration and Mapping for MASAI Framework

This module defines standardized parameter names and provides mapping functions
to convert between MASAI's standardized names and provider-specific names.

Design Philosophy:
1. Users configure models using standardized parameter names
2. Provider-specific parameters are prefixed (e.g., gemini_*, openai_*)
3. Automatic filtering prevents passing incompatible parameters to APIs
4. Automatic mapping converts standardized names to provider-specific names

Parameter Categories:
- SHARED: Same name and type for both providers
- MAPPED: Different names between providers, need mapping
- PROVIDER_SPECIFIC: Only supported by one provider
"""

from typing import Dict, Any, Set, Optional
import logging

# Get logger
logger = logging.getLogger(__name__)

# ============================================================================
# SHARED PARAMETERS (Same name, same type for both providers)
# ============================================================================
SHARED_PARAMS: Set[str] = {
    'temperature',      # float (0.0-2.0) - Controls randomness
    'top_p',           # float (0.0-1.0) - Nucleus sampling
    'presence_penalty', # float (-2.0 to 2.0) - Penalize repeated tokens
    'frequency_penalty',# float (-2.0 to 2.0) - Penalize frequent tokens
    'seed',            # int - Deterministic output
}

# ============================================================================
# MAPPED PARAMETERS (Different names between providers)
# ============================================================================
# Format: standardized_name -> {provider: provider_specific_name}
MAPPED_PARAMS: Dict[str, Dict[str, str]] = {
    # Output length control
    'max_output_tokens': {
        'openai': 'max_tokens',           # OpenAI uses 'max_tokens'
        'gemini': 'max_output_tokens'     # Gemini uses 'max_output_tokens'
    },
    
    # Stop sequences
    'stop_sequences': {
        'openai': 'stop',                 # OpenAI uses 'stop' (str or List[str])
        'gemini': 'stop_sequences'        # Gemini uses 'stop_sequences' (List[str])
    },
    
    # Log probabilities (enable/disable)
    'enable_logprobs': {
        'openai': 'logprobs',             # OpenAI: bool to enable/disable
        'gemini': 'response_logprobs'     # Gemini: bool to enable/disable
    },
    
    # Log probabilities (number of tokens)
    'num_logprobs': {
        'openai': 'top_logprobs',         # OpenAI: int (0-20) for number of tokens
        'gemini': 'logprobs'              # Gemini: int (1-20) for number of tokens
    },
}

# ============================================================================
# PROVIDER-SPECIFIC PARAMETERS
# ============================================================================
# These parameters are ONLY supported by specific providers
# They should be prefixed in user config (e.g., gemini_top_k, openai_reasoning_effort)

# Gemini-only parameters (remove 'gemini_' prefix before passing to API)
GEMINI_ONLY_PARAMS: Set[str] = {
    'gemini_top_k',                    # int (1-100) - Top-k sampling
    'gemini_safety_settings',          # List[Dict] - Safety category/threshold pairs
    'gemini_thinking_budget',          # int (-1=dynamic, 0=off, 1-10000=tokens)
    'gemini_candidate_count',          # int (1-8) - Number of response variations
}

# OpenAI-only parameters (remove 'openai_' prefix before passing to API)
OPENAI_ONLY_PARAMS: Set[str] = {
    'openai_reasoning_effort',         # str ("low", "medium", "high") - For reasoning models
}

# ============================================================================
# PARAMETER EXTRACTION AND MAPPING FUNCTIONS
# ============================================================================

def extract_gemini_params(kwargs: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Extract and map parameters for Gemini API.
    
    Process:
    1. Add shared parameters (no mapping needed)
    2. Map standardized names to Gemini-specific names
    3. Add Gemini-specific parameters (remove prefix)
    4. Filter out OpenAI-only parameters
    
    Args:
        kwargs: Dictionary of all parameters from user config
        verbose: Whether to log parameter extraction
        
    Returns:
        Dictionary of parameters ready for Gemini API
    """
    params = {}
    
    # Step 1: Add shared parameters (no mapping needed)
    for key in SHARED_PARAMS:
        if key in kwargs:
            params[key] = kwargs[key]
            if verbose:
                logger.debug(f"✅ Gemini: Added shared parameter '{key}' = {kwargs[key]}")
    
    # Step 2: Map standardized names to Gemini-specific names
    for std_name, provider_names in MAPPED_PARAMS.items():
        if std_name in kwargs:
            gemini_name = provider_names['gemini']
            params[gemini_name] = kwargs[std_name]
            if verbose:
                logger.debug(f"✅ Gemini: Mapped '{std_name}' → '{gemini_name}' = {kwargs[std_name]}")
    
    # Step 3: Add Gemini-specific parameters (remove 'gemini_' prefix)
    for prefixed_key in GEMINI_ONLY_PARAMS:
        if prefixed_key in kwargs:
            # Remove 'gemini_' prefix to get actual parameter name
            param_name = prefixed_key.replace('gemini_', '')
            params[param_name] = kwargs[prefixed_key]
            if verbose:
                logger.debug(f"✅ Gemini: Added provider-specific parameter '{param_name}' = {kwargs[prefixed_key]}")
    
    # Step 4: Warn about OpenAI-only parameters (they will be filtered out)
    openai_params_found = [key for key in kwargs if key in OPENAI_ONLY_PARAMS]
    if openai_params_found and verbose:
        logger.warning(f"⚠️ Gemini: OpenAI-only parameters {openai_params_found} found in config. These will be ignored by Gemini API.")
    
    # Step 5: Add backward compatibility for model_kwargs
    if 'model_kwargs' in kwargs:
        params['model_kwargs'] = kwargs['model_kwargs']
        if verbose:
            logger.debug(f"✅ Gemini: Added model_kwargs for backward compatibility")
    
    return params


def extract_openai_params(kwargs: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Extract and map parameters for OpenAI API.
    
    Process:
    1. Add shared parameters (no mapping needed)
    2. Map standardized names to OpenAI-specific names
    3. Add OpenAI-specific parameters (remove prefix)
    4. Filter out Gemini-only parameters (CRITICAL: prevents 400 errors)
    
    Args:
        kwargs: Dictionary of all parameters from user config
        verbose: Whether to log parameter extraction
        
    Returns:
        Dictionary of parameters ready for OpenAI API
    """
    params = {}
    
    # Step 1: Add shared parameters (no mapping needed)
    for key in SHARED_PARAMS:
        if key in kwargs:
            params[key] = kwargs[key]
            if verbose:
                logger.debug(f"✅ OpenAI: Added shared parameter '{key}' = {kwargs[key]}")
    
    # Step 2: Map standardized names to OpenAI-specific names
    for std_name, provider_names in MAPPED_PARAMS.items():
        if std_name in kwargs:
            openai_name = provider_names['openai']
            params[openai_name] = kwargs[std_name]
            if verbose:
                logger.debug(f"✅ OpenAI: Mapped '{std_name}' → '{openai_name}' = {kwargs[std_name]}")
    
    # Step 3: Add OpenAI-specific parameters (remove 'openai_' prefix)
    for prefixed_key in OPENAI_ONLY_PARAMS:
        if prefixed_key in kwargs:
            # Remove 'openai_' prefix to get actual parameter name
            param_name = prefixed_key.replace('openai_', '')
            params[param_name] = kwargs[prefixed_key]
            if verbose:
                logger.debug(f"✅ OpenAI: Added provider-specific parameter '{param_name}' = {kwargs[prefixed_key]}")
    
    # Step 4: CRITICAL - Warn about Gemini-only parameters (they MUST be filtered out)
    gemini_params_found = [key for key in kwargs if key in GEMINI_ONLY_PARAMS]
    if gemini_params_found:
        if verbose:
            logger.warning(f"🔴 OpenAI: Gemini-only parameters {gemini_params_found} found in config. These are FILTERED OUT to prevent 400 errors.")
        # These are NOT added to params - OpenAI will reject them with 400 error
    
    # Step 5: Add backward compatibility for model_kwargs
    if 'model_kwargs' in kwargs:
        params['model_kwargs'] = kwargs['model_kwargs']
        if verbose:
            logger.debug(f"✅ OpenAI: Added model_kwargs for backward compatibility")
    
    return params


def validate_parameters(kwargs: Dict[str, Any], category: str) -> None:
    """
    Validate parameters and warn about potential issues.
    
    Args:
        kwargs: Dictionary of all parameters
        category: Model category ('openai' or 'gemini')
    """
    if 'openai' in category:
        # Check for Gemini-only parameters
        gemini_params = [key for key in kwargs if key in GEMINI_ONLY_PARAMS]
        if gemini_params:
            logger.warning(
                f"⚠️ Configuration contains Gemini-only parameters {gemini_params} "
                f"but model category is '{category}'. These will be filtered out to prevent API errors."
            )
    
    elif 'gemini' in category:
        # Check for OpenAI-only parameters
        openai_params = [key for key in kwargs if key in OPENAI_ONLY_PARAMS]
        if openai_params:
            logger.info(
                f"ℹ️ Configuration contains OpenAI-only parameters {openai_params} "
                f"but model category is '{category}'. These will be ignored by Gemini API."
            )


def get_all_supported_params() -> Dict[str, Set[str]]:
    """
    Get all supported parameters for each provider.
    
    Returns:
        Dictionary mapping provider to set of supported parameter names
    """
    # Shared parameters
    shared = SHARED_PARAMS.copy()
    
    # Mapped parameters (standardized names)
    mapped = set(MAPPED_PARAMS.keys())
    
    # Provider-specific parameters (with prefixes)
    gemini_specific = GEMINI_ONLY_PARAMS.copy()
    openai_specific = OPENAI_ONLY_PARAMS.copy()
    
    return {
        'shared': shared,
        'mapped': mapped,
        'gemini_only': gemini_specific,
        'openai_only': openai_specific,
        'all_standardized': shared | mapped | gemini_specific | openai_specific
    }


def print_parameter_guide():
    """Print a user-friendly guide to parameter naming."""
    print("\n" + "="*80)
    print("MASAI PARAMETER NAMING GUIDE")
    print("="*80)
    
    print("\n📋 SHARED PARAMETERS (Work for both OpenAI and Gemini):")
    for param in sorted(SHARED_PARAMS):
        print(f"  • {param}")
    
    print("\n🔄 MAPPED PARAMETERS (Standardized names, auto-mapped to provider-specific):")
    for std_name, providers in MAPPED_PARAMS.items():
        print(f"  • {std_name}")
        print(f"    → OpenAI: {providers['openai']}")
        print(f"    → Gemini: {providers['gemini']}")
    
    print("\n🔵 GEMINI-ONLY PARAMETERS (Prefix with 'gemini_'):")
    for param in sorted(GEMINI_ONLY_PARAMS):
        actual_name = param.replace('gemini_', '')
        print(f"  • {param} → {actual_name}")
    
    print("\n🟢 OPENAI-ONLY PARAMETERS (Prefix with 'openai_'):")
    for param in sorted(OPENAI_ONLY_PARAMS):
        actual_name = param.replace('openai_', '')
        print(f"  • {param} → {actual_name}")
    
    print("\n" + "="*80)
    print("✅ Use standardized names in model_config.json")
    print("✅ Prefix provider-specific parameters (e.g., gemini_top_k)")
    print("✅ MASAI automatically filters and maps parameters")
    print("="*80 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Print parameter guide
    print_parameter_guide()
    
    # Example: Extract parameters for Gemini
    example_config = {
        'temperature': 0.7,
        'max_output_tokens': 2048,
        'top_p': 0.95,
        'stop_sequences': ['STOP!', 'END'],
        'enable_logprobs': True,
        'num_logprobs': 5,
        'gemini_top_k': 20,
        'gemini_safety_settings': [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'}
        ],
        'gemini_thinking_budget': -1,
        'openai_reasoning_effort': 'medium',  # Will be filtered out for Gemini
    }
    
    print("\n📦 Example Config:")
    for k, v in example_config.items():
        print(f"  {k}: {v}")
    
    print("\n🔵 Extracted Gemini Parameters:")
    gemini_params = extract_gemini_params(example_config, verbose=True)
    for k, v in gemini_params.items():
        print(f"  {k}: {v}")
    
    print("\n🟢 Extracted OpenAI Parameters:")
    openai_params = extract_openai_params(example_config, verbose=True)
    for k, v in openai_params.items():
        print(f"  {k}: {v}")

