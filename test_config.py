#!/usr/bin/env python3
"""
Configuration file for MiniRAG testing
Easy to modify for different test scenarios
"""

# Test Configuration for MiniRAG Light vs Mini Mode Comparison
# This file centralizes all test parameters for easy modification

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================
SERVER_URL = "http://localhost:9721"
SERVER_TIMEOUT = 60  # seconds

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Base log directory
LOG_DIR = Path("./test_logs")

# Log file settings
LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
DEBUG_TO_PRIMARY_ONLY = False  # Since you're only using local directory

# ============================================================================
# TEST QUERIES
# ============================================================================
# Test Queries (modify these for different tests)
TEST_QUERIES = [
    "What were some of Kalpana Chawla's hobbies?",
    "Explain about Kalpana Chawla's husband",
    "Explain Kalpana Chawla's educational background",
    "When did Kalpana Chawla start working at NASA?"
]

# ============================================================================
# PDF SELECTION
# ============================================================================
# PDF Selection (modify to test with different PDFs)
# Options: "random", "all", or specific PDF names
PDF_SELECTION = ["kalpana_chawla_nasa.pdf"]  # Focus on the shortest PDF

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================
EXPECTED_RESPONSE_TIME = 5.0  # seconds
MIN_RESPONSE_LENGTH = 100  # characters

# ============================================================================
# RESPONSE STORAGE
# ============================================================================
# Base directory for storing test responses
RESPONSE_BASE_DIR = Path("./light_mini_comparison")

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
# Load from .env file without exposing values
def get_env_var(key, default=None):
    """Get environment variable safely"""
    return os.getenv(key, default)

# API keys and configuration (loaded from .env)
LLM_API_KEY = get_env_var("LLM_BINDING_API_KEY")
EMBEDDING_API_KEY = get_env_var("EMBEDDING_BINDING_API_KEY")
LLM_MODEL = get_env_var("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = get_env_var("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_BINDING = get_env_var("LLM_BINDING", "openai")
EMBEDDING_BINDING = get_env_var("EMBEDDING_BINDING", "openai")

# RAG Configuration from .env
TOP_K = int(get_env_var("TOP_K", "60"))
COSINE_THRESHOLD = float(get_env_var("COSINE_THRESHOLD", "0.4"))
MAX_TOKENS = int(get_env_var("MAX_TOKENS", "32768"))
EMBEDDING_DIM = int(get_env_var("EMBEDDING_DIM", "1536"))
MAX_EMBED_TOKENS = int(get_env_var("MAX_EMBED_TOKENS", "1000"))
CHUNK_SIZE = int(get_env_var("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP_SIZE = int(get_env_var("CHUNK_OVERLAP_SIZE", "100"))

# ============================================================================
# CHUNK SIZE TESTING CONFIGURATION
# ============================================================================
# Different chunk sizes to test for performance comparison
CHUNK_SIZE_VARIANTS = [400, 800, 1200]  # Different chunk sizes to test
CHUNK_OVERLAP_VARIANTS = [50, 100, 150]  # Corresponding overlap sizes

# Default chunk configuration (used when not testing variants)
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 100

# ============================================================================
# PDF TESTING CONFIGURATION
# ============================================================================
# Configuration for testing specific PDFs
PDF_CONFIGS = {
    "kalpana_chawla_nasa.pdf": {
        "description": "NASA biography of Kalpana Chawla",
        "expected_topics": ["space", "NASA", "astronaut", "engineering"],
        "test_queries": [
            "What were some of Kalpana Chawla's hobbies?",
            "Explain about Kalpana Chawla's husband"
        ]
    },
    # Add more PDFs here as needed
    # "federal_reserve_mpr_2024.pdf": {
    #     "description": "Federal Reserve Monetary Policy Report",
    #     "expected_topics": ["economics", "monetary policy", "Federal Reserve"],
    #     "test_queries": [
    #         "What are the main economic indicators discussed?",
    #         "What monetary policy changes were announced?"
    #     ]
    # }
}

# ============================================================================
# LOGGING NAMING CONVENTION
# ============================================================================
# Log file naming pattern: miniraq_test_{pdf_name}.log
LOG_FILE_PATTERN = "miniraq_test_{pdf_name}.log"

# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================
def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = [
        "LLM_BINDING_API_KEY",
        "EMBEDDING_BINDING_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not get_env_var(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("   Make sure your .env file is properly configured")
        return False
    
    print("✅ Environment variables loaded successfully")
    return True

# ============================================================================
# CHUNK SIZE TESTING UTILITIES
# ============================================================================
def get_chunk_size_config(chunk_size, chunk_overlap=None):
    """Get configuration for a specific chunk size"""
    if chunk_overlap is None:
        # Find corresponding overlap size
        try:
            idx = CHUNK_SIZE_VARIANTS.index(chunk_size)
            chunk_overlap = CHUNK_OVERLAP_VARIANTS[idx] if idx < len(CHUNK_OVERLAP_VARIANTS) else CHUNK_OVERLAP_VARIANTS[-1]
        except ValueError:
            chunk_overlap = DEFAULT_CHUNK_OVERLAP
    
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "description": f"Chunk size {chunk_size} with overlap {chunk_overlap}"
    }

def print_chunk_size_variants():
    """Print available chunk size variants"""
    print("Available chunk size variants:")
    for i, chunk_size in enumerate(CHUNK_SIZE_VARIANTS):
        overlap = CHUNK_OVERLAP_VARIANTS[i] if i < len(CHUNK_OVERLAP_VARIANTS) else CHUNK_OVERLAP_VARIANTS[-1]
        print(f"  {chunk_size} tokens (overlap: {overlap})")

# Validate environment on import
if __name__ == "__main__":
    validate_environment()
    print_chunk_size_variants() 