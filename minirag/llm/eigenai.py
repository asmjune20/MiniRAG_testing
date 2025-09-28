#!/usr/bin/env python3
"""
EigenAI LLM implementation for MiniRAG
"""

import asyncio
import json
import os
import logging
from typing import Optional, Union, AsyncIterator
import httpx
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from minirag.utils import wrap_embedding_func_with_attrs
from minirag.llm.hf import hf_embed

# Set up logging
logger = logging.getLogger(__name__)


class EigenAIError(Exception):
    """Custom exception for EigenAI errors"""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
)
async def eigenai_complete(
    prompt: str,
    model: str = "gpt-oss-120b-f16",
    base_url: str = "http://192.222.58.254:8001",
    api_key: str = "sk-key-aalps",
    system_prompt: Optional[str] = None,
    max_tokens: int = 4000,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Complete a prompt using EigenAI
    
    Args:
        prompt: The input prompt
        model: The model to use (default: gpt-oss-120b-f16)
        base_url: EigenAI server URL
        api_key: API key for authentication
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional parameters
    
    Returns:
        The generated response text
    """
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Prepare request payload
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    # Remove hashing_kv if present (not needed for EigenAI)
    kwargs.pop("hashing_kv", None)
    
    # Add any additional parameters
    payload.update(kwargs)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Log the request
            logger.debug(f"EigenAI Request: {json.dumps(payload, indent=2)}")
            
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Log the response
            logger.debug(f"EigenAI Response: {json.dumps(result, indent=2)}")
            
            if "choices" not in result or not result["choices"]:
                raise EigenAIError("No choices in response")
            
            content = result["choices"][0]["message"]["content"]
            logger.info(f"EigenAI Content: {content}")
            
            return content
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 413:
                raise EigenAIError(f"Request too large: {e.response.text}")
            elif e.response.status_code == 429:
                raise EigenAIError(f"Rate limit exceeded: {e.response.text}")
            else:
                raise EigenAIError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise EigenAIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise EigenAIError(f"Invalid JSON response: {str(e)}")


async def eigenai_complete_if_cache(
    prompt: str,
    model: str = "gpt-oss-120b-f16",
    base_url: str = "http://192.222.58.254:8001",
    api_key: str = "sk-key-aalps",
    system_prompt: Optional[str] = None,
    max_tokens: int = 4000,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Complete a prompt using EigenAI with caching support
    (Same as eigenai_complete for now, but can be extended with caching)
    """
    return await eigenai_complete(
        prompt=prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )


# Convenience functions for specific models
async def eigenai_gpt_oss_120b_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4000,
    **kwargs
) -> str:
    """Complete using GPT-OSS 120B model"""
    return await eigenai_complete(
        prompt=prompt,
        model="gpt-oss-120b-f16",
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        **kwargs
    )


async def eigenai_llama_3_2_1b_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4000,
    **kwargs
) -> str:
    """Complete using Llama 3.2 1B model"""
    return await eigenai_complete(
        prompt=prompt,
        model="llama-3-2-1b-instruct-q3",
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        **kwargs
    )


async def eigenai_gemma_3_27b_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4000,
    **kwargs
) -> str:
    """Complete using Gemma 3 27B model"""
    return await eigenai_complete(
        prompt=prompt,
        model="gemma-3-27b-it-q4",
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        **kwargs
    )


async def eigenai_llama_3_1_8b_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4000,
    **kwargs
) -> str:
    """Complete using Llama 3.1 8B model"""
    return await eigenai_complete(
        prompt=prompt,
        model="meta-llama-3-1-8b-instruct-q3",
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        **kwargs
    )


# Embedding function for MiniRAG
@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
)
async def eigenai_embed(texts: list[str], **kwargs) -> np.ndarray:
    """
    Embedding function for MiniRAG using HuggingFace models
    This is separate from the LLM and used for vector similarity search
    """
    # For now, use a simple embedding approach
    # In production, you might want to use a proper embedding service
    import hashlib
    
    # Create simple hash-based embeddings for testing
    embeddings = []
    for text in texts:
        # Create a hash-based embedding (not ideal but works for testing)
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 1024-dimensional vector
        embedding = []
        for i in range(1024):
            byte_idx = i % len(hash_bytes)
            embedding.append(float(hash_bytes[byte_idx]) / 255.0)
        
        embeddings.append(embedding)
    
    return np.array(embeddings, dtype=np.float32)


# Test function
async def test_eigenai():
    """Test EigenAI connection and basic functionality"""
    print("Testing EigenAI connection...")
    
    try:
        response = await eigenai_gpt_oss_120b_complete("Hello! What is 2+2?")
        print(f"✅ Success: {response}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_eigenai())
