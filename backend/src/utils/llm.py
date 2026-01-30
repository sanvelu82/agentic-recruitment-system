"""
LLM utility functions for the recruitment system.

Provides helper functions for interacting with LLM providers.
"""

import os
from typing import Optional

from groq import Groq


def get_agent_decision(system_prompt: str, user_payload: str) -> str:
    """
    Get a JSON decision from the LLM using Groq.
    
    Args:
        system_prompt: The system prompt to set the LLM's behavior
        user_payload: The user message/payload to process
    
    Returns:
        Raw JSON string from the LLM response
    
    Raises:
        ValueError: If GROQ_API_KEY environment variable is not set
        groq.APIError: If the API call fails
    
    Example:
        >>> system = "You are a skill extractor. Return JSON with 'skills' array."
        >>> payload = "Looking for Python and JavaScript developers"
        >>> result = get_agent_decision(system, payload)
        >>> # result: '{"skills": ["Python", "JavaScript"]}'
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        response_format={"type": "json_object"},
    )
    
    return response.choices[0].message.content
