import os
from groq import Groq # or OpenAI
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = True):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", # High performance for reasoning
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"} if json_mode else None
    )
    return response.choices[0].message.content


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
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    groq_client = Groq(api_key=api_key)
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        response_format={"type": "json_object"},
    )
    
    return response.choices[0].message.content