from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv() 


def chat_with_llm(
    prompt: str,
    deployment_name: str = "gpt-4o",
    max_tokens: int = 4096
) -> str:
    """
    Send a prompt to an OpenAI chat model and return the generated response text.

    Args:
        prompt (str): The user prompt to send.
        deployment_name (str): Model name (e.g., 'gpt-4o', 'gpt-4o-mini').
        max_tokens (int): Max tokens to generate.

    Returns:
        str: The model's reply text.
    """

    # Initialize the client (auto-reads OPENAI_API_KEY from environment)
    client = OpenAI()

    # Make the chat completion request
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    # Extract the assistant message
    return response.choices[0].message.content

def chat_with_llm2(
    prompt: str,
    deployment_name: str = "gpt-4.1",
    max_tokens: int = 16384
) -> str:
    """
    Send a prompt to an OpenAI chat model and return the generated response text.

    Args:
        prompt (str): The user prompt to send.
        deployment_name (str): Model name (e.g., 'gpt-4o', 'gpt-4o-mini').
        max_tokens (int): Max tokens to generate.

    Returns:
        str: The model's reply text.
    """

    # Initialize the client (auto-reads OPENAI_API_KEY from environment)
    client = OpenAI()

    # Make the chat completion request
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    # Extract the assistant message
    return response.choices[0].message.content