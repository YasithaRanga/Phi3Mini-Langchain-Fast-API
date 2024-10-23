import asyncio
import ollama
from typing import AsyncGenerator

ollama_client = ollama

async def generate_ollama_response(prompt: str, model: str, stream: bool) -> AsyncGenerator[str, None]:
    """Asynchronous generator for streaming responses from Ollama."""
    try:
        # Streaming Ollama's response
        if(stream):
            stream = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=stream,
            )
            for chunk in stream:  # Assuming `stream` is a regular generator
                yield chunk['message']['content'] # Yield the content part of each streamed chunk
                await asyncio.sleep(0.1)  # Adjust or remove delay based on actual response time
        else:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ])
            yield response['message']['content']
        # Ollama's stream might not support async natively, so convert it using asyncio

    except Exception as e:
        yield f"Error: {str(e)}"  # Stream an error message back to the client
