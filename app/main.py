# app/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.models.custom_phi3mini_slm import CustomPhi3MiniSLM
from app.schemas.generate import GenerateRequest
import time

# Initialize FastAPI
app = FastAPI()

# Load Phi-3 Mini model
phi3_llm = CustomPhi3MiniSLM(model_name="your_local_phi3_model_path")


@app.post("/v1/completions")
async def generate_text(request: GenerateRequest):
    """
    API to generate text with optional streaming.
    Mimics GPT-like completion endpoints.
    """
    prompt = request.prompt
    if request.stream:
        # Streaming response using a generator
        def token_streamer():
            for token in phi3_llm._stream_call(prompt):
                time.sleep(0.05)  # Optional delay to simulate streaming speed
                yield token

        return StreamingResponse(token_streamer(), media_type="text/plain")
    
    else:
        # Non-streaming response
        response = phi3_llm._call(prompt)
        return {"response": response}


@app.get("/")
async def health_check():
    """
    Basic health check endpoint to verify API is running.
    """
    return {"status": "API is running!"}
