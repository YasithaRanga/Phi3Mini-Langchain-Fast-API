from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.model.ollama_model import generate_ollama_response

app = FastAPI()

model_name = "phi3"

@app.post("/v1/completions")
async def generate_completion(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    model = body.get("model", model_name)
    stream = body.get("stream", False)

    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided.")

    # StreamingResponse needs a generator function to stream data in chunks
    return StreamingResponse(generate_ollama_response(prompt, model, stream), media_type="text/plain")
   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
