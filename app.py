from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import os

app = FastAPI()

# Enable CORS for all origins (change to specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TOKENS = 200
MODEL_PATH = os.getenv('MODEL_PATH', './model')

# Global model state
model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    if not load_model():
        print("Warning: Model failed to load. API will return error responses.")

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_tokens: int = MAX_TOKENS

class GenerateResponse(BaseModel):
    text: str
    tokens: list[int]

@app.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    if model is None or tokenizer is None:
        return GenerateResponse(text="error: model not loaded", tokens=[])

    try:
        # Tokenize input
        input_ids = tokenizer.encode(request.prompt, return_tensors='pt').to(device)
        prompt_len = input_ids.shape[1]

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=prompt_len + request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract only the generated tokens (after prompt)
        generated_ids = output[0][prompt_len:].tolist()

        # Decode the full response
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove the prompt from the start
        response_text = full_text[len(request.prompt):].lstrip()

        return GenerateResponse(
            text=response_text,
            tokens=generated_ids
        )

    except Exception as e:
        return GenerateResponse(
            text=f"error: {str(e)}",
            tokens=[]
        )

@app.get("/status")
async def status():
    return {
        "ready": model is not None and tokenizer is not None,
        "device": str(device),
        "model_path": MODEL_PATH
    }

@app.get("/")
async def root():
    return {
        "service": "stylift-api",
        "version": "1.0",
        "endpoints": {
            "POST /generate": "Generate text from prompt",
            "GET /status": "Check model status"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
