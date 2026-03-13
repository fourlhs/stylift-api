# Stylift API

A FastAPI backend for the Stylift language model demo. Serves a GPT-2 model fine-tuned on Gen Z slang.

## Setup

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## Endpoints

- `GET /` - Service info
- `GET /status` - Check if model is loaded and ready
- `POST /generate` - Generate text from a prompt

### Generate Endpoint

**Request:**
```json
{
  "prompt": "yo this is",
  "temperature": 0.8,
  "top_p": 0.9,
  "max_tokens": 200
}
```

**Response:**
```json
{
  "text": "gen z slang text here...",
  "tokens": [1234, 5678, ...]
}
```

## Deployment

### Render.com

Push to GitHub, then on Render.com:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python app.py`

Note: You'll need to upload your model to the Render environment or provide an S3 URL.

## Model

The model should be placed in a `model/` directory at the root. Currently expects a PyTorch model compatible with HuggingFace transformers.
