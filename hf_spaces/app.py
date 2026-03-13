#!/usr/bin/env python3
"""NanoGPT inference app using Flask."""
import sys
import os
import torch
import tiktoken
from flask import Flask, request, jsonify
from model import GPT
from urllib.request import urlretrieve

# Load model
print("Loading model...")

# Try multiple paths
ckpt_paths = [
    'checkpoints/finetune_genz_1000k_best.pt',
    'finetune_genz_1000k_best.pt',
    '/app/checkpoints/finetune_genz_1000k_best.pt',
    '/app/finetune_genz_1000k_best.pt',
]

ckpt_path = None
for path in ckpt_paths:
    if os.path.exists(path):
        ckpt_path = path
        print(f"✓ Found checkpoint at: {path}")
        break

if not ckpt_path:
    print("Checkpoint not found locally. Attempting download from GitHub...")
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/finetune_genz_1000k_best.pt'

    try:
        url = "https://github.com/fourlhs/nano-gpt-z/releases/download/v1.0/finetune_genz_1000k_best.pt"
        print(f"Downloading: {url}")
        urlretrieve(url, ckpt_path, timeout=30)
        print(f"✓ Downloaded to {ckpt_path}")
    except Exception as e:
        print(f"✗ Download failed: {e}")

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model = GPT(vocab_size=50257)
state = ckpt['model']
state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
enc = tiktoken.encoding_for_model('gpt2')
print("✅ Model loaded")

app = Flask(__name__)

def generate(prompt, max_tokens=60, temperature=0.9, top_p=0.6):
    """Generate text from prompt."""
    print(f"[GENERATE] Prompt: '{prompt}'")

    # Normalize all-caps input to lowercase
    if prompt.isupper() and len(prompt) > 1:
        prompt = prompt.lower()

    # Tokenize
    print("[GENERATE] Encoding...")
    tokens = enc.encode(prompt)
    num_prompt_tokens = len(tokens)
    idx = torch.tensor(tokens).unsqueeze(0)

    # Generate
    print(f"[GENERATE] Generating {int(max_tokens)} tokens...")
    with torch.no_grad():
        idx = model.generate(idx, max_new_tokens=int(max_tokens),
                            temperature=float(temperature), top_p=float(top_p))
    print("[GENERATE] Done")

    # Decode only the newly generated tokens
    generated_tokens = idx[0, num_prompt_tokens:].tolist()
    generated = enc.decode(generated_tokens)

    # Post-processing
    # 1. Remove <|endoftext|> tokens
    generated = generated.replace('<|endoftext|>', '')
    generated = generated.strip()

    # 2. Skip first sentence (often fragment) only if there are multiple
    sentences = generated.split('. ')
    if len(sentences) > 2:
        generated = '. '.join(sentences[1:])

    # 3. Keep first ~180 characters and cut at natural boundary
    if len(generated) > 180:
        truncated = generated[:180]
        best_cut = None
        for punct in ['.', '!', '?']:
            last_punct = truncated.rfind(punct)
            if last_punct > 80:
                best_cut = last_punct + 1
                break

        if best_cut:
            generated = generated[:best_cut]
        else:
            for i in range(min(len(truncated), 200), 80, -1):
                if i < len(generated) and generated[i] == ' ':
                    if i > 80 and (generated[i-1].isalnum() or ord(generated[i-1]) > 127):
                        generated = generated[:i].rstrip() + '.'
                        break
            else:
                generated = truncated + '.'
    elif not generated.endswith('.') and not generated.endswith('!') and not generated.endswith('?'):
        generated += '.'

    # 4. Stop at newline
    if '\n' in generated:
        generated = generated.split('\n')[0]

    # 5. Remove line-level repetition
    generated = generated.strip()
    lines = generated.split('\n')
    seen = set()
    deduped = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped:
            if line_stripped not in seen:
                deduped.append(line)
                seen.add(line_stripped)
        else:
            deduped.append(line)

    generated = '\n'.join(deduped).strip()
    return generated

@app.route('/generate', methods=['POST'])
def api_generate():
    """API endpoint for text generation."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 60)
        temperature = data.get('temperature', 0.9)
        top_p = data.get('top_p', 0.6)

        result = generate(prompt, max_tokens, temperature, top_p)
        return jsonify({'text': result, 'error': None})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'text': '', 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    """Home page with simple UI."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>NanoGPT</title>
        <style>
            body { font-family: sans-serif; max-width: 600px; margin: 50px auto; }
            input, textarea { width: 100%; padding: 10px; margin: 10px 0; }
            button { padding: 10px 20px; background: #667eea; color: white; border: none; cursor: pointer; }
            #output { margin-top: 20px; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>NanoGPT</h1>
        <input type="text" id="prompt" placeholder="Type your message..." />
        <button onclick="generate()">Generate</button>
        <div id="output"></div>
        <script>
            async function generate() {
                const prompt = document.getElementById('prompt').value;
                const output = document.getElementById('output');
                output.textContent = 'Generating...';

                try {
                    const res = await fetch('/generate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt, max_tokens: 60, temperature: 0.9, top_p: 0.6})
                    });
                    const data = await res.json();
                    output.textContent = data.text || data.error;
                } catch (e) {
                    output.textContent = 'Error: ' + e.message;
                }
            }
            document.getElementById('prompt').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') generate();
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)