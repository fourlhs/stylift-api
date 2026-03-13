# Deploy NanoGPT to HuggingFace Spaces

## Quick Start

1. **Create a HuggingFace Spaces repo:**
   - Go to https://huggingface.co/new-space
   - Name: `nanogpt` (or whatever you want)
   - License: MIT
   - Space type: **Docker** (for custom setup with model files)

2. **Upload your model checkpoint:**
   - In the Files tab, create a `checkpoints/` folder
   - Upload `finetune_genz_1000k_best.pt` to `checkpoints/`
   - Also upload `model.py` to the root

3. **Push the code:**
   ```bash
   cd nanogpt
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/nanogpt
   git push hf main
   ```

   This will push:
   - `app.py` - Gradio interface
   - `requirements.txt` - Dependencies
   - `model.py` - Model definition

4. **Done!**
   - HuggingFace will automatically build the Docker image and start the app
   - Your space will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/nanogpt`

## File Structure

```
nanogpt/
├── app.py                    # Gradio interface (entry point)
├── requirements.txt          # Python dependencies
├── model.py                  # GPT model definition
├── checkpoints/
│   └── finetune_genz_1000k_best.pt  # Model weights
└── HF_SPACES.md             # This file
```

## Notes

- **First load time:** ~30-60 seconds (model loading)
- **Inference time:** ~3-5 seconds per generation
- **Storage:** The model checkpoint should fit in the free tier (typically 50GB available)
- **GPU:** Spaces includes a free GPU with 16GB VRAM (plenty for this model)

## Environment Variables (Optional)

You can add environment variables in the Space settings:
- `HF_TOKEN` - If you want to log in to HuggingFace Hub

## Troubleshooting

**"Model file not found"**
- Make sure `checkpoints/finetune_genz_1000k_best.pt` is uploaded
- Path should be exactly: `checkpoints/finetune_genz_1000k_best.pt`

**Space is very slow**
- First request might be slow due to model loading
- Subsequent requests are faster
- If persistent, the GPU might be busy - try again later

**Generation takes too long**
- Normal for CPU inference, even with GPU it's 3-5s per generation
- This is expected behavior for this model size