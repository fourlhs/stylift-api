import math
import numpy as np
import torch
from model import GPT

# hyperparameters
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size    = 50257       # GPT-2 tokenizer (tiktoken)
batch_size    = 32
block_size    = 64
eval_iters    = 300
eval_interval = 2000
save_interval = 5000
max_steps     = 200000
learning_rate = 3e-4
min_lr        = 3e-5        # cosine decays down to this
drive_path    = '/content/drive/MyDrive/nanogpt'

# data
def get_batch(split):
    path = f'data/{"train" if split == "train" else "val"}.bin'
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([torch.from_numpy(data[i   : i+block_size  ].astype(np.int64)) for i in ix])
    y    = torch.stack([torch.from_numpy(data[i+1 : i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# evaluation
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate scheduler (cosine decay)
def get_lr(step):
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * step / max_steps))

# model + optimizer
model     = GPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

n_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {n_params/1e6:.2f}M | device: {device}")

# training loop
best_val_loss = float('inf')

for step in range(max_steps):

    # evaluate + checkpoint
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), f'{drive_path}/best_model.pt')

    if step % save_interval == 0:
        torch.save(model.state_dict(), f'{drive_path}/ckpt_{step:05d}.pt')

    # update lr
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # forward + backward
    x, y         = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("training complete.")