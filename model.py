import torch
import torch.nn as nn

from runtimes.configure_agent import EMBED_DIM, N_HEAD, N_LAYER, BLOCK_SIZE

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=N_HEAD,
            dim_feedforward=4 * EMBED_DIM,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYER)
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        causal_mask = torch.triu(
            torch.ones(T, T, device=idx.device, dtype=torch.bool),
            diagonal=1,
        )
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.encoder(x, mask=causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -min(idx.size(1), BLOCK_SIZE):]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
