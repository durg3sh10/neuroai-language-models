"""Shared GPT-2 helpers for parts 2 and 3."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"


def get_device():
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            if free > 1.5e9:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def load_model(name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval().to(get_device())
    return tok, model


@torch.no_grad()
def sentence_logprobs(model, tok, sentences, batch_size=16):
    device = next(model.parameters()).device
    sums, counts = [], []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        logits = model(**enc).logits[:, :-1]
        targets = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].float()
        lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        sums += (lp * mask).sum(1).tolist()
        counts += mask.sum(1).tolist()
    return sums, counts


def perplexity(model, tok, sentences):
    s, c = sentence_logprobs(model, tok, sentences)
    return float(torch.exp(-torch.tensor(s).sum() / torch.tensor(c).sum()))


@torch.no_grad()
def next_token_logits(model, tok, prompts):
    device = next(model.parameters()).device
    out = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        out.append(model(**enc).logits[0, -1].float().cpu())
    return torch.stack(out)
