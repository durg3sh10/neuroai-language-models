"""Part 2: ablation study on GPT-2 (layers, attention heads, input embeddings)."""
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_model, perplexity, next_token_logits


TEXT = [
    "The sun rises in the east and sets in the west.",
    "She opened the door and walked into the quiet room.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The children played in the park until it started to rain.",
    "He forgot his umbrella, so he got wet on the way home.",
    "The museum was closed on Monday, so we visited on Tuesday instead.",
    "Scientists have discovered a new species of frog in the rainforest.",
    "The recipe calls for two eggs, a cup of flour, and a pinch of salt.",
    "After a long day at work, she enjoyed a cup of tea on the balcony.",
    "The train to London departs every hour from platform nine.",
    "My grandfather taught me how to fish when I was a little boy.",
    "The company announced record profits for the third quarter.",
    "Please remember to lock the door before you leave the house.",
    "The old bridge was replaced by a modern steel structure last year.",
    "The students listened carefully as the teacher explained the lesson.",
    "A gentle breeze blew through the open window on a warm summer evening.",
]

CLOZE = [
    ("The capital of France is", " Paris"),
    ("Two plus two equals", " four"),
    ("The opposite of hot is", " cold"),
    ("The sky is", " blue"),
    ("A week has seven", " days"),
    ("Cats like to chase", " mice"),
    ("The first month of the year is", " January"),
    ("Water is made of hydrogen and", " oxygen"),
    ("Barack Obama was the president of the United", " States"),
    ("Peanut butter and", " jelly"),
    ("The Earth revolves around the", " sun"),
    ("Once upon a", " time"),
    ("Roses are red, violets are", " blue"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("He drank a glass of", " water"),
    ("She put the book back on the", " shelf"),
    ("The dog wagged its", " tail"),
    ("Thank you very", " much"),
    ("It was raining, so he opened his", " umbrella"),
    ("The baby started to", " cry"),
]

SENTIMENT = [
    ("This movie was absolutely wonderful and I loved every minute.", "positive"),
    ("A brilliant, moving, and beautifully acted film.", "positive"),
    ("The food was delicious and the service was excellent.", "positive"),
    ("I had a fantastic time, highly recommended!", "positive"),
    ("What a delightful and charming little story.", "positive"),
    ("The acting was superb and the plot kept me hooked.", "positive"),
    ("Great product, works perfectly and arrived early.", "positive"),
    ("An amazing experience from start to finish.", "positive"),
    ("The hotel was clean, comfortable and the staff were friendly.", "positive"),
    ("One of the best books I have read this year.", "positive"),
    ("This movie was terrible and a complete waste of time.", "negative"),
    ("Boring, predictable, and badly acted.", "negative"),
    ("The food was cold and the service was awful.", "negative"),
    ("I hated it, do not bother watching.", "negative"),
    ("A dull and disappointing story with no charm.", "negative"),
    ("The plot made no sense and the acting was wooden.", "negative"),
    ("Broken on arrival, the worst purchase I have made.", "negative"),
    ("An unpleasant experience from start to finish.", "negative"),
    ("The hotel was dirty, noisy and the staff were rude.", "negative"),
    ("One of the worst books I have ever read.", "negative"),
]


def task_perplexity(model, tok):
    return perplexity(model, tok, TEXT)


def task_cloze(model, tok):
    prompts = [p for p, _ in CLOZE]
    logits = next_token_logits(model, tok, prompts)
    pred = logits.argmax(-1)
    gold = torch.tensor([tok.encode(w)[0] for _, w in CLOZE])
    return (pred == gold).float().mean().item()


def task_sentiment(model, tok):
    prompts = [f"Review: {t}\nThe sentiment of this review is" for t, _ in SENTIMENT]
    logits = next_token_logits(model, tok, prompts)
    pos, neg = tok.encode(" positive")[0], tok.encode(" negative")[0]
    score = logits[:, pos] - logits[:, neg]
    score = score - score.median()
    pred = np.where(score.numpy() > 0, "positive", "negative")
    return float(np.mean(pred == np.array([l for _, l in SENTIMENT])))


def evaluate(model, tok):
    return {"perplexity": task_perplexity(model, tok),
            "cloze_acc": task_cloze(model, tok),
            "sentiment_acc": task_sentiment(model, tok)}


class Ablation:
    def __init__(self, model, skip_layers=(), zero_heads=(), embed_mask_frac=0.0, seed=0):
        self.model = model
        self.skip_layers = set(skip_layers)
        self.zero_heads = set(zero_heads)
        self.embed_mask_frac = embed_mask_frac
        self.seed = seed
        self.handles = []

    def __enter__(self):
        cfg = self.model.config
        head_dim = cfg.n_embd // cfg.n_head
        for li, block in enumerate(self.model.transformer.h):
            if li in self.skip_layers:
                def skip(mod, inp, out):
                    return (inp[0],) + tuple(out[1:])
                self.handles.append(block.register_forward_hook(skip))
            heads = [h for (l, h) in self.zero_heads if l == li]
            if heads:
                def zero(mod, inp, heads=heads):
                    x = inp[0].clone()
                    for h in heads:
                        x[..., h * head_dim:(h + 1) * head_dim] = 0
                    return (x,)
                self.handles.append(block.attn.c_proj.register_forward_pre_hook(zero))
        if self.embed_mask_frac > 0:
            g = torch.Generator().manual_seed(self.seed)
            keep = (torch.rand(cfg.n_embd, generator=g) >= self.embed_mask_frac).float()
            def mask(mod, inp, out):
                return out * keep.to(out.device)
            self.handles.append(self.model.transformer.wte.register_forward_hook(mask))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def main():
    tok, model = load_model()
    L, H = model.config.n_layer, model.config.n_head
    rows = []

    def record(exp, x, res):
        rows.append({"experiment": exp, "x": x, **res})
        print(f"{exp:22s} x={x:<6} ppl={res['perplexity']:8.1f}  cloze={res['cloze_acc']:.2f}  sent={res['sentiment_acc']:.2f}")

    base = evaluate(model, tok)
    record("baseline", 0, base)


    for l in range(L):
        with Ablation(model, skip_layers=[l]):
            record("single_layer", l, evaluate(model, tok))

    for k in range(1, L):
        with Ablation(model, skip_layers=range(L - k, L)):
            record("cum_top", k, evaluate(model, tok))
        with Ablation(model, skip_layers=range(0, k)):
            record("cum_bottom", k, evaluate(model, tok))

    all_heads = [(l, h) for l in range(L) for h in range(H)]
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        acc = {"perplexity": [], "cloze_acc": [], "sentiment_acc": []}
        for seed in range(3):
            rng = np.random.default_rng(seed)
            chosen = [all_heads[i] for i in rng.choice(len(all_heads), int(frac * len(all_heads)), replace=False)]
            with Ablation(model, zero_heads=chosen):
                for k, v in evaluate(model, tok).items():
                    acc[k].append(v)
        record("random_heads", frac, {k: float(np.mean(v)) for k, v in acc.items()})

    for l in range(L):
        with Ablation(model, zero_heads=[(l, h) for h in range(H)]):
            record("layer_all_heads", l, evaluate(model, tok))

    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with Ablation(model, embed_mask_frac=frac):
            record("embed_mask", frac, evaluate(model, tok))

    with open("results/part2_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


    def panel(ax, exp, xlabel, metric, ylabel, logy=False):
        r = [x for x in rows if x["experiment"] == exp]
        ax.plot([x["x"] for x in r], [x[metric] for x in r], "o-")
        ax.axhline(base[metric], color="gray", ls="--", label="baseline")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(exp)
        if logy: ax.set_yscale("log")
        ax.legend()

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for i, (metric, ylabel, logy) in enumerate([("perplexity", "perplexity", True),
                                                ("cloze_acc", "cloze top-1 acc", False),
                                                ("sentiment_acc", "sentiment acc", False)]):
        panel(axes[i, 0], "single_layer", "layer removed", metric, ylabel, logy)
        r1 = [x for x in rows if x["experiment"] == "cum_top"]
        r2 = [x for x in rows if x["experiment"] == "cum_bottom"]
        axes[i, 1].plot([x["x"] for x in r1], [x[metric] for x in r1], "o-", label="remove from top")
        axes[i, 1].plot([x["x"] for x in r2], [x[metric] for x in r2], "s-", label="remove from bottom")
        axes[i, 1].axhline(base[metric], color="gray", ls="--", label="baseline")
        axes[i, 1].set_xlabel("# layers removed"); axes[i, 1].set_ylabel(ylabel)
        axes[i, 1].set_title("cumulative layer removal"); axes[i, 1].legend()
        if logy: axes[i, 1].set_yscale("log")
        r3 = [x for x in rows if x["experiment"] == "random_heads"]
        r4 = [x for x in rows if x["experiment"] == "embed_mask"]
        axes[i, 2].plot([x["x"] for x in r3], [x[metric] for x in r3], "o-", label="random heads zeroed")
        axes[i, 2].plot([x["x"] for x in r4], [x[metric] for x in r4], "s-", label="embedding dims masked")
        axes[i, 2].axhline(base[metric], color="gray", ls="--", label="baseline")
        axes[i, 2].set_xlabel("fraction ablated"); axes[i, 2].set_ylabel(ylabel)
        axes[i, 2].set_title("heads / embeddings"); axes[i, 2].legend()
        if logy: axes[i, 2].set_yscale("log")
    plt.tight_layout(); plt.savefig("results/part2_ablation.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    r = [x for x in rows if x["experiment"] == "layer_all_heads"]
    ax.bar([x["x"] for x in r], [x["perplexity"] for x in r])
    ax.axhline(base["perplexity"], color="gray", ls="--", label="baseline")
    ax.set_yscale("log"); ax.set_xlabel("layer whose 12 heads are zeroed"); ax.set_ylabel("perplexity")
    ax.set_title("Attention removed from a single layer"); ax.legend()
    plt.tight_layout(); plt.savefig("results/part2_layer_heads.png", dpi=150)
    print("saved results/part2_ablation.csv, part2_ablation.png, part2_layer_heads.png")


if __name__ == "__main__":
    main()
