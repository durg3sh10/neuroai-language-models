"""Part 3: syntax vs. semantics sensitivity of GPT-2."""
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_model, sentence_logprobs, perplexity

rng = np.random.default_rng(0)


FRAMES = [
    ("dog", "chases", "chase", "ball", "across the yard"),
    ("teacher", "reads", "read", "book", "to the class"),
    ("chef", "cooks", "cook", "soup", "in the kitchen"),
    ("girl", "throws", "throw", "stone", "into the river"),
    ("farmer", "drives", "drive", "tractor", "through the field"),
    ("cat", "catches", "catch", "mouse", "under the table"),
    ("boy", "kicks", "kick", "ball", "over the fence"),
    ("pilot", "flies", "fly", "plane", "above the clouds"),
    ("doctor", "examines", "examine", "patient", "in the clinic"),
    ("mother", "feeds", "feed", "baby", "at the table"),
    ("student", "writes", "write", "essay", "before the deadline"),
    ("painter", "paints", "paint", "wall", "with a brush"),
    ("driver", "parks", "park", "car", "near the station"),
    ("gardener", "waters", "water", "plant", "every morning"),
    ("singer", "sings", "sing", "song", "on the stage"),
    ("carpenter", "builds", "build", "table", "in the workshop"),
    ("soldier", "guards", "guard", "gate", "during the night"),
    ("waiter", "serves", "serve", "meal", "to the guests"),
    ("baker", "bakes", "bake", "bread", "in the oven"),
    ("nurse", "helps", "help", "patient", "at the hospital"),
]
NONSENSE_SUBJ = ["idea", "cloud", "number", "silence", "window", "theory", "spoon", "river",
                 "Tuesday", "shadow", "pillow", "mountain", "hour", "colour", "sock", "sentence",
                 "corner", "bottle", "noise", "ladder"]
NONSENSE_OBJ = ["mountain", "freedom", "ocean", "carpet", "melody", "stapler", "dream", "volcano",
                "cucumber", "calendar", "galaxy", "tooth", "whisper", "ceiling", "pebble", "symphony",
                "elbow", "glacier", "rumour", "pancake"]


def build():
    subj_perm = rng.permutation(len(FRAMES))
    conds = {k: [] for k in "ABCDS"}
    for i, (s, v3, vb, o, pp) in enumerate(FRAMES):
        ns, no = NONSENSE_SUBJ[subj_perm[i]], NONSENSE_OBJ[i]
        A = f"The {s} {v3} the {o} {pp}."
        conds["A"].append(A)
        conds["B"].append(f"The {ns} {v3} the {no} {pp}.")
        conds["C"].append(f"The {s} {vb} the {o} {pp}.")
        conds["D"].append(f"The {ns} {vb} the {no} {pp}.")
        words = A[:-1].split()
        conds["S"].append(" ".join(rng.permutation(words)) + ".")
    return conds


@torch.no_grad()
def mean_hidden_per_layer(model, tok, sentences):
    device = next(model.parameters()).device
    enc = tok(sentences, return_tensors="pt", padding=True).to(device)
    hs = model(**enc, output_hidden_states=True).hidden_states
    m = enc.attention_mask.unsqueeze(-1).float()
    return torch.stack([(h * m).sum(1) / m.sum(1) for h in hs]).cpu()


def main():
    tok, model = load_model()
    conds = build()
    names = {"A": "grammatical+meaningful", "B": "grammatical+nonsense",
             "C": "ungrammatical+meaningful", "D": "ungrammatical+nonsense", "S": "scrambled"}


    ppl = {k: perplexity(model, tok, v) for k, v in conds.items()}
    print("Perplexity per condition:")
    for k in "ABCDS":
        print(f"  {k} {names[k]:26s} {ppl[k]:8.1f}")


    lp = {k: np.array(sentence_logprobs(model, tok, v)[0]) for k, v in conds.items()}
    pairs = {"syntax  (A>C)": np.mean(lp["A"] > lp["C"]),
             "semantics (A>B)": np.mean(lp["A"] > lp["B"]),
             "syntax in nonsense (B>D)": np.mean(lp["B"] > lp["D"]),
             "semantics in ungrammatical (C>D)": np.mean(lp["C"] > lp["D"]),
             "order (A>S)": np.mean(lp["A"] > lp["S"])}
    print("Minimal-pair accuracy (fraction of frames where the model prefers the 'good' sentence):")
    for k, v in pairs.items():
        print(f"  {k:34s} {v:.2f}")
    print("Mean log-prob penalty vs A:  B %.2f   C %.2f   D %.2f   S %.2f" % tuple(
        (lp[k] - lp["A"]).mean() for k in "BCDS"))


    H = {k: mean_hidden_per_layer(model, tok, v) for k, v in conds.items()}
    sims = {}
    for k in "BCDS":
        sims[k] = torch.nn.functional.cosine_similarity(H["A"], H[k], dim=-1).mean(1).numpy()

    shift = torch.roll(H["A"], 1, dims=1)
    sims["unrelated"] = torch.nn.functional.cosine_similarity(H["A"], shift, dim=-1).mean(1).numpy()

    with open("results/part3_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "perplexity"]); [w.writerow([names[k], ppl[k]]) for k in "ABCDS"]
        w.writerow([]); w.writerow(["minimal pair", "accuracy"]); [w.writerow([k, v]) for k, v in pairs.items()]
        w.writerow([]); w.writerow(["layer"] + list(sims.keys()))
        for l in range(len(sims["B"])):
            w.writerow([l] + [sims[k][l] for k in sims])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar([names[k].replace("+", "\n+") for k in "ABCDS"], [ppl[k] for k in "ABCDS"],
                color=["C2", "C0", "C1", "C3", "C7"])
    axes[0].set_ylabel("perplexity"); axes[0].set_yscale("log"); axes[0].set_title("Perplexity by condition")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].bar(list(pairs.keys()), list(pairs.values()))
    axes[1].axhline(0.5, color="gray", ls="--", label="chance"); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Minimal-pair preference"); axes[1].tick_params(axis="x", rotation=30, labelsize=7); axes[1].legend()
    for k, lab in [("B", "semantic violation (B)"), ("C", "syntactic violation (C)"),
                   ("D", "both (D)"), ("S", "scrambled (S)"), ("unrelated", "unrelated sentence")]:
        axes[2].plot(sims[k], "o-", label=lab)
    axes[2].set_xlabel("layer"); axes[2].set_ylabel("cosine sim. to A (mean-pooled)")
    axes[2].set_title("Representational similarity across layers"); axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("results/part3_syntax_semantics.png", dpi=150)
    print("saved results/part3_results.csv, part3_syntax_semantics.png")


if __name__ == "__main__":
    main()
