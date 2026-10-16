# ES 673 NeuroAI – Take Home Assignment 3: Language Models

Three small, self-contained Python/PyTorch experiments. Everything runs on CPU in a few
minutes (GPU is used automatically if one has >1.5 GB free). Figures and CSVs go to `results/`.

```bash
pip install torch transformers matplotlib numpy   # already installed on this machine
python part1_continuous_interaction.py            # ~20 s
python part2_ablation.py                          # ~5-10 min on CPU
python part3_syntax_semantics.py                  # ~30 s
```

| File | Part | What it does |
|------|------|--------------|
| `part1_continuous_interaction.py` | 1 | Simulated agent that explores a grid world, learns object–label associations with a prediction-error Hebbian rule, and re-adapts when the environment changes. Compares curiosity-driven vs. random exploration. |
| `part2_ablation.py` | 2 | Ablates GPT-2 (124M) with forward hooks: skip layers (single / cumulative), zero attention heads (random fraction, or all heads in one layer), mask input-embedding dimensions. Measures perplexity, cloze top-1 accuracy, zero-shot sentiment accuracy. |
| `part3_syntax_semantics.py` | 3 | Builds matched sentence sets (grammatical/ungrammatical × meaningful/nonsense + scrambled). Measures perplexity, minimal-pair preference, and layer-wise hidden-state similarity. |
| `common.py` | 2, 3 | Model loading and sentence log-prob / perplexity helpers. |

## Part 1 – model summary

* **Units**: perception units (one-hot colour, one-hot shape, conjunctive colour×shape units, sensory noise);
  an association memory `W` (features × labels); prediction units `softmax(Wᵀx)`; a curiosity unit
  (prediction entropy + time since last visit); an action unit (move one cell toward target).
* **Learning**: on every interaction, `W += lr · x ⊗ (y − p)` (Rescorla–Wagner / delta-rule Hebbian
  learning: strengthen feature→label links in proportion to the prediction error), plus slow decay so
  stale associations fade and the agent can re-learn after the world changes.
* **Continuous interaction**: a single endless loop of perceive → choose → move → interact → update.
  No train/test phases; half-way through, half of the colour/shape→label rules are re-drawn and the
  agent has to notice (its predictions start failing) and adapt. The active agent goes to the objects it
  is most uncertain about or hasn't visited for a long time; the baseline wanders randomly.
* Output: `results/part1_learning_curve.png` (mean ± sd over 10 seeds).

## Part 2 – ablation outputs

* `results/part2_ablation.csv` – every configuration and its three task scores.
* `results/part2_ablation.png` – 3×3 grid: rows = perplexity / cloze / sentiment, columns =
  single-layer removal, cumulative removal (top vs bottom), random heads vs embedding masking.
* `results/part2_layer_heads.png` – perplexity when all 12 heads of one layer are zeroed (attention-only ablation).

## Part 3 – outputs

* `results/part3_results.csv`, `results/part3_syntax_semantics.png`
* Conditions: A grammatical+meaningful, B grammatical+nonsense (content words swapped, syntax kept),
  C ungrammatical+meaningful (subject–verb agreement broken), D both, S scrambled word order.
* Note: mean-pooled GPT-2 states are strongly anisotropic, so all cosine similarities are close to 1;
  the *ordering* of the curves (which violation moves the representation furthest from A, and at which
  layer) is the informative part.

The written report is not included here – per the assignment it must be written by you.
