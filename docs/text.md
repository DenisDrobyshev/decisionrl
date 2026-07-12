# LLM alignment (RLHF on a language model)

`reinforce.text` provides a tiny character-level GPT and the full RLHF loop used
to align language models — pre-training (SFT) followed by reinforcement toward a
reward with a **KL penalty against the reference model**.

```python
from reinforce.text import CharTokenizer, CharGPT, sft_train, rlhf_finetune
from reinforce.text import char_frequency_reward, lexicon_reward

# 1. tokenize + build a small GPT
tok = CharTokenizer(corpus)
lm = CharGPT(tok.vocab_size, block_size=64, n_embd=64, n_head=4, n_layer=3)

# 2. supervised pre-training (SFT)
sft_train(lm, tok, corpus, n_iters=2000)

# 3. RLHF fine-tuning toward a reward, KL-regularized to the SFT model
result = rlhf_finetune(lm, tok, char_frequency_reward("o"), kl_coef=0.05, iters=120)
print(result["before"], "->", result["after"])   # e.g. 0.09 -> 0.47
```

## How it works

- **`CharGPT`** — a small causal Transformer LM (`forward`, `generate`).
- **`sft_train`** — next-character supervised pre-training on a corpus.
- **`rlhf_finetune`** — samples a group of completions, scores them with
  `reward_fn`, and updates the policy with **group-normalized advantages** (GRPO —
  no value network) plus a **KL penalty** (Schulman's `k3` estimator) to the frozen
  reference model, so it maximizes reward without drifting into gibberish.
- **Rewards** — `char_frequency_reward(target)` and `lexicon_reward(positive,
  negative)` (sentiment-style), or pass any `reward_fn(text) -> float`.

This mirrors `reinforce.rlhf` (preference-based RLHF for control) and
`reinforce.algorithms.GRPO`, extended to autoregressive text generation.
