"""Language modeling + RLHF on a tiny char-level GPT.

    from reinforce.text import CharTokenizer, CharGPT, sft_train, rlhf_finetune
    from reinforce.text import char_frequency_reward

    tok = CharTokenizer(corpus)
    lm = CharGPT(tok.vocab_size, block_size=64)
    sft_train(lm, tok, corpus, n_iters=2000)            # supervised pre-training
    rlhf_finetune(lm, tok, char_frequency_reward("a"))  # align with a reward + KL
"""

from .gpt import CharGPT, CharTokenizer, sft_train
from .rlhf import char_frequency_reward, lexicon_reward, rlhf_finetune

__all__ = [
    "CharTokenizer",
    "CharGPT",
    "sft_train",
    "rlhf_finetune",
    "char_frequency_reward",
    "lexicon_reward",
]
