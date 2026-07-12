"""Tests for the char-level GPT and RLHF fine-tuning."""

import math

import pytest
import torch

from reinforce.text import (
    CharGPT,
    CharTokenizer,
    char_frequency_reward,
    lexicon_reward,
    rlhf_finetune,
    sft_train,
)


def test_tokenizer_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"
    assert tok.vocab_size == len(set("hello world"))


def test_chargpt_forward_and_generate():
    tok = CharTokenizer("abcdefgh ")
    model = CharGPT(tok.vocab_size, block_size=16, n_embd=32, n_head=2, n_layer=2)
    idx = torch.zeros(2, 5, dtype=torch.long)
    assert model(idx).shape == (2, 5, tok.vocab_size)
    assert model.generate(idx, 7).shape == (2, 12)


def test_sft_reduces_loss_below_uniform():
    corpus = "abcabcabc " * 300
    tok = CharTokenizer(corpus)
    model = CharGPT(tok.vocab_size, block_size=16, n_embd=32, n_head=2, n_layer=2)
    res = sft_train(model, tok, corpus, n_iters=400, batch_size=32)
    assert res["loss"] < math.log(tok.vocab_size)  # learned structure beats uniform


def test_reward_helpers():
    assert char_frequency_reward("o")("foo") == pytest.approx(2 / 3)
    assert lexicon_reward(["good"], ["bad"])("good good bad") == 1


@pytest.mark.slow
def test_rlhf_steers_generation():
    corpus = "the quick brown fox jumps over the lazy dog. " * 40 + "pack my box with five dozen liquor jugs. " * 40
    tok = CharTokenizer(corpus)
    lm = CharGPT(tok.vocab_size, block_size=32, n_embd=64, n_head=4, n_layer=3)
    sft_train(lm, tok, corpus, n_iters=1500, batch_size=32, seed=0)
    res = rlhf_finetune(
        lm, tok, char_frequency_reward("o"), prompt=" ", gen_len=32, iters=120,
        group_size=24, kl_coef=0.05, seed=0,
    )
    # RLHF with a KL penalty steers generation toward the reward (more 'o's).
    assert res["after"] > res["before"] + 0.02
