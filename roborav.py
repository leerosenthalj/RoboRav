import os
import re
import math
from collections import defaultdict, Counter
from functools import partial
import sys
import json
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import datasets
from pathlib import Path

import torch as t
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import einops


# Tokenization and data preparation.
class TalmudTokenizer:
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.space_prefix = 'Ġ'

    def _get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, text: str):
        print("Starting tokenizer training...")
        
        # Preprocess text to add space prefix, including for the first word
        words = [self.space_prefix + word for word in text.split()]
        
        # Also add non-prefixed versions of words to the vocabulary
        non_prefixed_words = text.split()
        
        # Initialize vocab with character tokens
        chars = set(''.join(words + non_prefixed_words))
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = char

        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Convert words to space-separated character sequences
        vocab = Counter(' '.join(word) for word in words)
        vocab.update(' '.join(word) for word in non_prefixed_words)
        
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                print(f"No more pairs to merge after {i} iterations")
                break
            
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            self.merges[best] = ''.join(best)
            new_token = ''.join(best)
            
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = new_token
            
            if len(self.vocab) >= self.vocab_size:
                print(f"Reached target vocabulary size after {i+1} iterations")
                break
            
            if i % 100 == 0:
                print(f"Completed {i} merges. Current vocab size: {len(self.vocab)}")

        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")

    def _tokenize_word(self, word: str) -> List[str]:
        if word in self.vocab:
            return [word]
        
        word = ' '.join(word)
        tokens = []
        while len(word) > 0:
            subword = word
            while len(subword) > 0:
                if subword in self.vocab:
                    tokens.append(subword)
                    word = word[len(subword):].lstrip()
                    break
                subword = subword[:-1]
            if len(subword) == 0:
                tokens.append(word[0])
                word = word[1:].lstrip()
        return tokens

    def tokenize(self, text: str) -> List[int]:
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            if i == 0 or word.startswith(self.space_prefix):
                tokens.extend(self._tokenize_word(word))
            else:
                tokens.extend(self._tokenize_word(self.space_prefix + word))
        return [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(id, "<UNK>") for id in token_ids]
        text = ''.join(tokens).replace(self.space_prefix, ' ')
        return text.strip()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, 'merges.json'), 'w', encoding='utf-8') as f:
            json.dump({' '.join(k): v for k, v in self.merges.items()}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab_size': self.vocab_size, 'space_prefix': self.space_prefix}, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        tokenizer = cls(vocab_size=config['vocab_size'])
        tokenizer.space_prefix = config['space_prefix']
        
        with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as f:
            tokenizer.vocab = json.load(f)
        tokenizer.inverse_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        
        with open(os.path.join(path, 'merges.json'), 'r', encoding='utf-8') as f:
            merges = json.load(f)
            tokenizer.merges = {tuple(k.split()): v for k, v in merges.items()}
        
        return tokenizer


class SequenceDataset(Dataset):
    def __init__(self, tokens, sequence_length):
        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.sequence_length + 1]
        return t.tensor(chunk[:-1], dtype=t.long), t.tensor(chunk[1:], dtype=t.long)


def prepare_data_for_training(tokens, sequence_length, batch_size, val_split=0.1):
    dataset = SequenceDataset(tokens, sequence_length)
    
    # Split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

    
# Model classes
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 16000
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        mean = residual.mean(dim=-1, keepdim=True)
        vary = (residual.var(dim=-1, keepdim=True, unbiased=False) +
                            self.cfg.layer_norm_eps).sqrt()
        residual = (residual - mean)/vary
        return residual * self.w + self.b


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        shapey = tokens.shape
        batch = shapey[0]
        seq_len = shapey[1]
        return self.W_pos[:seq_len].unsqueeze(0).expand(batch, -1, -1)


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self, x: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]: #normalized_resid_pre
        # x: residual stream, (batch, seq_len, d_model)
        # Compute key, query, and value tensors.
        # n heads, b batches, sequence length s, embedding e (d_model), head size h.
        K = t.einsum('nmh,bsm->bsnh', self.W_K, x) + self.b_K # (b, s, n, h)
        Q = t.einsum('nmh,bsm->bsnh', self.W_Q, x) + self.b_Q # (b, s, n, h)
        V = t.einsum('nmh,bsm->bsnh', self.W_V, x) + self.b_V # (b, s, n, h)
        # Compute attention scores.
        QKt = t.einsum('bsnh,btnh->bnst', Q, K) # (b, n, s, s)
        # Compute attention probabilities, with causal mask.
        A = self.apply_causal_mask(QKt/self.cfg.d_head**0.5).softmax(-1) # (b, n, s, s)
        # Sum over value vectors, weighted by A.
        z = t.einsum("bnst,btnh->bsnh", A, V) # (b, s, n, h)
        # Sum over parallel attention heads.
        attention = t.einsum("nhm,bsnh->bsm", self.W_O, z)
        return attention

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        onesy = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(onesy, diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, x: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        neural = t.einsum('mn,bsm->bsn', self.W_in, x) + self.b_in
        neural = geluprox(neural)
        return t.einsum('nm,bsn->bsm', self.W_out, neural) + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, x: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        y = self.attn(self.ln1(x)) + x
        z = self.mlp(self.ln2(y)) + y
        return z


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, x: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return t.einsum('mv,bsm->bsv', self.W_U, x) + self.b_U


class RoboRav(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg       = cfg
        self.embed     = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks    = nn.ModuleList([TransformerBlock(cfg) for
                                        _ in range(cfg.n_layers)])
        self.ln_final  = LayerNorm(cfg)
        self.unembed   = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits