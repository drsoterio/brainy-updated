"""
Token-level LSTM text generator.
Uses GPT-2's subword tokenizer (vocab files only, ~2 MB, no model weights)
to tokenize the student's texts, then trains a compact LSTM on those tokens.
Result: word-aware generation without the memory cost of a pretrained model.
"""
from __future__ import annotations

import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

_TOKENIZER_NAME = 'distilgpt2'   # only vocab/merges files are loaded, not model weights


# ── Model ──────────────────────────────────────────────────────────────────────

class _TokenLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 256,
                 num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True,
                             dropout=0.1 if num_layers > 1 else 0.0)
        self.head  = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.head(out), hidden

    def init_hidden(self, batch_size: int, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c


# ── Trainer ────────────────────────────────────────────────────────────────────

class TextTrainer:
    def __init__(self):
        self.device = torch.device('cpu')
        self._texts: list[str]  = []
        self._model: _TokenLSTM | None = None
        self._tokenizer         = None
        self._tok2loc: dict[int, int] = {}   # GPT-2 token id  → local index
        self._loc2tok: dict[int, int] = {}   # local index     → GPT-2 token id
        self._eos_loc: int = 0
        self.trained  = False
        self.history: dict = {'loss': []}

    # ── Data ──────────────────────────────────────────────────────────────────

    def add_text(self, text: str) -> None:
        self._texts.append(text.strip())

    def remove_text(self, index: int) -> None:
        if index < 0 or index >= len(self._texts):
            raise IndexError('Text index out of range')
        self._texts.pop(index)
        self._model    = None
        self._tokenizer = None
        self._tok2loc  = {}
        self._loc2tok  = {}
        self.trained   = False
        self.history   = {'loss': []}

    def clear(self) -> None:
        self._texts.clear()
        self._model    = None
        self._tokenizer = None
        self._tok2loc  = {}
        self._loc2tok  = {}
        self.trained   = False
        self.history   = {'loss': []}

    def count(self) -> int:
        return len(self._texts)

    # ── Generation ────────────────────────────────────────────────────────────

    def _to_local(self, gpt_ids: list[int]) -> list[int]:
        return [self._tok2loc.get(t, 0) for t in gpt_ids]

    def _sample(self, prompt: str = '', max_new_tokens: int = 50,
                temperature: float = 0.9) -> str:
        if not self.trained or self._model is None:
            raise RuntimeError('Model not trained yet.')
        self._model.eval()

        if prompt:
            gpt_ids   = self._tokenizer.encode(prompt)
            local_ids = self._to_local(gpt_ids) or [0]
        else:
            non_eos   = [i for i in range(len(self._loc2tok)) if i != self._eos_loc]
            local_ids = [random.choice(non_eos)] if non_eos else [0]

        x      = torch.tensor([local_ids], device=self.device)
        hidden = self._model.init_hidden(1, self.device)
        generated = local_ids[:]

        with torch.no_grad():
            if len(local_ids) > 1:
                _, hidden = self._model(x[:, :-1], hidden)
                x = x[:, -1:]
            for _ in range(max_new_tokens):
                logits, hidden = self._model(x, hidden)
                logits = logits[0, -1] / max(temperature, 1e-6)
                probs  = F.softmax(logits, dim=-1)
                nxt    = torch.multinomial(probs, 1).item()
                if nxt == self._eos_loc:
                    break
                generated.append(nxt)
                x = torch.tensor([[nxt]], device=self.device)

        gpt_ids = [self._loc2tok[i] for i in generated if i in self._loc2tok]
        return self._tokenizer.decode(gpt_ids, skip_special_tokens=True).strip()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        epochs:      int   = 80,
        lr:          float = 0.001,
        hidden_size: int   = 40,   # repurposed: max tokens per window (clamped 20-60)
        seq_len:     int   = 50,   # kept for API compatibility, unused
        **_,
    ):
        if not self._texts:
            yield {'phase': 'error', 'message': 'Add at least one text example to start training.'}
            return

        try:
            from transformers import GPT2Tokenizer
        except ImportError:
            yield {'phase': 'error', 'message': 'transformers not found — run: pip install transformers'}
            return

        try:
            self.history = {'loss': []}
            self.trained = False

            # Load tokenizer vocab only (~2 MB, no model weights)
            self._tokenizer = GPT2Tokenizer.from_pretrained(_TOKENIZER_NAME, use_fast=False)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            eos_id = self._tokenizer.eos_token_id

            # Tokenize every training text
            all_seqs: list[list[int]] = []
            for text in self._texts:
                ids = self._tokenizer.encode(text.strip()) + [eos_id]
                all_seqs.append(ids)

            # Build compact vocabulary from only the tokens that appear
            flat      = [t for s in all_seqs for t in s]
            used      = sorted(set(flat))
            self._tok2loc = {t: i for i, t in enumerate(used)}
            self._loc2tok = {i: t for i, t in enumerate(used)}
            self._eos_loc = self._tok2loc[eos_id]
            vocab_size    = len(used)

            # Re-encode sequences in local vocab; build (x, y) pairs
            win = max(20, min(int(hidden_size), 60))
            xs, ys = [], []
            for seq in all_seqs:
                loc = self._to_local(seq)
                if len(loc) < 2:
                    continue
                # whole sequence as one sample
                xs.append(loc[:-1]);  ys.append(loc[1:])
                # sliding windows for longer texts
                if len(loc) > win + 1:
                    step = max(1, win // 2)
                    for i in range(0, len(loc) - win - 1, step):
                        xs.append(loc[i:i+win]);  ys.append(loc[i+1:i+win+1])

            if not xs:
                yield {'phase': 'error', 'message': 'Texts too short. Add more examples.'}
                return

            # Pad to uniform length
            pad_len = max(len(x) for x in xs)
            def pad(s): return s + [self._eos_loc] * (pad_len - len(s))
            x_t = torch.tensor([pad(x) for x in xs])
            y_t = torch.tensor([pad(y) for y in ys])

            loader = DataLoader(TensorDataset(x_t, y_t),
                                batch_size=min(8, len(xs)), shuffle=True, num_workers=0)

            self._model = _TokenLSTM(vocab_size=vocab_size).to(self.device)
            optimizer   = torch.optim.Adam(self._model.parameters(), lr=lr)
            n_params    = sum(p.numel() for p in self._model.parameters())
            ckpt_every  = max(1, epochs // 8)

            yield {
                'phase':       'start',
                'epochs':      epochs,
                'corpus_len':  sum(len(t) for t in self._texts),
                'vocab_size':  vocab_size,
                'n_params':    n_params,
                'hidden_size': 256,
                'device':      str(self.device),
            }

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                self._model.train()
                epoch_loss = 0.0
                n_batches  = 0

                for xb, yb in loader:
                    hidden = self._model.init_hidden(xb.size(0), self.device)
                    optimizer.zero_grad()
                    logits, _ = self._model(xb, hidden)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches  += 1

                epoch_dur_s = time.time() - epoch_start
                avg_loss = epoch_loss / max(n_batches, 1)
                self.history['loss'].append(round(avg_loss, 4))

                sample = None
                if epoch == 1 or epoch % ckpt_every == 0 or epoch == epochs:
                    try:
                        sample = self._sample(max_new_tokens=50, temperature=0.9)
                    except Exception:
                        pass

                yield {
                    'phase':      'epoch',
                    'epoch':      epoch,
                    'epochs':     epochs,
                    'loss':       round(avg_loss, 4),
                    'sample':     sample,
                    'epoch_dur_s': round(epoch_dur_s, 2),
                    'eta_s':      round(epoch_dur_s * (epochs - epoch)),
                }

            self.trained     = True
            final_sample     = self._sample(max_new_tokens=60, temperature=0.9)

            yield {
                'phase':      'done',
                'sample':     final_sample,
                'history':    self.history,
                'n_params':   n_params,
                'vocab_size': vocab_size,
            }

        except Exception as exc:
            yield {'phase': 'error', 'message': str(exc)}

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str = '', length: int = 200, temperature: float = 1.0) -> str:
        if not self.trained or self._model is None:
            raise RuntimeError('Model not trained yet.')
        return self._sample(prompt=prompt, max_new_tokens=max(15, min(length // 4, 80)),
                            temperature=temperature)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        if not self.trained or self._model is None:
            raise RuntimeError('No trained model to save.')
        torch.save({
            'model_state': self._model.state_dict(),
            'tok2loc':     self._tok2loc,
            'loc2tok':     self._loc2tok,
            'eos_loc':     self._eos_loc,
            'history':     self.history,
        }, path)

    def load(self, path) -> None:
        from transformers import GPT2Tokenizer
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._tok2loc   = ckpt['tok2loc']
        self._loc2tok   = {int(k): v for k, v in ckpt['loc2tok'].items()}
        self._eos_loc   = ckpt['eos_loc']
        self._tokenizer = GPT2Tokenizer.from_pretrained(_TOKENIZER_NAME, use_fast=False)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        vocab_size      = len(self._tok2loc)
        self._model     = _TokenLSTM(vocab_size=vocab_size).to(self.device)
        self._model.load_state_dict(ckpt['model_state'])
        self._model.eval()
        self.trained    = True
        self.history    = ckpt.get('history', {'loss': []})

    def get_info(self) -> dict:
        if self._model is None:
            return {'trained': False, 'count': len(self._texts)}
        return {
            'trained':    self.trained,
            'n_params':   sum(p.numel() for p in self._model.parameters()),
            'vocab_size': len(self._tok2loc),
            'count':      len(self._texts),
            'loss':       self.history['loss'][-1] if self.history['loss'] else None,
        }
