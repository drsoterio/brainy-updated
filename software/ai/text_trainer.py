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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

_TOKENIZER_NAME = 'distilgpt2'   # only vocab/merges files are loaded, not model weights


def _pca2d(arr: np.ndarray) -> list:
    """arr: (N, D) float. Returns [[x, y], ...] normalized to [-1, 1]."""
    arr = arr - arr.mean(axis=0, keepdims=True)
    n, d = arr.shape
    if n < 2 or d < 1:
        return [[0.0, 0.0]] * n
    _, _, Vt = np.linalg.svd(arr, full_matrices=False)
    proj = arr @ Vt[:min(2, d)].T
    if proj.shape[1] < 2:
        proj = np.hstack([proj, np.zeros((n, 1))])
    out = proj[:, :2].astype(float)
    for j in range(2):
        col = out[:, j]; lo, hi = col.min(), col.max(); rng = hi - lo
        if rng > 1e-8:
            out[:, j] = (col - lo) / rng * 2 - 1
    return out.tolist()


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


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
        self.device = _best_device()
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
        epochs:            int   = 80,
        lr:                float = 0.001,
        hidden_size:       int   = 40,   # repurposed: max tokens per window (clamped 20-60)
        seq_len:           int   = 50,   # kept for API compatibility, unused
        max_examples:      int   = 0,
        time_budget_s:     int   = 0,
        energy_budget_khw: float = 0,    # alias kept for compat
        energy_budget_kwh: float = 0,
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

            # Apply examples constraint: use first N texts if limit set
            texts_to_use = self._texts[:max_examples] if 0 < max_examples < len(self._texts) else self._texts

            # Tokenize every training text
            all_seqs: list[list[int]] = []
            for text in texts_to_use:
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
                                batch_size=min(32, len(xs)), shuffle=True, num_workers=0)

            self._model = _TokenLSTM(vocab_size=vocab_size).to(self.device)
            optimizer   = torch.optim.Adam(self._model.parameters(), lr=lr)
            n_params    = sum(p.numel() for p in self._model.parameters())
            ckpt_every  = max(1, epochs // 8)

            # Fixed anchor prompt: first 2 tokens of training data → honest per-epoch comparison
            _anchor_raw  = ' '.join(texts_to_use[0].strip().split()[:2]) if texts_to_use else ''
            _anchor_loc  = self._to_local(self._tokenizer.encode(_anchor_raw)) if _anchor_raw else [0]
            _anchor_loc  = _anchor_loc or [0]

            def _mid_sample() -> str | None:
                """Generate from anchor prompt mid-training (fixed RNG seed, no self.trained guard)."""
                if self._model is None:
                    return None
                rng = torch.get_rng_state()          # save — don't perturb training randomness
                torch.manual_seed(42)
                self._model.eval()
                try:
                    loc      = list(_anchor_loc)
                    x        = torch.tensor([loc], device=self.device)
                    h        = self._model.init_hidden(1, self.device)
                    gen      = list(loc)
                    with torch.no_grad():
                        if len(loc) > 1:
                            _, h = self._model(x[:, :-1], h)
                            x    = x[:, -1:]
                        for _ in range(40):
                            logits, h = self._model(x, h)
                            logits    = logits[0, -1] / 0.8
                            probs     = F.softmax(logits, dim=-1)
                            nxt       = torch.multinomial(probs, 1).item()
                            if nxt == self._eos_loc:
                                break
                            gen.append(nxt)
                            x = torch.tensor([[nxt]], device=self.device)
                    gpt_ids = [self._loc2tok[i] for i in gen if i in self._loc2tok]
                    return self._tokenizer.decode(gpt_ids, skip_special_tokens=True).strip()
                except Exception:
                    return None
                finally:
                    self._model.train()
                    torch.set_rng_state(rng)          # restore

            yield {
                'phase':       'start',
                'epochs':      epochs,
                'corpus_len':  sum(len(t) for t in self._texts),
                'vocab_size':  vocab_size,
                'n_params':    n_params,
                'hidden_size': 256,
                'device':      str(self.device),
            }

            # Resolve energy budget (accept both kwh and khw typo)
            _energy_budget_kwh = energy_budget_kwh or energy_budget_khw
            train_start_time = time.time()

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                self._model.train()
                epoch_loss = 0.0
                n_batches  = 0

                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
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

                anchor_sample = _mid_sample()

                yield {
                    'phase':         'epoch',
                    'epoch':         epoch,
                    'epochs':        epochs,
                    'loss':          round(avg_loss, 4),
                    'sample':        sample,
                    'anchor_sample': anchor_sample,
                    'epoch_dur_s':   round(epoch_dur_s, 2),
                    'eta_s':         round(epoch_dur_s * (epochs - epoch)),
                }

                if time_budget_s > 0 and (time.time() - train_start_time) >= time_budget_s:
                    yield {'phase': 'stopped', 'reason': 'time_budget', 'epoch': epoch, 'loss': round(avg_loss, 4)}
                    return
                if _energy_budget_kwh > 0:
                    elapsed_s = time.time() - train_start_time
                    used_kwh = elapsed_s * 25 / 3_600_000
                    if used_kwh >= _energy_budget_kwh:
                        yield {'phase': 'stopped', 'reason': 'energy_budget', 'epoch': epoch, 'loss': round(avg_loss, 4)}
                        return

            self.trained     = True
            final_sample     = self._sample(max_new_tokens=60, temperature=0.9)

            # PCA on LSTM hidden states for each training sequence
            pca_points: list = []
            pca_labels: list = []
            try:
                self._model.eval()
                hiddens = []
                with torch.no_grad():
                    for seq in all_seqs:
                        loc = self._to_local(seq)
                        if not loc:
                            continue
                        x = torch.tensor([loc], device=self.device)
                        _, (h, _) = self._model.lstm(self._model.embed(x))
                        hiddens.append(h[-1, 0].cpu().numpy())
                if len(hiddens) >= 2:
                    pca_points = _pca2d(np.array(hiddens, dtype=np.float32))
                    pca_labels = [f'text {i+1}' for i in range(len(hiddens))]
            except Exception:
                pass

            yield {
                'phase':      'done',
                'sample':     final_sample,
                'history':    self.history,
                'n_params':   n_params,
                'vocab_size': vocab_size,
                'pca_points': pca_points,
                'pca_labels': pca_labels,
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
            'texts':       self._texts,
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
        self._texts     = ckpt.get('texts', [])

    def get_info(self) -> dict:
        if self._model is None:
            return {'trained': False, 'count': len(self._texts), 'texts': self._texts}
        return {
            'trained':    self.trained,
            'n_params':   sum(p.numel() for p in self._model.parameters()),
            'vocab_size': len(self._tok2loc),
            'count':      len(self._texts),
            'loss':       self.history['loss'][-1] if self.history['loss'] else None,
            'texts':      self._texts,
        }

    def get_scatter_with_gen(self, generated_text: str) -> dict:
        """Return 2-D PCA points for training texts + generated text, plus distances and insight."""
        if not self.trained or self._model is None:
            raise RuntimeError('Model not trained yet.')
        self._model.eval()
        hiddens: list = []
        with torch.no_grad():
            for text in self._texts:
                tids = self._tokenizer.encode(text, add_special_tokens=False)
                loc  = self._to_local(tids) or [0]
                x    = torch.tensor([loc], device=self.device)
                _, (h, _) = self._model.lstm(self._model.embed(x))
                hiddens.append(h[-1, 0].cpu().numpy())
            gen_tids = self._tokenizer.encode(generated_text, add_special_tokens=False)
            gen_loc  = self._to_local(gen_tids) or [0]
            x_gen    = torch.tensor([gen_loc], device=self.device)
            _, (h_gen, _) = self._model.lstm(self._model.embed(x_gen))
            gen_hidden = h_gen[-1, 0].cpu().numpy()

        n_train  = len(hiddens)
        all_h    = np.array(hiddens + [gen_hidden], dtype=np.float32)
        pts_2d   = _pca2d(all_h)          # list of [x, y]
        train_pts = pts_2d[:n_train]
        gen_pt    = pts_2d[n_train]

        # Euclidean distances in 2-D PCA space — consistent with what the student sees
        dists = [
            float(np.sqrt((p[0] - gen_pt[0])**2 + (p[1] - gen_pt[1])**2))
            for p in train_pts
        ]

        insight = self._scatter_insight(dists, train_pts)
        return {
            'train_points': train_pts,
            'train_labels': [f'text {i+1}' for i in range(n_train)],
            'gen_point':    gen_pt,
            'distances':    dists,
            'insight':      insight,
        }

    def _scatter_insight(self, distances: list, train_pts: list) -> str:
        if not distances:
            return 'Add more training examples to see patterns.'
        dists    = np.array(distances, dtype=np.float32)
        mean_d   = float(dists.mean())
        min_d    = float(dists.min())
        max_d    = float(dists.max())

        # Spread of training points relative to each other
        arr = np.array(train_pts, dtype=np.float32)
        if len(arr) >= 2:
            # Mean pairwise distance among training examples
            pair_dists = []
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    d = float(np.sqrt(((arr[i] - arr[j])**2).sum()))
                    pair_dists.append(d)
            spread = float(np.std(pair_dists)) if pair_dists else 0.0
        else:
            spread = 0.0

        # Thresholds (PCA space normalised to [-1,1], max possible dist ≈ 2.83)
        # mean_d > 1.0  → generated point is far from every training example
        # min_d < 0.4 and max_d > 1.2 → one cluster is close, others are far
        # spread > 0.7  → training examples are very spread out
        # default       → generated point sits comfortably near the cluster
        if mean_d > 1.0:
            return 'Your AI is making something unlike anything you taught it. Add more examples like the output you want.'
        if min_d < 0.4 and max_d > 1.2:
            return 'Your AI is leaning hard on one type of example. Add more variety to your dataset.'
        if spread > 0.7:
            return 'Your examples are very different from each other. Add more examples in the style you want most.'
        return 'Your AI learned the pattern of your examples well. Try giving it harder examples to keep growing.'
