"""
DistilGPT-2 fine-tune trainer.
Loads the pretrained distilgpt2 model and fine-tunes it on the student's texts.
Yields the same event-dict protocol as TextTrainer so the frontend is unaffected.

NOTE: Requires ~500 MB free RAM. Saving a fine-tuned model takes ~330 MB on disk.
"""
from __future__ import annotations

import os

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_MODEL_NAME = 'distilgpt2'


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


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 64):
        self.examples = []
        eos = tokenizer.eos_token
        for text in texts:
            enc = tokenizer(
                text.strip() + eos,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
            )
            self.examples.append(enc['input_ids'].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class FinetuneTrainer:
    # Weights are saved as a HuggingFace model directory, not a single .pt file.
    WEIGHTS_SUBPATH = 'hf_weights'

    def __init__(self):
        # Use MPS on Apple Silicon to bypass the cblas_sgemm alignment fault (EXC_ARM_DA_ALIGN)
        # that crashes the server on MacBooks with ARM CPUs. PYTORCH_ENABLE_MPS_FALLBACK=1
        # (set in app.py) handles any ops not yet supported by MPS.
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self._texts: list[str] = []
        self._model    = None
        self._tokenizer = None
        self.trained   = False
        self.history: dict = {'loss': []}

    # ── Data ──────────────────────────────────────────────────────────────────

    def add_text(self, text: str) -> None:
        self._texts.append(text.strip())

    def remove_text(self, index: int) -> None:
        if index < 0 or index >= len(self._texts):
            raise IndexError('Text index out of range')
        self._texts.pop(index)
        self._model     = None
        self._tokenizer = None
        self.trained    = False
        self.history    = {'loss': []}

    def clear(self) -> None:
        self._texts.clear()
        self._model     = None
        self._tokenizer = None
        self.trained    = False
        self.history    = {'loss': []}

    def count(self) -> int:
        return len(self._texts)

    # ── Generation ────────────────────────────────────────────────────────────

    def _sample(self, prompt: str = '', max_new_tokens: int = 50,
                temperature: float = 0.9) -> str:
        if self._model is None:
            raise RuntimeError('Model not loaded.')
        self._model.eval()

        if prompt:
            input_ids = self._tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        else:
            input_ids = torch.tensor([[self._tokenizer.eos_token_id]], device=self.device)

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-6),
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        result = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # GPT-2 (a continuation model) often re-generates the prompt word as its
        # first new token before producing the actual content.  Strip it so the
        # response doesn't echo back whatever the user typed.
        if prompt:
            prompt_stripped = prompt.strip()
            if result.lower().startswith(prompt_stripped.lower()):
                result = result[len(prompt_stripped):].lstrip()
        print(f'[FINETUNE _sample] prompt={prompt!r} input_len={input_ids.shape[1]} output_len={output[0].shape[0]} returning={result!r}', flush=True)
        return result

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        epochs:     int   = 40,
        lr:         float = 5e-5,
        batch_size: int   = 4,
        max_length: int   = 64,
        **_,
    ):
        if not self._texts:
            yield {'phase': 'error', 'message': 'Add at least one text example to start training.'}
            return

        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
        except ImportError:
            yield {'phase': 'error', 'message': 'transformers not found — run: pip install transformers'}
            return

        try:
            self.history = {'loss': []}
            self.trained = False

            # Tokenizer
            self._tokenizer = GPT2Tokenizer.from_pretrained(_MODEL_NAME, use_fast=False)
            self._tokenizer.pad_token = self._tokenizer.eos_token

            # Model (full pretrained weights ~330 MB)
            self._model = GPT2LMHeadModel.from_pretrained(_MODEL_NAME)
            self._model.to(self.device).train()

            n_params   = sum(p.numel() for p in self._model.parameters())
            dataset    = _TextDataset(self._texts, self._tokenizer, max_length=max_length)
            loader     = DataLoader(dataset, batch_size=min(batch_size, len(dataset)),
                                    shuffle=True, num_workers=0)
            optimizer  = torch.optim.AdamW(self._model.parameters(), lr=lr)
            ckpt_every = max(1, epochs // 8)

            yield {
                'phase':      'start',
                'epochs':     epochs,
                'corpus_len': sum(len(t) for t in self._texts),
                'n_params':   n_params,
                'model_name': _MODEL_NAME,
                'device':     str(self.device),
            }

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                self._model.train()
                epoch_loss = 0.0
                n_batches  = 0

                for batch in loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = self._model(batch, labels=batch)
                    loss    = outputs.loss
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
                        sample = self._sample(max_new_tokens=30, temperature=0.9)
                    except Exception:
                        pass

                yield {
                    'phase':       'epoch',
                    'epoch':       epoch,
                    'epochs':      epochs,
                    'loss':        round(avg_loss, 4),
                    'sample':      sample,
                    'epoch_dur_s': round(epoch_dur_s, 2),
                    'eta_s':       round(epoch_dur_s * (epochs - epoch)),
                }

            self.trained = True
            final_sample = self._sample(max_new_tokens=50, temperature=0.9)

            # PCA on GPT-2 hidden states for each training text
            pca_points: list = []
            pca_labels: list = []
            try:
                self._model.eval()
                hiddens = []
                with torch.no_grad():
                    for text in self._texts:
                        ids = self._tokenizer.encode(text, return_tensors='pt').to(self.device)
                        out = self._model.transformer(ids)
                        h = out.last_hidden_state.mean(1).squeeze(0).float().cpu().numpy()
                        hiddens.append(h)
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
                'pca_points': pca_points,
                'pca_labels': pca_labels,
            }

        except Exception as exc:
            yield {'phase': 'error', 'message': str(exc)}

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str = '', length: int = 200, temperature: float = 1.0) -> str:
        print(f'[FINETUNE generate] called with prompt={prompt!r}', flush=True)
        if not self.trained or self._model is None:
            raise RuntimeError('Model not trained yet.')
        # The model was fine-tuned on complete standalone sentences, so it
        # generates best from a clean start (EOS seed) — the same way training
        # samples are produced.  Using a chat greeting like "hi" as a text seed
        # causes GPT-2 to continue the greeting instead of generating an unfortune.
        result = self._sample(
            prompt='',
            max_new_tokens=max(15, min(length // 4, 80)),
            temperature=temperature,
        )
        print(f'[FINETUNE generate] returning={result!r}', flush=True)
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        """path is a directory — HuggingFace save_pretrained format."""
        if not self.trained or self._model is None:
            raise RuntimeError('No trained model to save.')
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(p)
        self._tokenizer.save_pretrained(p)
        torch.save({'history': self.history}, p / '_training_meta.pt')

    def load(self, path) -> None:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        from pathlib import Path
        p = Path(path)
        self._tokenizer = GPT2Tokenizer.from_pretrained(str(p), use_fast=False)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = GPT2LMHeadModel.from_pretrained(str(p))
        self._model.to(self.device).eval()
        self.trained = True
        meta_path = p / '_training_meta.pt'
        if meta_path.exists():
            meta = torch.load(meta_path, map_location=self.device, weights_only=False)
            self.history = meta.get('history', {'loss': []})

    def get_info(self) -> dict:
        if self._model is None:
            return {'trained': False, 'count': len(self._texts)}
        return {
            'trained':    self.trained,
            'n_params':   sum(p.numel() for p in self._model.parameters()),
            'count':      len(self._texts),
            'loss':       self.history['loss'][-1] if self.history['loss'] else None,
            'model_name': _MODEL_NAME,
        }
