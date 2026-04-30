"""
Smart Prompt trainer — RAG pipeline: embeddings + cosine similarity + Ollama/LLaVA.
No gradient training. Examples are converted to vectors at "prepare" time;
at generation time the closest examples are retrieved and sent to LLaVA.

Dependencies:
  pip install sentence-transformers          # embedding model (~80 MB download on first use)
  ollama pull llava                          # or set OLLAMA_MODEL env var

Env vars (optional):
  OLLAMA_URL   — default http://localhost:11434
  OLLAMA_MODEL — default llava
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

_EMBED_MODEL  = 'all-MiniLM-L6-v2'
_OLLAMA_URL   = os.environ.get('OLLAMA_URL',   'http://localhost:11434')
_OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llava')

# Approximate parameter count of all-MiniLM-L6-v2
_EMBED_PARAMS = 22_700_000


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _pca_2d(matrix: list[list]) -> list[tuple[float, float]]:
    """Pure-numpy 2-component PCA. Falls back to evenly spaced if numpy absent."""
    try:
        import numpy as np
        arr     = np.array(matrix, dtype=np.float32)
        centered = arr - arr.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords   = centered @ vt[:2].T
        lo, hi   = coords.min(0), coords.max(0)
        rng      = hi - lo
        rng[rng == 0] = 1
        norm = (coords - lo) / rng * 0.88 + 0.06
        return [(float(x), float(y)) for x, y in norm]
    except Exception:
        n = len(matrix)
        return [(i / max(n - 1, 1), 0.5) for i in range(n)]


def _umap_2d(matrix: list[list]) -> list[tuple[float, float]]:
    """UMAP reduction — returns None if umap-learn not installed."""
    try:
        import numpy as np
        from umap import UMAP
        arr     = np.array(matrix, dtype=np.float32)
        n_nb    = min(len(matrix) - 1, 5)
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_nb)
        coords  = reducer.fit_transform(arr)
        lo, hi  = coords.min(0), coords.max(0)
        rng     = hi - lo
        rng[rng == 0] = 1
        norm = (coords - lo) / rng * 0.88 + 0.06
        return [(float(x), float(y)) for x, y in norm]
    except Exception:
        return None


class SmartPromptTrainer:
    # Stored as a single JSON file — tiny (~50 KB for 20 examples)
    WEIGHTS_SUBPATH = 'smart_data.json'

    def __init__(self):
        self._texts:      list[str]  = []
        self._embeddings: list[list] = []
        self._embed_model            = None
        self.trained                 = False
        self.history:     dict       = {'loss': [0.0]}
        self._viz_2d:     list       = []   # [{x, y, text, idx}]
        self._last_generate_meta: dict | None = None

    # ── Data ──────────────────────────────────────────────────────────────────

    def add_text(self, text: str) -> None:
        self._texts.append(text.strip())

    def remove_text(self, index: int) -> None:
        if index < 0 or index >= len(self._texts):
            raise IndexError('Text index out of range')
        self._texts.pop(index)
        self._embeddings = []
        self._embed_model            = None
        self.trained                 = False
        self.history                 = {'loss': [0.0]}
        self._viz_2d                 = []
        self._last_generate_meta     = None

    def clear(self) -> None:
        self._texts.clear()
        self._embeddings.clear()
        self._embed_model            = None
        self.trained                 = False
        self.history                 = {'loss': [0.0]}
        self._viz_2d                 = []
        self._last_generate_meta     = None

    def count(self) -> int:
        return len(self._texts)

    # ── Preparation (the "training" for Smart Prompt) ─────────────────────────

    def train(self, **_):
        """Entry point called by /api/train-stream — wraps prepare()."""
        yield from self._prepare()

    def _prepare(self):
        if not self._texts:
            yield {'phase': 'error', 'message': 'Add at least one text example first.'}
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            yield {
                'phase': 'error',
                'message': 'sentence-transformers not installed — run: pip install sentence-transformers',
            }
            return

        try:
            n = len(self._texts)

            yield {
                'phase':      'start',
                'epochs':     n,           # we "pretend" each text is an epoch for the progress bar
                'n_params':   _EMBED_PARAMS,
                'device':     'cpu',
                'corpus_len': sum(len(t) for t in self._texts),
            }

            # Stage 1 — load / reuse embedding model
            yield {'phase': 'smart_stage', 'stage': 1,
                   'message': '📥 Loading the embedding model (all-MiniLM-L6-v2)…'}
            t0 = time.time()
            if self._embed_model is None:
                self._embed_model = SentenceTransformer(_EMBED_MODEL)

            # Stage 2 — embed each example, emit one "epoch" per text
            yield {'phase': 'smart_stage', 'stage': 2,
                   'message': '🔢 Turning your examples into 384-dimensional vectors…'}
            embeddings = []
            for i, text in enumerate(self._texts):
                emb = self._embed_model.encode(text, normalize_embeddings=True).tolist()
                embeddings.append(emb)
                yield {
                    'phase': 'epoch', 'epoch': i + 1, 'epochs': n,
                    'loss': 0.0, 'sample': None, 'epoch_dur_s': 0, 'eta_s': 0,
                }
            self._embeddings = embeddings

            # Stage 3 — 2-D layout for visualization
            yield {'phase': 'smart_stage', 'stage': 3,
                   'message': '🗺️ Building similarity map…'}
            coords = (_umap_2d(embeddings) or _pca_2d(embeddings)) if len(embeddings) > 1 \
                     else [(0.5, 0.5)]
            self._viz_2d = [
                {'x': coords[i][0], 'y': coords[i][1],
                 'text': self._texts[i][:60], 'idx': i}
                for i in range(n)
            ]

            dur_s = round(time.time() - t0, 1)
            self.trained = True
            self.history = {'loss': [0.0]}

            yield {
                'phase':     'done',
                'sample':    f'Ready! {n} examples → {len(embeddings[0])}-dim vectors',
                'history':   self.history,
                'n_params':  _EMBED_PARAMS,
                'n_examples': n,
                'n_dims':    len(embeddings[0]),
                'viz_2d':    self._viz_2d,
                'dur_s':     dur_s,
            }

        except Exception as exc:
            yield {'phase': 'error', 'message': str(exc)}

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str = '', length: int = 200, temperature: float = 0.9,
                 **_) -> str:
        if not self.trained or not self._embeddings:
            raise RuntimeError('Model not prepared yet. Click "Prepare My Bot" first.')
        if self._embed_model is None:
            raise RuntimeError('Embedding model not loaded.')

        import urllib.request, urllib.error

        query = prompt.strip() or 'write a new example in the style of the training data'
        q_emb = self._embed_model.encode(query, normalize_embeddings=True).tolist()

        # Cosine similarity (embeddings already L2-normalised, so dot product = cosine)
        sims  = [(i, _cosine(q_emb, e)) for i, e in enumerate(self._embeddings)]
        sims.sort(key=lambda x: -x[1])
        top_k = min(5, len(sims))
        top   = sims[:top_k]

        matches = [
            {'text': self._texts[i], 'sim': round(s * 100, 1), 'idx': i}
            for i, s in top
        ]

        examples_block = '\n\n'.join(
            f'[Example {j + 1}]\n{m["text"]}' for j, m in enumerate(matches)
        )
        system = (
            'You are a creative writing assistant that generates new content matching a specific style.\n'
            'Below are example texts showing the style you must match.\n\n'
            f'{examples_block}\n\n'
            'Now write ONE new piece of text in EXACTLY this style. '
            'Output only the new text, nothing else.'
        )
        if prompt:
            system += f'\n\nTopic / starting point: "{prompt}"'

        payload = json.dumps({
            'model':  _OLLAMA_MODEL,
            'prompt': system,
            'stream': False,
            'options': {'temperature': max(0.1, min(float(temperature), 2.0))},
        }).encode()

        req = urllib.request.Request(
            f'{_OLLAMA_URL}/api/generate',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
            text = data.get('response', '').strip()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f'Cannot reach Ollama at {_OLLAMA_URL}. '
                'Make sure Ollama is running and try again.'
            ) from e
        except Exception as e:
            raise RuntimeError(f'Ollama error: {e}') from e

        self._last_generate_meta = {
            'matches':        matches,
            'viz_highlights': [i for i, _ in top],
        }
        return text

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            'texts':      self._texts,
            'embeddings': self._embeddings,
            'viz_2d':     self._viz_2d,
        }))

    def load(self, path) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError('sentence-transformers not installed') from e
        p    = Path(path)
        data = json.loads(p.read_text())
        self._texts      = data['texts']
        self._embeddings = data['embeddings']
        self._viz_2d     = data.get('viz_2d', [])
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(_EMBED_MODEL)
        self.trained = True
        self.history = {'loss': [0.0]}

    def get_info(self) -> dict:
        n_dims = len(self._embeddings[0]) if self._embeddings else 0
        return {
            'trained':    self.trained,
            'count':      len(self._texts),
            'n_params':   _EMBED_PARAMS,
            'loss':       None,
            'n_dims':     n_dims,
            'viz_2d':     self._viz_2d,
        }
