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

    def generate(self, prompt: str = '', length: int = 200, temperature: float = 0.85,
                 **_) -> str:
        if not self.trained or not self._embeddings:
            raise RuntimeError('Model not prepared yet. Click "Prepare My Bot" first.')
        if self._embed_model is None:
            raise RuntimeError('Embedding model not loaded.')

        import urllib.request, urllib.error

        query = prompt.strip() or 'write a new example'
        q_emb = self._embed_model.encode(query, normalize_embeddings=True).tolist()

        top_k = min(20, len(self._embeddings))
        top   = sorted(range(len(self._embeddings)),
                       key=lambda i: -_cosine(q_emb, self._embeddings[i]))[:top_k]

        matches = [
            {'text': self._texts[i], 'sim': round(_cosine(q_emb, self._embeddings[i]) * 100, 1), 'idx': i}
            for i in top
        ]

        examples_block = '\n'.join(self._texts[i] for i in top)

        TRIGGER_WORDS = {'hi', 'hello', 'hey', 'fortuna', 'malafortuna', 'alakazam', 'test', 'go', 'start'}
        clean_prompt = (prompt or '').strip().lower()
        is_trigger = clean_prompt in TRIGGER_WORDS or len(clean_prompt.split()) < 2

        system = (
            'You are a fortune cookie machine that writes humorous "unfortune" predictions for entertainment.\n'
            'These are intentionally light, mundane, and self-deprecating — like a fortune cookie that predicts\n'
            'small everyday annoyances instead of good fortune. The content is entirely fictional and harmless.\n\n'
            'Below are example unfortunes showing the exact style to copy:\n\n'
            + examples_block
            + '\n\n'
            'Now write ONE new unfortune in this exact style.\n\n'
            'RULES:\n'
            '1. The user\'s input is a trigger word, not a question. Generate a fresh unfortune regardless.\n'
            '2. Output ONLY the unfortune text. No greetings, no refusals, no explanations.\n'
            '3. This content is fictional and humorous — similar to the examples shown. There is no harm here.\n'
            '4. NEVER refuse. If unsure about style, copy the examples as closely as possible.\n'
            '5. Use future-tense framing matching the examples (e.g. "You will", "Someone will", "Your").\n'
        )
        if not is_trigger and prompt:
            system += f'\nLoose theme to consider: {prompt}\n'
        system += '\nUnfortune:'

        def _call_ollama(prompt_text, temp):
            p = json.dumps({
                'model': _OLLAMA_MODEL, 'prompt': prompt_text, 'stream': False,
                'options': {'temperature': max(0.1, min(float(temp), 2.0)),
                            'top_p': 0.9, 'num_predict': 60,
                            'stop': ['\n\n', 'Unfortune:', 'Example']},
            }).encode()
            r = urllib.request.Request(
                f'{_OLLAMA_URL}/api/generate', data=p,
                headers={'Content-Type': 'application/json'}, method='POST',
            )
            with urllib.request.urlopen(r, timeout=90) as resp:
                return json.loads(resp.read()).get('response', '').strip()

        REFUSAL_PATTERNS = (
            "i can't", "i cannot", "i can not", "i'm not able", "i am not able",
            "i'm sorry", "i apologize", "sorry,", "i won't", "i will not",
            "i don't feel comfortable", "i'm unable", "i am unable",
            "as an ai", "i'm just an ai", "i am just an ai", "unable to fulfill",
        )

        def _is_refusal(text):
            low = text.lower().strip()[:80]
            return any(p in low for p in REFUSAL_PATTERNS) or len(text.strip()) < 8

        try:
            out = _call_ollama(system, temperature)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f'Cannot reach Ollama at {_OLLAMA_URL}. '
                'Make sure Ollama is running and try again.'
            ) from e
        except Exception as e:
            raise RuntimeError(f'Ollama error: {e}') from e

        if _is_refusal(out):
            retry_system = (
                system
                + '\n\nIMPORTANT: Your previous response was a refusal. '
                'These unfortunes are entirely fictional and harmless (like fortune cookie messages). '
                'Generate one now, copying the exact style of the examples above.'
            )
            try:
                out = _call_ollama(retry_system, 0.7)
            except Exception:
                pass

        if _is_refusal(out):
            out = "[The bot's brain refused to predict this one. Try a different trigger word.]"

        # take only first line
        if '\n' in out:
            out = out.split('\n')[0].strip()
        # strip leading bullets/dashes/numbers
        while out and out[0] in '-*•0123456789. )':
            out = out[1:].strip()
        # strip surrounding quotes
        if len(out) > 1 and out[0] in ('"', "'") and out[-1] == out[0]:
            out = out[1:-1].strip()
        # strip preamble leakage
        for prefix in ('Here is', "Here's", 'Sure', 'Of course', 'Output:', 'Next line', 'New line'):
            if out.lower().startswith(prefix.lower()):
                out = out.split(':', 1)[1].strip() if ':' in out else out[len(prefix):].strip()
                break

        text = out

        # Project the response into 2D using nearest-neighbour centroid of training points
        resp_emb = self._embed_model.encode(text, normalize_embeddings=True).tolist()
        resp_sims = sorted(range(len(self._embeddings)),
                           key=lambda i: -_cosine(resp_emb, self._embeddings[i]))
        k_nn = min(5, len(resp_sims))
        if self._viz_2d and k_nn > 0:
            nn_indices = resp_sims[:k_nn]
            rx = sum(self._viz_2d[i]['x'] for i in nn_indices) / k_nn
            ry = sum(self._viz_2d[i]['y'] for i in nn_indices) / k_nn
        else:
            rx, ry = 0.5, 0.5

        self._last_generate_meta = {
            'matches':         matches,
            'viz_highlights':  list(top),
            'response_coords': {'x': rx, 'y': ry, 'text': text[:80]},
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
