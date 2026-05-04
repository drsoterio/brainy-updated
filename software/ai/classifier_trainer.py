"""
ClassifierTrainer — supervised image / text / audio classifier.

input_mode:    'image' | 'text' | 'audio'
training_mode: None | 'finetune' | 'scratch' (None until user chooses in the UI)

finetune → frozen pretrained backbone + trained linear head
  image/audio: MobileNetV3-small (torchvision)
  text:        DistilBERT embeddings (transformers) + small MLP

scratch → small architecture trained from random weights
  image/audio: tiny 3-layer CNN
  text:        word embedding (nn.Embedding) + MLP

Yield protocol matches the other trainers:
  {'phase':'start', 'epochs':N, 'n_params':N, 'n_classes':N, ...}
  {'phase':'epoch', 'epoch':N, 'loss':float, 'accuracy':float, ...}
  {'phase':'done',  'history':{...}, 'confusion_matrix':[[...]], ...}
  {'phase':'error', 'message':str}
"""
from __future__ import annotations

from typing import Optional

import base64
import io
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset



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


# ── Device selection ─────────────────────────────────────────────────────────

def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Image normalisation constants (ImageNet) ──────────────────────────────────

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD  = [0.229, 0.224, 0.225]


def _train_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(_IMG_MEAN, _IMG_STD),
    ])


def _val_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMG_MEAN, _IMG_STD),
    ])


# ── Scratch architectures ────────────────────────────────────────────────────

class _TinyCNN(nn.Module):
    """3-layer CNN for image/audio scratch mode. Input: 3×224×224."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class _TextMLP(nn.Module):
    """Word-embedding MLP for text scratch mode."""
    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) — mean-pool embeddings
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mask = (x != 0).float().unsqueeze(-1)        # (B, L, 1)
        emb  = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(emb)


class _BertMLP(nn.Module):
    """Small MLP classifier on top of frozen DistilBERT embeddings (768-dim)."""
    def __init__(self, n_classes: int, embed_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Datasets ─────────────────────────────────────────────────────────────────

class _ImageDataset(Dataset):
    def __init__(self, pil_images: list, label_idxs: list, transform):
        self.imgs      = pil_images
        self.labels    = label_idxs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.transform(self.imgs[i]), self.labels[i]


class _TensorDataset(Dataset):
    """Generic: list of tensors + list of int labels."""
    def __init__(self, tensors: list, label_idxs: list):
        self.x = tensors
        self.y = label_idxs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _TextScratchDataset(Dataset):
    """Pads token sequences to fixed length for batch training."""
    def __init__(self, tokens_list: list, label_idxs: list, max_len: int = 64):
        pad = lambda t: (t + [0] * max_len)[:max_len]
        self.x = [torch.tensor(pad(t), dtype=torch.long) for t in tokens_list]
        self.y = label_idxs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ── Utility ───────────────────────────────────────────────────────────────────

def _confusion_matrix(y_true: list, y_pred: list, n: int) -> list[list[int]]:
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t][p] += 1
    return cm


def _decode_image_b64(data: str) -> Image.Image:
    raw = base64.b64decode(data.split(',')[-1])
    return Image.open(io.BytesIO(raw)).convert('RGB')


def _audio_bytes_to_tensor(audio_bytes: bytes,
                            sample_rate: int = 22050,
                            duration: float = 5.0) -> torch.Tensor:
    """Convert raw audio bytes → 3-channel mel-spectrogram tensor (3×224×224)."""
    import librosa
    import numpy as np
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate,
                        mono=True, duration=duration)
    if len(y) < int(sample_rate * duration):
        y = np.pad(y, (0, int(sample_rate * duration) - len(y)))
    spec    = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_n  = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
    t       = torch.from_numpy(np.stack([spec_n] * 3, axis=0).astype('float32'))
    # Resize to 224×224
    t = F.interpolate(t.unsqueeze(0), size=(224, 224),
                      mode='bilinear', align_corners=False).squeeze(0)
    return t


# ── Main class ────────────────────────────────────────────────────────────────

class ClassifierTrainer:
    WEIGHTS_SUBPATH = 'classifier.pt'

    def __init__(self, input_mode: str = 'image', training_mode: Optional[str] = None):
        self.input_mode    = input_mode     # 'image' | 'text' | 'audio'
        self.training_mode = training_mode  # None | 'finetune' | 'scratch'
        self.device        = _best_device()

        self.labels: list[str]        = []
        self._examples: dict[str, list] = {}  # label → [PIL | str | bytes]

        self._model                = None
        self._vocab: dict[str, int] = {}  # scratch text vocab

        # Lazy-loaded for text-finetune prediction
        self._bert_model    = None
        self._bert_tokenizer = None

        self.trained  = False
        self.history: dict = {'loss': [], 'accuracy': []}
        self._confusion: list | None = None

    # ── Labels ───────────────────────────────────────────────────────────────

    def set_labels(self, labels: list[str]) -> None:
        self.labels = [l.strip() for l in labels if l.strip()]
        for lbl in self.labels:
            self._examples.setdefault(lbl, [])

    # ── Data ─────────────────────────────────────────────────────────────────

    def add_example(self, label: str, data: str) -> None:
        """
        data: base64 image/audio string for image/audio mode;
              raw text for text mode.
        """
        if label not in self.labels:
            raise ValueError(f'Label "{label}" not defined. Call set_labels() first.')
        self._examples.setdefault(label, [])
        if self.input_mode == 'image':
            self._examples[label].append(_decode_image_b64(data))
        elif self.input_mode == 'text':
            self._examples[label].append(data.strip())
        elif self.input_mode == 'audio':
            raw = base64.b64decode(data.split(',')[-1])
            self._examples[label].append(raw)

    def remove_example(self, label: str, idx: int) -> None:
        bucket = self._examples.get(label, [])
        if 0 <= idx < len(bucket):
            bucket.pop(idx)

    def clear(self) -> None:
        self.labels      = []
        self._examples   = {}
        self._model      = None
        self._vocab      = {}
        self._bert_model = None
        self._bert_tokenizer = None
        self.trained     = False
        self.history     = {'loss': [], 'accuracy': []}
        self._confusion  = None

    def count(self) -> int:
        return sum(len(v) for v in self._examples.values())

    def count_per_label(self) -> dict[str, int]:
        return {lbl: len(self._examples.get(lbl, [])) for lbl in self.labels}

    # ── Training dispatch ─────────────────────────────────────────────────────

    def train(self, epochs: int = 30, lr: float = 1e-3,
              batch_size: int = 32, **_):
        if len(self.labels) < 2:
            yield {'phase': 'error', 'message': 'Define at least 2 labels before training.'}
            return
        missing = [l for l in self.labels if not self._examples.get(l)]
        if missing:
            yield {'phase': 'error',
                   'message': f'No examples for: {", ".join(missing)}'}
            return
        if self.training_mode not in ('scratch', 'finetune'):
            yield {'phase': 'error',
                   'message': 'Choose Machine Learning or AI before training.'}
            return
        if self.input_mode == 'image':
            yield from self._train_image(epochs, lr, batch_size)
        elif self.input_mode == 'text':
            yield from self._train_text(epochs, lr, batch_size)
        elif self.input_mode == 'audio':
            yield from self._train_audio(epochs, lr, batch_size)
        else:
            yield {'phase': 'error', 'message': f'Unknown input_mode: {self.input_mode}'}

    # ── Image training ────────────────────────────────────────────────────────

    def _train_image(self, epochs, lr, batch_size):
        try:
            from torchvision import models
        except ImportError:
            yield {'phase': 'error',
                   'message': 'torchvision not installed — run: pip install torchvision'}
            return

        try:
            n_classes   = len(self.labels)
            pil_imgs, label_idxs = [], []
            for idx, lbl in enumerate(self.labels):
                for img in self._examples.get(lbl, []):
                    pil_imgs.append(img)
                    label_idxs.append(idx)

            dataset = _ImageDataset(pil_imgs, label_idxs, _train_transform())
            loader  = DataLoader(dataset,
                                 batch_size=min(batch_size, len(dataset)),
                                 shuffle=True, num_workers=0)

            if self.training_mode == 'finetune':
                backbone = models.mobilenet_v3_small(
                    weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                in_f = backbone.classifier[3].in_features
                backbone.classifier[3] = nn.Linear(in_f, n_classes)
                for name, p in backbone.named_parameters():
                    p.requires_grad = name.startswith('classifier')
                self._model = backbone.to(self.device)
            else:
                self._model = _TinyCNN(n_classes).to(self.device)

            n_params    = sum(p.numel() for p in self._model.parameters())
            n_trainable = sum(p.numel() for p in self._model.parameters()
                              if p.requires_grad)
            optimizer   = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self._model.parameters()), lr=lr)
            criterion   = nn.CrossEntropyLoss()
            self.history = {'loss': [], 'accuracy': []}
            self.trained  = False
            ckpt_every    = max(1, epochs // 8)

            yield {
                'phase':         'start',
                'epochs':        epochs,
                'n_params':      n_params,
                'n_trainable':   n_trainable,
                'n_classes':     n_classes,
                'labels':        self.labels,
                'training_mode': self.training_mode,
                'input_mode':    self.input_mode,
                'device':        str(self.device),
                'dataset_size':  len(dataset),
                'corpus_len':    len(dataset),
            }

            yield from self._epoch_loop(loader, optimizer, criterion,
                                        epochs, n_classes, 'image',
                                        pil_imgs, label_idxs, batch_size)

        except Exception as exc:
            yield {'phase': 'error', 'message': str(exc)}

    # ── Text training ─────────────────────────────────────────────────────────

    def _train_text(self, epochs, lr, batch_size):
        try:
            n_classes     = len(self.labels)
            all_texts, label_idxs = [], []
            for idx, lbl in enumerate(self.labels):
                for txt in self._examples.get(lbl, []):
                    all_texts.append(txt)
                    label_idxs.append(idx)

            if self.training_mode == 'finetune':
                try:
                    from transformers import AutoTokenizer, AutoModel
                except ImportError:
                    yield {'phase': 'error',
                           'message': 'transformers not installed — pip install transformers'}
                    return

                yield {
                    'phase': 'start', 'epochs': epochs,
                    'n_params': 0,   # MLP head only; BERT is frozen
                    'n_classes': n_classes, 'labels': self.labels,
                    'training_mode': self.training_mode, 'input_mode': self.input_mode,
                    'device': str(self.device), 'dataset_size': len(all_texts),
                    'corpus_len': sum(len(t) for t in all_texts),
                    'smart_stage': True,
                }
                yield {'phase': 'smart_stage', 'stage': 1,
                       'message': '📥 Loading DistilBERT (frozen)…'}

                tok  = AutoTokenizer.from_pretrained('distilbert-base-uncased')
                bert = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)
                bert.eval()
                # Keep references for prediction
                self._bert_tokenizer = tok
                self._bert_model     = bert

                yield {'phase': 'smart_stage', 'stage': 2,
                       'message': '🔢 Embedding training texts…'}

                embeddings = self._embed_texts(all_texts)
                embed_dim  = embeddings.shape[1]

                self._model = _BertMLP(n_classes, embed_dim).to(self.device)
                n_params    = sum(p.numel() for p in self._model.parameters())
                n_trainable = n_params

                dataset = _TensorDataset(
                    [embeddings[i] for i in range(len(embeddings))], label_idxs)

            else:  # scratch
                # Build word vocabulary
                vocab: dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
                for txt in all_texts:
                    for word in txt.lower().split():
                        if word not in vocab:
                            vocab[word] = len(vocab)
                self._vocab = vocab

                tokens_list = [
                    [vocab.get(w, 1) for w in txt.lower().split() or ['<UNK>']]
                    for txt in all_texts
                ]
                self._model = _TextMLP(len(vocab), n_classes).to(self.device)
                n_params    = sum(p.numel() for p in self._model.parameters())
                n_trainable = n_params
                dataset     = _TextScratchDataset(tokens_list, label_idxs)

                yield {
                    'phase': 'start', 'epochs': epochs,
                    'n_params': n_params, 'n_trainable': n_trainable,
                    'n_classes': n_classes, 'labels': self.labels,
                    'training_mode': self.training_mode, 'input_mode': self.input_mode,
                    'device': str(self.device), 'dataset_size': len(all_texts),
                    'corpus_len': sum(len(t) for t in all_texts),
                }

            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            self.history = {'loss': [], 'accuracy': []}
            self.trained  = False
            loader = DataLoader(dataset,
                                batch_size=min(batch_size, len(dataset)),
                                shuffle=True, num_workers=0)

            yield from self._epoch_loop(loader, optimizer, criterion,
                                        epochs, n_classes, 'text',
                                        all_texts, label_idxs, batch_size)

        except Exception as exc:
            yield {'phase': 'error', 'message': str(exc)}

    # ── Audio training ────────────────────────────────────────────────────────

    def _train_audio(self, epochs, lr, batch_size):
        try:
            import librosa  # noqa: F401
        except ImportError:
            yield {'phase': 'error',
                   'message': 'librosa not installed — run: pip install librosa soundfile'}
            return
        try:
            from torchvision import models
        except ImportError:
            yield {'phase': 'error',
                   'message': 'torchvision not installed — pip install torchvision'}
            return

        try:
            n_classes = len(self.labels)
            tensors, label_idxs = [], []
            for idx, lbl in enumerate(self.labels):
                for audio_bytes in self._examples.get(lbl, []):
                    tensors.append(_audio_bytes_to_tensor(audio_bytes))
                    label_idxs.append(idx)

            dataset = _TensorDataset(tensors, label_idxs)
            loader  = DataLoader(dataset,
                                 batch_size=min(batch_size, len(dataset)),
                                 shuffle=True, num_workers=0)

            if self.training_mode == 'finetune':
                backbone = models.mobilenet_v3_small(
                    weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                in_f = backbone.classifier[3].in_features
                backbone.classifier[3] = nn.Linear(in_f, n_classes)
                for name, p in backbone.named_parameters():
                    p.requires_grad = name.startswith('classifier')
                self._model = backbone.to(self.device)
            else:
                self._model = _TinyCNN(n_classes).to(self.device)

            n_params    = sum(p.numel() for p in self._model.parameters())
            n_trainable = sum(p.numel() for p in self._model.parameters()
                              if p.requires_grad)
            optimizer   = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self._model.parameters()), lr=lr)
            criterion   = nn.CrossEntropyLoss()
            self.history = {'loss': [], 'accuracy': []}
            self.trained  = False

            yield {
                'phase': 'start', 'epochs': epochs,
                'n_params': n_params, 'n_trainable': n_trainable,
                'n_classes': n_classes, 'labels': self.labels,
                'training_mode': self.training_mode, 'input_mode': self.input_mode,
                'device': str(self.device), 'dataset_size': len(dataset),
                'corpus_len': len(dataset),
            }

            yield from self._epoch_loop(loader, optimizer, criterion,
                                        epochs, n_classes, 'audio',
                                        tensors, label_idxs, batch_size)

        except Exception as exc:
            yield {'phase': 'error', 'message': str(exc)}

    # ── Shared training loop ──────────────────────────────────────────────────

    def _epoch_loop(self, loader, optimizer, criterion,
                    epochs, n_classes, data_type,
                    raw_data, label_idxs, batch_size):
        """
        Shared per-epoch logic for all three input modes.
        Yields 'epoch' events, then a 'done' event with confusion matrix.
        """
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self._model.train()
            total_loss, correct, total = 0.0, 0, 0

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = torch.tensor(yb, dtype=torch.long).to(self.device) \
                     if not isinstance(yb, torch.Tensor) else yb.to(self.device)
                optimizer.zero_grad()
                logits = self._model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                preds       = logits.argmax(dim=1)
                correct    += (preds == yb).sum().item()
                total      += yb.size(0)

            dur      = time.time() - t0
            avg_loss = total_loss / max(len(loader), 1)
            acc      = correct / max(total, 1)
            self.history['loss'].append(round(avg_loss, 4))
            self.history['accuracy'].append(round(acc, 4))

            yield {
                'phase':       'epoch',
                'epoch':       epoch,
                'epochs':      epochs,
                'loss':        round(avg_loss, 4),
                'accuracy':    round(acc * 100, 1),
                'sample':      f'{round(acc*100,1)}% accuracy',
                'epoch_dur_s': round(dur, 2),
                'eta_s':       round(dur * (epochs - epoch)),
            }

        # ── Confusion matrix on full training set ─────────────────────────
        self._model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = self._model(xb)
                y_pred.extend(logits.argmax(1).cpu().tolist())
                y_true.extend(
                    yb.tolist() if isinstance(yb, torch.Tensor)
                    else list(yb)
                )
        self._confusion = _confusion_matrix(y_true, y_pred, n_classes)
        self.trained = True

        # PCA on model logits for each training example
        pca_points: list = []
        pca_labels_out: list = []
        try:
            reps, rep_labels = [], []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    logits = self._model(xb)
                    reps.append(logits.cpu().numpy())
                    yl = yb.tolist() if isinstance(yb, torch.Tensor) else list(yb)
                    rep_labels.extend([self.labels[i] for i in yl
                                       if isinstance(i, int) and i < len(self.labels)])
            if reps and len(rep_labels) >= 2:
                arr = np.vstack(reps)
                pca_points = _pca2d(arr)
                pca_labels_out = rep_labels
        except Exception:
            pass

        n_params = sum(p.numel() for p in self._model.parameters())
        yield {
            'phase':            'done',
            'history':          self.history,
            'n_params':         n_params,
            'labels':           self.labels,
            'confusion_matrix': self._confusion,
            'final_accuracy':   round(self.history['accuracy'][-1] * 100, 1),
            'sample':           f'{round(self.history["accuracy"][-1]*100,1)}% accuracy',
            'pca_points':       pca_points,
            'pca_labels':       pca_labels_out,
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, data: str) -> dict:
        """
        data: base64 image/audio string, or plain text for text mode.
        Returns: {'label': str, 'confidence': float, 'probs': {label: float}}
        """
        if not self.trained or self._model is None:
            raise RuntimeError('Model not trained yet.')
        self._model.eval()
        with torch.no_grad():
            if self.input_mode == 'image':
                probs = self._predict_image(data)
            elif self.input_mode == 'text':
                probs = self._predict_text(data)
            elif self.input_mode == 'audio':
                probs = self._predict_audio(data)
            else:
                raise RuntimeError(f'Unknown input_mode: {self.input_mode}')
        return self._fmt(probs)

    def _predict_image(self, data_b64: str) -> list[float]:
        img = _decode_image_b64(data_b64)
        x   = _val_transform()(img).unsqueeze(0).to(self.device)
        return F.softmax(self._model(x), dim=1)[0].cpu().tolist()

    def _predict_text(self, text: str) -> list[float]:
        if self.training_mode == 'finetune':
            emb = self._embed_texts([text]).to(self.device)
            return F.softmax(self._model(emb), dim=1)[0].cpu().tolist()
        else:
            tokens = [self._vocab.get(w, 1) for w in text.lower().split() or ['<UNK>']]
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            return F.softmax(self._model(x), dim=1)[0].cpu().tolist()

    def _predict_audio(self, data_b64: str) -> list[float]:
        raw = base64.b64decode(data_b64.split(',')[-1])
        t   = _audio_bytes_to_tensor(raw).unsqueeze(0).to(self.device)
        return F.softmax(self._model(t), dim=1)[0].cpu().tolist()

    def _fmt(self, probs: list[float]) -> dict:
        best = int(max(range(len(probs)), key=lambda i: probs[i]))
        return {
            'label':      self.labels[best],
            'confidence': round(probs[best] * 100, 1),
            'probs':      {self.labels[i]: round(p * 100, 1)
                           for i, p in enumerate(probs)},
        }

    # ── DistilBERT helper ─────────────────────────────────────────────────────

    def _load_bert_if_needed(self) -> None:
        if self._bert_model is None:
            from transformers import AutoTokenizer, AutoModel
            self._bert_tokenizer = AutoTokenizer.from_pretrained(
                'distilbert-base-uncased')
            self._bert_model = AutoModel.from_pretrained(
                'distilbert-base-uncased').to(self.device)
            self._bert_model.eval()

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        self._load_bert_if_needed()
        enc  = self._bert_tokenizer(
            texts, padding=True, truncation=True, max_length=64,
            return_tensors='pt').to(self.device)
        out  = self._bert_model(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return emb.cpu()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        if not self.trained or self._model is None:
            raise RuntimeError('No trained model to save.')
        torch.save({
            'model_state':   self._model.state_dict(),
            'labels':        self.labels,
            'input_mode':    self.input_mode,
            'training_mode': self.training_mode,
            'history':       self.history,
            'confusion':     self._confusion,
            'vocab':         self._vocab,
            # architecture hints for load()
            'n_classes':     len(self.labels),
            'arch':          self._arch_tag(),
        }, path)

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.labels        = ckpt['labels']
        self.input_mode    = ckpt['input_mode']
        self.training_mode = ckpt['training_mode']
        self.history       = ckpt.get('history', {'loss': [], 'accuracy': []})
        self._confusion    = ckpt.get('confusion')
        self._vocab        = ckpt.get('vocab', {})
        n_classes          = ckpt['n_classes']

        if self.input_mode in ('image', 'audio'):
            if self.training_mode == 'finetune':
                from torchvision import models
                bb   = models.mobilenet_v3_small(weights=None)
                in_f = bb.classifier[3].in_features
                bb.classifier[3] = nn.Linear(in_f, n_classes)
                self._model = bb
            else:
                self._model = _TinyCNN(n_classes)
        elif self.input_mode == 'text':
            if self.training_mode == 'finetune':
                self._model = _BertMLP(n_classes, embed_dim=768)
            else:
                vocab_size  = len(self._vocab) or max(self._vocab.values(), default=1) + 1
                self._model = _TextMLP(vocab_size, n_classes)

        self._model.load_state_dict(ckpt['model_state'])
        self._model.to(self.device).eval()
        self.trained = True

    def _arch_tag(self) -> str:
        tm = self.training_mode or 'unset'
        return f'{self.input_mode}-{tm}'

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        base = {
            'trained':        self.trained,
            'count':          self.count(),
            'count_per_label': self.count_per_label(),
            'labels':         self.labels,
            'n_classes':      len(self.labels),
            'input_mode':     self.input_mode,
            'training_mode':  self.training_mode,
        }
        if self._model is not None:
            base['n_params']         = sum(p.numel() for p in self._model.parameters())
            base['loss']             = (self.history['loss'][-1]
                                        if self.history['loss'] else None)
            base['accuracy']         = (self.history['accuracy'][-1]
                                        if self.history['accuracy'] else None)
            base['confusion_matrix'] = self._confusion
        return base
