"""
VAE-based image trainer. Students upload photos → model trains → generates new ones.
Pure torch + Pillow, no torchvision needed.

Performance notes (96×96 images, Mac M-series):
  - MPS backend gives ~8× speedup over CPU for the backward pass.
  - Augmented tensors are pre-built once before the epoch loop;
    DataLoader simply indexes into the cache — zero per-batch augmentation cost.
  - augment_factor is capped at 8 internally so per-epoch batch count stays low
    (~2-3 batches for a typical 5-10 image demo), keeping epoch time ~1-2s on MPS.
"""
from __future__ import annotations

import base64
import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 96×96 inputs → encoder bottleneck 512×3×3 = 4608 floats.
IMAGE_SIZE = 96

# Cap augment_factor to keep per-epoch batch count manageable.
# 5 imgs × 8 = 40 samples → 2 batches at batch_size=32.
_AUG_CAP = 8


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


_BOTTLENECK_SPATIAL = 3   # 96 → 48 → 24 → 12 → 6 → 3 after 5 stride-2 convs
_BOTTLENECK_FLAT    = 512 * _BOTTLENECK_SPATIAL * _BOTTLENECK_SPATIAL  # 4608


class _VAE(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    nn.ReLU(),  # 96→48
            nn.Conv2d(32, 64, 4, 2, 1),   nn.ReLU(),  # 48→24
            nn.Conv2d(64, 128, 4, 2, 1),  nn.ReLU(),  # 24→12
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),  # 12→6
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),  # 6→3
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(_BOTTLENECK_FLAT, latent_dim)
        self.fc_logvar = nn.Linear(_BOTTLENECK_FLAT, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, _BOTTLENECK_FLAT)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),    # 3→6
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),    # 6→12
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(),    # 12→24
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(),    # 24→48
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    nn.Sigmoid(), # 48→96
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 512, _BOTTLENECK_SPATIAL, _BOTTLENECK_SPATIAL))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def _augment(img: Image.Image) -> torch.Tensor:
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = float(np.random.uniform(-10, 10))
    img = img.rotate(angle, resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * float(np.random.uniform(0.8, 1.2)), 0, 255)
    mean = arr.mean()
    arr = np.clip((arr - mean) * float(np.random.uniform(0.8, 1.2)) + mean, 0, 255)
    t = torch.from_numpy(arr.astype(np.uint8)).float() / 255.0
    return t.permute(2, 0, 1)


def _build_aug_cache(pil_imgs: list, augment_factor: int) -> torch.Tensor:
    """Generate all augmented tensors once on CPU. Returns (N*aug, 3, H, W)."""
    tensors = []
    for img in pil_imgs:
        for _ in range(augment_factor):
            tensors.append(_augment(img))
    return torch.stack(tensors)


class _CachedDataset(Dataset):
    """Wraps a pre-built tensor so DataLoader returns plain tensors (not tuples)."""
    def __init__(self, cache: torch.Tensor):
        self.cache = cache

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        return self.cache[i]


def _ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Structural similarity loss via AvgPool2d (~4× faster than Gaussian on CPU/MPS)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    p    = window_size // 2
    pool = lambda t: F.avg_pool2d(t, window_size, stride=1, padding=p, count_include_pad=False)
    mu1  = pool(pred);   mu2  = pool(target)
    mu1s = mu1 * mu1;    mu2s = mu2 * mu2;    mu12 = mu1 * mu2
    s1   = pool(pred   * pred)   - mu1s
    s2   = pool(target * target) - mu2s
    s12  = pool(pred   * target) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s1 + s2 + C2))
    return (1 - ssim_map).mean()


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


def _to_b64_grid(tensors: list, cols: int = 4) -> str:
    n    = len(tensors)
    rows = (n + cols - 1) // cols
    cell = IMAGE_SIZE
    canvas = Image.new('RGB', (cols * cell, rows * cell), (18, 10, 36))
    for i, t in enumerate(tensors):
        arr = (t.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        r, c = divmod(i, cols)
        canvas.paste(img, (c * cell, r * cell))
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


class ImageTrainer:
    def __init__(self):
        self.device     = _best_device()
        self._pil_imgs: list     = []
        self._model: _VAE | None = None
        self.trained    = False
        self.latent_dim = 256
        self.history: dict = {'loss': []}

    def add_image(self, source: str) -> None:
        if source.startswith('data:'):
            _, data = source.split(',', 1)
            raw  = base64.b64decode(data)
            img  = Image.open(io.BytesIO(raw)).convert('RGB')
        else:
            img = Image.open(source).convert('RGB')
        self._pil_imgs.append(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

    def remove_image(self, index: int) -> None:
        if index < 0 or index >= len(self._pil_imgs):
            raise IndexError('Image index out of range')
        self._pil_imgs.pop(index)
        self._model  = None
        self.trained = False
        self.history = {'loss': []}

    def clear(self) -> None:
        self._pil_imgs.clear()
        self._model  = None
        self.trained = False
        self.history = {'loss': []}

    def count(self) -> int:
        return len(self._pil_imgs)

    def thumbnail_b64(self, index: int, size: int = 80) -> str:
        img = self._pil_imgs[index].resize((size, size), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    def train(
        self,
        epochs:         int   = 60,
        lr:             float = 1e-3,
        batch_size:     int   = 32,
        latent_dim:     int   = 256,
        augment_factor: int   = 50,
        preview_every:  int   = 5,
    ):
        if len(self._pil_imgs) < 2:
            yield {'phase': 'error', 'message': 'Upload at least 2 images to start training.'}
            return

        try:
            self.latent_dim = latent_dim
            self.history    = {'loss': []}
            self.trained    = False

            # Cap augmentation to keep per-epoch batch count small (fast epochs).
            aug = min(augment_factor, _AUG_CAP)

            # Build the full augmented dataset ONCE on CPU, then reuse every epoch.
            cache      = _build_aug_cache(self._pil_imgs, aug)
            dataset    = _CachedDataset(cache)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            self._model = _VAE(latent_dim=latent_dim).to(self.device)
            optimizer   = torch.optim.Adam(self._model.parameters(), lr=lr)
            n_params    = sum(p.numel() for p in self._model.parameters())
            _preview_n    = batch_size              # match student's batch-size selection exactly
            _preview_cols = max(1, min(_preview_n, 8))
            _fixed_z      = torch.randn(_preview_n, latent_dim, device=self.device)

            yield {
                'phase': 'start', 'epochs': epochs,
                'n_images': len(self._pil_imgs), 'dataset_size': len(dataset),
                'n_params': n_params, 'latent_dim': latent_dim,
                'device': str(self.device),
                'batches_per_epoch': len(dataloader),
            }

            epoch_dur_s = None

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                self._model.train()
                epoch_loss = 0.0
                kl_w = min(1.0, 2.0 * (epoch - 1) / max(1, epochs - 1)) * 0.1
                for batch in dataloader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    recon, mu, logvar = self._model(batch)
                    rec_loss = 0.75 * _ssim(recon, batch) + 0.25 * F.l1_loss(recon, batch)
                    kl_loss  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss     = rec_loss + kl_w * kl_loss
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                epoch_dur_s = time.time() - epoch_start
                avg_loss = epoch_loss / max(len(dataloader), 1)
                self.history['loss'].append(round(avg_loss, 4))

                preview_b64 = None
                if epoch == 1 or epoch % preview_every == 0 or epoch == epochs:
                    self._model.eval()
                    with torch.no_grad():
                        samples = self._model.decode(_fixed_z).cpu()
                    preview_b64 = _to_b64_grid(list(samples), cols=_preview_cols)
                    self._model.train()

                yield {
                    'phase': 'epoch', 'epoch': epoch, 'epochs': epochs,
                    'loss': round(avg_loss, 4), 'preview': preview_b64,
                    'preview_n': _preview_n,
                    'epoch_dur_s': round(epoch_dur_s, 2),
                    'eta_s': round(epoch_dur_s * (epochs - epoch)),
                }

            self.trained = True
            self._model.eval()
            with torch.no_grad():
                z_big = torch.randn(16, latent_dim, device=self.device)
                final = self._model.decode(z_big).cpu()

            # PCA, diversity, and per-image reconstruction errors
            pca_points: list = []
            pca_labels: list = []
            diversity: float = 0.0
            recon_errors: list = []
            try:
                with torch.no_grad():
                    raw = [torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
                           for img in self._pil_imgs]
                    xs  = torch.stack(raw).to(self.device)
                    mu, _ = self._model.encode(xs)
                    # Diversity: std across 16 random samples
                    z_div = torch.randn(16, latent_dim, device=self.device)
                    samples_div = self._model.decode(z_div)
                    diversity = round(float(samples_div.std(dim=0).mean().item()), 4)
                    # Per-image MSE reconstruction error
                    recons = self._model.decode(mu)
                    for orig, rec in zip(xs, recons):
                        mse = float(F.mse_loss(rec, orig).item())
                        recon_errors.append(round(mse, 5))
                pca_points = _pca2d(mu.cpu().numpy())
                pca_labels = [f'img {i+1}' for i in range(len(self._pil_imgs))]
            except Exception:
                pass

            yield {
                'phase':        'done',
                'grid':         _to_b64_grid(list(final), cols=4),
                'history':      self.history,
                'n_params':     n_params,
                'pca_points':   pca_points,
                'pca_labels':   pca_labels,
                'diversity':    diversity,
                'recon_errors': recon_errors,
            }

        except Exception as exc:
            yield {'phase': 'error', 'message': str(exc)}

    def generate(self, n: int = 16) -> str:
        if not self.trained or self._model is None:
            raise RuntimeError('Model not trained yet.')
        self._model.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            s = self._model.decode(z).cpu()
        return _to_b64_grid(list(s), cols=1 if n == 1 else 4)

    def interpolate(self, steps: int = 10) -> str:
        if not self.trained or self._model is None:
            raise RuntimeError('Model not trained yet.')
        self._model.eval()
        with torch.no_grad():
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)
            imgs = [
                self._model.decode((1 - i / (steps - 1)) * z1 + (i / (steps - 1)) * z2).cpu().squeeze()
                for i in range(steps)
            ]
        return _to_b64_grid(imgs, cols=steps)

    def save(self, path) -> None:
        if not self.trained or self._model is None:
            raise RuntimeError('No trained model to save.')
        torch.save({
            'model_state': self._model.state_dict(),
            'latent_dim':  self.latent_dim,
            'history':     self.history,
        }, path)

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.latent_dim = ckpt['latent_dim']
        self._model = _VAE(latent_dim=self.latent_dim).to(self.device)
        self._model.load_state_dict(ckpt['model_state'])
        self._model.eval()
        self.trained = True
        self.history = ckpt.get('history', {'loss': []})

    def get_info(self) -> dict:
        if self._model is None:
            return {'trained': False, 'count': len(self._pil_imgs)}
        return {
            'trained':    self.trained,
            'n_params':   sum(p.numel() for p in self._model.parameters()),
            'latent_dim': self.latent_dim,
            'count':      len(self._pil_imgs),
            'loss':       self.history['loss'][-1] if self.history['loss'] else None,
        }
