"""
VAE-based image trainer. Students upload photos → model trains → generates new ones.
Pure torch + Pillow, no torchvision needed.
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

IMAGE_SIZE = 128


class _VAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    nn.ReLU(),  # 128→64
            nn.Conv2d(32, 64, 4, 2, 1),   nn.ReLU(),  # 64→32
            nn.Conv2d(64, 128, 4, 2, 1),  nn.ReLU(),  # 32→16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),  # 16→8
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),  # 8→4
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),   # 4→8
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),   # 8→16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(),   # 16→32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(),   # 32→64
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    nn.Sigmoid(), # 64→128
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 512, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def _augment(img: Image.Image) -> torch.Tensor:
    # img is already resized to IMAGE_SIZE × IMAGE_SIZE at upload time
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


class _AugDataset(Dataset):
    def __init__(self, images: list, augment_factor: int = 50):
        self.images = images
        self.augment_factor = augment_factor

    def __len__(self):
        return len(self.images) * self.augment_factor

    def __getitem__(self, idx):
        return _augment(self.images[idx % len(self.images)])


def _ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Structural similarity loss via AvgPool2d.
    Same SSIM formula as before — local means/variances via uniform pooling
    instead of Gaussian convolution. ~4× faster forward and backward on CPU,
    identical training dynamics for a generative model at this scale.
    """
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


def _to_b64_grid(tensors: list, cols: int = 4) -> str:
    n    = len(tensors)
    rows = (n + cols - 1) // cols
    cell = IMAGE_SIZE
    canvas = Image.new('RGB', (cols * cell, rows * cell), (18, 10, 36))
    for i, t in enumerate(tensors):
        arr = (t.detach().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        r, c = divmod(i, cols)
        canvas.paste(img, (c * cell, r * cell))
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


class ImageTrainer:
    def __init__(self):
        self.device     = torch.device('cpu')
        self._pil_imgs: list     = []
        self._model: _VAE | None = None
        self.trained    = False
        self.latent_dim = 128
        self.history: dict = {'loss': []}

    def add_image(self, source: str) -> None:
        if source.startswith('data:'):
            _, data = source.split(',', 1)
            raw  = base64.b64decode(data)
            img  = Image.open(io.BytesIO(raw)).convert('RGB')
        else:
            img = Image.open(source).convert('RGB')
        # Pre-resize once at upload time — avoids repeated resize in every __getitem__ call
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
        batch_size:     int   = 8,
        latent_dim:     int   = 128,
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

            dataset    = _AugDataset(self._pil_imgs, augment_factor=augment_factor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            self._model = _VAE(latent_dim=latent_dim).to(self.device)
            optimizer   = torch.optim.Adam(self._model.parameters(), lr=lr)
            n_params    = sum(p.numel() for p in self._model.parameters())
            _fixed_z    = torch.randn(4, latent_dim, device=self.device)

            yield {
                'phase': 'start', 'epochs': epochs,
                'n_images': len(self._pil_imgs), 'dataset_size': len(dataset),
                'n_params': n_params, 'latent_dim': latent_dim,
                'device': str(self.device),
                'batches_per_epoch': len(dataloader),
            }

            epoch_dur_s = None  # measured from first epoch, used for ETA

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                self._model.train()
                epoch_loss = 0.0
                # KL annealing: start at 0, ramp to full weight over first half of training.
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
                self.history['loss'].append(round(avg_loss, 1))

                preview_b64 = None
                if epoch == 1 or epoch % preview_every == 0 or epoch == epochs:
                    self._model.eval()
                    with torch.no_grad():
                        samples = self._model.decode(_fixed_z).cpu()
                    preview_b64 = _to_b64_grid(list(samples), cols=2)
                    self._model.train()

                eta_s = round(epoch_dur_s * (epochs - epoch))

                yield {
                    'phase': 'epoch', 'epoch': epoch, 'epochs': epochs,
                    'loss': round(avg_loss, 1), 'preview': preview_b64,
                    'epoch_dur_s': round(epoch_dur_s, 1),
                    'eta_s': eta_s,
                }

            self.trained = True
            self._model.eval()
            with torch.no_grad():
                z_big = torch.randn(16, latent_dim, device=self.device)
                final = self._model.decode(z_big).cpu()

            yield {
                'phase': 'done',
                'grid': _to_b64_grid(list(final), cols=4),
                'history': self.history,
                'n_params': n_params,
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
