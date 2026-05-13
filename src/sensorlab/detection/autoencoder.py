"""LSTM autoencoder — recurrent deep detector that captures temporal dependence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sensorlab.detection.base import BaseDetector


class _LSTMAEModule(nn.Module):
    """Encoder-decoder reconstructing the full input window."""

    def __init__(self, n_features: int, hidden: int = 32, latent: int = 8, n_layers: int = 1):
        super().__init__()
        self.hidden = hidden
        self.latent = latent
        self.encoder = nn.LSTM(n_features, hidden, num_layers=n_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)  # h: (n_layers, B, hidden)
        z = self.to_latent(h[-1])  # (B, latent)
        # Replicate latent across timesteps as decoder input
        seq_len = x.size(1)
        dec_in = self.from_latent(z).unsqueeze(1).expand(-1, seq_len, -1)
        y, _ = self.decoder(dec_in)
        return self.head(y)


@dataclass
class LSTMAutoencoder(BaseDetector):
    """Train on normal windows, score by mean squared reconstruction error per window.

    Designed to be small and CPU-friendly so the test suite stays fast.
    """

    window: int = 20
    hidden: int = 32
    latent: int = 8
    n_layers: int = 1
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 0
    device: str = "cpu"
    name: str = "LSTM-AE"
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._model: _LSTMAEModule | None = None
        self._n_features: int | None = None

    # ----- public API -------------------------------------------------------

    def fit(self, X_normal: np.ndarray) -> Self:
        """Fit on a 3D array of normal windows ``(n_windows, window, n_features)``."""
        X = self._ensure_windows(X_normal)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._n_features = int(X.shape[-1])
        self._model = _LSTMAEModule(
            self._n_features, hidden=self.hidden, latent=self.latent, n_layers=self.n_layers
        ).to(self.device)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        self._model.train()
        self.history.clear()
        for _ in range(self.epochs):
            running = 0.0
            n = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                recon = self._model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                running += float(loss.item()) * batch.size(0)
                n += batch.size(0)
            self.history.append(running / max(n, 1))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-window mean squared reconstruction error."""
        if self._model is None:
            raise RuntimeError("LSTMAutoencoder must be fit before scoring.")
        X = self._ensure_windows(X)
        self._model.eval()
        scores = np.empty(X.shape[0], dtype=np.float32)
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                batch = torch.from_numpy(X[start:end]).to(self.device)
                recon = self._model(batch)
                err = ((recon - batch) ** 2).mean(dim=(1, 2))
                scores[start:end] = err.cpu().numpy()
        return scores

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _ensure_windows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"LSTMAutoencoder expects a 3D array (n_windows, window, n_features); "
                f"got shape {X.shape}"
            )
        return X
