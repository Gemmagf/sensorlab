import numpy as np
import pytest

from sensorlab.detection import LSTMAutoencoder


def test_ae_requires_3d_input():
    ae = LSTMAutoencoder(epochs=1, window=8, hidden=4, latent=2)
    bad = np.zeros((10, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        ae.fit(bad)


def test_ae_score_shape(tiny_windows):
    W = tiny_windows["windows"]
    ae = LSTMAutoencoder(window=W.shape[1], epochs=2, hidden=8, latent=4, batch_size=32)
    ae.fit(W[:200])
    scores = ae.score(W[:50])
    assert scores.shape == (50,)
    assert (scores >= 0).all()


def test_ae_loss_decreases(tiny_windows):
    W = tiny_windows["windows"]
    ae = LSTMAutoencoder(window=W.shape[1], epochs=8, hidden=8, latent=4, batch_size=32)
    ae.fit(W[:200])
    assert len(ae.history) == 8
    assert ae.history[-1] < ae.history[0]


def test_ae_requires_fit_first(tiny_windows):
    ae = LSTMAutoencoder()
    with pytest.raises(RuntimeError):
        ae.score(tiny_windows["windows"][:5])
