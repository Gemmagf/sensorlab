from pathlib import Path

from sensorlab.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    TEP,
    ensure_dirs,
)


def test_tep_spec_constants():
    assert TEP.n_xmeas + TEP.n_xmv == TEP.n_total
    assert TEP.n_fault_types == 21
    assert TEP.sample_minutes > 0


def test_tep_fault_names_count():
    # 1 Normal + 21 faults
    assert len(TEP.fault_names) == TEP.n_fault_types + 1
    assert TEP.fault_names[0] == "Normal"


def test_project_root_is_directory():
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.is_dir()


def test_ensure_dirs_creates_paths(tmp_path, monkeypatch):
    """Use monkeypatch so we don't touch the real project dirs."""
    from sensorlab import config as cfg_mod

    fake_dirs = {
        "RAW_DIR": tmp_path / "raw",
        "PROCESSED_DIR": tmp_path / "processed",
        "MODELS_DIR": tmp_path / "models",
        "ARTIFACTS_DIR": tmp_path / "artifacts",
    }
    for name, p in fake_dirs.items():
        monkeypatch.setattr(cfg_mod, name, p)
    ensure_dirs()
    for p in fake_dirs.values():
        assert p.is_dir()


def test_constants_are_absolute_paths():
    for d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
        assert d.is_absolute()
