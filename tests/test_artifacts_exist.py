from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

def test_scaler_exists():
    assert (ART / "scaler.json").is_file()
