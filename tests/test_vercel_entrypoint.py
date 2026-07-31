"""Assert Vercel FastAPI entrypoint exports a top-level app."""
from pathlib import Path
import ast


def test_app_py_exports_fastapi_app():
    src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    assigns = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigns.add(node.target.id)
    assert "app" in assigns


def test_requirements_include_fastapi():
    text = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    assert "fastapi" in text.lower()
    assert "streamlit" not in text.lower()


def test_requirements_include_uvicorn():
    text = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    assert "uvicorn" in text.lower()
