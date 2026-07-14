from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_metadata_is_synchronized() -> None:
    metadata = load_module(
        "_mioxpektron_metadata", ROOT / "mioXpektron" / "_metadata.py"
    )

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    author_entry = f'{{name = "{metadata.AUTHOR}", email = "{metadata.EMAIL}"}}'
    assert pyproject.count(author_entry) == 2
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "mioXpektron._metadata.VERSION"}' in pyproject

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert metadata.LICENSE_COPYRIGHT in license_text

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"- **Developed by**: {metadata.AUTHOR}" in readme
    assert f"- **Contact**: {metadata.EMAIL}" in readme
    assert f"- **Copyright**: {metadata.AUTHOR}" in readme
    assert f"  author = {{{metadata.AUTHOR}}}," in readme
    assert f"  year = {{{metadata.COPYRIGHT_YEAR}}}," in readme

    installation = (ROOT / "docs" / "installation.rst").read_text(encoding="utf-8")
    assert f"# {metadata.VERSION}" in installation


def test_docs_config_reads_central_metadata() -> None:
    metadata = load_module(
        "_mioxpektron_metadata", ROOT / "mioXpektron" / "_metadata.py"
    )
    docs_conf = load_module("_mioxpektron_docs_conf", ROOT / "docs" / "conf.py")

    assert docs_conf.project == metadata.PROJECT_NAME
    assert docs_conf.author == metadata.AUTHOR
    assert docs_conf.copyright == metadata.COPYRIGHT
    assert docs_conf.release == metadata.VERSION
