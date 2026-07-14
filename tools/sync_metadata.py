"""Synchronize static metadata mirrors from ``mioXpektron/_metadata.py``."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "mioXpektron" / "_metadata.py"


def load_metadata():
    spec = importlib.util.spec_from_file_location("_mioxpektron_metadata", METADATA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load metadata from {METADATA_PATH}")
    metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata)
    return metadata


def replace_pattern(
    path: Path,
    pattern: str,
    replacement: str | Callable[[re.Match[str]], str],
    *,
    expected_count: int = 1,
) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} replacements in {path}, found {count}"
        )
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    metadata = load_metadata()
    author_entry = f'{{name = "{metadata.AUTHOR}", email = "{metadata.EMAIL}"}}'

    replace_pattern(
        ROOT / "pyproject.toml",
        r'\{name = "[^"]+", email = "[^"]+"\}',
        author_entry,
        expected_count=2,
    )
    replace_pattern(
        ROOT / "LICENSE",
        r"Copyright \(c\) \d{4} .+",
        metadata.LICENSE_COPYRIGHT,
    )
    replace_pattern(
        ROOT / "README.md",
        r"- \*\*Developed by\*\*: .+",
        f"- **Developed by**: {metadata.AUTHOR}",
    )
    replace_pattern(
        ROOT / "README.md",
        r"- \*\*Contact\*\*: .+",
        f"- **Contact**: {metadata.EMAIL}",
    )
    replace_pattern(
        ROOT / "README.md",
        r"- \*\*Copyright\*\*: .+",
        f"- **Copyright**: {metadata.AUTHOR}",
    )
    replace_pattern(
        ROOT / "README.md",
        r"  author = \{[^}]+\},",
        f"  author = {{{metadata.AUTHOR}}},",
    )
    replace_pattern(
        ROOT / "README.md",
        r"  year = \{\d{4}\},",
        f"  year = {{{metadata.COPYRIGHT_YEAR}}},",
    )
    replace_pattern(
        ROOT / "docs" / "installation.rst",
        r"(print\(mx\.__version__\)\n\s+# )\S+",
        lambda match: f"{match.group(1)}{metadata.VERSION}",
    )


if __name__ == "__main__":
    main()
