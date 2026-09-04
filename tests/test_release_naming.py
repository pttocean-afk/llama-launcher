"""Release artifacts must be versioned and immutable."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _project_version():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version = "([^"]+)"$', text, re.MULTILINE).group(1)


def test_inno_output_filename_contains_version():
    text = (ROOT / "installer" / "LlamaLauncher.iss").read_text(
        encoding="utf-8")
    version = _project_version()
    assert f'#define MyAppVersion "{version}"' in text
    assert "OutputBaseFilename=LlamaLauncher-Setup-{#MyAppVersion}-x64" in text


def test_canonical_build_uses_versioned_immutable_artifacts():
    text = (ROOT / "scripts" / "build-release-windows.sh").read_text(
        encoding="utf-8")
    assert 'SETUP_NAME="LlamaLauncher-Setup-${PROJECT_VERSION}-x64.exe"' in text
    assert 'PORTABLE_NAME="LlamaLauncher-Portable-${PROJECT_VERSION}-x64.zip"' in text
    assert "Refusing to overwrite existing release artifact" in text
    # Canonical build must not delete stable artifacts before copying a new one.
    assert 'rm -f "$REPO/dist/LlamaLauncher-Setup' not in text


def test_development_guide_records_release_naming_rule():
    text = (ROOT / "DEVELOPMENT.md").read_text(encoding="utf-8")
    assert "所有正式產物檔名必須包含完整版本號" in text
    assert "不得覆蓋或刪除既有版本的正式產物" in text
    assert "LlamaLauncher-Setup-X.Y.Z-x64.exe" in text
