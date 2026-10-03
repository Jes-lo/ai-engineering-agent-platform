"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def isolated_settings_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Isolate settings tests from host environment and repository dotenv files."""

    for key in tuple(os.environ):
        if key.startswith("AI_PLATFORM_"):
            monkeypatch.delenv(key)

    monkeypatch.chdir(tmp_path)
