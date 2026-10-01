"""Keep persistence tests and application startup away from real user chats."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_chat_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
