import os
from pathlib import Path

from data.env_loader import load_dotenv_if_present


def test_loads_key_value_pairs_into_environ(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEST_ENV_LOADER_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_ENV_LOADER_KEY=hello\n", encoding="utf-8")

    load_dotenv_if_present(env_file)

    assert os.environ["TEST_ENV_LOADER_KEY"] == "hello"


def test_missing_file_is_a_silent_no_op(tmp_path: Path):
    load_dotenv_if_present(tmp_path / "does_not_exist.env")  # should not raise


def test_comments_and_blank_lines_are_skipped(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEST_ENV_LOADER_SKIP", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\n\nTEST_ENV_LOADER_SKIP=value\n", encoding="utf-8")

    load_dotenv_if_present(env_file)

    assert os.environ["TEST_ENV_LOADER_SKIP"] == "value"


def test_quoted_values_are_unquoted(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEST_ENV_LOADER_QUOTED", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('TEST_ENV_LOADER_QUOTED="quoted value"\n', encoding="utf-8")

    load_dotenv_if_present(env_file)

    assert os.environ["TEST_ENV_LOADER_QUOTED"] == "quoted value"


def test_does_not_override_already_set_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_ENV_LOADER_PRIORITY", "real_shell_value")
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_ENV_LOADER_PRIORITY=dotenv_value\n", encoding="utf-8")

    load_dotenv_if_present(env_file)

    assert os.environ["TEST_ENV_LOADER_PRIORITY"] == "real_shell_value"
