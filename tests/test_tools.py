"""Dispatch validation, project containment, and complete versus bounded output."""

import pytest

from lclaude.tools import ToolError, dispatch


def test_read_file_range_and_bounded_output(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree", encoding="utf-8")
    output = dispatch(tmp_path, "read_file", {
        "path": "a.txt", "start_line": 2, "end_line": 3, "max_chars": 4,
    })
    assert output.content == "two\nthree"
    assert output.excerpt == "two\n"
    assert output.truncated


@pytest.mark.parametrize("name,args", [
    ("shell", {}),
    ("read_file", {}),
    ("read_file", {"path": "../outside.txt"}),
    ("read_file", {"path": "missing.txt"}),
    ("read_file", {"path": "a.txt", "start_line": 0}),
    ("read_file", {"path": "a.txt", "start_line": True}),
    ("read_file", {"path": "a.txt", "start_line": 3, "end_line": 2}),
    ("read_file", {"path": "a.txt", "end_line": 99}),
    ("read_file", {"path": "a.txt", "max_chars": "10"}),
    ("read_file", {"path": "a.txt", "extra": 1}),
    ("read_file", {"path": 1}),
    ("read_file", {"path": "a.txt", "max_chars": 12001}),
    ("read_file", []),
    ("read_file", '{"path":"a.txt"}'),
    ("list_files", {"limit": False}),
    ("list_files", {"path": "../"}),
    ("search_text", {"query": ""}),
    ("search_text", {"query": "a", "limit": -1}),
    ("read_tool_output", {"artifact_id": "x", "start": -1}),
])
def test_argument_errors_are_clear(tmp_path, name, args):
    (tmp_path / "a.txt").write_text("one\ntwo", encoding="utf-8")
    with pytest.raises(ToolError) as error:
        dispatch(tmp_path, name, args)
    assert str(error.value)


def test_list_and_search_keep_all_output_beyond_excerpt_limits(tmp_path):
    (tmp_path / "a.py").write_text("needle\nNEEDLE\nneedle", encoding="utf-8")
    (tmp_path / "b.py").write_text("other", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.py").write_text("needle", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "private").write_text("needle", encoding="utf-8")
    listed = dispatch(tmp_path, "list_files", {"limit": 1})
    assert listed.excerpt == "a.py\n"
    assert "nested/c.py" in listed.content
    assert ".git" not in listed.content
    assert listed.truncated
    found = dispatch(tmp_path, "search_text", {"query": "needle", "limit": 1})
    assert found.excerpt == "a.py:1:needle\n"
    assert "a.py:3:needle" in found.content and "nested/c.py:1:needle" in found.content
    assert found.truncated


def test_binary_read_error_and_search_skip(tmp_path):
    (tmp_path / "bin").write_bytes(b"\xff\x00")
    with pytest.raises(ToolError):
        dispatch(tmp_path, "read_file", {"path": "bin"})
    assert dispatch(tmp_path, "search_text", {"query": "x"}).content == ""


def test_symlinks_cannot_escape_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    try:
        (project / "link.txt").symlink_to(secret)
        (project / "outside").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires permission on this Windows installation")
    with pytest.raises(ToolError, match="outside"):
        dispatch(project, "read_file", {"path": "link.txt"})
    assert dispatch(project, "search_text", {"query": "secret"}).content == ""
    assert dispatch(project, "list_files", {}).content == ""
