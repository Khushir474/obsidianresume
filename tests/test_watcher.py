"""Tests for the JD file watcher."""
import time
import pathlib
import pytest

from obsidianresumeforge.watcher import JDWatcher


def _make_watcher(callback):
    return JDWatcher(jds_dir="/fake/JDs", on_new_jd=callback)


def _fake_event(path, is_directory=False):
    class _Ev:
        src_path = path
        is_directory = False

    e = _Ev()
    e.is_directory = is_directory
    e.src_path = path
    return e


def test_md_file_triggers_callback(tmp_path):
    fired = []
    watcher = JDWatcher(jds_dir=str(tmp_path), on_new_jd=lambda p: fired.append(p))

    jd_file = tmp_path / "test.md"
    jd_file.write_text("# Job Description\nContent here.")

    watcher.on_created(_fake_event(str(jd_file)))
    time.sleep(2.5)  # wait for debounce

    assert len(fired) == 1
    assert str(jd_file) in fired[0]


def test_non_md_file_does_not_trigger(tmp_path):
    fired = []
    watcher = JDWatcher(jds_dir=str(tmp_path), on_new_jd=lambda p: fired.append(p))

    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("not a JD")

    watcher.on_created(_fake_event(str(txt_file)))
    time.sleep(2.5)

    assert fired == []


def test_empty_file_does_not_trigger(tmp_path):
    fired = []
    watcher = JDWatcher(jds_dir=str(tmp_path), on_new_jd=lambda p: fired.append(p))

    empty = tmp_path / "empty.md"
    empty.write_text("")

    watcher.on_created(_fake_event(str(empty)))
    time.sleep(2.5)

    assert fired == [], "Empty .md file should not trigger kickoff"


def test_directory_event_ignored(tmp_path):
    fired = []
    watcher = JDWatcher(jds_dir=str(tmp_path), on_new_jd=lambda p: fired.append(p))

    watcher.on_created(_fake_event(str(tmp_path / "subdir"), is_directory=True))
    time.sleep(2.5)

    assert fired == []


def test_rapid_events_debounced(tmp_path):
    """Multiple events for same file within debounce window → only one callback."""
    fired = []
    watcher = JDWatcher(jds_dir=str(tmp_path), on_new_jd=lambda p: fired.append(p))

    jd_file = tmp_path / "role.md"
    jd_file.write_text("# Role\nDetails.")

    for _ in range(5):
        watcher.on_created(_fake_event(str(jd_file)))
        time.sleep(0.1)

    time.sleep(2.5)

    assert len(fired) == 1, f"Expected 1 callback after debounce, got {len(fired)}"
