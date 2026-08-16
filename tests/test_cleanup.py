import time

from webapp.cleanup import sweep_directory


def test_sweep_directory_deletes_old_files(tmp_path):
    old_file = tmp_path / "old.mp4"
    old_file.write_bytes(b"data")
    now = time.time()

    sweep_directory(tmp_path, max_age_seconds=3600, now=now + 7200)

    assert not old_file.exists()


def test_sweep_directory_keeps_recent_files(tmp_path):
    recent_file = tmp_path / "recent.mp4"
    recent_file.write_bytes(b"data")
    now = time.time()

    sweep_directory(tmp_path, max_age_seconds=3600, now=now)

    assert recent_file.exists()


def test_sweep_directory_keeps_gitkeep(tmp_path):
    gitkeep = tmp_path / ".gitkeep"
    gitkeep.write_bytes(b"")
    now = time.time()

    sweep_directory(tmp_path, max_age_seconds=3600, now=now + 7200)

    assert gitkeep.exists()


def test_sweep_directory_missing_dir_is_noop(tmp_path):
    sweep_directory(tmp_path / "does_not_exist", max_age_seconds=3600)
