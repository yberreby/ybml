from pathlib import Path

from ymc.git import get_git_metadata


def test_get_git_metadata_in_repo():
    """Smoketest: get_git_metadata returns expected keys in a git repo."""
    repo_path = Path(__file__).parent.parent.parent
    metadata = get_git_metadata(repo_path)

    # Should have commit if we're in a repo
    assert "commit" in metadata
    assert isinstance(metadata["commit"], str)
    assert len(metadata["commit"]) == 40  # SHA-1 hash length

    # Should have dirty status
    assert "dirty" in metadata
    assert isinstance(metadata["dirty"], bool)

    # If dirty, should have status
    if metadata["dirty"]:
        assert "status" in metadata


def test_get_git_metadata_invalid_path():
    """Test graceful handling of non-git directory."""
    metadata = get_git_metadata(Path("/tmp"))
    # Should return empty dict or dict without keys (warnings logged)
    assert isinstance(metadata, dict)
