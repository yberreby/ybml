import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_git_metadata(repo_path: Path) -> dict[str, Any]:
    """Return commit hash and dirty status for reproducibility metadata."""
    metadata: dict[str, Any] = {}

    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path)
            .decode()
            .strip()
        )
        metadata["commit"] = commit
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Unable to determine git commit: %s", exc)

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_path
        ).decode()
        status = status_output.strip()
        metadata["dirty"] = bool(status)
        if status:
            metadata["status"] = status
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Unable to determine git working tree status: %s", exc)

    return metadata
