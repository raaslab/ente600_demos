"""Tiny bootstrap helper for the feature matching Colab notebook."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/raaslab/ente600_demos.git"
REPO_DIR = "ente600_demos"


def setup_demo() -> None:
    if Path("feature_matching_utils.py").exists():
        print("Already in the project folder.")
        return

    if not Path(REPO_DIR).exists():
        subprocess.run(["git", "clone", REPO_URL], check=True)

    os.chdir(REPO_DIR)
    print("Changed working directory to:", os.getcwd())
