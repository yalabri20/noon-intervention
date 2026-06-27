#!/usr/bin/env python3
"""One-command bootstrap: install deps and launch the Streamlit dashboard."""
import subprocess
import sys


def main() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    )
    subprocess.check_call(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
    )


if __name__ == "__main__":
    main()
