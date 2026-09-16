"""
Pipeline orchestrator.
Runs all three stages in a loop, every RUN_INTERVAL seconds.

Stage 1 + 2 are executed through DVC (dvc repro), which skips a stage
if neither its code nor its data has changed.
Stage 3 rebuilds and restarts the containers so the API always serves
the freshest model.
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = REPO_ROOT / "code" / "deployment"

RUN_INTERVAL = 300


def run(command: list[str], cwd: Path) -> bool:
    """Runs a shell command and streams its output. Returns True on success."""
    print(f"\n>>> {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"!!! Command failed with code {result.returncode}")
        return False
    return True


def run_pipeline() -> None:
    """One full pass: data -> model -> deployment."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 60}\n[{timestamp}] Starting pipeline run\n{'=' * 60}")

    # Stages 1 and 2: DVC decides what actually needs to be re-executed
    if not run(["dvc", "repro"], cwd=REPO_ROOT):
        print("Data/model stages failed, skipping deployment")
        return

    # Stage 3: rebuild images and restart containers with the new model.
    # -d runs them in the background so the loop can continue.
    run(["docker", "compose", "up", "--build", "-d"], cwd=DEPLOYMENT_DIR)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Pipeline run finished")


def main() -> None:
    print(f"Pipeline scheduler started. Interval: {RUN_INTERVAL} seconds.")
    print("Press Ctrl+C to stop.\n")

    while True:
        start = time.time()
        run_pipeline()

        elapsed = time.time() - start
        sleep_for = max(0, RUN_INTERVAL - elapsed)
        print(f"Next run in {sleep_for:.0f} seconds...\n")
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()