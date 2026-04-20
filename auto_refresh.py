#!/usr/bin/env python3
"""
Auto refresh pipeline for ENISO RAG data.

What it does:
1) Watches raw data files for changes.
2) Re-runs preprocessing to regenerate LLM-friendly markdown files.
3) Rebuilds the Chroma index so updates (e.g. new timetable versions) are used.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rag_system import CHROMA_DIR, GOLD_CHROMA_DIR, ENISO_RAW_DATA_DIR


SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg"}


def _snapshot_directory(data_dir: Path) -> str:
    """Return a hash fingerprint for all supported files."""
    records = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = p.stat()
        rel = p.relative_to(data_dir).as_posix()
        records.append(f"{rel}|{stat.st_mtime_ns}|{stat.st_size}")

    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    return digest


def _run_preprocessing(input_dir: Path, output_dir: Path):
    """Run preprocess_data.py in clean mode."""
    cmd = [
        sys.executable,
        "preprocess_data.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--clean-output",
    ]
    subprocess.run(cmd, check=True)


def _rebuild_index():
    """Delete main index and rebuild it from processed data."""
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # Keep gold index by default; delete only if explicitly needed.
    from rag_system import RAGSystem

    # force_reindex rebuilds the main vector DB.
    RAGSystem(force_reindex=True)


def run_once(input_dir: Path, output_dir: Path):
    print("\n[Pipeline] Regenerating processed data...")
    _run_preprocessing(input_dir, output_dir)
    print("[Pipeline] Rebuilding vector index...")
    _rebuild_index()
    print("[Pipeline] Completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Watch ENISO raw files and auto-refresh processed data + index."
    )
    parser.add_argument(
        "--input-dir",
        default=ENISO_RAW_DATA_DIR,
        help="Raw source directory to watch.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "processed_data"),
        help="Directory where processed markdown files are written.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run refresh one time then exit.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if args.once:
        run_once(input_dir, output_dir)
        return

    print("Auto-refresh watcher started")
    print(f"Watching: {input_dir}")
    print(f"Output  : {output_dir}")
    print(f"Interval: {args.interval}s")
    print("Press Ctrl+C to stop.\n")

    last_fp = None
    while True:
        try:
            current_fp = _snapshot_directory(input_dir)
            if current_fp != last_fp:
                if last_fp is None:
                    print("[Watcher] Initial scan detected files.")
                else:
                    print("[Watcher] Change detected in source files.")
                run_once(input_dir, output_dir)
                last_fp = current_fp
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nWatcher stopped by user.")
            break
        except Exception as exc:
            print(f"[Watcher] Error: {exc}")
            # Keep process alive and retry next cycle.
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
