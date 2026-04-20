import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._last_status = "never_run"
        self._last_started_at: str | None = None
        self._last_finished_at: str | None = None
        self._last_error: str | None = None

    def status(self) -> dict:
        return {
            "running": self._running,
            "last_status": self._last_status,
            "last_started_at": self._last_started_at,
            "last_finished_at": self._last_finished_at,
            "last_error": self._last_error,
        }

    def run_refresh_pipeline(self) -> dict:
        with self._lock:
            if self._running:
                return {"accepted": False, "detail": "Job already running", "status": self.status()}
            self._running = True
            self._last_status = "running"
            self._last_started_at = datetime.utcnow().isoformat()
            self._last_error = None

        try:
            project_root = Path(__file__).resolve().parents[2]
            subprocess.run(
                [sys.executable, "auto_refresh.py", "--once"],
                cwd=str(project_root),
                check=True,
            )
            self._last_status = "success"
            return {"accepted": True, "detail": "Refresh pipeline completed", "status": self.status()}
        except Exception as exc:
            self._last_status = "failed"
            self._last_error = str(exc)
            return {"accepted": True, "detail": "Refresh pipeline failed", "status": self.status()}
        finally:
            self._running = False
            self._last_finished_at = datetime.utcnow().isoformat()


job_manager = JobManager()
