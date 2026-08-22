"""One-command local launcher for the PDF retriever.

Run ``python app.py`` from the repository root. The launcher reuses healthy
services, starts missing ones, opens the browser, and stops only child services
it created when Ctrl+C is pressed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
APP_URL = "http://127.0.0.1:5173"
API_URL = "http://127.0.0.1:8000"
OLLAMA_URL = "http://127.0.0.1:11434"


def service_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except (OSError, URLError, TimeoutError):
        return False


def project_python() -> Path:
    candidates = (
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / "venv" / "bin" / "python",
    )
    return next((path for path in candidates if path.is_file()), Path(sys.executable))


def ollama_executable() -> Path | None:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = (
        local_app_data / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
    )
    return next((path for path in candidates if path.is_file()), None)


def hidden_creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def wait_for(url: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if service_ready(url):
            return True
        time.sleep(0.25)
    return False


def start_services() -> list[subprocess.Popen]:
    started: list[subprocess.Popen] = []
    python = str(project_python())

    if not service_ready(f"{OLLAMA_URL}/api/tags"):
        executable = ollama_executable()
        if executable is None:
            raise RuntimeError(
                "Ollama is not running and ollama.exe was not found. Start Ollama first."
            )
        started.append(
            subprocess.Popen(
                [str(executable), "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=hidden_creation_flags(),
            )
        )
        if not wait_for(f"{OLLAMA_URL}/api/tags", 15):
            raise RuntimeError("Ollama did not become ready within 15 seconds.")

    if not service_ready(f"{API_URL}/health"):
        started.append(
            subprocess.Popen(
                [
                    python,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                cwd=BACKEND_DIR,
                creationflags=hidden_creation_flags(),
            )
        )
        if not wait_for(f"{API_URL}/health", 30):
            raise RuntimeError(
                "The backend did not start. Check backend/logs/app.log for details."
            )

    if not service_ready(APP_URL):
        started.append(
            subprocess.Popen(
                [
                    python,
                    "-m",
                    "http.server",
                    "5173",
                    "--bind",
                    "127.0.0.1",
                    "--directory",
                    str(FRONTEND_DIR),
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=hidden_creation_flags(),
            )
        )
        if not wait_for(APP_URL, 15):
            raise RuntimeError("The frontend did not start on port 5173.")

    return started


def stop_started_services(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local PDF retriever")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the app")
    parser.add_argument("--check", action="store_true", help="Only check running services")
    args = parser.parse_args()

    statuses = {
        "Ollama": service_ready(f"{OLLAMA_URL}/api/tags"),
        "Backend": service_ready(f"{API_URL}/health"),
        "Frontend": service_ready(APP_URL),
    }
    if args.check:
        for name, ready in statuses.items():
            print(f"{name}: {'ready' if ready else 'not running'}")
        return 0 if all(statuses.values()) else 1

    processes: list[subprocess.Popen] = []
    try:
        processes = start_services()
        print("LangChain PDF Retriever is ready.")
        print(f"App:      {APP_URL}")
        print(f"API docs: {API_URL}/docs")
        print("Press Ctrl+C to stop services started by this launcher.")
        if not args.no_browser:
            webbrowser.open(APP_URL)
        if not processes:
            print("All services were already running.")
            return 0
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        return 1
    except KeyboardInterrupt:
        print("\nStopping local services...")
        return 0
    except RuntimeError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_started_services(processes)


if __name__ == "__main__":
    raise SystemExit(main())
