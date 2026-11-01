"""Extract model code and verify it in an isolated pytest subprocess."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str | None:
    """Return the first Python fenced block, or the text if it parses as code."""
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip() + "\n"
    stripped = text.strip()
    if not stripped:
        return None
    try:
        compile(stripped, "<model>", "exec")
    except SyntaxError:
        return None
    return stripped + "\n"


@dataclass
class SandboxResult:
    passed: bool
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str


class Sandbox:
    """Runs candidate code + tests via `python -m pytest` with a timeout.

    The candidate is never imported into this process. It still executes
    with your user permissions, so treat the first loop as trusted-machine
    only (ideally inside a VM or disposable account).
    """

    def __init__(self, timeout: float = 2.0) -> None:
        self.timeout = timeout

    def run(self, code: str, test_source: str) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="ouroboros-") as tmp:
            workdir = Path(tmp)
            (workdir / "solution.py").write_text(code, encoding="utf-8")
            (workdir / "test_solution.py").write_text(
                test_source, encoding="utf-8"
            )
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "test_solution.py",
                    ],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    passed=False,
                    exit_code=None,
                    timed_out=True,
                    stdout=_decode(exc.stdout),
                    stderr=_decode(exc.stderr)
                    or f"Timed out after {self.timeout}s",
                )
            return SandboxResult(
                passed=proc.returncode == 0,
                exit_code=proc.returncode,
                timed_out=False,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )


def _decode(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def tail(text: str, chars: int = 2000) -> str:
    return text[-chars:] if len(text) > chars else text
