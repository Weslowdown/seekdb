#!/usr/bin/env python3
"""Run the public MiniVectorDB SQL cases against server.py."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
class Client:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.process: subprocess.Popen[str] | None = None
        self.next_id = 1
        self.search_mode = "exact"
        self.start()

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "server.py"),
                f"--data-dir={self.data_dir}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self.process.stdout is not None
        ready = json.loads(self.process.stdout.readline())
        if ready != {"type": "ready"}:
            raise RuntimeError(f"invalid ready handshake: {ready!r}")

    def stop(self) -> None:
        if self.process is None:
            return
        assert self.process.stdin is not None
        request_id = self.next_id
        self.next_id += 1
        self.process.stdin.write(
            json.dumps({"id": request_id, "op": "shutdown"}) + "\n"
        )
        self.process.stdin.flush()
        assert self.process.stdout is not None
        response = json.loads(self.process.stdout.readline())
        if response != {"id": request_id, "ok": True}:
            raise RuntimeError(f"invalid shutdown response: {response!r}")
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None

    def execute(self, sql: str) -> list[str]:
        assert self.process is not None and self.process.stdin is not None
        assert self.process.stdout is not None
        request_id = self.next_id
        self.next_id += 1
        self.process.stdin.write(
            json.dumps(
                {
                    "id": request_id,
                    "op": "execute",
                    "sql": sql.rstrip(";") + ";",
                    "search_mode": self.search_mode,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if line == "":
            raise RuntimeError("database exited unexpectedly")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise RuntimeError(f"response id mismatch: {response!r}")
        if not response.get("ok"):
            error = response.get("error") or {}
            return [f"ERROR {error.get('code', '')}: {error.get('message', '')}"]
        if "rows" in response:
            return [
                "\t".join(_render_test_value(value) for value in row)
                for row in response["rows"]
            ]
        if "affected_rows" in response:
            return [f"OK {response['affected_rows']}"]
        return ["OK"]


def _render_test_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def cast_like(raw: str, expected: Any) -> Any:
    if isinstance(expected, bool):
        return raw.lower() == "true"
    if isinstance(expected, int):
        return int(raw)
    if isinstance(expected, float):
        return float(raw)
    if isinstance(expected, str):
        return raw
    return raw


def compare(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if "error" in expected:
        return (
            bool(actual.get("lines"))
            and actual["lines"][0].startswith(f"ERROR {expected['error']}:")
        )
    if expected == {}:
        return bool(actual.get("lines")) and actual["lines"][0].startswith("OK")
    if "affected_rows" in expected:
        return actual.get("lines") == [f"OK {expected['affected_rows']}"]
    columns = expected.get("columns", [])
    expected_rows = expected.get("rows", [])
    actual_rows: list[list[Any]] = []
    for line in actual.get("lines", []):
        if not line:
            actual_rows.append([])
        else:
            fields = line.split("\t")
            if len(fields) != len(columns):
                return False
            actual_rows.append(fields)
    if len(actual_rows) != len(expected_rows):
        return False
    for actual_row, expected_row in zip(actual_rows, expected_rows):
        for got_raw, want in zip(actual_row, expected_row):
            try:
                got = cast_like(got_raw, want)
            except ValueError:
                return False
            if isinstance(want, float):
                if not math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-5):
                    return False
            elif got != want:
                return False
    if expected.get("unordered"):
        return sorted(map(str, actual_rows)) == sorted(map(str, expected_rows))
    return True


def run_case(client: Client, test_path: Path, result_path: Path) -> tuple[bool, str]:
    expected = json.loads(result_path.read_text(encoding="utf-8"))
    lines = [line.strip() for line in test_path.read_text(encoding="utf-8").splitlines()]
    expected_index = 0
    for sql in lines:
        if not sql:
            continue
        if sql == "--restart":
            client.stop()
            client.start()
            continue
        if sql.startswith("--mode "):
            client.search_mode = sql.split()[1]
            continue
        if sql.startswith("--"):
            continue
        actual = {"lines": client.execute(sql)}
        if expected_index >= len(expected):
            return False, f"extra result for {sql}"
        if not compare(actual, expected[expected_index]):
            return False, (
                f"statement {expected_index + 1}: {sql}\n"
                f"expected: {expected[expected_index]}\n"
                f"actual: {actual['lines']}"
            )
        expected_index += 1
    if expected_index != len(expected):
        return False, f"missing {len(expected) - expected_index} results"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "mysqltest" / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
    failures = 0
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="mvd-") as tmp:
            client = Client(Path(tmp))
            try:
                ok, message = run_case(
                    client,
                    ROOT / "mysqltest" / case["test"],
                    ROOT / "mysqltest" / case["result"],
                )
            finally:
                client.stop()
        print(f"{'PASS' if ok else 'FAIL'} {case['id']}")
        if not ok:
            failures += 1
            print(message)
    print(f"{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
