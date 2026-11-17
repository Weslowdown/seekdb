#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from run_public import Client


def main() -> int:
    results: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="mvd-learning-") as tmp:
        client = Client(Path(tmp))
        try:
            client.execute(
                "CREATE TABLE tie_case "
                "(id BIGINT PRIMARY KEY, v VECTOR(2))"
            )
            client.execute(
                "INSERT INTO tie_case VALUES "
                "(4,'[0,0]'),(2,'[1,0]'),(5,'[-1,0]'),(1,'[2,0]')"
            )
            results["tie_offset"] = client.execute(
                "SELECT id, L2_DISTANCE(v, '[0,0]') AS d "
                "FROM tie_case ORDER BY d ASC LIMIT 2 OFFSET 1"
            )
        finally:
            client.stop()

        client = Client(Path(tmp))
        try:
            client.execute(
                "CREATE TABLE durable_delete "
                "(id BIGINT PRIMARY KEY, content VARCHAR(8))"
            )
            client.execute("INSERT INTO durable_delete VALUES (1,'row')")
            client.execute("DELETE FROM durable_delete WHERE id = 1")
            assert client.process is not None
            client.process.kill()
            client.process.wait(timeout=10)
            client.process = None
            client.start()
            results["committed_delete_after_kill"] = client.execute(
                "SELECT id FROM durable_delete"
            )
        finally:
            client.stop()

        client = Client(Path(tmp))
        try:
            client.execute(
                "CREATE TABLE float_case "
                "(id BIGINT PRIMARY KEY, v VECTOR(2))"
            )
            client.execute("INSERT INTO float_case VALUES (1,'[16777217,0]')")
            results["float32_rounding"] = client.execute(
                "SELECT id, L2_DISTANCE(v, '[16777216,0]') AS d "
                "FROM float_case"
            )
        finally:
            client.stop()

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
