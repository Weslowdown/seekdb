#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from run_public import Client


def restart_after_kill(client: Client) -> Client:
    assert client.process is not None
    client.process.kill()
    client.process.wait(timeout=10)
    client.process = None
    client.start()
    return client


def index_maintenance() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="mvd-index-evidence-") as tmp:
        client = Client(Path(tmp))
        try:
            client.execute(
                "CREATE TABLE docs (id BIGINT PRIMARY KEY, v VECTOR(2))"
            )
            client.execute(
                "INSERT INTO docs VALUES "
                "(1,'[0,0]'),(2,'[1,0]'),(3,'[0,2]'),(4,'[0,0]')"
            )
            client.execute("CREATE VECTOR INDEX docs_vec ON docs(v)")
            client.execute("UPDATE docs SET v = '[3,4]' WHERE id = 1")
            client.execute("DELETE FROM docs WHERE id = 2")
            restart_after_kill(client)
            return client.execute(
                "SELECT id, L2_DISTANCE(v, '[0,0]') AS d "
                "FROM docs ORDER BY d ASC LIMIT 3"
            )
        finally:
            client.stop()


def batch_write(per_batch: int) -> tuple[float, list[str]]:
    with tempfile.TemporaryDirectory(prefix="mvd-batch-evidence-") as tmp:
        client = Client(Path(tmp))
        started = time.perf_counter()
        try:
            client.execute(
                "CREATE TABLE events (id BIGINT PRIMARY KEY, payload VARCHAR(16))"
            )
            for start in range(0, 1000, per_batch):
                values = ",".join(
                    f"({i},'row{i}')" for i in range(start, start + per_batch)
                )
                response = client.execute(
                    "INSERT INTO events VALUES " + values
                )
                if response != [f"OK {per_batch}"]:
                    raise RuntimeError(response)
            elapsed = time.perf_counter() - started
            restart_after_kill(client)
            rows = client.execute("SELECT id FROM events ORDER BY id")
            return elapsed, rows
        finally:
            client.stop()


def main() -> int:
    batch_elapsed, batch_rows = batch_write(100)
    single_elapsed, single_rows = batch_write(1)
    result = {
        "index_maintenance_after_restart": index_maintenance(),
        "batch_100_commit": {
            "seconds": round(batch_elapsed, 3),
            "rows_after_sigkill": len(batch_rows),
        },
        "single_row_commit": {
            "seconds": round(single_elapsed, 3),
            "rows_after_sigkill": len(single_rows),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
