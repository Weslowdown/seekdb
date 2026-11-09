#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from run_public import Client


def vector_sql(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in vector) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    rng = np.random.default_rng(20261109)
    rows = rng.normal(size=(args.rows, args.dim)).astype(np.float32)
    queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="mvd-bench-") as tmp:
        client = Client(Path(tmp))
        try:
            started = time.perf_counter()
            client.execute(
                f"CREATE TABLE bench (id BIGINT PRIMARY KEY, cat VARCHAR(8), v VECTOR({args.dim}))"
            )
            for start in range(0, len(rows), 100):
                values = []
                for row in rows[start : start + 100]:
                    pk = start + len(values)
                    values.append(f"({pk},'g{pk % 100}',{vector_sql(row)})")
                response = client.execute(
                    "INSERT INTO bench VALUES " + ",".join(values)
                )
                if response != ["OK 100"]:
                    raise RuntimeError(response)
            write_time = time.perf_counter() - started

            started = time.perf_counter()
            for i, query in enumerate(queries):
                where = f" WHERE cat = 'g{i % 100}'" if i % 2 else ""
                response = client.execute(
                    "SELECT id, L2_DISTANCE(v, '"
                    + vector_sql(query)
                    + "') AS d FROM bench"
                    + where
                    + " ORDER BY d LIMIT 10"
                )
                if len(response) != 10:
                    raise RuntimeError(response)
            query_time = time.perf_counter() - started
        finally:
            client.stop()
    print(
        f"{args.rows:,} writes: {write_time:.3f}s; "
        f"{args.queries:,} TopK queries: {query_time:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
