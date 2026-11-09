#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

from run_public import Client


def assert_response(actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mvd-extended-") as tmp:
        data_dir = Path(tmp)
        client = Client(data_dir)
        try:
            assert_response(
                client.execute(
                    "CREATE TABLE t (id BIGINT, s VARCHAR(2), v VECTOR(2), PRIMARY KEY (id))"
                ),
                ["OK"],
            )
            assert_response(
                client.execute("CREATE TABLE bad_dim (id BIGINT PRIMARY KEY, v VECTOR(0))"),
                ["ERROR invalid_table: v requires a positive size"],
            )
            assert_response(
                client.execute(
                    "CREATE TABLE bad_dup (id BIGINT PRIMARY KEY, x BIGINT, x VARCHAR(2))"
                ),
                ["ERROR invalid_table: duplicate column name"],
            )
            assert_response(
                client.execute(
                    "INSERT INTO t VALUES (-9223372036854775808, 'a😀', '[1,2]'), "
                    "(9223372036854775807, 'z', '[-1,2]')"
                ),
                ["OK 2"],
            )
            assert_response(
                client.execute("INSERT INTO t VALUES (3, 'abc', '[0,0]')"),
                ["ERROR varchar_too_long: s is too long"],
            )
            assert_response(
                client.execute("INSERT INTO t VALUES (3, 'x', '[1]')"),
                ["ERROR dimension_mismatch: v must have 2 dimensions"],
            )
            assert_response(
                client.execute("INSERT INTO t VALUES (3, 'x', '[NaN,0]')"),
                ["ERROR invalid_vector: vector contains an invalid component"],
            )
            assert_response(
                client.execute("INSERT INTO t VALUES (3, 'x', '[1e100,0]')"),
                ["ERROR vector_overflow: vector component overflows float32"],
            )
            assert_response(
                client.execute("SELECT id FROM t ORDER BY s"),
                ["-9223372036854775808", "9223372036854775807"],
            )
            assert_response(client.execute("BEGIN"), ["OK"])
            assert_response(
                client.execute("UPDATE t SET s = 'q' WHERE id = -9223372036854775808"),
                ["OK 1"],
            )
            assert_response(
                client.execute("INSERT INTO t VALUES (5, 'n', '[2,1]')"), ["OK 1"]
            )
            assert_response(
                client.execute("SELECT id, s FROM t ORDER BY id"),
                ["-9223372036854775808\tq", "5\tn", "9223372036854775807\tz"],
            )
            assert_response(client.execute("ROLLBACK"), ["OK"])
            assert_response(
                client.execute("SELECT id, s FROM t ORDER BY id"),
                ["-9223372036854775808\ta😀", "9223372036854775807\tz"],
            )
            assert_response(client.execute("BEGIN"), ["OK"])
            assert_response(
                client.execute("DELETE FROM t WHERE id = 9223372036854775807"), ["OK 1"]
            )
            assert_response(
                client.execute("INSERT INTO t VALUES (7, 'c', '[0,0]')"), ["OK 1"]
            )
            assert_response(client.execute("COMMIT"), ["OK"])
            assert_response(
                client.execute("SELECT id FROM t ORDER BY id"),
                ["-9223372036854775808", "7"],
            )
            assert_response(client.execute("BEGIN"), ["OK"])
            assert_response(client.execute("DELETE FROM t"), ["OK 2"])
            client.stop()
            client.start()
            assert_response(
                client.execute("SELECT id, s FROM t ORDER BY id"),
                ["-9223372036854775808\ta😀", "7\tc"],
            )
            assert_response(
                client.execute("SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t WHERE s = 'c'"),
                ["7\t0.000000"],
            )
            assert_response(
                client.execute(
                    "SELECT t.id, L2_DISTANCE('[0,0]', t.v) AS d "
                    "FROM t ORDER BY d ASC LIMIT 0, 1"
                ),
                ["7\t0.000000"],
            )
        finally:
            client.stop()
        print("PASS extended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
