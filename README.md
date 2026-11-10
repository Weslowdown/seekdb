# MiniVectorDB

Standalone implementation for the 2026 OceanBase Database Competition
tie-breaker round.

## Build and run

```bash
docker build -t minivectordb .
docker run -i --rm -v "$PWD/data:/data" minivectordb
```

The process uses the official JSON Lines protocol over stdin/stdout. Persistent
state and the append-only transaction journal are written below `/data`. All
diagnostics go to stderr.

## Supported SQL

- `CREATE TABLE` with `BIGINT`, `VARCHAR(n)`, `VECTOR(n)` and one BIGINT primary key.
- `INSERT`, `SELECT`, `UPDATE`, `DELETE`, `DROP TABLE`.
- `WHERE` comparisons with `=`, `<`, `<=`, `>`, `>=` and `AND`.
- `ORDER BY`, `LIMIT`, `OFFSET`.
- `L2_DISTANCE(column, '[...]') AS alias` followed by TopK ordering.
- `CREATE VECTOR INDEX`.
- `BEGIN`, `COMMIT`, `ROLLBACK`, and `SET vector_mode = exact|ann`.

Exact top-k is used for both modes. This deliberately preserves Recall@10
equal to 1.0 while avoiding approximate-index recall risk.

## I/O protocol

Startup emits exactly:

```json
{"type":"ready"}
```

Each input line is an `execute` request with an `id`, SQL text, and
`search_mode`. Query responses contain `id`, `ok`, `columns`, and `rows`;
mutation responses additionally contain `affected_rows`; failures contain a
structured `error` object. `{"op":"shutdown"}` rolls back unfinished work,
returns `{"id":...,"ok":true}`, and exits with status zero.

## Durability

Every automatically committed mutation is appended as a complete JSON record
and fsynced before `OK` is printed. Explicit transactions are buffered and
become durable as one record at `COMMIT`. A partial final journal record is
ignored during recovery, preventing an interrupted transaction from becoming
visible.

## Known limitations

The implementation keeps committed table rows in memory and performs exact
vector scans. It favors semantic correctness over very large datasets and
approximate-index latency. `ann` currently uses the same exact path so filtered
and unfiltered recall remain complete.

## Design

`server.py` is one self-contained process. It tokenizes and parses the supported
SQL subset, stores each table as typed rows keyed by its BIGINT primary key, and
keeps NumPy `float32` vectors in a cached matrix for fast exact scans. A SELECT
builds its visibility mask before calculating distances, so filtered TopK is
computed inside the filtered set. Distance ties are resolved by ascending
primary key. `VARCHAR` lengths and ordering use Python's Unicode code-point
semantics.

Each successful auto-commit mutation is appended to `/data/journal.ndjson` as
one complete JSON record and fsynced before a success response. Explicit
transactions buffer their mutations, become visible to reads immediately, and
are appended as one record only at COMMIT. ROLLBACK restores the transaction
snapshot. Startup replays complete journal records and ignores a torn final
record, so an interrupted transaction cannot become partially visible.

`CREATE VECTOR INDEX` is accepted as metadata. Both `exact` and `ann` modes use
the exact path, which avoids ANN recall risk and keeps filtered and unfiltered
recall at 1.0. The cost is lower benchmark performance than a tuned approximate
index for larger datasets.

## Verification

The supplied public suite and additional destructive tests are included:

```bash
python tests/run_public.py   # 20/20 public cases
python tests/run_extended.py # limits, Unicode, transaction visibility
python tests/run_crash.py    # SIGKILL and torn-journal recovery
python tests/bench_small.py  # 10k x 128-d smoke benchmark
```

The official `protocol_check/run.py` from
`oceanbase/seekdb@2026-competition-round2` was also run with the trusted
`--command` mode. All 20 public cases passed; the recorded report is in
`verification/official-public-protocol-result.json`.

The latest local benchmark measured 18.612 seconds for 10,000
128-dimensional writes and 3.034 seconds for 1,000 TopK queries on a Windows
development host. The development machine does not have Docker, so the image
itself was not built locally. The Dockerfile uses a pinned NumPy release and
does not require network access at runtime.
