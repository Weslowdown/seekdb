#!/usr/bin/env python3
"""MiniVectorDB: a small, self-contained SQL vector database."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class DBError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    size: int = 0


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


def sql_error(code: str, message: str) -> DBError:
    return DBError(code, message)


def tokenize(sql: str) -> list[Token]:
    result: list[Token] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            if end < 0:
                raise sql_error("invalid_sql", "unterminated block comment")
            i = end + 2
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            end = sql.find("\n", i + 2)
            i = n if end < 0 else end + 1
            continue
        if ch == "#":
            end = sql.find("\n", i + 1)
            i = n if end < 0 else end + 1
            continue
        if ch in ("'", '"'):
            quote = ch
            start = i
            i += 1
            search_from = i
            end = -1
            while True:
                end = sql.find(quote, search_from)
                if end < 0:
                    raise sql_error("invalid_sql", "unterminated string")
                if end + 1 < n and sql[end + 1] == quote:
                    search_from = end + 2
                    continue
                break
            raw = sql[i:end]
            if "\\" not in raw:
                value = raw.replace(quote + quote, quote)
            else:
                out: list[str] = []
                j = 0
                while j < len(raw):
                    if raw[j] == "\\" and j + 1 < len(raw):
                        esc = raw[j + 1]
                        mapping = {"n": "\n", "r": "\r", "t": "\t", "0": "\0"}
                        out.append(mapping.get(esc, esc))
                        j += 2
                    else:
                        out.append(raw[j])
                        j += 1
                value = "".join(out)
            i = end + 1
            if end < start:
                raise sql_error("invalid_sql", "unterminated string")
            result.append(Token("STRING", value, start))
            continue
        if ch == "`":
            start = i
            i += 1
            out: list[str] = []
            while i < n:
                if sql[i] == "`":
                    if i + 1 < n and sql[i + 1] == "`":
                        out.append("`")
                        i += 2
                        continue
                    i += 1
                    break
                out.append(sql[i])
                i += 1
            else:
                raise sql_error("invalid_sql", "unterminated identifier")
            result.append(Token("WORD", "".join(out), start))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (sql[i].isalnum() or sql[i] in "_$"):
                i += 1
            result.append(Token("WORD", sql[start:i], start))
            continue
        if ch.isdigit() or (
            ch == "." and i + 1 < n and sql[i + 1].isdigit()
        ):
            start = i
            i += 1
            while i < n and (sql[i].isdigit() or sql[i] in ".eE+-"):
                if sql[i] in "+-" and i > start and sql[i - 1] not in "eE":
                    break
                i += 1
            result.append(Token("NUMBER", sql[start:i], start))
            continue
        if ch in "<>!=":
            if i + 1 < n and sql[i : i + 2] in ("<=", ">=", "!=", "<>"):
                result.append(Token("OP", sql[i : i + 2], i))
                i += 2
            else:
                result.append(Token("OP", ch, i))
                i += 1
            continue
        if ch in "(),[]*+-.":
            result.append(Token(ch, ch, i))
            i += 1
            continue
        raise sql_error("unsupported_sql", f"unsupported character {ch!r}")
    result.append(Token("EOF", "", n))
    return result


class Parser:
    def __init__(self, sql: str):
        self.tokens = tokenize(sql)
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def take(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def accept(self, value: str) -> bool:
        tok = self.peek()
        if tok.kind.upper() == value.upper() or tok.value.upper() == value.upper():
            self.pos += 1
            return True
        return False

    def expect(self, value: str) -> Token:
        tok = self.peek()
        if tok.kind.upper() == value.upper() or tok.value.upper() == value.upper():
            self.pos += 1
            return tok
        raise sql_error("invalid_sql", f"expected {value}, got {tok.value or tok.kind}")

    def ident(self) -> str:
        tok = self.take()
        if tok.kind != "WORD":
            raise sql_error("invalid_sql", "expected identifier")
        name = tok.value
        while self.accept("."):
            name = self.ident()
        return name

    def integer(self) -> int:
        sign = 1
        if self.accept("-"):
            sign = -1
        tok = self.take()
        if tok.kind != "NUMBER" or not re.fullmatch(r"\d+", tok.value):
            raise sql_error("type_mismatch", "expected integer")
        return sign * int(tok.value)

    def number(self) -> float:
        sign = 1.0
        if self.accept("-"):
            sign = -1.0
        tok = self.take()
        if tok.kind != "NUMBER":
            raise sql_error("invalid_vector", "expected numeric vector component")
        try:
            return sign * float(tok.value)
        except ValueError:
            raise sql_error("invalid_vector", "invalid vector component")

    def vector(self) -> np.ndarray:
        if self.peek().kind == "STRING":
            return parse_vector_text(self.take().value)
        else:
            self.expect("[")
            values: list[str] = []
            while not self.accept("]"):
                if values:
                    self.expect(",")
                if self.accept("-"):
                    component = self.take()
                    if component.kind != "NUMBER":
                        raise sql_error("invalid_vector", "expected numeric vector component")
                    values.append("-" + component.value)
                elif self.accept("+"):
                    component = self.take()
                    if component.kind != "NUMBER":
                        raise sql_error("invalid_vector", "expected numeric vector component")
                    values.append(component.value)
                else:
                    component = self.take()
                    if component.kind != "NUMBER":
                        raise sql_error("invalid_vector", "expected numeric vector component")
                    values.append(component.value)
            try:
                vals = [float(x) for x in values]
            except ValueError:
                raise sql_error("invalid_vector", "invalid vector")
            return validate_vector(np.asarray(vals, dtype=np.float64))
        raise sql_error("invalid_vector", "invalid vector literal")

    def string(self) -> str:
        tok = self.take()
        if tok.kind != "STRING":
            raise sql_error("type_mismatch", "expected string")
        return tok.value

    def parse(self) -> dict[str, Any]:
        tok = self.peek()
        first = tok.value.upper() if tok.kind == "WORD" else tok.kind.upper()
        if first == "CREATE":
            statement = self.parse_create()
        elif first == "INSERT":
            statement = self.parse_insert()
        elif first == "SELECT":
            statement = self.parse_select()
        elif first == "UPDATE":
            statement = self.parse_update()
        elif first == "DELETE":
            statement = self.parse_delete()
        elif first == "BEGIN":
            self.take()
            self.accept("WORK")
            statement = {"kind": "begin"}
        elif first == "START":
            self.take()
            self.expect("TRANSACTION")
            statement = {"kind": "begin"}
        elif first in ("COMMIT", "END"):
            self.take()
            self.accept("WORK")
            statement = {"kind": "commit"}
        elif first == "ROLLBACK":
            self.take()
            self.accept("WORK")
            statement = {"kind": "rollback"}
        elif first == "SET":
            statement = self.parse_set()
        elif first == "DROP":
            statement = self.parse_drop()
        else:
            raise sql_error("unsupported_sql", f"unsupported statement {self.peek().value}")
        if self.peek().kind != "EOF":
            raise sql_error("unsupported_sql", f"unsupported trailing syntax {self.peek().value}")
        return statement

    def parse_create(self) -> dict[str, Any]:
        self.expect("CREATE")
        if self.accept("VECTOR"):
            self.expect("INDEX")
            index_name = self.ident()
            self.expect("ON")
            table = self.ident()
            self.expect("(")
            column = self.ident()
            self.expect(")")
            return {
                "kind": "create_index",
                "index": index_name,
                "table": table,
                "column": column,
            }
        self.expect("TABLE")
        if self.accept("IF"):
            self.expect("NOT")
            self.expect("EXISTS")
            if_not_exists = True
        else:
            if_not_exists = False
        table_name = self.ident()
        self.expect("(")
        columns: list[Column] = []
        primary: str | None = None
        primary_count = 0
        while True:
            if self.accept("PRIMARY"):
                self.expect("KEY")
                self.expect("(")
                primary = self.ident()
                primary_count += 1
                self.expect(")")
            else:
                name = self.ident()
                typ = self.ident().upper()
                if typ in ("INT", "INTEGER"):
                    typ = "BIGINT"
                size = 0
                if typ == "VARCHAR":
                    self.expect("(")
                    size = self.integer()
                    self.expect(")")
                elif typ == "VECTOR":
                    self.expect("(")
                    size = self.integer()
                    self.expect(")")
                elif typ != "BIGINT":
                    raise sql_error("unsupported_sql", f"unsupported type {typ}")
                if self.accept("PRIMARY"):
                    self.expect("KEY")
                    primary = name
                    primary_count += 1
                columns.append(Column(name, typ, size))
            if self.accept(")"):
                break
            self.expect(",")
        return {
            "kind": "create_table",
            "table": table_name,
            "columns": columns,
            "primary": primary,
            "primary_count": primary_count,
            "if_not_exists": if_not_exists,
        }

    def parse_insert(self) -> dict[str, Any]:
        self.expect("INSERT")
        self.expect("INTO")
        table = self.ident()
        target_columns: list[str] | None = None
        if self.accept("("):
            target_columns = [self.ident()]
            while self.accept(","):
                target_columns.append(self.ident())
            self.expect(")")
        self.expect("VALUES")
        rows: list[list[Any]] = []
        while True:
            self.expect("(")
            values: list[Any] = []
            if not self.accept(")"):
                while True:
                    values.append(self.literal())
                    if self.accept(")"):
                        break
                    self.expect(",")
            rows.append(values)
            if not self.accept(","):
                break
        return {
            "kind": "insert",
            "table": table,
            "columns": target_columns,
            "rows": rows,
        }

    def literal(self) -> Any:
        tok = self.peek()
        if tok.kind == "STRING":
            return self.take().value
        if tok.kind == "NUMBER" or tok.value == "-":
            sign = -1 if self.accept("-") else 1
            tok = self.take()
            if tok.kind != "NUMBER":
                raise sql_error("invalid_sql", "expected number")
            if any(c in tok.value for c in ".eE"):
                return sign * float(tok.value)
            return sign * int(tok.value)
        if tok.kind == "[":
            return self.vector()
        raise sql_error("invalid_sql", "expected literal")

    def parse_where(self) -> list[tuple[str, str, Any]]:
        conditions: list[tuple[str, str, Any]] = []
        while True:
            self.accept("(")
            column = self.ident()
            op = self.take().value
            if op not in ("=", "<", "<=", ">", ">=", "!=", "<>"):
                raise sql_error("unsupported_sql", f"unsupported operator {op}")
            conditions.append((column, "=" if op == "=" else op, self.literal()))
            self.accept(")")
            if not self.accept("AND"):
                break
        return conditions

    def parse_select(self) -> dict[str, Any]:
        self.expect("SELECT")
        select_all = self.accept("*")
        projections: list[dict[str, Any]] = []
        if not select_all:
            while True:
                if self.peek().kind == "WORD" and self.peek().value.upper() == "L2_DISTANCE":
                    self.take()
                    self.expect("(")
                    if self.peek().kind == "STRING" or self.peek().kind == "[":
                        vector = self.vector()
                        self.expect(",")
                        column = self.ident()
                    else:
                        column = self.ident()
                        self.expect(",")
                        vector = self.vector()
                    self.expect(")")
                    proj: dict[str, Any] = {
                        "type": "distance",
                        "column": column,
                        "vector": vector,
                    }
                else:
                    proj = {"type": "column", "column": self.ident()}
                if self.accept("AS"):
                    proj["alias"] = self.ident()
                projections.append(proj)
                if not self.accept(","):
                    break
        self.expect("FROM")
        table = self.ident()
        where: list[tuple[str, str, Any]] = []
        if self.accept("WHERE"):
            where = self.parse_where()
        order: list[tuple[str, bool]] = []
        if self.accept("ORDER"):
            self.expect("BY")
            while True:
                name = self.ident()
                desc = self.accept("DESC")
                if not desc:
                    self.accept("ASC")
                order.append((name, desc))
                if not self.accept(","):
                    break
        limit = None
        offset = 0
        if self.accept("LIMIT"):
            first = self.integer()
            if self.accept(","):
                offset = first
                limit = self.integer()
            else:
                limit = first
            if self.accept("OFFSET"):
                offset = self.integer()
        return {
            "kind": "select",
            "table": table,
            "all": select_all,
            "projections": projections,
            "where": where,
            "order": order,
            "limit": limit,
            "offset": offset,
        }

    def parse_update(self) -> dict[str, Any]:
        self.expect("UPDATE")
        table = self.ident()
        self.expect("SET")
        assignments: list[tuple[str, Any]] = []
        while True:
            name = self.ident()
            self.expect("=")
            assignments.append((name, self.literal()))
            if not self.accept(","):
                break
        where = []
        if self.accept("WHERE"):
            where = self.parse_where()
        return {
            "kind": "update",
            "table": table,
            "assignments": assignments,
            "where": where,
        }

    def parse_delete(self) -> dict[str, Any]:
        self.expect("DELETE")
        self.expect("FROM")
        table = self.ident()
        where = []
        if self.accept("WHERE"):
            where = self.parse_where()
        return {"kind": "delete", "table": table, "where": where}

    def parse_set(self) -> dict[str, Any]:
        self.expect("SET")
        name = self.ident().lower()
        self.expect("=")
        value = self.take().value.lower()
        return {"kind": "set", "name": name, "value": value}

    def parse_drop(self) -> dict[str, Any]:
        self.expect("DROP")
        self.expect("TABLE")
        if self.accept("IF"):
            self.expect("EXISTS")
            if_exists = True
        else:
            if_exists = False
        return {"kind": "drop_table", "table": self.ident(), "if_exists": if_exists}


def validate_vector(raw: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(raw)):
        raise sql_error("invalid_vector", "vector contains NaN or Infinity")
    with np.errstate(over="ignore", invalid="ignore"):
        vec = raw.astype(np.float32)
    if not np.all(np.isfinite(vec)):
        raise sql_error("vector_overflow", "vector component overflows float32")
    return vec


_VECTOR_NUMBER = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)
_VECTOR_BODY = re.compile(
    rf"\s*{_VECTOR_NUMBER.pattern}\s*(?:,\s*{_VECTOR_NUMBER.pattern}\s*)*"
)


def parse_vector_text(raw: str) -> np.ndarray:
    text = raw.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise sql_error("invalid_vector", "vector must be enclosed in brackets")
    body = text[1:-1].strip()
    if not body:
        return validate_vector(np.asarray([], dtype=np.float64))
    if _VECTOR_BODY.fullmatch(body) is not None:
        return validate_vector(np.fromstring(body, sep=",", dtype=np.float64))
    raise sql_error("invalid_vector", "vector contains an invalid component")


def compare(left: Any, op: str, right: Any) -> bool:
    if op == "=":
        return left == right
    if op == "!=" or op == "<>":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    return False


class Table:
    def __init__(self, name: str, columns: list[Column], primary: str):
        self.name = name
        self.columns = columns
        self.column_by_name = {c.name.lower(): c for c in columns}
        self.primary_index = self.index_of(primary)
        self.pk = columns[self.primary_index].name.lower()
        self.rows: dict[int, tuple[Any, ...]] = {}
        self.vector_indexes: dict[str, str] = {}
        self.version = 0
        self._cache: tuple[
            np.ndarray, np.ndarray, np.ndarray, dict[int, int], list[np.ndarray]
        ] | None = None

    def index_of(self, name: str) -> int:
        low = name.lower()
        for i, col in enumerate(self.columns):
            if col.name.lower() == low:
                return i
        raise sql_error("column_not_found", f"unknown column {name}")

    def vector_index_of(self) -> int:
        found = [i for i, c in enumerate(self.columns) if c.kind == "VECTOR"]
        if len(found) != 1:
            raise sql_error("invalid_table", "table must have exactly one vector column")
        return found[0]

    def cached_matrix(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[int, int], list[np.ndarray]
    ]:
        if self._cache is not None:
            return self._cache
        primary: list[int] = []
        vector_indexes = [
            i for i, column in enumerate(self.columns) if column.kind == "VECTOR"
        ]
        dimension = (
            self.columns[vector_indexes[0]].size if vector_indexes else 0
        )
        matrix = np.empty((len(self.rows), dimension), dtype=np.float32)
        for i, (pk, row) in enumerate(self.rows.items()):
            primary.append(pk)
            if vector_indexes:
                matrix[i] = row[vector_indexes[0]]
        pks = np.asarray(primary, dtype=np.int64)
        vector_norms = np.einsum(
            "ij,ij->i", matrix.astype(np.float64), matrix.astype(np.float64)
        )
        position = {int(pk): i for i, pk in enumerate(primary)}
        columns: list[np.ndarray] = []
        rows = list(self.rows.values())
        for column_index, column in enumerate(self.columns):
            if column.kind == "BIGINT":
                columns.append(
                    np.fromiter(
                        (row[column_index] for row in rows),
                        dtype=np.int64,
                        count=len(rows),
                    )
                )
            elif column.kind == "VECTOR":
                columns.append(matrix)
            else:
                columns.append(np.asarray([row[column_index] for row in rows], dtype=object))
        self._cache = (pks, matrix, vector_norms, position, columns)
        return self._cache

    def changed(self) -> None:
        self.version += 1
        self._cache = None


class Database:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = data_dir / "journal.ndjson"
        self.tables: dict[str, Table] = {}
        self.vector_mode = "exact"
        self.tx_active = False
        self.tx_snapshot: dict[str, Table] | None = None
        self.tx_pending: list[str] = []
        self._journal = None
        self._recover()

    def _recover(self) -> None:
        if self.journal_path.exists():
            with self.journal_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    statements = record.get("tx", [])
                    for sql in statements:
                        try:
                            self.execute(Parser(sql).parse(), sql=sql, replay=True)
                        except DBError:
                            # A torn/unusable historical statement must not make the
                            # whole database unavailable.
                            continue
        self._journal = self.journal_path.open("a", encoding="utf-8")

    def _persist(self, statements: list[str]) -> None:
        payload = json.dumps(
            {"tx": statements}, ensure_ascii=False, separators=(",", ":")
        )
        assert self._journal is not None
        self._journal.write(payload + "\n")
        self._journal.flush()
        os.fsync(self._journal.fileno())

    def close(self) -> None:
        if self.tx_active:
            self.rollback()
        if self._journal is not None:
            self._journal.close()
            self._journal = None

    def begin(self) -> None:
        if self.tx_active:
            raise sql_error("transaction_error", "nested transaction is not supported")
        self.tx_active = True
        self.tx_snapshot = copy.deepcopy(self.tables)
        self.tx_pending = []

    def rollback(self) -> None:
        if not self.tx_active:
            return
        assert self.tx_snapshot is not None
        self.tables = self.tx_snapshot
        self.tx_active = False
        self.tx_snapshot = None
        self.tx_pending = []

    def commit(self) -> None:
        if not self.tx_active:
            return
        if self.tx_pending:
            self._persist(self.tx_pending)
        self.tx_active = False
        self.tx_snapshot = None
        self.tx_pending = []

    def execute(
        self, stmt: dict[str, Any], sql: str | None = None, replay: bool = False
    ) -> tuple[list[str] | None, list[list[Any]] | None, int | None]:
        kind = stmt["kind"]
        if kind == "begin":
            self.begin()
            return None, None, None
        if kind == "rollback":
            self.rollback()
            return None, None, None
        if kind == "commit":
            self.commit()
            return None, None, None
        if kind == "set":
            if stmt["name"] not in ("vector_mode", "mvd_vector_mode"):
                raise sql_error("unsupported_sql", "unsupported setting")
            if stmt["value"] not in ("exact", "ann"):
                raise sql_error("invalid_sql", "invalid vector mode")
            self.vector_mode = stmt["value"]
            return None, None, None

        mutating = kind in (
            "create_table",
            "drop_table",
            "insert",
            "update",
            "delete",
            "create_index",
        )
        if kind == "create_table":
            self.create_table(stmt)
            result = (None, None, None)
        elif kind == "drop_table":
            key = stmt["table"].lower()
            if key not in self.tables and not stmt["if_exists"]:
                raise sql_error("table_not_found", f"unknown table {stmt['table']}")
            self.tables.pop(key, None)
            result = (None, None, None)
        elif kind == "create_index":
            table = self.table(stmt["table"])
            col = table.column_by_name.get(stmt["column"].lower())
            if col is None or col.kind != "VECTOR":
                raise sql_error("column_not_found", "vector column not found")
            table.vector_indexes[stmt["index"].lower()] = col.name
            result = (None, None, None)
        elif kind == "insert":
            result = (None, None, self.insert(stmt))
        elif kind == "update":
            result = (None, None, self.update(stmt))
        elif kind == "delete":
            result = (None, None, self.delete(stmt))
        elif kind == "select":
            columns, rows = self.select(stmt)
            result = (columns, rows, None)
        else:
            raise sql_error("unsupported_sql", "unsupported statement")
        if mutating and not replay:
            if self.tx_active:
                self.tx_pending.append(sql or "")
            else:
                self._persist([sql or ""])
        return result

    def table(self, name: str) -> Table:
        table = self.tables.get(name.lower())
        if table is None:
            raise sql_error("table_not_found", f"unknown table {name}")
        return table

    def create_table(self, stmt: dict[str, Any]) -> None:
        name = stmt["table"]
        key = name.lower()
        if key in self.tables:
            if stmt["if_not_exists"]:
                return
            raise sql_error("table_already_exists", f"table {name} already exists")
        columns = stmt["columns"]
        primary = stmt["primary"]
        if primary is None or stmt.get("primary_count") != 1:
            raise sql_error("invalid_table", "a primary key is required")
        names = [column.name.lower() for column in columns]
        if len(names) != len(set(names)):
            raise sql_error("invalid_table", "duplicate column name")
        if not any(c.name.lower() == primary.lower() and c.kind == "BIGINT" for c in columns):
            raise sql_error("invalid_table", "primary key must be BIGINT")
        if sum(1 for c in columns if c.kind == "VECTOR") > 1:
            raise sql_error("invalid_table", "at most one vector column is allowed")
        for column in columns:
            if column.kind in ("VARCHAR", "VECTOR") and column.size <= 0:
                raise sql_error("invalid_table", f"{column.name} requires a positive size")
        self.tables[key] = Table(name, columns, primary)

    def convert_value(self, col: Column, raw: Any) -> Any:
        if col.kind == "BIGINT":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise sql_error("type_mismatch", f"{col.name} requires BIGINT")
            if raw < -(1 << 63) or raw > (1 << 63) - 1:
                raise sql_error("integer_overflow", f"{col.name} is out of range")
            return int(raw)
        if col.kind == "VARCHAR":
            if not isinstance(raw, str):
                raise sql_error("type_mismatch", f"{col.name} requires VARCHAR")
            if len(raw) > col.size:
                raise sql_error("varchar_too_long", f"{col.name} is too long")
            return raw
        if col.kind == "VECTOR":
            if isinstance(raw, str):
                vec = parse_vector_text(raw)
            elif isinstance(raw, np.ndarray):
                vec = validate_vector(raw)
            else:
                raise sql_error("type_mismatch", f"{col.name} requires VECTOR")
            if vec.size != col.size:
                raise sql_error("dimension_mismatch", f"{col.name} must have {col.size} dimensions")
            return vec
        raise sql_error("type_mismatch", "unknown type")

    def insert(self, stmt: dict[str, Any]) -> int:
        table = self.table(stmt["table"])
        target_names = stmt["columns"]
        if target_names is None:
            target_indexes = list(range(len(table.columns)))
        else:
            target_indexes = [table.index_of(name) for name in target_names]
        converted: list[tuple[Any, ...]] = []
        seen: set[int] = set()
        for raw_row in stmt["rows"]:
            if len(raw_row) != len(target_indexes):
                raise sql_error("column_count_mismatch", "wrong number of values")
            row: list[Any] = [None] * len(table.columns)
            specified = set()
            for index, raw in zip(target_indexes, raw_row):
                if index in specified:
                    raise sql_error("invalid_sql", "duplicate target column")
                specified.add(index)
                row[index] = self.convert_value(table.columns[index], raw)
            if len(specified) != len(table.columns):
                raise sql_error("invalid_sql", "all columns are required")
            pk = int(row[table.primary_index])
            if pk in table.rows or pk in seen:
                raise sql_error("duplicate_key", f"duplicate primary key {pk}")
            seen.add(pk)
            converted.append(tuple(row))
        for row in converted:
            table.rows[int(row[table.primary_index])] = row
        if converted:
            table.changed()
        return len(converted)

    def rows_matching(
        self, table: Table, where: Iterable[tuple[str, str, Any]]
    ) -> list[tuple[int, tuple[Any, ...]]]:
        predicates: list[tuple[int, str, Any]] = []
        for name, op, raw in where:
            index = table.index_of(name)
            predicates.append((index, op, self.convert_value(table.columns[index], raw)))
        result = []
        for pk, row in table.rows.items():
            ok = True
            for index, op, expected in predicates:
                try:
                    if not compare(row[index], op, expected):
                        ok = False
                        break
                except TypeError:
                    ok = False
                    break
            if ok:
                result.append((pk, row))
        return result

    def update(self, stmt: dict[str, Any]) -> int:
        table = self.table(stmt["table"])
        indexes: list[tuple[int, Any]] = []
        for name, raw in stmt["assignments"]:
            index = table.index_of(name)
            if index == table.primary_index:
                raise sql_error("unsupported_sql", "primary key updates are not supported")
            indexes.append((index, self.convert_value(table.columns[index], raw)))
        matched = self.rows_matching(table, stmt["where"])
        new_rows: list[tuple[int, tuple[Any, ...]]] = []
        for pk, row in matched:
            mutable = list(row)
            for index, value in indexes:
                mutable[index] = value
            new_rows.append((pk, tuple(mutable)))
        for pk, row in new_rows:
            table.rows[pk] = row
        if new_rows:
            table.changed()
        return len(new_rows)

    def delete(self, stmt: dict[str, Any]) -> int:
        table = self.table(stmt["table"])
        matched = self.rows_matching(table, stmt["where"])
        for pk, _ in matched:
            table.rows.pop(pk, None)
        if matched:
            table.changed()
        return len(matched)

    def select(self, stmt: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
        table = self.table(stmt["table"])
        if stmt["all"]:
            projections = [
                {"type": "column", "column": col.name, "alias": col.name}
                for col in table.columns
            ]
        else:
            projections = list(stmt["projections"])
            for proj in projections:
                proj.setdefault("alias", proj["column"])
        available = {col.name.lower() for col in table.columns}
        available.update(str(p["alias"]).lower() for p in projections)
        for name, _ in stmt["order"]:
            if name.lower() not in available:
                raise sql_error("column_not_found", f"unknown column {name}")
        pks, matrix, vector_norms, _positions, cached_columns = table.cached_matrix()
        matched_positions = self.filter_positions(table, stmt["where"], cached_columns)

        distance_projection = next(
            (p for p in projections if p["type"] == "distance"), None
        )
        distances: np.ndarray | None = None
        if distance_projection is not None:
            vector_index = table.index_of(distance_projection["column"])
            if table.columns[vector_index].kind != "VECTOR":
                raise sql_error("type_mismatch", "L2_DISTANCE requires a vector column")
            query = distance_projection["vector"]
            if query.size != table.columns[vector_index].size:
                raise sql_error("dimension_mismatch", "query vector dimension mismatch")
            if matched_positions.size:
                candidate_vectors = matrix[matched_positions]
                if candidate_vectors.size and np.max(np.abs(candidate_vectors)) > 1.0e18:
                    products = candidate_vectors.astype(np.float64) @ query.astype(np.float64)
                else:
                    products = candidate_vectors @ query
                squared = (
                    vector_norms[matched_positions]
                    + np.dot(query.astype(np.float64), query.astype(np.float64))
                    - 2.0 * products
                )
                distances = np.sqrt(np.maximum(squared, 0.0))
            else:
                distances = np.empty(0, dtype=np.float64)

        order_keys: list[np.ndarray] = []
        for name, desc in stmt["order"]:
            low = name.lower()
            if distance_projection is not None and low == str(
                distance_projection["alias"]
            ).lower():
                assert distances is not None
                values = distances
            else:
                column_index = table.index_of(name)
                values = cached_columns[column_index][matched_positions]
            if desc:
                if values.dtype.kind in "iuf":
                    values = -values
                else:
                    # Object/string keys use the stable Python fallback below.
                    values = values
            order_keys.append(values)

        matched_pks = pks[matched_positions]
        if order_keys:
            topk_order = (
                distance_projection is not None
                and len(stmt["order"]) == 1
                and stmt["order"][0][0].lower()
                == str(distance_projection["alias"]).lower()
                and stmt["limit"] is not None
            )
            if topk_order:
                assert distances is not None
                wanted = max(int(stmt["offset"]) + int(stmt["limit"]), 0)
                if wanted < len(matched_positions):
                    if wanted == 0:
                        candidate_indexes = np.empty(0, dtype=np.int64)
                    else:
                        partition_key = (
                            -distances
                            if stmt["order"][0][1]
                            else distances
                        )
                        threshold = np.partition(
                            partition_key, wanted - 1
                        )[wanted - 1]
                        # Keep every row tied at the Kth distance; the primary-key
                        # tie-break must run before the final LIMIT.
                        candidate_indexes = np.flatnonzero(
                            partition_key <= threshold
                        )
                    matched_positions = matched_positions[candidate_indexes]
                    distances = distances[candidate_indexes]
                    matched_pks = pks[matched_positions]
                    order_keys = [
                        -distances if stmt["order"][0][1] else distances
                    ]
                object_descending = False
            else:
                object_descending = any(
                    values.dtype.kind == "O" and desc
                    for (_, desc), values in zip(stmt["order"], order_keys)
                )
            if object_descending:
                order_positions = sorted(
                    range(len(matched_positions)),
                    key=lambda i: int(matched_pks[i]),
                )
                for (_, desc), values in reversed(
                    list(zip(stmt["order"], order_keys))
                ):
                    order_positions.sort(
                        key=lambda i: values[i], reverse=desc
                    )
                order_positions_array = np.asarray(order_positions, dtype=np.int64)
                matched_positions = matched_positions[order_positions_array]
                if distances is not None:
                    distances = distances[order_positions_array]
            else:
                # lexsort's final key is primary; pks also provide the required
                # ascending primary-key tie-break.
                sorted_indexes = np.lexsort(tuple([matched_pks] + list(reversed(order_keys))))
                matched_positions = matched_positions[sorted_indexes]
                if distances is not None:
                    distances = distances[sorted_indexes]
        else:
            sorted_indexes = np.argsort(matched_pks, kind="stable")
            matched_positions = matched_positions[sorted_indexes]
            if distances is not None:
                distances = distances[sorted_indexes]

        offset = max(int(stmt["offset"]), 0)
        if stmt["limit"] is not None:
            limit = max(int(stmt["limit"]), 0)
            matched_positions = matched_positions[offset : offset + limit]
            if distances is not None:
                distances = distances[offset : offset + limit]
        else:
            matched_positions = matched_positions[offset:]
            if distances is not None:
                distances = distances[offset:]

        rows: list[list[Any]] = []
        result_pks = pks[matched_positions]
        distance_iter = iter(distances.tolist()) if distances is not None else None
        for pk in result_pks.tolist():
            row = table.rows[int(pk)]
            values: list[Any] = []
            for projection in projections:
                if projection["type"] == "column":
                    values.append(row[table.index_of(projection["column"])])
                else:
                    assert distance_iter is not None
                    values.append(next(distance_iter))
            rows.append(values)
        return [str(p["alias"]) for p in projections], rows

    def filter_positions(
        self,
        table: Table,
        where: Iterable[tuple[str, str, Any]],
        cached_columns: list[np.ndarray],
    ) -> np.ndarray:
        count = len(table.rows)
        mask = np.ones(count, dtype=bool)
        for name, op, raw in where:
            index = table.index_of(name)
            column = table.columns[index]
            if column.kind == "VECTOR":
                raise sql_error("type_mismatch", "vector comparisons are not supported")
            expected = self.convert_value(column, raw)
            values = cached_columns[index]
            if op == "=":
                mask &= values == expected
            elif op in ("!=", "<>"):
                mask &= values != expected
            elif op == "<":
                mask &= values < expected
            elif op == "<=":
                mask &= values <= expected
            elif op == ">":
                mask &= values > expected
            elif op == ">=":
                mask &= values >= expected
            else:
                raise sql_error("unsupported_sql", f"unsupported operator {op}")
        return np.flatnonzero(mask)


def split_statements(text: str, at_eof: bool) -> tuple[list[str], str]:
    statements: list[str] = []
    start = 0
    quote: str | None = None
    block_comment = False
    line_comment = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
            i += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue
        if quote:
            if ch == quote:
                if nxt == quote:
                    i += 2
                    continue
                quote = None
            elif ch == "\\":
                i += 2
                continue
            i += 1
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        if (ch == "-" and nxt == "-") or ch == "#":
            line_comment = True
            i += 1 if ch == "#" else 2
            continue
        if ch == ";":
            piece = text[start:i].strip()
            if piece:
                statements.append(piece)
            start = i + 1
        i += 1
    remainder = text[start:]
    if at_eof:
        piece = remainder.strip()
        if piece:
            statements.append(piece)
        remainder = ""
    return statements, remainder


def render_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, np.ndarray):
        return "[" + ",".join(f"{float(x):.6g}" for x in value) + "]"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def execute_statement(db: Database, sql: str) -> tuple[list[str] | None, int | None]:
    stmt = Parser(sql).parse()
    columns, rows, affected = db.execute(stmt, sql=sql)
    if columns is not None and rows is not None:
        for row in rows:
            print("\t".join(render_value(value) for value in row), flush=True)
    elif affected is not None:
        print(f"OK {affected}", flush=True)
    else:
        print("OK", flush=True)
    return columns, affected


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    sys.stdout.reconfigure(encoding="utf-8", errors="strict", newline="\n")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--ready-marker", default="")
    parser.add_argument("--vector-mode", choices=("exact", "ann"), default="exact")
    parser.add_argument("--header", action="store_true")
    parser.add_argument("--row-separator", default="\t")
    parser.add_argument("--numeric-errors", action="store_true")
    parser.add_argument("--page-size", default=None)
    parser.add_argument("--pool-pages", default=None)
    parser.add_argument("--ivf-nprobe", default=None)
    parser.add_argument("--checkpoint-mb", default=None)
    args, _unknown = parser.parse_known_args()

    db = Database(Path(args.data_dir))
    db.vector_mode = args.vector_mode

    def stop(_signum: int, _frame: Any) -> None:
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    pending = ""
    try:
        while True:
            line = sys.stdin.readline()
            if line == "":
                statements, pending = split_statements(pending, True)
                for sql in statements:
                    process_one(db, sql, args)
                break
            pending += line
            statements, pending = split_statements(pending, False)
            for sql in statements:
                process_one(db, sql, args)
    finally:
        db.close()
    return 0


def process_one(db: Database, sql: str, args: argparse.Namespace) -> None:
    try:
        execute_statement(db, sql)
    except DBError as exc:
        code = str(exc.code)
        print(f"ERROR {code}: {exc.message}", flush=True)
        print("", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"ERROR internal_error: {exc}", flush=True)
        print(f"internal error: {exc}", file=sys.stderr, flush=True)
    if args.ready_marker:
        print(args.ready_marker, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
