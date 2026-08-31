from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

class PostgreSQLError(RuntimeError): pass

@dataclass
class PGSettings:
    dsn: str
    statement_timeout_ms: int = 60000
    application_name: str = "qdbo-index-selection"


class PGClient:
    def __init__(self, settings: PGSettings):
        if psycopg is None:
            raise RuntimeError("psycopg is required. Install with: pip install 'psycopg[binary]'")
        self.settings = settings
        self.conn = None
        self.server_version_num: Optional[int] = None

    def __enter__(self) -> "PGClient":
        self.connect(); return self
    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        kwargs: Dict[str, Any] = {"autocommit": True}
        if self.settings.application_name:
            kwargs["application_name"] = self.settings.application_name
        self.conn = psycopg.connect(self.settings.dsn, **kwargs)
        with self.conn.cursor() as cur:
            cur.execute("SHOW server_version_num")
            self.server_version_num = int(cur.fetchone()[0])
            if self.settings.statement_timeout_ms > 0:
                cur.execute("SELECT set_config('statement_timeout', %s, false)", (f"{self.settings.statement_timeout_ms}ms",))

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close(); self.conn = None

    def cursor(self):
        if self.conn is None:
            raise PostgreSQLError("PGClient is not connected")
        return self.conn.cursor()

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
            return row[0] if row else None

    def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> List[Tuple[Any, ...]]:
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            return list(cur.fetchall())

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        with self.cursor() as cur:
            cur.execute(sql, params or ())

    def ensure_extension(self, extension: str) -> None:
        self.scalar(f"CREATE EXTENSION IF NOT EXISTS {extension}")

    def has_extension(self, extension: str) -> bool:
        return bool(self.scalar("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname=%s)", (extension,)))

    def existing_indexes(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT schemaname, tablename, indexname, indexdef FROM pg_indexes WHERE schemaname NOT IN ('pg_catalog','information_schema')"
        params: List[Any] = []
        if schema:
            sql += " AND schemaname=%s"; params.append(schema)
        rows = self.fetchall(sql + " ORDER BY schemaname, tablename, indexname", params)
        return [{"schema": r[0], "table": r[1], "indexname": r[2], "indexdef": r[3]} for r in rows]

    def table_columns(self, schema: str, table: str) -> List[str]:
        rows = self.fetchall(
            "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
            (schema, table),
        )
        return [str(r[0]) for r in rows]
