from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import List
from .models import WorkloadQuery
from .pg import PGClient


def workload_from_pg_stat_statements(pg: PGClient, *, limit: int = 50, min_calls: int = 1, order_by: str = "total_exec_time") -> List[WorkloadQuery]:
    order = {"total_exec_time": "total_exec_time", "mean_exec_time": "mean_exec_time", "calls": "calls"}.get(order_by, "total_exec_time")
    rows = pg.fetchall(f"""
        SELECT queryid, calls, total_exec_time, mean_exec_time, rows, query
        FROM pg_stat_statements
        WHERE dbid=(SELECT oid FROM pg_database WHERE datname=current_database())
          AND calls >= %s
          AND query ~* '^\\s*(select|with)'
          AND query NOT ILIKE '%%pg_stat_statements%%'
        ORDER BY {order} DESC LIMIT %s
    """, (min_calls, limit))
    return [WorkloadQuery(str(r[0]), str(r[5]), int(r[1]), float(r[2]), float(r[3]), int(r[4]), "pg_stat_statements") for r in rows]


def workload_from_jsonable(data) -> List[WorkloadQuery]:
    return [WorkloadQuery.from_dict(x) for x in data]


def workload_to_jsonable(workload: List[WorkloadQuery]):
    return [q.to_dict() for q in workload]


def _split_sql(text: str) -> List[str]:
    out=[]; cur=[]; single=False; double=False; prev=""
    for ch in text:
        cur.append(ch)
        if ch=="'" and not double and prev != "\\": single = not single
        elif ch=='"' and not single and prev != "\\": double = not double
        elif ch==";" and not single and not double:
            s="".join(cur).strip().rstrip(";").strip()
            if s: out.append(s)
            cur=[]
        prev=ch
    tail="".join(cur).strip().rstrip(";").strip()
    if tail: out.append(tail)
    return out


def workload_from_sql_file(path: str, default_calls: int = 1) -> List[WorkloadQuery]:
    text = Path(path).read_text(encoding="utf-8")
    weight_re = re.compile(r"--\s*(calls|weight)\s*:\s*(\d+)", re.I)
    out=[]
    for i, stmt in enumerate(_split_sql(text)):
        m = weight_re.search(stmt)
        calls = int(m.group(2)) if m else default_calls
        sql = re.sub(weight_re, "", stmt).strip()
        if not sql: continue
        qid = hashlib.sha1(sql.encode("utf-8")).hexdigest()[:16]
        out.append(WorkloadQuery(f"sqlfile_{i}_{qid}", sql, calls, 0.0, 0.0, 0, "sql_file"))
    return out
