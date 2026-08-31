from __future__ import annotations
import hashlib, re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
from .models import CandidateIndex, WorkloadQuery, quote_ident
try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None; exp = None


def _parse_table_name(table_name: str):
    if "." in table_name:
        schema, table = table_name.split(".", 1)
        return schema.strip('"'), table.strip('"')
    return "public", table_name.strip('"')


def _make_name(schema: str, table: str, cols: List[str], prefix: str = "qdbo_idx") -> str:
    base = f"{schema}_{table}_{'_'.join(cols)}"
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).lower().strip("_")
    return f"{prefix}_{base}_{hashlib.sha1(base.encode()).hexdigest()[:8]}"[:60]


def _ddl(name: str, schema: str, table: str, cols: List[str], method="btree") -> str:
    return f"CREATE INDEX {quote_ident(name)} ON {quote_ident(schema)}.{quote_ident(table)} USING {method} ({', '.join(quote_ident(c) for c in cols)})"


def load_candidates_json(data) -> List[CandidateIndex]:
    out=[]
    for i, raw in enumerate(data):
        d=dict(raw)
        if "schema" not in d:
            schema, table = _parse_table_name(d.get("table", ""))
            d["schema"], d["table"] = schema, table
        d.setdefault("candidate_id", int(d.get("id", i)))
        d.setdefault("columns", [])
        d.setdefault("method", "btree")
        d.setdefault("name", _make_name(d["schema"], d["table"], d["columns"] or [str(i)]))
        d.setdefault("ddl", _ddl(d["name"], d["schema"], d["table"], d["columns"], d["method"]))
        d.setdefault("hypopg_ddl", d["ddl"])
        d.setdefault("source", "manual")
        d.setdefault("reason", "")
        d.setdefault("metadata", {})
        out.append(CandidateIndex.from_dict(d))
    return out


def candidates_to_jsonable(candidates: List[CandidateIndex]):
    return [c.to_dict() for c in candidates]


class CandidateExtractor:
    """Lightweight prototype candidate generator from SQL text.

    This is intentionally conservative. For a paper prototype, manual candidate lists
    are usually easier to defend; auto-generation is provided as a convenience.
    """
    def __init__(self, default_schema="public", max_columns=2, include_order_by=True, prefix="qdbo_idx", column_to_tables: Optional[Dict[str, Set[Tuple[str,str]]]] = None):
        self.default_schema=default_schema; self.max_columns=max_columns; self.include_order_by=include_order_by; self.prefix=prefix
        self.column_to_tables=column_to_tables or {}

    def _table_aliases(self, tree) -> Dict[str, tuple[str,str]]:
        aliases={}
        for t in tree.find_all(exp.Table):
            schema = t.db or self.default_schema
            table = t.name
            aliases[t.alias_or_name] = (schema, table)
            aliases[t.name] = (schema, table)
        return aliases

    def extract(self, workload: List[WorkloadQuery]) -> List[CandidateIndex]:
        if sqlglot is None or exp is None:
            raise RuntimeError("sqlglot is required for candidate extraction")
        scores: Dict[tuple[str,str], Counter[str]] = defaultdict(Counter)
        for q in workload:
            try: tree=sqlglot.parse_one(q.sql, read="postgres")
            except Exception: continue
            aliases=self._table_aliases(tree); tables=set(aliases.values()); default=next(iter(tables)) if len(tables)==1 else None
            nodes=[tree.args.get("where")]
            for j in tree.find_all(exp.Join): nodes.append(j.args.get("on"))
            if self.include_order_by: nodes.append(tree.args.get("order"))
            for node in nodes:
                if node is None: continue
                for col in node.find_all(exp.Column):
                    if col.table:
                        st=aliases.get(col.table)
                    elif default is not None:
                        st=default
                    else:
                        cands=self.column_to_tables.get(col.name.lower(), set()) & tables
                        st=next(iter(cands)) if len(cands)==1 else None
                    if st: scores[st][col.name]+=max(1,q.calls)
        candidates=[]; cid=0
        for (schema,table), counter in sorted(scores.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
            cols=[c for c,_ in counter.most_common()]
            for c in cols:
                name=_make_name(schema,table,[c],self.prefix); candidates.append(CandidateIndex(cid,name,schema,table,[c],_ddl(name,schema,table,[c]),None,"btree","auto",f"column seen in workload",{})); cid+=1
            if self.max_columns>=2:
                for a,b in zip(cols, cols[1:]):
                    name=_make_name(schema,table,[a,b],self.prefix); candidates.append(CandidateIndex(cid,name,schema,table,[a,b],_ddl(name,schema,table,[a,b]),None,"btree","auto",f"two-column candidate from workload",{})); cid+=1
            if self.max_columns>=3:
                for a,b,c in zip(cols, cols[1:], cols[2:]):
                    name=_make_name(schema,table,[a,b,c],self.prefix); candidates.append(CandidateIndex(cid,name,schema,table,[a,b,c],_ddl(name,schema,table,[a,b,c]),None,"btree","auto",f"three-column candidate from workload",{})); cid+=1
        return candidates


def prune_duplicate_existing(candidates: List[CandidateIndex], existing_indexdefs: List[str]) -> List[CandidateIndex]:
    norm_existing={re.sub(r"\s+"," ",x.lower()) for x in existing_indexdefs}
    out=[]
    for c in candidates:
        key_cols=", ".join(x.lower() for x in c.columns)
        if any(c.table.lower() in e and key_cols in e for e in norm_existing): continue
        out.append(c)
    return out
