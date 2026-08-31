#!/usr/bin/env python3
"""
add_pk_candidates_nopk.py

Append the 8 TPC-H PRIMARY KEY columns as explicit B-tree candidates to a
candidates.json file. Required for the strict-Kossmann no-PK experiment
(Task B), where we removed all PKs from the database and want them as
selectable candidates.

Dedupe rule: if a candidate with the same (table, columns) already exists,
skip; otherwise append with a fresh candidate_id at the end.

Usage:
  add_pk_candidates_nopk.py <input_candidates.json> <output_candidates.json>
"""

import json
import sys
import hashlib
from pathlib import Path

PK_COLUMNS = [
    ("region",   ["r_regionkey"]),
    ("nation",   ["n_nationkey"]),
    ("part",     ["p_partkey"]),
    ("supplier", ["s_suppkey"]),
    ("customer", ["c_custkey"]),
    ("orders",   ["o_orderkey"]),
    ("partsupp", ["ps_partkey", "ps_suppkey"]),
    ("lineitem", ["l_orderkey", "l_linenumber"]),
]


def column_set_key(table, cols):
    return (table.lower(), tuple(c.lower() for c in cols))


def name_for(table, cols):
    sig = "_".join(cols)
    h = hashlib.md5(f"{table}_{sig}".encode()).hexdigest()[:8]
    return f"qdbo_tpch_idx_public_{table}_{sig}_{h}"


def ddl_for(table, cols, name):
    qcols = ", ".join(f'"{c}"' for c in cols)
    return f'CREATE INDEX "{name}" ON "public"."{table}" USING btree ({qcols})'


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cands = json.loads(src.read_text())
    if not isinstance(cands, list):
        print(f"ERR: expected list, got {type(cands).__name__}"); sys.exit(2)

    existing = {column_set_key(c["table"], c["columns"]) for c in cands}
    next_id = max((c["candidate_id"] for c in cands), default=-1) + 1

    added = []
    for table, cols in PK_COLUMNS:
        key = column_set_key(table, cols)
        if key in existing:
            print(f"[skip] PK ({table}, {cols}) already in candidates")
            continue
        name = name_for(table, cols)
        cand = {
            "candidate_id": next_id,
            "name": name,
            "schema": "public",
            "table": table,
            "columns": cols,
            "ddl": ddl_for(table, cols, name),
            "hypopg_ddl": None,
            "origin": "pk_augment",
        }
        cands.append(cand)
        existing.add(key)
        added.append((next_id, table, cols))
        next_id += 1

    dst.write_text(json.dumps(cands, indent=2))
    print(f"[ok] wrote {dst} (n={len(cands)}, added {len(added)} PK candidates)")
    for i, t, c in added:
        print(f"  +cid={i} {t}({','.join(c)})")


if __name__ == "__main__":
    main()
