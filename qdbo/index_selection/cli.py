from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from .candidates import (
    CandidateExtractor,
    candidates_to_jsonable,
    prune_duplicate_existing,
)
from .pg import PGClient, PGSettings
from .qubo import (
    build_index_selection_qubo,
    normalize_qubo,
    qubo_to_ising,
    qubo_to_jsonable,
)
from .solvers import solve_problem_dict
from .utils import config_get, load_config, read_json, write_json
from .workload import (
    workload_from_jsonable,
    workload_from_pg_stat_statements,
    workload_from_sql_file,
    workload_to_jsonable,
)


def _pg(config: Dict[str, Any]) -> PGClient:
    dsn = (
        os.getenv("PG_DSN")
        or config_get(config, "postgres.dsn")
        or config_get(config, "database.dsn")
    )
    if not dsn:
        raise SystemExit("Missing PG_DSN or postgres/database.dsn in config")
    timeout = int(
        os.getenv("PG_STATEMENT_TIMEOUT_MS")
        or config_get(
            config,
            "postgres.statement_timeout_ms",
            config_get(config, "database.statement_timeout_ms", 60000),
        )
    )
    return PGClient(
        PGSettings(
            dsn,
            timeout,
            str(
                config_get(config, "postgres.application_name", "qdbo-index-selection")
            ),
        )
    )


def _problem_dict(path: str) -> Dict[str, Any]:
    problem = read_json(path)
    if "max_wgt" in problem and "max_weight" not in problem:
        problem["max_weight"] = problem.pop("max_wgt")
    problem.setdefault("evaluations", [])
    problem.setdefault("metadata", {})
    problem.setdefault(
        "pairwise_interactions",
        problem.pop("interactions", {}) if "interactions" in problem else {},
    )
    return problem


def cmd_doctor(args):
    config = load_config(args.config)
    print("Python package import: OK")
    if (
        os.getenv("PG_DSN")
        or config_get(config, "postgres.dsn")
        or config_get(config, "database.dsn")
    ):
        with _pg(config) as pg:
            print(f"PostgreSQL server_version_num={pg.server_version_num}")
            print(
                f"pg_stat_statements installed={pg.has_extension('pg_stat_statements')}"
            )
            print(f"hypopg installed={pg.has_extension('hypopg')}")
    else:
        print("No PG_DSN configured; skipped PG checks")


def cmd_collect_workload(args):
    config = load_config(args.config)
    if args.sql_file:
        workload = workload_from_sql_file(args.sql_file, args.default_calls)
    else:
        with _pg(config) as pg:
            workload = workload_from_pg_stat_statements(
                pg,
                limit=args.limit or int(config_get(config, "workload.limit", 50)),
                min_calls=args.min_calls
                or int(config_get(config, "workload.min_calls", 1)),
                order_by=args.order_by
                or str(config_get(config, "workload.order_by", "total_exec_time")),
            )
    write_json(args.out, workload_to_jsonable(workload))
    print(f"wrote {args.out} ({len(workload)} queries)")


def _workload_table_set(workload, default_schema: str):
    import sqlglot
    from sqlglot import exp

    out = set()
    for q in workload:
        try:
            tree = sqlglot.parse_one(q.sql, read="postgres")
        except Exception:
            continue
        for t in tree.find_all(exp.Table):
            out.add((t.db or default_schema, t.name))
    return out


def _build_column_to_tables(pg: PGClient, tables):
    m: Dict[str, set] = {}
    for schema, table in tables:
        for col in pg.table_columns(schema, table):
            m.setdefault(col.lower(), set()).add((schema, table))
    return m


def cmd_generate_candidates(args):
    config = load_config(args.config)
    workload = workload_from_jsonable(read_json(args.workload))
    default_schema = args.default_schema or str(
        config_get(config, "candidate_generation.default_schema", "public")
    )
    have_pg = bool(
        os.getenv("PG_DSN")
        or config_get(config, "postgres.dsn")
        or config_get(config, "database.dsn")
    )
    column_to_tables: Dict[str, set] = {}
    existing: List[Dict[str, Any]] = []
    if have_pg:
        with _pg(config) as pg:
            column_to_tables = _build_column_to_tables(
                pg, _workload_table_set(workload, default_schema)
            )
            if args.prune_existing:
                existing = pg.existing_indexes(schema=default_schema)
    elif args.prune_existing:
        raise SystemExit(
            "--prune-existing requires PG_DSN or postgres/database.dsn in config"
        )
    else:
        print(
            "warning: no PG configured; unqualified columns in multi-table queries will be dropped"
        )
    extractor = CandidateExtractor(
        default_schema=default_schema,
        max_columns=args.max_columns
        or int(config_get(config, "candidate_generation.max_columns", 2)),
        include_order_by=not args.no_order_by,
        prefix=args.prefix
        or str(config_get(config, "candidate_generation.prefix", "qdbo_idx")),
        column_to_tables=column_to_tables,
    )
    candidates = extractor.extract(workload)
    if args.prune_existing:
        candidates = prune_duplicate_existing(
            candidates, [item["indexdef"] for item in existing]
        )
    max_candidates = args.max_candidates or int(
        config_get(config, "candidate_generation.max_candidates", 20)
    )
    candidates = candidates[:max_candidates]
    for index, candidate in enumerate(candidates):
        candidate.candidate_id = index
    write_json(args.out, candidates_to_jsonable(candidates))
    print(f"wrote {args.out} ({len(candidates)} candidates)")


def cmd_build_qubo(args):
    problem = _problem_dict(args.problem)
    if args.budget is not None:
        problem["max_weight"] = args.budget
    qubo, meta = build_index_selection_qubo(
        problem["profits"],
        problem["weights"],
        problem["max_weight"],
        penalty=args.penalty,
        pairwise_interactions=problem.get("pairwise_interactions") or None,
    )
    if args.normalize is not None:
        qubo, scale = normalize_qubo(qubo, args.normalize)
        meta["normalization_factor"] = scale
    out = {"Q": qubo_to_jsonable(qubo), "meta": meta}
    if args.include_ising:
        h, j_couplers, offset = qubo_to_ising(qubo, int(meta["n_total_vars"]))
        out["ising"] = {
            "h": {str(k): v for k, v in h.items()},
            "J": [
                {"i": i, "j": j, "value": v} for (i, j), v in sorted(j_couplers.items())
            ],
            "offset": offset,
        }
    write_json(args.out, out)
    print(f"wrote {args.out} ({meta['n_total_vars']} vars, {len(qubo)} terms)")


def cmd_solve(args):
    problem = _problem_dict(args.problem)
    if args.budget is not None:
        problem["max_weight"] = args.budget
    qubo_doc = read_json(args.qubo) if args.qubo else None
    result = solve_problem_dict(
        problem,
        solver=args.solver,
        qubo_doc=qubo_doc,
        penalty=args.penalty,
        normalize=args.normalize,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        seed=args.seed,
        time_limit=args.time_limit,
        label=args.label,
        chain_strength=args.chain_strength,
    )
    write_json(args.out, result)
    print(
        f"wrote {args.out}; solver={result['solver']} selected_ids={result['selected_ids']} "
        f"weight={result['selected_weight']} profit={result['selected_profit']} feasible={result['feasible']}"
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="qdbo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("doctor")
    sp.add_argument("--config")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("collect-workload")
    sp.add_argument("--config")
    sp.add_argument("--out", required=True)
    sp.add_argument("--sql-file")
    sp.add_argument("--default-calls", type=int, default=1)
    sp.add_argument("--limit", type=int)
    sp.add_argument("--min-calls", type=int)
    sp.add_argument(
        "--order-by", choices=["total_exec_time", "mean_exec_time", "calls"]
    )
    sp.set_defaults(func=cmd_collect_workload)

    sp = sub.add_parser("generate-candidates")
    sp.add_argument("--config")
    sp.add_argument("--workload", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--default-schema")
    sp.add_argument("--max-columns", type=int)
    sp.add_argument("--prefix")
    sp.add_argument("--max-candidates", type=int)
    sp.add_argument("--no-order-by", action="store_true")
    sp.add_argument("--prune-existing", action="store_true")
    sp.set_defaults(func=cmd_generate_candidates)

    sp = sub.add_parser("build-qubo")
    sp.add_argument("--problem", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--penalty", type=float)
    sp.add_argument("--budget", type=float, help="override problem.max_weight")
    sp.add_argument("--normalize", type=float)
    sp.add_argument("--include-ising", action="store_true")
    sp.set_defaults(func=cmd_build_qubo)

    sp = sub.add_parser("solve")
    sp.add_argument("--problem", required=True)
    sp.add_argument("--qubo")
    sp.add_argument("--out", required=True)
    sp.add_argument(
        "--solver", required=True, choices=["exact-knapsack", "greedy", "dwave-hybrid"]
    )
    sp.add_argument("--budget", type=float, help="override problem.max_weight")
    sp.add_argument("--penalty", type=float)
    sp.add_argument("--normalize", type=float)
    sp.add_argument("--num-reads", type=int, default=1000)
    sp.add_argument("--num-sweeps", type=int, default=1000)
    sp.add_argument("--seed", type=int)
    sp.add_argument("--time-limit", type=int)
    sp.add_argument("--label", default="qdbo-index-selection")
    sp.add_argument("--chain-strength", type=float)
    sp.set_defaults(func=cmd_solve)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
