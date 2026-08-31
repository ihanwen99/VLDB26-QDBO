# QDBO: A Real-time Quantum-augmented Database System Optimizer

[Project page](https://ihanwen99.github.io/VLDB26-QDBO/) ·
[Paper](https://doi.org/10.14778/3836663.3836712)

QDBO is a sampling-centric framework for running database-optimization QUBOs on
quantum annealers. This repository contains the shared QDBO implementation and
the two use cases evaluated in the paper: join ordering (Q²O-QDBO) and index
selection (IS-QDBO).

## Install

```bash
git clone https://github.com/ihanwen99/VLDB26-QDBO.git
cd VLDB26-QDBO

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

# The platform-specific shared library is built locally and is not committed.
bash qdbo/embedding/build_cpp_embed.sh

cp config.example.env config.env
source config.env
```

Verify the local installation without a database or QPU call:

```bash
python -c 'import qdbo; from qdbo.core import SEBREMforBQM; print(qdbo.__version__)'
qdbo doctor
python -m qdbo.join_ordering.actual_benchmark_pipeline --help
```

## Repository layout

```text
VLDB26-QDBO/
├── qdbo/
│   ├── core/                 # shared iterative sampling/correction loop
│   ├── embedding/            # Python/C++ variable-qubit mapping
│   ├── backend/              # shared database and result helpers
│   ├── join_ordering/        # Q²O-QDBO implementation
│   └── index_selection/      # IS-QDBO implementation and CLI
├── workloads/
│   ├── join_ordering/        # JOB, CEB, and synthetic inputs
│   └── index_selection/      # TPC-H SQL, schema, and fixed paper input
├── scripts/index_selection/  # database setup, solve, replay, aggregation
├── README.md
├── LICENSE
├── config.example.env
├── pyproject.toml
└── requirements.txt
```

`workloads/index_selection/fixtures/` contains immutable inputs, not generated
results. Every run writes under the ignored `results/` directory.

## D-Wave

Create a local Leap configuration before running a quantum solver:

```bash
dwave config create
dwave ping
```

Never commit a Leap token or `config.env`.

## Join ordering (Q²O-QDBO)

The workload parser and CLI help work without PostgreSQL. Solver runs require the
corresponding D-Wave service; `--execute` additionally requires PostgreSQL with
the IMDB data and `pg_hint_plan`.

```bash
# Inspect the complete CLI.
python -m qdbo.join_ordering.actual_benchmark_pipeline --help

# Run the iterative QDBO path on a paper workload.
python -m qdbo.join_ordering.actual_benchmark_pipeline \
  --benchmark JOB --solvers iter --embedding semanticV_greedy --iterations 1
```

Use `--benchmark CEB` for CEB. The available solver values are `iter`, `bqm`,
`cqm`, and `nl`. The synthetic entry point is:

```bash
python -m qdbo.join_ordering.synthetic_benchmark_pipeline --help
```

## Index selection (IS-QDBO)

The tracked fixture can rebuild the QUBO and run the classical baselines without
PostgreSQL or a QPU:

```bash
FIXTURE=workloads/index_selection/fixtures/sf1_nopk_19q_e2_runtime
OUT=results/index_selection/quickstart
mkdir -p "$OUT"

BUDGET=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["budgets"]["10pct"]["units"])' \
  "$FIXTURE/budgets.json")

qdbo build-qubo --problem "$FIXTURE/problem.json" --budget "$BUDGET" \
  --out "$OUT/qubo.json" --include-ising
qdbo solve --solver greedy --problem "$FIXTURE/problem.json" \
  --budget "$BUDGET" --qubo "$OUT/qubo.json" --out "$OUT/greedy.json"
qdbo solve --solver exact-knapsack --problem "$FIXTURE/problem.json" \
  --budget "$BUDGET" --qubo "$OUT/qubo.json" --out "$OUT/exact.json"
```

The full paper workflow adds TPC-H, PostgreSQL, D-Wave Leap, and replay:

```bash
bash workloads/index_selection/configs/tpch_load.sh
bash scripts/index_selection/setup_sf1_nopk.sh
python scripts/index_selection/compute_real_profits.py --help
bash scripts/index_selection/sweep_solve.sh
bash scripts/index_selection/sweep_replay.sh
python scripts/index_selection/compute_table_csv.py "$IS_QDBO_EXP_DIR"
```

The scripts have one role each: prepare the database, measure candidate profits,
solve the paper configurations, replay selected indexes, and aggregate the output.
External service paths and DSNs belong in `config.env`, not in source code.

## Results

Generated experiment results are not committed. The repository ships code,
workloads, a fixed small input, and the scripts needed to produce new results.
Raw logs, QPU samples, database files, caches, and historical internal runs stay
outside Git. A reference result should be published only after it is mapped to a
canonical run and checked against the accepted paper.

## Workload sources

- [Join Order Benchmark](https://github.com/gregrahn/join-order-benchmark)
- [Cardinality Estimation Benchmark](https://github.com/learnedsystems/CEB)
- [TPC-H](https://www.tpc.org/tpch/) (`dbgen` is obtained separately)

## Citation

```bibtex
@article{liu2026qdbo,
  author  = {Hanwen Liu and Abhishek Kumar and Federico Spedalieri and Ibrahim Sabek},
  title   = {{QDBO}: A Real-time Quantum-augmented Database System Optimizer},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {19},
  number  = {11},
  pages   = {3606--3620},
  year    = {2026},
  doi     = {10.14778/3836663.3836712}
}
```

## License

GPL-3.0-or-later. See `LICENSE`.
