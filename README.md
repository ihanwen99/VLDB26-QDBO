# QDBO: A Real-time Quantum-augmented Database System Optimizer

PVLDB, 2026. Paper: https://doi.org/10.14778/3836663.3836712

QDBO is a white-box quantum-augmented database optimizer that first reduces a QUBO to an embeddable instance with one of several variable-to-qubit mapping strategies, and then applies an online iterative relaxation correction that uses sampling feedback to correct the reduced QUBO across iterations. This repository contains QDBO and its two use cases, join ordering on JOB and CEB and index selection on TPC-H.

## Requirements

- Python 3.10
- A C++ compiler
- PostgreSQL with the IMDB database and the `pg_hint_plan` extension
- A D-Wave Leap account with QPU access

## Install

```bash
conda create -n qdbo python=3.10
conda activate qdbo
pip install -r requirements.txt
bash qdbo/embedding/build_cpp_embed.sh
dwave config create
```

Set the database connection in `qdbo/config.py`.

## Run

```bash
python -m qdbo.join_ordering.run
python -m qdbo.index_selection.run
```

The first command optimizes the join order of every JOB and CEB query with QDBO, runs each query on PostgreSQL with that join order and with the default plan, and writes the timings to CSV files. The second command builds the index selection QUBO of the TPC-H workload for storage budgets of 10%, 20%, 40% and 50% of the total index size and solves each one with QDBO.

## Repository layout

```text
VLDB26-QDBO/
├── qdbo/
│   ├── iterative_solver.py  Iterative QPU solver
│   ├── embedding/           Variable-to-qubit mapping
│   ├── join_ordering/
│   │   ├── qubo.py          Join order QUBO
│   │   ├── join_order.py    Join order decoding and cost
│   │   ├── run.py           Join ordering experiment
│   │   ├── pipeline.py      Per-solver runs
│   │   ├── bqm.py           BQM baseline
│   │   ├── cqm.py           CQM baseline
│   │   ├── nl.py            NL baseline
│   │   ├── postgres.py      PostgreSQL access
│   │   ├── hints.py         Leading hints
│   │   └── inputs.py        Workload loading
│   ├── index_selection/
│   │   ├── qubo.py          Index selection QUBO
│   │   └── run.py           Index selection experiment
│   ├── output.py            Output and timing
│   └── config.py            Settings
└── workloads/
    ├── join_ordering/       JOB and CEB queries
    └── index_selection/
        └── problem.json     TPC-H index candidates
```

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
