# VLDB26-QDBO

QDBO: A Real-time Quantum-augmented Database System Optimizer

## Repository Overview

- `actual_benchmark_pipeline.py`: main entry for running CEB/JOB workloads on real queries.
- `synthetic_benchmark_pipeline.py`: main entry for synthetic workloads.
- `algorithm/`: iterative solver and embedding implementation.
- `backend/`: solver backends (NL/CQM/BQM), problem generation, and shared utilities.
- `benchmark/`: benchmark inputs (kept out of edits in this repo).

## Main Entry (Pipeline)

The pipeline scripts are the primary entry points. Typical runs:

```bash
python3 actual_benchmark_pipeline.py --benchmark CEB --solvers bqm
python3 actual_benchmark_pipeline.py --benchmark JOB --solvers bqm
```

Use `--solvers` to select any combination of `nl`, `cqm`, `bqm`, and `iter`.
