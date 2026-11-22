from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "workloads" / "join_ordering"
RESULTS_ROOT = REPO_ROOT / "results"
EMBEDDING_DIR = RESULTS_ROOT / "embeddings"

JOB_DSN = None

INDEX_SELECTION_PROBLEM = REPO_ROOT / "workloads" / "index_selection" / "problem.json"
INDEX_SELECTION_RESULTS_DIR = RESULTS_ROOT / "index_selection"
