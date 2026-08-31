import os

import psycopg2

BENCHMARK_DSN_ENV = {
    "CEB": "QDBO_JOB_DSN",
    "JOB": "QDBO_JOB_DSN",
}


def get_connection(benchmark="JOB"):
    """Create a PostgreSQL connection from an explicit DSN.

    Args:
        benchmark: Logical benchmark name used to select a benchmark-specific
            DSN variable. ``QDBO_DB_DSN`` is the fallback.

    Returns:
        A psycopg2 connection, or ``None`` when configuration/connection fails.
    """
    benchmark_name = (benchmark or "JOB").upper()
    benchmark_env = BENCHMARK_DSN_ENV.get(benchmark_name)
    dsn = (os.environ.get(benchmark_env) if benchmark_env else None) or os.environ.get(
        "QDBO_DB_DSN"
    )
    if not dsn:
        expected = benchmark_env or "QDBO_DB_DSN"
        print(
            f"Database configuration error: set {expected} or QDBO_DB_DSN "
            "to a PostgreSQL connection string."
        )
        return None

    try:
        return psycopg2.connect(dsn)
    except psycopg2.Error as e:
        print(f"Database connection error for {benchmark_name}: {e}")
        return None
