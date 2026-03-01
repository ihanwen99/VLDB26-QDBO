import os

import psycopg2

# Database mapping based on benchmark
BENCHMARK_DB_MAPPING = {
    "TPC-H": "tpch",
    "TPC-DS": "tpcds",
    "JOB": "imdbload",
    # Add more mappings as needed
}

DEFAULT_DATABASE = "imdbload"

def get_database_name(benchmark=None):
    """
    Get database name based on benchmark
    """
    if benchmark and benchmark in BENCHMARK_DB_MAPPING:
        return BENCHMARK_DB_MAPPING[benchmark]
    return DEFAULT_DATABASE

def get_connection(benchmark="JOB"):
    """
    Create and return one database connection based on benchmark
    
    Args:
        benchmark (str): The benchmark name (e.g., 'TPC-H', 'JOB')
    
    Returns:
        psycopg2.connection: Database connection or None if failed
    """
    database_name = get_database_name(benchmark)
    
    try:
        conn = psycopg2.connect(
            host=os.environ.get("QDBO_DB_HOST", "localhost"),
            port=os.environ.get("QDBO_DB_PORT", "5433"),
            database=database_name,
            user=os.environ.get("QDBO_DB_USER", ""),
            password=os.environ.get("QDBO_DB_PASSWORD", ""),
        )
        # print(f"Connected to database: {database_name} (benchmark: {benchmark})")
        return conn
    except psycopg2.Error as e:
        print(f"Database Connection Error for {database_name}: {e}")
        return None
