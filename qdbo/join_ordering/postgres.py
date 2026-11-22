import psycopg2
from psycopg2.extras import RealDictCursor

from qdbo import config
from qdbo.join_ordering.hints import (
    build_postgres_hint,
    build_table_alias_mapping,
    prepend_hint_to_sql,
)


def open_postgres_connection(benchmark="JOB"):
    benchmark_name = (benchmark or "JOB").upper()
    dsn = config.JOB_DSN
    if not dsn:
        print(
            "Database configuration error: set JOB_DSN in qdbo/config.py "
            "to a PostgreSQL connection string."
        )
        return None

    try:
        return psycopg2.connect(dsn)
    except psycopg2.Error as e:
        print(f"Database connection error for {benchmark_name}: {e}")
        return None


def extract_execution_time(explain_analyze_lines):
    execution_time = None
    for line in reversed(explain_analyze_lines):
        if "Execution Time" in line:
            try:
                execution_time_str = line.strip()
                execution_time = float(
                    execution_time_str.split("Execution Time:")[1].strip().split(" ")[0]
                )
                break
            except (IndexError, ValueError) as e:
                print(f"Error extracting execution time: {e}")
                execution_time = None
                break
    return execution_time


def extract_planning_time(explain_analyze_lines):
    for line in explain_analyze_lines:
        if "Planning Time" in line:
            return float(line.split(":")[1].strip().replace("ms", ""))
    return None


def explain_analyze(sql_query, time_out, benchmark=None):
    conn = open_postgres_connection(benchmark)
    if conn is None:
        print("Failed to establish a database connection.")
        return None, None, None

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET statement_timeout = {}".format(time_out))

            explain_query = f"EXPLAIN ANALYZE {sql_query}"
            cursor.execute(explain_query)
            explain_result = cursor.fetchall()

            explain_analyze_lines = [row["QUERY PLAN"] for row in explain_result]
            explain_analyze = "\n".join(explain_analyze_lines)

            execution_time = extract_execution_time(explain_analyze_lines)
            planning_time = extract_planning_time(explain_analyze_lines)

        conn.commit()
        return explain_analyze, execution_time, planning_time
    except psycopg2.Error:
        return None, None, None
    finally:
        conn.close()


def explain_analyze_with_hint(join_order: list, query: str, time_out: int):
    mapping = build_table_alias_mapping(query)
    hint = build_postgres_hint(mapping, join_order)
    hinted_query = prepend_hint_to_sql(query, hint)
    plan, execution_time, planning_time = explain_analyze(hinted_query, time_out)
    return planning_time, execution_time, plan, hint
