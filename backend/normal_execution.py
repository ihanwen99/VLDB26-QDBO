import psycopg2
from psycopg2.extras import RealDictCursor

from backend.database_connection import get_connection


def extract_execution_time(explain_analyze_lines):
    """
    Extracts execution time from the output of EXPLAIN ANALYZE.

    Args:
        explain_analyze_lines (list): List of lines from the EXPLAIN ANALYZE output.

    Returns:
        execution_time (float): Execution time in milliseconds.
    """
    execution_time = None
    for line in reversed(explain_analyze_lines):
        if 'Execution Time' in line:
            try:
                execution_time_str = line.strip()
                execution_time = float(execution_time_str.split('Execution Time:')[1].strip().split(' ')[0])
                break
            except (IndexError, ValueError) as e:
                print(f"Error extracting execution time: {e}")
                execution_time = None
                break
    return execution_time


def extract_planning_time(explain_analyze_lines):
    """
    Extracts planning time from the output of EXPLAIN ANALYZE.

    Args:
        explain_analyze_lines (list): List of lines from the EXPLAIN ANALYZE output.

    Returns:
        planning_time (float): Planning time in milliseconds.
    """
    for line in explain_analyze_lines:
        if "Planning Time" in line:
            return float(line.split(":")[1].strip().replace("ms", ""))
    return None


def execute_query(sql_query, benchmark=None):
    """
    Executes the given SQL query and returns the results, EXPLAIN ANALYZE output, and execution time.

    Args:
        sql_query (str): The SQL query to be executed.
        benchmark (str): The benchmark name to determine database connection.

    Returns:
        result (list of dict): Query results.
        explain_analyze (str): Output of EXPLAIN ANALYZE.
        execution_time (float): Execution time in milliseconds.
        planning_time (float): Planning time in milliseconds.
    """
    conn = get_connection(benchmark)
    if conn is None:
        print("Failed to establish a database connection.")
        return None, None, None, None

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Execute the query and fetch results
            cursor.execute(sql_query)
            result = cursor.fetchall()

            # Execute EXPLAIN ANALYZE
            explain_query = f"EXPLAIN ANALYZE {sql_query}"

            cursor.execute(explain_query)
            explain_result = cursor.fetchall()

            # Get EXPLAIN ANALYZE output
            explain_analyze_lines = [row['QUERY PLAN'] for row in explain_result]
            explain_analyze = "\n".join(explain_analyze_lines)

            # Extract execution time and planning time
            execution_time = extract_execution_time(explain_analyze_lines)
            planning_time = extract_planning_time(explain_analyze_lines)

        conn.commit()
        return result, explain_analyze, execution_time, planning_time
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")
        return None, None, None, None
    finally:
        conn.close()


def execute_quantum_query(sql_query, time_out, benchmark=None):
    """
    Executes the given EXPLAIN ANALYZE SQL query and returns the output, and execution time.

    Args:
        sql_query (str): The SQL query to be executed.
        time_out: Timeout for the query execution.
        benchmark (str): The benchmark name to determine database connection.

    Returns:
        explain_analyze (str): Output of EXPLAIN ANALYZE.
        execution_time (float): Execution time in milliseconds.
        planning_time (float): Planning time in milliseconds.
    """
    conn = get_connection(benchmark)
    if conn is None:
        print("Failed to establish a database connection.")
        return None, None, None

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Execute the query and fetch results
            cursor.execute("SET statement_timeout = {}".format(time_out))

            # Execute EXPLAIN ANALYZE
            explain_query = f"EXPLAIN ANALYZE {sql_query}"
            cursor.execute(explain_query)
            explain_result = cursor.fetchall()

            # Get EXPLAIN ANALYZE output
            explain_analyze_lines = [row['QUERY PLAN'] for row in explain_result]
            explain_analyze = "\n".join(explain_analyze_lines)

            # Extract execution time and planning time
            execution_time = extract_execution_time(explain_analyze_lines)
            planning_time = extract_planning_time(explain_analyze_lines)

        conn.commit()
        return explain_analyze, execution_time, planning_time
    except psycopg2.Error as e:
        # print(f"Error executing query: {e}")
        return None, None, None
    finally:
        conn.close()
