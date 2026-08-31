"""Helpers for join-ordering workloads and PostgreSQL execution.

Functions here read SQL workload files, generate PostgreSQL hint syntax,
compute left-deep-tree DB cost, and fetch real-execution results — all
PostgreSQL- and join-ordering-specific.
"""
import json
import os

# The PostgreSQL execution harness is imported lazily inside the fetch_*/execute_*
# helpers below, so the cost-model-only pipelines keep working without psycopg2.


def process_input(sql_folder_path):
    cardinalities_file_path = "{}/cardinalities.json".format(sql_folder_path)
    selectivities_file_path = "{}/selectivities.json".format(sql_folder_path)
    with open(cardinalities_file_path, "r") as file:
        cardinalities_content = json.load(file)

    with open(selectivities_file_path, "r") as file:
        selectivities_content = json.load(file)
    return cardinalities_content, selectivities_content


def parse_selectivities(sel):
    """Extract predicates and corresponding selectivities"""
    pred = []
    pred_sel = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            if sel[i][j] != 1:
                pred.append((i, j))
                pred_sel.append(sel[i][j])
    return pred, pred_sel


def generate_table_mapping(sql_query):
    """
    Robust table mapping using sqlparse for alias extraction

    Args:
        sql_query (str): The SQL query to parse

    Returns:
        dict: Mapping of indices to table aliases, or empty dict if parsing fails
    """
    try:
        import sqlparse
        from sqlparse.sql import IdentifierList, Identifier
        from sqlparse.tokens import Keyword
    except ImportError:
        print("Warning: sqlparse not available")
        return {}

    try:
        # Parse the SQL query
        parsed = sqlparse.parse(sql_query)[0]
        from_seen = False
        tables = []

        # Iterate over the parsed tokens
        for token in parsed.tokens:
            # Check if the FROM keyword is encountered
            if from_seen:
                # Handle multiple tables
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        alias = identifier.get_alias()
                        table_name = identifier.get_real_name()
                        # Prefer alias, fallback to table name
                        tables.append(alias if alias else table_name)

                # Handle a single table
                elif isinstance(token, Identifier):
                    alias = token.get_alias()
                    table_name = token.get_real_name()
                    # Prefer alias, fallback to table name
                    tables.append(alias if alias else table_name)

                # Stop when encountering WHERE or other clause keywords
                elif token.ttype is Keyword and token.value.upper() in ('WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT'):
                    break

            elif token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True

        # Generate the mapping
        mapping = {i: table for i, table in enumerate(tables) if table}
        return mapping

    except Exception as e:
        print(f"sqlparse failed: {e}")
        return {}


def generate_postgres_hint(mapping, join_order):
    """
    Generates a PostgreSQL hint based on the given mapping and join order.

    Args:
        mapping (dict): A dictionary mapping indices to table aliases.
        join_order (list): A list of integers representing the join order.

    Returns:
        str: A PostgreSQL hint string.
    """
    # Retrieve table aliases based on the join order
    table_aliases = [mapping[int(i)] for i in join_order]
    # Construct the hint string
    hint = "/*+\n    Leading (" + " ".join(table_aliases) + ")\n*/"
    return hint


def insert_hint_into_sql(sql_query, hint):
    return hint + "\n" + sql_query


def compute_db_cost(join_order: list, cardinalities, selectivities):
    """
    Compute the db perspective cost based on the join order - using the sum of intermediate cardinalities
    """
    total_cost = 0

    num_relations = len(join_order)
    curr_rid = join_order[0]
    intermediate_cardinality = cardinalities[curr_rid]
    involved_relations = [curr_rid]

    for i in range(1, num_relations):
        right_rid = join_order[i]  # Right side -  relation id (incoming join)
        current_selectivity = 1
        for left_rid in involved_relations:
            if abs(selectivities[left_rid][right_rid] - 1) == 0.00000001:
                return "JOIN UNPROPER RELATIONS..."
            current_selectivity *= selectivities[left_rid][right_rid]
        intermediate_cardinality = intermediate_cardinality * cardinalities[right_rid] * current_selectivity
        total_cost += intermediate_cardinality

    return total_cost


def fetch_query(sql_folder_path: str):
    """
    Read the str from the sql_folder_path
    """
    sql_files = [f for f in os.listdir(sql_folder_path) if f.endswith('.sql')]
    sql_file_path = os.path.join(sql_folder_path, sql_files[0])
    with open(sql_file_path, 'r', encoding='utf-8') as file:
        query = file.read()

    return query


def fetch_latest_result_timeout(join_order: list, query: str, time_out: int):
    from qdbo.join_ordering.backend.normal_execution import execute_quantum_query

    mapping = generate_table_mapping(query)
    hint = generate_postgres_hint(mapping, join_order)
    quantum_QUERY = insert_hint_into_sql(query, hint)
    quantum_explain_analyze, quantum_execution_time, quantum_planning_time = execute_quantum_query(quantum_QUERY, time_out)
    return quantum_planning_time, quantum_execution_time, quantum_explain_analyze, hint

def fetch_real_execution_result_with_hint(join_order: list, query: str):
    from qdbo.join_ordering.backend.normal_execution import execute_query

    mapping = generate_table_mapping(query)
    hint = generate_postgres_hint(mapping, join_order)
    quantum_QUERY = insert_hint_into_sql(query, hint)
    quantum_result, quantum_explain_analyze, quantum_execution_time, quantum_planning_time = execute_query(quantum_QUERY)
    return quantum_planning_time, quantum_execution_time


def execute_quantum_with_hint_and_timeout(join_order: list, query: str, time_out: int):
    from qdbo.join_ordering.backend.normal_execution import execute_quantum_query

    mapping = generate_table_mapping(query)
    hint = generate_postgres_hint(mapping, join_order)
    quantum_QUERY = insert_hint_into_sql(query, hint)
    quantum_explain_analyze, quantum_execution_time, quantum_planning_time = execute_quantum_query(quantum_QUERY, time_out)
    return quantum_planning_time, quantum_execution_time


def get_all_folders_in_target_directory_and_sorted(directory):
    """Sort in the real-number manners"""
    all_items = os.listdir(directory)
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    sorted_folders = sorted(folders,
                            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf'))
    return [os.path.join(directory, folder) for folder in sorted_folders]
