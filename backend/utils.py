import glob
import json
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pytz

from backend.normal_execution import execute_query, execute_quantum_query


def _json_default(o):
    """Make numpy / unknown types JSON-serializable."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _stringify_mapping(mapping: Dict[Any, Any]) -> Dict[str, int]:
    """
    JSON keys must be strings; safest is stringify logical variable labels.
    Values are hardware qubit ids -> int.
    """
    out: Dict[str, int] = {}
    for k, v in mapping.items():
        out[str(k)] = int(v)
    return out


def save_embedding_json(
        *,
        base_dir: str,
        custom_embedding: str,
        function_name: str,
        query_id: str,
        run_id: str,
        orig_bqm,
        target_bqm,
        mapping: Dict[Any, Any],
        stats: Dict[str, Any],
        extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save mapping/stats + (orig_bqm, target_bqm) into a JSON file:
      {base_dir}/{custom_embedding}/{query_id}/{run_id}.json
    Return the saved path.
    """
    la = pytz.timezone("America/Los_Angeles")
    created_at = datetime.now(la).isoformat()

    folder = os.path.join(base_dir, custom_embedding, query_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{run_id}.json")

    payload: Dict[str, Any] = {
        "meta": {
            "created_at": created_at,
            "function": function_name,
            "custom_embedding": custom_embedding,
            "query_id": query_id,
            "run_id": run_id,
        },
        "mapping": _stringify_mapping(mapping),
        "stats": stats,
        "orig_bqm": orig_bqm.to_serializable(),
        "target_bqm": target_bqm.to_serializable(),
    }

    if extra_meta:
        payload["meta"].update(extra_meta)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

    return path


def make_query_id(full_problem_path: str) -> str:
    # /.../benchmarks/jobq1/q1  -> jobq1_q1
    parts = os.path.normpath(full_problem_path).split(os.sep)
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


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


def process_input_demo(sql_folder_path):
    demo_file_path = "{}/demo.json".format(sql_folder_path)
    with open(demo_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    join_order = data["join_order"]
    hint = data["hint"]
    quantum_QUERY = data["quantum_QUERY"]
    return join_order, hint, quantum_QUERY


def fetchSQLContent(sql_folder_path):
    sql_files = glob.glob(os.path.join(sql_folder_path, '*.sql'))

    if not sql_files:
        print(f"No .sql file found in {sql_folder_path}")
        return

    sql_file_path = sql_files[0]

    try:
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()
        return sql_content
    except Exception as e:
        print(f"Error reading SQL file {sql_file_path}: {e}")

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

# def generate_table_mapping(sql_query):
#     """
#     Generates a mapping of tables used in the SQL query.
#
#     Args:
#         sql_query (str): The SQL query to parse.
#
#     Returns:
#         dict: A dictionary mapping indices to table aliases.
#     """
#     # Parse the SQL query
#     parsed = sqlparse.parse(sql_query)[0]
#     from_seen = False
#     tables = []
#
#     # Iterate over the parsed tokens
#     for token in parsed.tokens:
#         # Check if the FROM keyword is encountered
#         if from_seen:
#             # Handle multiple tables
#             if isinstance(token, IdentifierList):
#                 for identifier in token.get_identifiers():
#                     alias = identifier.get_alias()
#                     if alias:
#                         tables.append(alias)
#             # Handle a single table
#             elif isinstance(token, Identifier):
#                 alias = token.get_alias()
#                 if alias:
#                     tables.append(alias)
#             # Stop when encountering WHERE or other clause keywords
#             elif token.ttype is Keyword and token.value.upper() in ('WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT'):
#                 break
#         elif token.ttype is Keyword and token.value.upper() == 'FROM':
#             from_seen = True
#
#     # Generate the mapping
#     mapping = {i: table for i, table in enumerate(tables)}
#     print(f"inside: mapping: {mapping}")
#     return mapping


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
    mapping = generate_table_mapping(query)
    # print(mapping)

    hint = generate_postgres_hint(mapping, join_order)
    # print(hint)

    quantum_QUERY = insert_hint_into_sql(query, hint)
    # print(quantum_QUERY)

    # return 0,0
    quantum_explain_analyze, quantum_execution_time, quantum_planning_time = execute_quantum_query(quantum_QUERY, time_out)
    # print(quantum_explain_analyze)

    # print(quantum_execution_time)
    return quantum_planning_time, quantum_execution_time, quantum_explain_analyze, hint

def fetch_real_execution_result_with_hint(join_order: list, query: str):
    mapping = generate_table_mapping(query)
    # print(f"mapping: {mapping}")

    hint = generate_postgres_hint(mapping, join_order)
    # print(hint)

    quantum_QUERY = insert_hint_into_sql(query, hint)
    # print(quantum_QUERY)

    # return 0,0

    quantum_result, quantum_explain_analyze, quantum_execution_time, quantum_planning_time = execute_query(quantum_QUERY)
    # print(quantum_explain_analyze)

    # print(quantum_execution_time)
    return quantum_planning_time, quantum_execution_time


def execute_quantum_with_hint_and_timeout(join_order: list, query: str, time_out: int):
    mapping = generate_table_mapping(query)
    # print(mapping)

    hint = generate_postgres_hint(mapping, join_order)
    # print(hint)

    quantum_QUERY = insert_hint_into_sql(query, hint)
    # print(quantum_QUERY)

    # return 0,0

    quantum_explain_analyze, quantum_execution_time, quantum_planning_time = execute_quantum_query(quantum_QUERY, time_out)
    # print(quantum_explain_analyze)

    # print(quantum_execution_time)
    return quantum_planning_time, quantum_execution_time


def get_all_folders_in_target_directory_and_sorted(directory):
    """Sort in the real-number manners"""
    all_items = os.listdir(directory)
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    sorted_folders = sorted(folders,
                            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf'))
    return [os.path.join(directory, folder) for folder in sorted_folders]


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        return result

    return wrapper


def measure_time_return(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper
