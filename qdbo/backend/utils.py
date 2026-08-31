import json
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pytz


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


def measure_time_return(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper
