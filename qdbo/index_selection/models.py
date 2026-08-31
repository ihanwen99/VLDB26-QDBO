from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def quote_ident(identifier: str) -> str:
    identifier = str(identifier).strip()
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier
    return '"' + identifier.replace('"', '""') + '"'


@dataclass
class WorkloadQuery:
    query_id: str
    sql: str
    calls: int = 1
    total_exec_time_ms: float = 0.0
    mean_exec_time_ms: float = 0.0
    rows: int = 0
    source: str = "manual"
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkloadQuery":
        d = dict(d)
        if "query" in d and "sql" not in d:
            d["sql"] = d.pop("query")
        if "queryid" in d and "query_id" not in d:
            d["query_id"] = str(d.pop("queryid"))
        return WorkloadQuery(**d)


@dataclass
class CandidateIndex:
    candidate_id: int
    name: str
    schema: str
    table: str
    columns: List[str]
    ddl: str
    hypopg_ddl: Optional[str] = None
    method: str = "btree"
    source: str = "manual"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CandidateIndex":
        d = dict(d)
        if "candidate_id" not in d:
            d["candidate_id"] = int(d.get("id", 0))
        if "schema" not in d:
            # Accept table as public.table or table.
            table = d.get("table", "")
            if "." in table:
                schema, tab = table.split(".", 1)
                d["schema"], d["table"] = schema, tab
            else:
                d["schema"] = "public"
        if "hypopg_ddl" not in d or not d.get("hypopg_ddl"):
            d["hypopg_ddl"] = d.get("ddl")
        # Drop aliases not in dataclass.
        d.pop("id", None)
        return CandidateIndex(**d)


@dataclass
class IndexSelectionProblem:
    name: str
    profits: List[int]
    weights: List[int]
    max_weight: int
    candidates: List[Dict[str, Any]]
    workload: List[Dict[str, Any]]
    evaluations: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    pairwise_interactions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IndexSelectionProblem":
        d = dict(d)
        if "max_wgt" in d and "max_weight" not in d:
            d["max_weight"] = d.pop("max_wgt")
        d.setdefault("evaluations", [])
        d.setdefault("metadata", {})
        d.setdefault("pairwise_interactions", d.pop("interactions", {}) if "interactions" in d else {})
        return IndexSelectionProblem(**d)


@dataclass
class SolverResult:
    solver: str
    selected_ids: List[int]
    selected_bitstring: List[int]
    selected_weight: int
    selected_profit: int
    energy: Optional[float] = None
    feasible: bool = True
    raw: Dict[str, Any] = field(default_factory=dict)
    selected_profit_float: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
