"""An exact polynomial environment for explicit algebra proofs."""

from .environment import (
    Action, InvalidAction, ProofStep, State, UnsupportedExpression,
    VerificationError, apply_action, equivalent, enumerate_actions, is_solved, iter_actions,
)
from .search import (
    SearchLimit, SearchLimits, SearchResult, SearchStatus, breadth_first_search, expression_size,
)

__all__ = [
    "Action", "InvalidAction", "ProofStep", "State", "UnsupportedExpression",
    "VerificationError", "apply_action", "equivalent", "enumerate_actions", "is_solved",
    "iter_actions", "SearchLimit", "SearchLimits", "SearchResult", "SearchStatus",
    "breadth_first_search", "expression_size",
]
