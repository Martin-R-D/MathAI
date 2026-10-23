"""Deterministic, bounded breadth-first search over verified environment moves."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from time import monotonic

import sympy as sp

from .environment import ProofStep, State, apply_action, is_solved, iter_actions, normalize_rules


class SearchStatus(str, Enum):
    SOLVED = "solved"
    EXHAUSTED = "exhausted"
    INCOMPLETE = "incomplete"
    STATE_LIMIT = "state_limit"
    TIMEOUT = "timeout"


class SearchLimit(str, Enum):
    DEPTH = "depth"
    STATES = "states"
    EXPRESSION_SIZE = "expression_size"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class SearchLimits:
    max_depth: int = 6
    max_states: int = 1000
    max_expression_size: int = 200
    timeout_seconds: float | None = None

    def __post_init__(self):
        for name, minimum in (("max_depth", 0), ("max_states", 1),
                              ("max_expression_size", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        timeout = self.timeout_seconds
        if timeout is not None and (
            type(timeout) not in (int, float)
            or (type(timeout) is float and not isfinite(timeout)) or timeout < 0
        ):
            raise ValueError("timeout_seconds must be a finite nonnegative number or None.")


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    proof: tuple[ProofStep, ...]
    final_state: State | None
    admitted_states: int
    expanded_states: int
    attempted_transitions: int
    duplicate_successors: int
    size_pruned_successors: int
    max_admitted_depth: int | None
    elapsed_seconds: float
    restricted_by: frozenset[SearchLimit]


@dataclass(frozen=True)
class _Node:
    state: State
    depth: int
    parent: sp.Expr | None
    step: ProofStep | None


def expression_size(expression: sp.Expr) -> int:
    """Count tree-node occurrences, including atoms and power exponents.

    Shared subexpressions are counted once for each occurrence, not once per
    Python object. Integer magnitude and target size are not part of this count.
    """
    return sum(1 for _ in sp.preorder_traversal(expression))


def breadth_first_search(initial: State, limits: SearchLimits | None = None, *, rules=None) -> SearchResult:
    """Find a shortest proof in the size-pruned graph, within resource limits.

    See README for exact budget boundaries and counter definitions. Exceptions
    from the environment, including failed verification, deliberately propagate.
    """
    if limits is None:
        limits = SearchLimits()
    if not isinstance(limits, SearchLimits):
        raise TypeError("limits must be a SearchLimits instance.")
    enabled = normalize_rules(rules)
    started = monotonic()
    nodes: dict[sp.Expr, _Node] = {}
    queue = deque()
    restricted = set()
    expanded = attempted = duplicates = size_pruned = 0
    deepest = None

    def finish(status, key=None):
        proof = []
        final = nodes[key].state if key is not None else None
        while key is not None:
            node = nodes[key]
            if node.step is not None:
                proof.append(node.step)
            key = node.parent
        return SearchResult(
            status=status, proof=tuple(reversed(proof)), final_state=final,
            admitted_states=len(nodes), expanded_states=expanded,
            attempted_transitions=attempted, duplicate_successors=duplicates,
            size_pruned_successors=size_pruned, max_admitted_depth=deepest,
            elapsed_seconds=monotonic() - started, restricted_by=frozenset(restricted),
        )

    def timed_out():
        if limits.timeout_seconds is not None and monotonic() - started >= limits.timeout_seconds:
            restricted.add(SearchLimit.TIMEOUT)
            return True
        return False

    if expression_size(initial.expression) > limits.max_expression_size:
        restricted.add(SearchLimit.EXPRESSION_SIZE)
        return finish(SearchStatus.INCOMPLETE)

    # SymPy Expr hashing/equality are structural; no expansion or polynomial key.
    first_key = initial.expression
    nodes[first_key] = _Node(initial, 0, None, None)
    queue.append(first_key)
    deepest = 0
    if is_solved(initial):
        return finish(SearchStatus.SOLVED, first_key)

    while queue:
        if timed_out():
            return finish(SearchStatus.TIMEOUT)
        parent_key = queue.popleft()
        parent = nodes[parent_key]
        actions = iter(iter_actions(parent.state) if rules is None else iter_actions(parent.state, enabled))
        expanded += 1
        while True:
            if timed_out():
                return finish(SearchStatus.TIMEOUT)
            action = next(actions, None)
            if timed_out():
                return finish(SearchStatus.TIMEOUT)
            if action is None:
                break
            attempted += 1
            successor, step = apply_action(parent.state, action)
            if timed_out():
                return finish(SearchStatus.TIMEOUT)
            key = successor.expression
            if key in nodes:
                duplicates += 1
                continue
            oversized = expression_size(key) > limits.max_expression_size
            if timed_out():
                return finish(SearchStatus.TIMEOUT)
            if oversized:
                size_pruned += 1
                restricted.add(SearchLimit.EXPRESSION_SIZE)
                continue
            if parent.depth == limits.max_depth:
                restricted.add(SearchLimit.DEPTH)
                continue
            if len(nodes) == limits.max_states:
                restricted.add(SearchLimit.STATES)
                return finish(SearchStatus.STATE_LIMIT)
            depth = parent.depth + 1
            # Record visited on admission, before adding to the FIFO frontier.
            nodes[key] = _Node(successor, depth, parent_key, step)
            queue.append(key)
            deepest = max(deepest, depth)
            if is_solved(successor):
                return finish(SearchStatus.SOLVED, key)

    return finish(SearchStatus.INCOMPLETE if restricted else SearchStatus.EXHAUSTED)
