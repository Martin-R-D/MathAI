"""Automatically discover a proof of the default identity with bounded BFS."""

import argparse

import sympy as sp

from .environment import State
from .search import SearchLimits, SearchStatus, breadth_first_search


def main(argv=None):
    defaults = SearchLimits()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=defaults.max_depth)
    parser.add_argument("--max-states", type=int, default=defaults.max_states)
    parser.add_argument("--max-expression-size", type=int, default=defaults.max_expression_size)
    parser.add_argument("--timeout-seconds", type=float, default=defaults.timeout_seconds)
    args = parser.parse_args(argv)
    try:
        limits = SearchLimits(**vars(args))
    except ValueError as error:
        parser.error(str(error))

    x, y = sp.symbols("x y")
    initial = State((x+y)**2 - (x-y)**2, 4*x*y)
    result = breadth_first_search(initial, limits)
    print(f"Start: {initial.expression}")
    print(f"Target: {initial.target}")
    print(f"Search status: {result.status.value}")
    for index, step in enumerate(result.proof, 1):
        print(f"{index}. {step.action.rule} at {step.action.path}: {step.after}")
        print(f"   Verified: {step.verified}")
    print(f"Proof length: {len(result.proof)}")
    print(f"Admitted states: {result.admitted_states}")
    print(f"Expanded states: {result.expanded_states}")
    print(f"Attempted transitions: {result.attempted_transitions}")
    print(f"Duplicate successors: {result.duplicate_successors}")
    print(f"Size-pruned successors: {result.size_pruned_successors}")
    print(f"Maximum admitted depth: {result.max_admitted_depth}")
    print(f"Elapsed seconds: {result.elapsed_seconds:.6f}")
    if result.restricted_by:
        print("Limits restricting exploration: " + ", ".join(
            sorted(limit.value for limit in result.restricted_by)))
    if result.status == SearchStatus.SOLVED:
        print("Target reached: True")
    elif result.restricted_by:
        print("No proof found within these limits; the task may still have a proof.")
    else:
        print("No proof exists in the reachable graph of the current transformation rules.")


if __name__ == "__main__":
    main()
