"""Real algebra integration plus small directed graphs isolating BFS mechanics."""

from dataclasses import replace

import pytest
import sympy as sp

import mathai.search as search
from mathai import (
    Action, ProofStep, SearchLimit, SearchLimits, SearchStatus, State,
    VerificationError, apply_action, breadth_first_search, enumerate_actions,
    equivalent, expression_size, is_solved, iter_actions,
)
from mathai.search_demo import main

x, y, z = sp.symbols("x y z")


@pytest.fixture
def graph(monkeypatch):
    """Replace only enumeration/transitions; labels need not be algebra identities.

    All proof-verification assertions use the real environment in other tests.
    """
    def install(edges, start="a", goal="goal"):
        expanded = []

        def actions(state):
            label = str(state.expression)
            expanded.append(label)
            for index, destination in enumerate(edges.get(label, ())):
                yield Action(destination, (index,))

        def transition(state, action):
            following = State(sp.Symbol(action.rule), state.target)
            return following, ProofStep(action, state.expression, following.expression)

        monkeypatch.setattr(search, "iter_actions", actions)
        monkeypatch.setattr(search, "apply_action", transition)
        return State(sp.Symbol(start), sp.Symbol(goal)), expanded

    return install


def test_default_identity_search_connects_and_replays_verified_steps():
    initial = State((x+y)**2 - (x-y)**2, 4*x*y)
    original = State(initial.expression, initial.target)
    assert equivalent(initial.expression, initial.target)
    assert not is_solved(initial)
    result = breadth_first_search(initial)
    assert result.status == SearchStatus.SOLVED
    assert result.proof
    current = initial
    for step in result.proof:
        assert step.before == current.expression
        assert step.verified
        assert step.action in enumerate_actions(current)
        current, replay = apply_action(current, step.action)
        assert replay == step
    assert current == result.final_state
    assert is_solved(current)
    assert initial == original
    assert result.max_admitted_depth == len(result.proof)


def test_initial_goal_has_empty_proof_even_with_zero_depth_and_timeout():
    initial = State(4*x*y, 4*x*y)
    result = breadth_first_search(initial, SearchLimits(max_depth=0, max_states=1,
                                                     timeout_seconds=0))
    assert result.status == SearchStatus.SOLVED
    assert result.proof == ()
    assert result.final_state == initial
    assert result.admitted_states == 1
    assert result.expanded_states == result.attempted_transitions == 0
    assert result.max_admitted_depth == 0
    assert not result.restricted_by


def test_bfs_takes_shorter_second_branch(graph):
    initial, order = graph({"a": ("b", "c"), "b": ("d",),
                            "d": ("e",), "e": ("goal",), "c": ("goal",)})
    result = breadth_first_search(initial)
    # The first branch reaches the goal in four moves; the second in two.
    assert result.status == SearchStatus.SOLVED
    assert [step.action.rule for step in result.proof] == ["c", "goal"]
    assert [step.action.path for step in result.proof] == [(1,), (0,)]
    assert order == ["a", "b", "c"]
    assert result.admitted_states == 5
    assert result.expanded_states == 3
    assert result.attempted_transitions == 4


def test_diamond_duplicates_and_cycles_are_visited_only_once(graph):
    initial, order = graph({"a": ("b", "c"), "b": ("a", "d"),
                            "c": ("d",), "d": ("b",)})
    result = breadth_first_search(initial)
    assert result.status == SearchStatus.EXHAUSTED
    assert order == ["a", "b", "c", "d"]
    assert result.admitted_states == result.expanded_states == 4
    assert result.attempted_transitions == 6
    assert result.duplicate_successors == 3
    assert result.max_admitted_depth == 2
    assert result.final_state is None and result.proof == ()


def test_real_reversible_rules_terminate_without_polynomial_visited_keys():
    # Two mathematically equal but structurally distinct states form a cycle.
    result = breadth_first_search(State(x*y+x*z, 0), SearchLimits(max_states=2))
    assert result.status == SearchStatus.EXHAUSTED
    assert result.admitted_states == result.expanded_states == 2
    assert result.duplicate_successors == 1
    assert not result.restricted_by


@pytest.mark.parametrize("depth,status,admitted", [
    (0, SearchStatus.INCOMPLETE, 1),
    (1, SearchStatus.INCOMPLETE, 2),
    (2, SearchStatus.SOLVED, 3),
])
def test_depth_is_inclusive(graph, depth, status, admitted):
    initial, _ = graph({"a": ("b",), "b": ("goal",)})
    result = breadth_first_search(initial, SearchLimits(max_depth=depth))
    assert result.status == status
    assert result.admitted_states == admitted
    assert result.max_admitted_depth == depth
    assert result.restricted_by == (frozenset() if status == SearchStatus.SOLVED
                                    else frozenset({SearchLimit.DEPTH}))


@pytest.mark.parametrize("budget,status", [
    (1, SearchStatus.STATE_LIMIT), (2, SearchStatus.STATE_LIMIT), (3, SearchStatus.SOLVED),
])
def test_state_budget_includes_initial_and_goal(graph, budget, status):
    initial, _ = graph({"a": ("b",), "b": ("goal",)})
    result = breadth_first_search(initial, SearchLimits(max_states=budget))
    assert result.status == status
    assert result.admitted_states == budget
    assert (SearchLimit.STATES in result.restricted_by) == (status == SearchStatus.STATE_LIMIT)
    if status != SearchStatus.SOLVED:
        assert result.final_state is None and not result.proof


def test_full_budget_does_not_prevent_proving_exhaustion(graph):
    initial, _ = graph({"a": ("b",), "b": ("a",)})
    result = breadth_first_search(initial, SearchLimits(max_states=2, max_depth=1))
    assert result.status == SearchStatus.EXHAUSTED
    assert not result.restricted_by


def test_equivalent_unreachable_target_is_genuine_exhaustion_at_depth_zero():
    initial = State(x**2-y**2, (x-y)*(x+y))
    # Preserve this exhaustion regression for the original rule graph. The new
    # explicit difference-of-squares rule makes this task reachable otherwise.
    original_rules = ("expand_mul", "expand_power", "factor_common", "collect")
    assert equivalent(initial.expression, initial.target)
    assert enumerate_actions(initial, rules=original_rules) == ()
    result = breadth_first_search(initial, SearchLimits(max_depth=0, max_states=1), rules=original_rules)
    assert result.status == SearchStatus.EXHAUSTED
    assert result.admitted_states == result.expanded_states == 1
    assert result.attempted_transitions == 0
    assert not result.restricted_by


def test_tree_size_counts_occurrences_atoms_and_exponents():
    assert expression_size(x) == 1
    assert expression_size((x+y)**2) == 5  # Pow, Add, x, y, 2
    assert expression_size((x+y)*(x+z)) == 7  # x occurs twice
    assert expression_size(sp.Integer(10**100)) == 1


def test_initial_expression_size_boundary_and_precedence_over_goal():
    initial = State(x+y, x+y)
    rejected = breadth_first_search(initial, SearchLimits(max_expression_size=2))
    assert rejected.status == SearchStatus.INCOMPLETE
    assert rejected.restricted_by == {SearchLimit.EXPRESSION_SIZE}
    assert rejected.admitted_states == rejected.expanded_states == 0
    assert rejected.attempted_transitions == rejected.size_pruned_successors == 0
    assert rejected.max_admitted_depth is None
    assert rejected.final_state is None
    assert breadth_first_search(initial, SearchLimits(max_expression_size=3)).status == SearchStatus.SOLVED


def test_successor_size_boundary_including_goal():
    initial = State((x+y)**2, x**2+2*x*y+y**2)
    target_size = expression_size(initial.target)
    blocked = breadth_first_search(initial, SearchLimits(max_expression_size=target_size-1))
    assert blocked.status == SearchStatus.INCOMPLETE
    assert blocked.admitted_states == 1
    assert blocked.size_pruned_successors == 1
    assert blocked.restricted_by == {SearchLimit.EXPRESSION_SIZE}
    assert breadth_first_search(initial, SearchLimits(max_expression_size=target_size)).status == SearchStatus.SOLVED


def test_size_pruning_does_not_skip_other_actions():
    # The cubic expansion is too large; the later product expansion still fits.
    initial = State((x+y)**3 + z*(x+y), (x+y)**3 + x*z + y*z)
    result = breadth_first_search(initial, SearchLimits(max_expression_size=expression_size(initial.target)))
    assert result.status == SearchStatus.SOLVED
    assert result.size_pruned_successors > 0
    assert SearchLimit.EXPRESSION_SIZE in result.restricted_by


def test_multiple_restrictions_are_preserved():
    initial = State((x+y)**2 + z*(x+y), 0)
    result = breadth_first_search(initial, SearchLimits(
        max_depth=0, max_expression_size=expression_size(initial.expression)))
    assert result.status == SearchStatus.INCOMPLETE
    assert result.restricted_by == {SearchLimit.DEPTH, SearchLimit.EXPRESSION_SIZE}


def test_zero_timeout_is_deterministic(monkeypatch):
    monkeypatch.setattr(search, "monotonic", lambda: 100.0)
    result = breadth_first_search(State((x+y)**2, 0), SearchLimits(timeout_seconds=0))
    assert result.status == SearchStatus.TIMEOUT
    assert result.admitted_states == 1
    assert result.expanded_states == result.attempted_transitions == 0
    assert result.elapsed_seconds == 0
    assert result.restricted_by == {SearchLimit.TIMEOUT}


def test_deadline_after_transition_preserves_prior_size_pruning(monkeypatch, graph):
    initial, _ = graph({"a": ("b", "c")})
    clock = [0.0]
    monkeypatch.setattr(search, "monotonic", lambda: clock[0])
    transitions = []

    def transition(state, action):
        transitions.append(action.rule)
        if action.rule == "b":
            successor = State(x+y, state.target)  # Pruned by size.
        else:
            clock[0] = 1.0  # Exactly the deadline, no real wait.
            successor = State(sp.Symbol("c"), state.target)
        return successor, ProofStep(action, state.expression, successor.expression)

    monkeypatch.setattr(search, "apply_action", transition)
    result = breadth_first_search(initial, SearchLimits(max_expression_size=1, timeout_seconds=1))
    assert result.status == SearchStatus.TIMEOUT
    assert transitions == ["b", "c"]
    assert result.admitted_states == 1
    assert result.attempted_transitions == 2
    assert result.size_pruned_successors == 1
    assert result.elapsed_seconds == 1
    assert result.restricted_by == {SearchLimit.EXPRESSION_SIZE, SearchLimit.TIMEOUT}


def test_timeout_after_action_enumeration(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(search, "monotonic", lambda: clock[0])

    def slow_enumeration(state):
        clock[0] = 2.0
        yield Action("expand_power")

    monkeypatch.setattr(search, "iter_actions", slow_enumeration)
    result = breadth_first_search(State((x+y)**2, 0), SearchLimits(timeout_seconds=1))
    assert result.status == SearchStatus.TIMEOUT
    assert result.expanded_states == 1
    assert result.attempted_transitions == 0


def test_actions_are_consumed_incrementally(monkeypatch):
    def actions(state):
        yield Action("expand_power")
        raise AssertionError("Actions after a solved successor must not be consumed.")

    monkeypatch.setattr(search, "iter_actions", actions)
    initial = State((x+y)**2, x**2+2*x*y+y**2)
    assert breadth_first_search(initial).status == SearchStatus.SOLVED


def test_repeated_runs_have_same_proof_and_counters():
    initial = State((x+y)**2-(x-y)**2, 4*x*y)
    first, second = breadth_first_search(initial), breadth_first_search(initial)
    assert replace(first, elapsed_seconds=0) == replace(second, elapsed_seconds=0)


@pytest.mark.parametrize("field,value", [
    ("max_depth", -1), ("max_depth", 1.0), ("max_depth", True),
    ("max_states", 0), ("max_states", -1), ("max_states", None),
    ("max_expression_size", 0), ("max_expression_size", 1.5),
    ("max_expression_size", False), ("timeout_seconds", -1),
    ("timeout_seconds", float("nan")), ("timeout_seconds", float("inf")),
    ("timeout_seconds", "1"), ("timeout_seconds", True),
])
def test_invalid_limits_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        SearchLimits(**{field: value})


@pytest.mark.parametrize("failure", [VerificationError("bad rule"), RuntimeError("bug")])
@pytest.mark.parametrize("phase", ["iter_actions", "apply_action"])
def test_environment_failures_propagate(monkeypatch, phase, failure):
    def fail(*args):
        raise failure

    monkeypatch.setattr(search, phase, fail)
    with pytest.raises(type(failure), match=str(failure)):
        breadth_first_search(State((x+y)**2, 0))


def test_lazy_and_eager_enumeration_match():
    initial = State(z*(x+y)**2 + 2*z*(x+y), 0)
    assert tuple(iter_actions(initial)) == enumerate_actions(initial)


def test_cli_prints_searched_proof_and_limit_explanation(capsys):
    main([])
    output = capsys.readouterr().out
    assert "Search status: solved" in output
    assert "expand_power at" in output
    assert "Target reached: True" in output
    main(["--max-depth", "0"])
    output = capsys.readouterr().out
    assert "Search status: incomplete" in output
    assert "Limits restricting exploration: depth" in output
    assert "may still have a proof" in output
    with pytest.raises(SystemExit) as error:
        main(["--max-states", "0"])
    assert error.value.code == 2
