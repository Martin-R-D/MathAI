"""Explicit identity patterns, selective moves, and filtered proof search."""

from dataclasses import FrozenInstanceError, replace

import pytest
import sympy as sp

import mathai.environment as environment
from mathai import (
    RULES, Action, InvalidAction, SearchLimits, SearchStatus, State,
    VerificationError, apply_action, breadth_first_search, enumerate_actions,
    equivalent, is_solved, iter_actions,
)
from mathai.search_demo import main

a, b, c, x, y, z = sp.symbols("a b c x y z")


IDENTITIES = [
    ("factor_difference_squares", x**2-y**2, (x-y)*(x+y)),
    ("factor_difference_squares", 4*x**4-y**2/9, (2*x**2-y/3)*(2*x**2+y/3)),
    ("factor_difference_squares", x**2-1, (x-1)*(x+1)),
    ("factor_perfect_square", x**2+2*x*y+y**2, (x+y)**2),
    ("factor_perfect_square", x**2-2*x*y+y**2, (x-y)**2),
    ("factor_perfect_square", x**2-2*x+1, (x-1)**2),
    ("factor_perfect_square", x**2/4+x*y/3+y**2/9, (x/2+y/3)**2),
    ("factor_perfect_square", 4*x**4-4*x**2*y+y**2, (2*x**2-y)**2),
    ("factor_sum_cubes", x**3+y**3, (x+y)*(x**2-x*y+y**2)),
    ("factor_sum_cubes", x**3/8+y**3/27, (x/2+y/3)*(x**2/4-x*y/6+y**2/9)),
    ("factor_difference_cubes", x**3-y**3, (x-y)*(x**2+x*y+y**2)),
    ("factor_difference_cubes", 8*x**6-y**3/27,
     (2*x**2-y/3)*(4*x**4+2*x**2*y/3+y**2/9)),
    ("factor_difference_cubes", x**3-1, (x-1)*(x**2+x+1)),
]


@pytest.mark.parametrize("rule,expression,target", IDENTITIES)
def test_explicit_identities_and_filtered_search(rule, expression, target):
    initial = State(expression, target)
    assert equivalent(expression, target) and not is_solved(initial)
    final, step = apply_action(initial, Action(rule))
    assert is_solved(final)
    assert step.verified
    assert initial.expression == expression
    result = breadth_first_search(initial, rules=(rule,))
    assert result.status == SearchStatus.SOLVED
    assert result.proof == (step,)
    assert result.final_state == final


@pytest.mark.parametrize("rule,expression", [
    ("factor_difference_squares", x**2+y**2),
    ("factor_difference_squares", 2*x**2-y**2),
    ("factor_difference_squares", x**3-y**2),
    ("factor_difference_squares", (x+y)**2-z**2),
    ("factor_difference_squares", x**2-y**2+z),
    ("factor_perfect_square", x**2+3*x*y+y**2),
    ("factor_perfect_square", x**2+2*x*y-y**2),
    ("factor_perfect_square", 2*x**2+2*x*y+y**2),
    ("factor_perfect_square", x**2+2*x*y+y**2+1),
    ("factor_perfect_square", x**2+sp.Rational(2000001, 1000000)*x*y+y**2),
    ("factor_perfect_square", x**2+2*x*(y+z)+(y+z)**2),
    ("factor_sum_cubes", x**3-y**3),
    ("factor_sum_cubes", 2*x**3+y**3),
    ("factor_sum_cubes", x**2+y**3),
    ("factor_sum_cubes", (x+y)**3+z**3),
    ("factor_sum_cubes", -x**3-y**3),
    ("factor_difference_cubes", x**3+y**3),
    ("factor_difference_cubes", x**3-2*y**3),
    ("factor_difference_cubes", x**3-y**2),
    ("factor_difference_cubes", (x+y)**3-z**3),
])
def test_identity_mismatches_are_not_approximate_or_general_factorization(rule, expression):
    with pytest.raises(InvalidAction, match="inapplicable"):
        apply_action(State(expression, 0), Action(rule))


@pytest.mark.parametrize("rule,expression,target", IDENTITIES[::3])
def test_identity_application_is_local(rule, expression, target):
    before = State(z*(expression), z*target)
    path = (before.expression.args.index(expression),)
    after, step = apply_action(before, Action(rule, path))
    assert after.expression == z*target
    assert before.expression == z*expression
    assert step.action.path == path
    assert step.verified


def test_selective_distribution_preserves_grouped_remainder_and_metadata():
    initial = State(x*(a+b+c), a*x+x*(b+c))
    actions = enumerate_actions(initial, rules=("distribute_term",))
    assert len(actions) == 3  # One choice per summand, never arbitrary subsets.
    selected = next(action for action in actions
                    if apply_action(initial, action)[0].expression == initial.target)
    final, step = apply_action(initial, selected)
    assert is_solved(final) and step.verified
    assert x*(b+c) in final.expression.args
    assert selected.expected == initial.expression
    assert selected.selection == (initial.expression.args.index(a+b+c), 0)
    assert len(set(actions)) == 3  # Dataclass equality/hash include metadata.
    assert len({action.selection for action in actions}) == 3
    with pytest.raises(FrozenInstanceError):
        selected.selection = (0, 0)
    assert apply_action(initial, Action("expand_mul"))[0].expression == a*x+b*x+c*x


def test_distribution_selects_additive_factor_and_acts_at_nested_path():
    inner = (a+b+c)*(x+y+z)
    initial = State(1+inner, 0)
    path = (initial.expression.args.index(inner),)
    actions = [action for action in enumerate_actions(initial, rules=("distribute_term",))
               if action.path == path]
    assert len(actions) == 6  # Three summands in each of two selectable factors.
    i, j = inner.args.index(a+b+c), (a+b+c).args.index(a)
    action = Action("distribute_term", path, (i, j), inner)
    final, step = apply_action(initial, action)
    assert final.expression == 1+a*(x+y+z)+(b+c)*(x+y+z)
    assert step.verified


def test_pair_factoring_enables_later_shared_binomial_factoring_and_replay():
    initial = State(a*x+a*y+b*x+b*y, (a+b)*(x+y))
    state = initial
    stages = (a*(x+y)+b*x+b*y, a*(x+y)+b*(x+y), (a+b)*(x+y))
    recorded = []
    for target in stages:
        actions = enumerate_actions(state, rules=("factor_pair", "factor_common"))
        action = next(action for action in actions if apply_action(state, action)[0].expression == target)
        state, step = apply_action(state, action)
        assert step.verified
        recorded.append(step)
    assert is_solved(state)
    assert [step.action.rule for step in recorded] == ["factor_pair", "factor_pair", "factor_common"]
    current = initial
    for step in recorded:
        assert step.before == current.expression
        current, replay = apply_action(current, step.action)
        assert replay == step
    assert current == state
    assert initial.expression == a*x+a*y+b*x+b*y


def test_filtered_bfs_uses_selective_factoring_and_keeps_parent_selection_context():
    initial = State(a*x+a*y+b*x+b*y, (a+b)*(x+y))
    rules = ("factor_pair", "factor_common")
    result = breadth_first_search(initial, SearchLimits(max_depth=3, max_states=100), rules=rules)
    assert result.status == SearchStatus.SOLVED
    assert [step.action.rule for step in result.proof] == ["factor_pair", "factor_pair", "factor_common"]
    current = initial
    for step in result.proof:
        if step.action.selection:
            assert step.action.expected == current.expression
        current, replay = apply_action(current, step.action)
        assert replay == step and replay.verified
    assert current == result.final_state
    repeated = breadth_first_search(initial, SearchLimits(max_depth=3, max_states=100), rules=rules)
    assert replace(result, elapsed_seconds=0) == replace(repeated, elapsed_seconds=0)


def test_nested_pair_factoring_with_rational_content():
    selected = 2*a*x/3+4*a*y/9+b
    initial = State(selected**2, 0)
    terms = selected.args
    pair = tuple(sorted((terms.index(2*a*x/3), terms.index(4*a*y/9))))
    final, step = apply_action(initial, Action("factor_pair", (0,), pair, selected))
    assert final.expression == (2*a*(3*x+2*y)/9+b)**2
    assert step.verified


@pytest.mark.parametrize("selection", [(), (0,), (0, 1, 2), [0, 1], (-1, 0), (True, 0), (0.0, 0), (100, 100)])
@pytest.mark.parametrize("rule,expression", [("distribute_term", x*(a+b+c)), ("factor_pair", a*x+a*y+b)])
def test_invalid_selective_metadata_is_rejected(rule, expression, selection):
    with pytest.raises(InvalidAction):
        apply_action(State(expression, 0), Action(rule, selection=selection, expected=expression))


def test_pair_selection_requires_order_distinct_terms_and_actual_common_factor():
    expression = a*x+a*y+b
    for pair in ((1, 1), (2, 0)):
        with pytest.raises(InvalidAction):
            apply_action(State(expression, 0), Action("factor_pair", selection=pair, expected=expression))
    no_common = a+b+c
    with pytest.raises(InvalidAction, match="nontrivial common factor"):
        apply_action(State(no_common, 0), Action("factor_pair", selection=(0, 1), expected=no_common))
    rational_only = 2*x+2*y+z
    pair = tuple(sorted((rational_only.args.index(2*x), rational_only.args.index(2*y))))
    with pytest.raises(InvalidAction):
        apply_action(State(rational_only, 0), Action("factor_pair", selection=pair, expected=rational_only))


@pytest.mark.parametrize("expression,rule", [(x*(a+b), "distribute_term"), (a*x+a*y, "factor_pair")])
def test_selective_rules_require_a_proper_remainder(expression, rule):
    assert enumerate_actions(State(expression, 0), rules=(rule,)) == ()
    with pytest.raises(InvalidAction):
        apply_action(State(expression, 0), Action(rule, selection=(0, 1), expected=expression))


def test_distribution_rejects_nonadditive_factor_selection():
    expression = x*(a+b+c)
    with pytest.raises(InvalidAction, match="select one term of a sum"):
        apply_action(State(expression, 0), Action("distribute_term", selection=(0, 0), expected=expression))


@pytest.mark.parametrize("rule,expression", [("distribute_term", x*(a+b+c)), ("factor_pair", a*x+a*y+b)])
def test_missing_or_stale_selection_context_is_rejected(rule, expression):
    action = enumerate_actions(State(expression, 0), rules=(rule,))[0]
    with pytest.raises(InvalidAction, match="missing or stale"):
        apply_action(State(expression, 0), replace(action, expected=None))
    # Same node shape and valid indices, but a different original subexpression.
    changed = expression.xreplace({a: z})
    with pytest.raises(InvalidAction, match="missing or stale"):
        apply_action(State(changed, 0), action)


def test_ordinary_rules_reject_unexpected_metadata():
    with pytest.raises(InvalidAction, match="does not accept"):
        apply_action(State((x+y)**2, 0), Action("expand_power", selection=(0, 1)))


@pytest.mark.parametrize("expression", [a*x+a*y+b*x+b*y, x*(a+b+c), x**6-y**6, x**2+2*x*y+y**2])
def test_new_enumeration_is_deterministic_and_has_distinct_verified_successors(expression):
    initial = State(expression, 0)
    actions = enumerate_actions(initial)
    assert actions == enumerate_actions(initial) == tuple(iter_actions(initial))
    successors = [apply_action(initial, action)[0].expression for action in actions]
    assert len(successors) == len(set(successors))
    assert all(expression != following and equivalent(expression, following) for following in successors)


def test_original_rule_wins_duplicate_successor_but_filter_exposes_pair_action():
    initial = State(a*x+a*y+b, a*(x+y)+b)
    all_actions = enumerate_actions(initial)
    matching = [action for action in all_actions if apply_action(initial, action)[0].expression == initial.target]
    assert len(matching) == 1 and matching[0].rule == "collect"
    assert enumerate_actions(initial, rules=("factor_pair",))[0].rule == "factor_pair"
    forward = enumerate_actions(initial, rules=("collect", "factor_pair"))
    assert forward == enumerate_actions(initial, rules=("factor_pair", "collect", "factor_pair"))


def test_no_general_factorization_was_added_to_existing_rule():
    with pytest.raises(InvalidAction):
        apply_action(State(x**2-y**2, 0), Action("factor_common"))
    with pytest.raises(InvalidAction):
        apply_action(State(x**3+y**3, 0), Action("factor_common"))


def test_incorrect_identity_result_is_caught_by_independent_verifier(monkeypatch):
    monkeypatch.setattr(environment, "factor_difference_squares", lambda node: (x-y)**2)
    with pytest.raises(VerificationError):
        apply_action(State(x**2-y**2, 0), Action("factor_difference_squares"))


def test_all_rule_graph_still_has_genuine_equivalent_dead_ends():
    initial = State(x**2+y**2, (x+y)**2-2*x*y)
    assert equivalent(initial.expression, initial.target) and not is_solved(initial)
    result = breadth_first_search(initial)
    assert result.status == SearchStatus.EXHAUSTED
    assert result.attempted_transitions == 0


@pytest.mark.parametrize("rules", [("not_a_rule",), "factor_pair", (5,)])
def test_api_rule_filters_are_validated_even_for_initial_goals(rules):
    with pytest.raises(ValueError):
        breadth_first_search(State(x, x), rules=rules)
    with pytest.raises(ValueError):
        enumerate_actions(State(x, x), rules=rules)


def test_empty_api_filter_is_a_valid_graph_with_no_edges():
    result = breadth_first_search(State(x**2-y**2, (x-y)*(x+y)), rules=())
    assert result.status == SearchStatus.EXHAUSTED
    assert result.admitted_states == 1


def test_cli_rules_help_unknown_identifiers_and_filtered_identity(capsys):
    main(["--start", "x**2-y**2", "--target", "(x-y)*(x+y)", "--rules", "factor_difference_squares"])
    output = capsys.readouterr().out
    assert "Search status: solved" in output
    assert "factor_difference_squares at ()" in output
    main(["--start", "x**2-y**2", "--target", "(x-y)*(x+y)", "--rules", "factor_common"])
    assert "Search status: exhausted" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["--rules", "invented"])
    assert error.value.code == 2
    assert "invalid choice: 'invented'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert all(rule in help_text for rule in RULES)


def test_cli_prints_selective_metadata(capsys):
    main(["--start", "x*(a+b+c)", "--target", "a*x+x*(b+c)", "--rules", "distribute_term"])
    output = capsys.readouterr().out
    assert "Search status: solved" in output
    assert "selection=(1, 0)" in output
