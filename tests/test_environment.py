from dataclasses import FrozenInstanceError

import pytest
import sympy as sp

import mathai.environment as environment
from mathai import (
    Action, InvalidAction, State, UnsupportedExpression, VerificationError,
    apply_action, enumerate_actions, equivalent, is_solved,
)
from mathai.demo import run_demo

x, y, z = sp.symbols("x y z")


def test_identity_is_not_initially_solved():
    state = State((x + y)**2 - (x - y)**2, 4*x*y)
    assert equivalent(state.expression, state.target)
    assert not is_solved(state)


def test_demo_is_a_connected_verified_proof():
    initial, final, steps = run_demo()
    assert len(steps) == 2
    current = initial
    for step in steps:
        assert step.before == current.expression
        assert step.action in enumerate_actions(current)
        current, replayed = apply_action(current, step.action)
        assert replayed == step
        assert step.verified
    assert current == final
    assert is_solved(final)


@pytest.mark.parametrize("rule,expression,expected", [
    ("expand_mul", (x + y)*(x + z), x**2 + x*y + x*z + y*z),
    ("expand_mul", -2*z*(x + y), -2*x*z - 2*y*z),
    ("expand_power", (x + y)**3, x**3 + 3*x**2*y + 3*x*y**2 + y**3),
    ("factor_common", 6*x**2*y + 9*x*y**2, 3*x*y*(2*x + 3*y)),
    ("factor_common", sp.Rational(2, 3)*x*z + sp.Rational(4, 9)*y*z,
     sp.Rational(2, 9)*z*(3*x + 2*y)),
    ("factor_common", z*(x+y)**2 + z*(x+y), z*(x+y)*(1+x+y)),
    ("collect", x*y + x*z + y, x*(y + z) + y),
    ("collect", x**2*y + x**2*z + x*y, x**2*(y + z) + x*y),
])
def test_rules_preserve_value_and_produce_expected_form(rule, expression, expected):
    before = State(expression, expected)
    after, step = apply_action(before, Action(rule))
    assert after.expression == expected
    assert after.target == before.target
    assert step.verified
    assert is_solved(after)


def test_nested_path_changes_only_selected_subtree():
    expression = z*(x + y)**2
    path = (expression.args.index((x + y)**2),)
    before = State(expression, 0)
    after, step = apply_action(before, Action("expand_power", path))
    assert after.expression == z*(x**2 + 2*x*y + y**2)
    assert isinstance(after.expression, sp.Mul)  # Parent was not distributed.
    assert before.expression == expression
    assert step.before == expression
    with pytest.raises(FrozenInstanceError):
        before.expression = x
    with pytest.raises(FrozenInstanceError):
        step.action.path = ()


@pytest.mark.parametrize("action", [
    Action("unknown"), Action("expand_mul"), Action("expand_power", (99,)),
    Action("expand_power", (-1,)), Action("expand_power", (True,)),
    Action("expand_power", (0.5,)), Action("expand_power", [0]),
    Action("expand_power", (0, 0, 0, 0)),
])
def test_invalid_actions_are_rejected(action):
    with pytest.raises(InvalidAction):
        apply_action(State((x + y)**2, 0), action)


@pytest.mark.parametrize("expression", [
    0, x, x + y, x*y, sp.Integer(2)**0, x**2,
])
def test_enumeration_is_stable_and_contains_only_distinct_changes(expression):
    state = State(expression, 0)
    actions = enumerate_actions(state)
    assert actions == enumerate_actions(state)
    successors = [apply_action(state, action)[0].expression for action in actions]
    assert all(result != state.expression for result in successors)
    assert len(successors) == len(set(successors))


def test_enumeration_handles_competing_and_nested_rules():
    state = State(z*(x+y)**2 + 2*z*(x+y), 0)
    actions = enumerate_actions(state)
    assert actions
    assert actions == enumerate_actions(state)
    assert any(action.path for action in actions)
    successors = [apply_action(state, action)[0].expression for action in actions]
    assert len(successors) == len(set(successors))
    assert state.expression not in successors
    assert all(equivalent(state.expression, result) for result in successors)


def test_incorrect_transform_is_caught_independently(monkeypatch):
    assert not equivalent((x + y)**2, x**2 + y**2)
    monkeypatch.setattr(environment, "_transform", lambda rule, node: x**2 + y**2)
    with pytest.raises(VerificationError):
        apply_action(State((x + y)**2, 0), Action("expand_power"))


@pytest.mark.parametrize("expression", [
    sp.sin(x), 1/x, x**-1, x**sp.Rational(1, 2), x**y,
    sp.Float("0.5")*x, sp.pi*x, sp.I*x, sp.Eq(x, y), x < y,
    sp.Symbol("a", positive=True), sp.Symbol("b", commutative=False),
    "x + y", 0.5, True,
    sp.Mul(0, sp.sin(x), evaluate=False),
])
def test_unsupported_expressions_are_rejected_in_both_state_fields(expression):
    with pytest.raises(UnsupportedExpression):
        State(expression, 0)
    with pytest.raises(UnsupportedExpression):
        State(0, expression)


def test_exact_rationals_and_constant_equivalence():
    expression = sp.Rational(1, 3)*z*(x + y)
    target = x*z/3 + y*z/3
    final, step = apply_action(State(expression, target), Action("expand_mul"))
    assert is_solved(final)
    assert step.verified
    assert not final.expression.atoms(sp.Float)
    assert equivalent(sp.Rational(1, 3) + sp.Rational(2, 3), 1)
    assert not equivalent(sp.Rational(1, 3), sp.Rational(1, 2))


def test_canonicalization_does_not_expand_goals():
    unevaluated_sum = sp.Add(x, x, evaluate=False)
    assert is_solved(State(unevaluated_sum, 2*x))
    assert not is_solved(State((x+y)**2, x**2 + 2*x*y + y**2))
    assert enumerate_actions(State(x + x, 2*x)) == ()


def test_automatic_rational_distribution_is_not_an_explicit_action():
    state = State(sp.Rational(2, 3)*(x+y), 2*x/3 + 2*y/3)
    assert is_solved(state)
    with pytest.raises(InvalidAction):
        apply_action(state, Action("expand_mul"))
    # Factoring rational content alone is undone by normal SymPy evaluation.
    with pytest.raises(InvalidAction):
        apply_action(state, Action("factor_common"))
