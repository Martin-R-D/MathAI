"""Local, value-preserving moves on canonical SymPy polynomial trees.

Paths index ``expr.args``; () is the root. Paths belong to the state in
which they were generated, since rebuilding a parent may reorder its children.
All rebuilding uses SymPy's normal evaluation, never automatic expansion.
"""

import ast

from dataclasses import dataclass
from collections.abc import Iterator
from functools import reduce
from math import gcd, lcm

import sympy as sp


class UnsupportedExpression(ValueError):
    """The expression is outside the rational polynomial language."""


class InvalidAction(ValueError):
    """An unknown rule, invalid location, or inapplicable move."""


class VerificationError(ValueError):
    """A proposed move changed the polynomial value."""


def _canonical(expression: sp.Expr | int) -> sp.Expr:
    # Do not parse strings or convert floats implicitly. Validate before rebuilding
    # so unsupported nodes in unevaluated trees cannot disappear during evaluation.
    if isinstance(expression, int) and not isinstance(expression, bool):
        expression = sp.Integer(expression)
    if not isinstance(expression, sp.Expr):
        raise UnsupportedExpression("Use SymPy expressions or exact integers.")

    def visit(node):
        if isinstance(node, sp.Symbol):
            if node.assumptions0 != {"commutative": True}:
                raise UnsupportedExpression("Symbols with assumptions are unsupported.")
            return node
        if isinstance(node, sp.Rational):
            return node
        if isinstance(node, sp.Pow):
            base, exponent = node.args
            if not isinstance(exponent, sp.Integer) or exponent < 0:
                raise UnsupportedExpression("Powers require nonnegative integer exponents.")
            return visit(base) ** exponent
        if isinstance(node, (sp.Add, sp.Mul)):
            return node.func(*(visit(arg) for arg in node.args))
        raise UnsupportedExpression(f"Unsupported polynomial node: {node}")

    return visit(expression)


def parse_expression(text: str) -> sp.Expr:
    """Parse polynomial text without evaluating Python or SymPy source code.

    Only integer literals, symbol names, arithmetic, and parentheses are
    accepted. Division requires a nonzero rational constant denominator;
    powers require a nonnegative integer constant exponent. Constant arithmetic
    is allowed in denominators/exponents, but symbol-containing syntax is not,
    even if it would cancel. The resulting tree uses normal state conventions.
    """
    if not isinstance(text, str):
        raise UnsupportedExpression("Expression input must be text.")

    def constant_part(node, purpose):
        if any(isinstance(part, ast.Name) for part in ast.walk(node)):
            raise UnsupportedExpression(f"{purpose} must not contain symbols.")
        return convert(node)

    def convert(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node, ast.Name):
            return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = convert(node.left)
            if isinstance(node.op, ast.Div):
                right = constant_part(node.right, "Denominators")
                if not isinstance(right, sp.Rational) or right == 0:
                    raise UnsupportedExpression("Denominators must be nonzero rational constants.")
                return left / right
            if isinstance(node.op, ast.Pow):
                right = constant_part(node.right, "Exponents")
                if not isinstance(right, sp.Integer) or right < 0:
                    raise UnsupportedExpression("Powers require nonnegative integer exponents.")
                return left ** right
            right = convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise UnsupportedExpression(
            "Use only symbols, integers, parentheses, +, -, *, / for rational "
            "coefficients, and ** for nonnegative integer powers."
        )

    try:
        tree = ast.parse(text.strip(), mode="eval")
        return _canonical(convert(tree.body))
    except SyntaxError as error:
        raise UnsupportedExpression(f"Invalid expression syntax: {error.msg}.") from None
    except RecursionError:
        raise UnsupportedExpression("Expression nesting is too deep.") from None


@dataclass(frozen=True)
class State:
    expression: sp.Expr
    target: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "expression", _canonical(self.expression))
        object.__setattr__(self, "target", _canonical(self.target))


@dataclass(frozen=True)
class Action:
    rule: str
    path: tuple[int, ...] = ()


@dataclass(frozen=True)
class ProofStep:
    action: Action
    before: sp.Expr
    after: sp.Expr

    @property
    def verified(self) -> bool:
        return equivalent(self.before, self.after)


def equivalent(left: sp.Expr | int, right: sp.Expr | int) -> bool:
    """Independently compare exact polynomials over QQ, without sampling."""
    left, right = _canonical(left), _canonical(right)
    symbols = sorted(left.free_symbols | right.free_symbols, key=sp.default_sort_key)
    return sp.Poly(left - right, *(symbols or [sp.Dummy("constant")]), domain=sp.QQ).is_zero


def is_solved(state: State) -> bool:
    """Structural equality under normal SymPy evaluation; no expansion."""
    return state.expression == state.target


def _factor_common(node):
    if not isinstance(node, sp.Add):
        return node
    coefficients, remainders = zip(*(term.as_coeff_Mul() for term in node.args))
    content = sp.Rational(
        reduce(gcd, (int(c.p) for c in coefficients)),
        reduce(lcm, (int(c.q) for c in coefficients)),
    )
    powers = [term.as_powers_dict() for term in remainders]
    common = content
    for base in sorted(set.intersection(*(set(p) for p in powers)), key=sp.default_sort_key):
        exponent = min(p[base] for p in powers)
        common *= base ** exponent
    if common == 1:
        return node
    return common * sp.Add(*(term / common for term in node.args))


def _transform(rule, node):
    if rule == "expand_mul":
        # Distribute all additive factors of this one multiplication node.
        return sp.expand_mul(node, deep=False) if isinstance(node, sp.Mul) else node
    if rule == "expand_power":
        # Full multinomial expansion of this one power of a sum.
        if isinstance(node, sp.Pow) and isinstance(node.base, sp.Add):
            return sp.expand_multinomial(node, deep=False)
        return node
    if rule == "factor_common":
        # Extract positive rational content and factors shared by every summand.
        return _factor_common(node)
    if rule == "collect":
        # Group existing summands by powers of symbols, in deterministic order.
        if isinstance(node, sp.Add):
            symbols = sorted(node.free_symbols, key=sp.default_sort_key)
            return sp.collect(node, symbols) if symbols else node
        return node
    raise InvalidAction(f"Unknown rule: {rule!r}")


RULES = ("expand_mul", "expand_power", "factor_common", "collect")


def _at_path(expression, path):
    if not isinstance(path, tuple) or any(type(i) is not int or i < 0 for i in path):
        raise InvalidAction("Path must be a tuple of nonnegative integer indices.")
    node = expression
    for index in path:
        if index >= len(node.args):
            raise InvalidAction(f"Path {path!r} is outside the expression tree.")
        node = node.args[index]
    return node


def _replace(expression, path, replacement):
    if not path:
        return replacement
    children = list(expression.args)
    children[path[0]] = _replace(children[path[0]], path[1:], replacement)
    return expression.func(*children)


def apply_action(state: State, action: Action) -> tuple[State, ProofStep]:
    """Apply exactly the selected rule; verify the full result independently."""
    if action.rule not in RULES:
        raise InvalidAction(f"Unknown rule: {action.rule!r}")
    node = _at_path(state.expression, action.path)
    replacement = _transform(action.rule, node)
    after = _canonical(_replace(state.expression, action.path, replacement))
    if replacement == node or after == state.expression:
        raise InvalidAction(f"Rule {action.rule!r} is inapplicable at {action.path!r}.")
    if not equivalent(state.expression, after):
        raise VerificationError(f"Rule {action.rule!r} changed the polynomial value.")
    return State(after, state.target), ProofStep(action, state.expression, after)


def iter_actions(state: State) -> Iterator[Action]:
    """Yield actions lazily in preorder, keeping the first of duplicate successors.

    Like enumerate_actions, candidate testing uses verified transitions. One
    iterator advance may test multiple inapplicable or duplicate candidates.
    """
    seen = set()

    def visit(node, path):
        for rule in RULES:
            action = Action(rule, path)
            try:
                successor, _ = apply_action(state, action)
            except InvalidAction:
                continue
            if successor.expression not in seen:
                seen.add(successor.expression)
                yield action
        for index, child in enumerate(node.args):
            yield from visit(child, path + (index,))

    yield from visit(state.expression, ())


def enumerate_actions(state: State) -> tuple[Action, ...]:
    """Preorder paths, then RULES order; keep the first of duplicate successors."""
    return tuple(iter_actions(state))
