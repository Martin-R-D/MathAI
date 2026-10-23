"""Narrow identity recognition using exact monomial roots, not factorization.

Roots here are internal integer-root calculations: no radical expressions are
introduced. A monomial is a rational coefficient times nonnegative powers of
symbols, including a rational constant. Composite bases are deliberately outside
the matching scope. Callers independently verify every resulting transition.
"""

from itertools import combinations

import sympy as sp


def _monomial_root(term, degree):
    coefficient, remainder = term.as_coeff_Mul()
    if coefficient <= 0:
        return None
    numerator, exact_numerator = sp.integer_nthroot(int(coefficient.p), degree)
    denominator, exact_denominator = sp.integer_nthroot(int(coefficient.q), degree)
    if not (exact_numerator and exact_denominator):
        return None
    root = sp.Rational(numerator, denominator)
    if remainder == 1:
        return root
    for base, exponent in remainder.as_powers_dict().items():
        if (not isinstance(base, sp.Symbol) or not isinstance(exponent, sp.Integer)
                or exponent < 0 or exponent % degree):
            return None
        root *= base ** (exponent // degree)
    return root


def factor_difference_squares(node):
    if not isinstance(node, sp.Add) or len(node.args) != 2:
        return node
    positive = [term for term in node.args if term.as_coeff_Mul()[0] > 0]
    negative = [term for term in node.args if term.as_coeff_Mul()[0] < 0]
    if len(positive) != 1 or len(negative) != 1:
        return node
    a, b = _monomial_root(positive[0], 2), _monomial_root(-negative[0], 2)
    return (a-b)*(a+b) if a is not None and b is not None else node


def factor_perfect_square(node):
    if not isinstance(node, sp.Add) or len(node.args) != 3:
        return node
    for i, j in combinations(range(3), 2):
        a, b = _monomial_root(node.args[i], 2), _monomial_root(node.args[j], 2)
        if a is None or b is None:
            continue
        middle = node.args[3-i-j]
        if middle == 2*a*b:
            return (a+b)**2
        if middle == -2*a*b:
            # Choose a stable sign orientation without expanding the square.
            a, b = sorted((a, b), key=sp.default_sort_key)
            base = a-b
            if base.could_extract_minus_sign():
                base = -base
            return base**2
    return node


def factor_cubes(node, *, difference):
    if not isinstance(node, sp.Add) or len(node.args) != 2:
        return node
    terms = node.args
    if difference:
        positive = [term for term in terms if term.as_coeff_Mul()[0] > 0]
        negative = [term for term in terms if term.as_coeff_Mul()[0] < 0]
        if len(positive) != 1 or len(negative) != 1:
            return node
        terms = (positive[0], -negative[0])
    a, b = (_monomial_root(term, 3) for term in terms)
    if a is None or b is None:
        return node
    if difference:
        return (a-b)*(a**2+a*b+b**2)
    return (a+b)*(a**2-a*b+b**2)
