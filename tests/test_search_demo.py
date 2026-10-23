"""Restricted polynomial text parsing and custom search CLI inputs."""

import pytest
import sympy as sp

from mathai import UnsupportedExpression, parse_expression
from mathai.search_demo import main

x, y = sp.symbols("x y")


@pytest.mark.parametrize("text,expected", [
    (" (x + y)**2 ", (x+y)**2),
    ("x**2 + 2*x*y + y**2", x**2 + 2*x*y + y**2),
    ("-x + +y - 2", -x+y-2),
    ("1/3*x + y/2", x/3+y/2),
    ("(x+y)/(-2)", -(x+y)/2),
    ("(x+y)**(1+1)", (x+y)**2),
    ("x**0 + 7", sp.Integer(8)),
    ("alpha_2*beta", sp.Symbol("alpha_2")*sp.Symbol("beta")),
    ("pi + I", sp.Symbol("pi")+sp.Symbol("I")),
])
def test_parser_constructs_exact_supported_expressions(text, expected):
    parsed = parse_expression(text)
    assert parsed == expected
    assert not parsed.atoms(sp.Float)


@pytest.mark.parametrize("text", [
    "", "x +", "2x", "0.5*x", "1e3", "True", "None", "'x'",
    "x/y", "x/(y-y+1)", "x/0", "x/(2-2)", "x**-1", "x**(1/2)",
    "x**y", "x**(y-y)", "x^2", "x//2", "x%2", "x < y", "x == y",
    "sin(x)", "0*sin(x)", "x.real", "x[0]", "[x]", "(x,y)",
    "__import__('os').system('echo unsafe')", "(lambda: 1)()",
])
def test_parser_rejects_unsupported_syntax_before_cancellation(text):
    with pytest.raises(UnsupportedExpression):
        parse_expression(text)


def test_custom_cli_discovers_square_expansion_and_prints_statistics(capsys):
    main(["--start", "(x + y)**2", "--target", "x**2 + 2*x*y + y**2",
          "--max-depth", "4", "--max-states", "100", "--timeout-seconds", "5"])
    output = capsys.readouterr().out
    assert "Start: (x + y)**2" in output
    assert "Target: x**2 + 2*x*y + y**2" in output
    assert "Search status: solved" in output
    assert "expand_power at (): x**2 + 2*x*y + y**2" in output
    assert "Verified: True" in output
    assert "Proof length: 1" in output
    assert "Admitted states: 2" in output
    assert "Attempted transitions: 1" in output


def test_custom_cli_supports_exact_rational_coefficients(capsys):
    main(["--start", "(x+y)**2/3", "--target", "x**2/3+2*x*y/3+y**2/3"])
    output = capsys.readouterr().out
    assert "Search status: solved" in output
    assert "Verified: True" in output


@pytest.mark.parametrize("option", ["--start", "--target"])
def test_cli_requires_both_problem_arguments(option, capsys):
    with pytest.raises(SystemExit) as error:
        main([option, "x"])
    assert error.value.code == 2
    assert "--start and --target must be supplied together" in capsys.readouterr().err


@pytest.mark.parametrize("start,target,option,message", [
    ("sin(x)", "x", "--start", "Use only"),
    ("x", "x/0", "--target", "nonzero rational constants"),
    ("", "x", "--start", "Invalid expression syntax"),
    ("x", "x**-1", "--target", "nonnegative integer exponents"),
    ("x +", "x", "--start", "Invalid expression syntax"),
])
def test_cli_reports_which_input_is_invalid(start, target, option, message, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--start", start, "--target", target])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert f"error: {option}:" in output.err
    assert message in output.err
    assert "Traceback" not in output.err
    assert output.out == ""


@pytest.mark.parametrize("limit,value,status,reason", [
    ("--max-depth", "0", "incomplete", "depth"),
    ("--max-states", "1", "state_limit", "states"),
    ("--max-expression-size", "1", "incomplete", "expression_size"),
    ("--timeout-seconds", "0", "timeout", "timeout"),
])
def test_custom_cli_preserves_each_search_limit(limit, value, status, reason, capsys):
    main(["--start", "(x+y)**2", "--target", "x**2+2*x*y+y**2", limit, value])
    output = capsys.readouterr().out
    assert f"Search status: {status}" in output
    assert f"Limits restricting exploration: {reason}" in output
    assert "may still have a proof" in output


def test_custom_cli_explains_exhaustion(capsys):
    main(["--start", "x", "--target", "y"])
    output = capsys.readouterr().out
    assert "Search status: exhausted" in output
    assert "No proof exists in the reachable graph" in output


def test_default_cli_keeps_original_identity_and_trace(capsys):
    main([])
    output = capsys.readouterr().out
    assert "Start: -(x - y)**2 + (x + y)**2" in output
    assert "Target: 4*x*y" in output
    assert "Search status: solved" in output
    assert "1. expand_power at (0,):" in output
    assert "2. expand_power at (2, 1): 4*x*y" in output
    assert "Proof length: 2" in output
