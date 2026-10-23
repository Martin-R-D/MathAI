# MathAI

MathAI will construct step-by-step proofs of algebraic identities. Eventually,
a small Transformer will rank transformations, bounded search will explore them,
and an independent symbolic verifier will check each step.

This first milestone implements only the exact polynomial environment, tests,
and a scripted command-line proof. There is no neural training, automated proof
search, web interface, or Google Colab integration.

## Setup and commands

Use Python 3.10 or newer. From the repository root, with your chosen Python
environment active:

```console
python -m pip install -r requirements.txt
python -m mathai.demo
python -m pytest -q
```

An isolated environment is recommended: create it with `python -m venv .venv`.
On standard Windows Python, activate it with
`.\.venv\Scripts\Activate.ps1`; on POSIX, use `source .venv/bin/activate`.
Some Windows Python distributions also place the executables in `.venv/bin`;
in that case, use `.\.venv\bin\python.exe` directly in place of `python`.

## State, actions, and correctness

```python
import sympy as sp
from mathai import State, enumerate_actions, apply_action, equivalent, is_solved

x, y = sp.symbols("x y")
state = State((x + y)**2 - (x - y)**2, 4*x*y)
assert equivalent(state.expression, state.target)
assert not is_solved(state)
actions = enumerate_actions(state)
next_state, step = apply_action(state, actions[0])
assert step.verified
```

`State(expression, target)`, `Action(rule, path)`, and
`ProofStep(action, before, after)` are frozen dataclasses. State construction
validates both expressions and rebuilds them with normal SymPy evaluation.
Use SymPy expressions or Python integers as inputs; strings are not parsed.
Applying an action returns a new state and leaves the old state intact.

A path is a tuple of indices into SymPy's `expr.args`: `()` selects the root,
`(1,)` its second child, and `(1, 0)` that child's first child. These are tree
positions, not positions in the printed text. Inspect `expr.args` when selecting
a location. A path applies to the state where its action was generated;
rebuilding an expression can flatten, remove, and reorder children. Generate
new actions after each step. Actions do not carry a state identifier, so callers
must keep track of their originating state.

Enumeration walks the tree in preorder, trying rules in the order shown below.
It returns an immutable tuple, omits structural no-ops, and retains only the
first action for each distinct full-expression successor. Applying a different
valid action with the same successor is still allowed. Unknown rules, malformed
or out-of-range paths, and inapplicable rules raise `InvalidAction`.
Unsupported input raises `UnsupportedExpression`.

Verification is independent of the transformation code: `equivalent(a, b)`
constructs the polynomial difference over the exact rational domain `QQ` and
checks that it is zero. Every applied action passes this check before a new
state or proof step is returned; failure raises `VerificationError`.
`step.verified` recomputes that same check. No random substitutions are used.

`is_solved(state)` checks **structural equality** of the canonical current and
target expressions. It never expands or factors either expression. Thus a
correct identity can be mathematically equivalent before its target form is
reached. A task may also have inequivalent endpoints; construction accepts it,
but value-preserving actions cannot solve it.

## Explicit transformation granularity

| Rule | Operation at the selected node |
| --- | --- |
| `expand_mul` | Distribute all additive factors of one multiplication node, including products of several sums. Does not recursively expand powers or child products. |
| `expand_power` | Fully expand one nonnegative integer power of a sum using the multinomial formula. Does not recursively expand unrelated subtrees. Powers zero and one normally disappear during construction. |
| `factor_common` | At a sum, extract the positive rational coefficient content and existing factors shared by every term, using their minimum exponents. Does not discover new polynomial factors such as a difference of squares. |
| `collect` | At a sum, group existing summands by powers of symbols in SymPy's sorted symbol order, e.g. `x*y + x*z + y` becomes `x*(y + z) + y`. Does not expand the sum first. |

These are local algebra operations, not necessarily single textbook elementary
steps. In particular, a power expansion may produce many terms, and normal
parent reconstruction can combine or cancel them. No unrestricted `simplify()`
action is provided. Factoring and expansion may reverse each other; this
environment does not attempt to choose a proof strategy or prevent cycles.

## Supported mathematics and representation limits

The language consists of ordinary commutative symbols, exact integer/rational
coefficients, addition, subtraction, multiplication, and nonnegative integer
powers. Use `sp.Rational(1, 3)` or `x/3` for exact fractions; Python's `1/3`
creates a float and is rejected. Functions, inequalities, equations, symbolic
or negative exponents, variable denominators, floats, non-rational constants,
noncommutative symbols, and symbols with assumptions are rejected.

SymPy evaluates expressions before this API sees them, and states consistently
use its normal canonical `Add`, `Mul`, and `Pow` trees. It orders operands,
flattens sums/products, combines identical terms, folds numeric arithmetic,
removes zero/unit factors, and combines powers. It also distributes a rational
scalar over a sum, including a minus sign. Those changes are **automatic**, not
separate proof actions. For example, `x+x` already has target form `2*x`, and
`-(x+y)` already has form `-x-y`. Rational-only common factoring may immediately
evaluate back to the original sum, so it is a no-op and is not offered.
Rational content can remain factored when accompanied by a common symbolic
factor, as in `2*z*(3*x+2*y)/9`.

General products and powers of sums remain unexpanded, preserving meaningful
proof states. Unevaluated input trees are validated before rebuilding but are
not preserved as distinct states. Unsupported syntax that SymPy has already
eliminated before state construction cannot be recovered or rejected (for
example, an already evaluated `0*sp.sin(x)` is simply `0`). Paths and exact
printed traces depend on the SymPy representation, rather than on the user's
original spelling. Large degrees and expression trees can be expensive; this
milestone provides no resource limits or untrusted-expression parser.

## Actual demonstration trace

Generated with SymPy 1.14.0:

```text
Start: -(x - y)**2 + (x + y)**2
Target: 4*x*y
Initially equivalent: True
Initially reached target: False
1. expand_power at (0,): x**2 + 2*x*y + y**2 - (x - y)**2
   Verified: True
2. expand_power at (2, 1): 4*x*y
   Verified: True
All steps verified: True
Target reached: True
```

The script prescribes the two square expansions and looks up each prescribed
subtree in the current state. This is tree location lookup, not proof search.
During the second action, rebuilding the negative term distributes its minus
sign and the enclosing sum cancels terms automatically. There is no extra
invented distribution or collection step in the trace.

The next milestone is bounded symbolic search with depth/node budgets and
cycle detection. Transformer guidance comes later, once the environment and
search provide dependable proof trajectories.
