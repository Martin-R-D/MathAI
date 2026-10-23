# MathAI

MathAI will construct step-by-step proofs of algebraic identities. Eventually,
a small Transformer will rank transformations, bounded search will explore them,
and an independent symbolic verifier will check each step.

MathAI implements an exact polynomial environment with explicit identities and
selective transformations, bounded breadth-first proof search, tests, and
scripted and automatic command-line demonstrations. There is no neural model, training pipeline, heuristic search,
web interface, or Google Colab integration.

## Setup and commands

Use Python 3.10 or newer. From the repository root, with your chosen Python
environment active:

```console
python -m pip install -r requirements.txt
python -m mathai.demo
python -m mathai.search_demo
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

`State(expression, target)`, `Action(rule, path, selection=(), expected=None)`, and
`ProofStep(action, before, after)` are frozen dataclasses. State construction
validates both expressions and rebuilds them with normal SymPy evaluation.
Use SymPy expressions or Python integers as state inputs. For text, first call
`parse_expression(text)` to construct a validated SymPy expression.
Applying an action returns a new state and leaves the old state intact.

A path is a tuple of indices into SymPy's `expr.args`: `()` selects the root,
`(1,)` its second child, and `(1, 0)` that child's first child. These are tree
positions, not positions in the printed text. Inspect `expr.args` when selecting
a location. A path applies to the state where its action was generated;
rebuilding an expression can flatten, remove, and reorder children. Generate
new actions after each step. Actions do not carry a state identifier, so callers
must keep track of their originating state.

Existing two-argument actions remain valid. Selective actions additionally
record an immutable tuple `selection` and the original selected subexpression
as `expected`. The metadata participates in action equality and hashing and is
retained verbatim in proof steps. Applying a selective action requires its
`expected` expression to match the current node, rejecting stale selections
even when their indices are still in range. Nonselective rules reject selection
metadata. The expectation guards the selected node, not the entire state;
callers must still respect the originating-state path convention.

Enumeration walks the tree in preorder, trying rules in the order shown below.
`enumerate_actions` returns an immutable tuple; `iter_actions` yields the same
actions lazily. Both omit structural no-ops and retain only the
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

An audit of the original rules found that `factor_common` only intersects
existing factor powers across all summands and extracts rational coefficient
content; it never calls a general polynomial factorizer. It can extract a
shared composite factor such as `(x+y)`, but cannot recognize `x**2-y**2`.
`collect` can group multiple pairs at once, so it can bypass a pair-by-pair
factoring proof. Full distribution can also coincide with selective distribution
when constructors flatten the result. Enumeration retains the first distinct
successor in preorder and rule order, with the four original rules preceding
the new rules at each node. Rule filtering happens before this deduplication,
so a selective-only experiment still exposes its otherwise overlapping moves.

| Rule (in preference order at each node) | Operation and example |
| --- | --- |
| `expand_mul` | Distribute all additive factors of one multiplication node, including products of several sums: `x*(a+b) -> a*x+b*x`. Does not recursively expand powers or child products. |
| `expand_power` | Fully expand one nonnegative integer power of a sum using the multinomial formula: `(x+y)**2 -> x**2+2*x*y+y**2`. Does not recursively expand unrelated subtrees. Powers zero and one normally disappear during construction. |
| `factor_common` | At a sum, extract positive rational content and existing factors shared by every term using minimum exponents: `a*x+a*y -> a*(x+y)`. No general polynomial factorization. |
| `collect` | At a sum, group existing summands by powers of symbols in SymPy's sorted symbol order: `x*y+x*z+y -> x*(y+z)+y`. Does not expand first. |
| `factor_difference_squares` | Recognize exactly two opposite-sign square monomials: `x**2-y**2 -> (x-y)*(x+y)`. |
| `factor_perfect_square` | Recognize exactly three monomials with square endpoints and an exact signed cross term: `x**2+2*x*y+y**2 -> (x+y)**2`, or `x**2-2*x*y+y**2 -> (x-y)**2`. |
| `factor_sum_cubes` | Recognize exactly two positive cube monomials: `x**3+y**3 -> (x+y)*(x**2-x*y+y**2)`. |
| `factor_difference_cubes` | Recognize exactly two opposite-sign cube monomials: `x**3-y**3 -> (x-y)*(x**2+x*y+y**2)`. |
| `distribute_term` | At a product, distribute the other factors over one summand of one selected additive factor: `x*(a+b+c) -> a*x+x*(b+c)`. Leave the other summands grouped. |
| `factor_pair` | At a sum, extract a common factor from exactly two selected summands, leaving other terms alone: `a*x+a*y+b -> a*(x+y)+b`. |

Identity matching is intentionally narrower than arbitrary substitution. A
monomial is a positive rational coefficient times nonnegative integer powers
of symbols, including a positive rational constant. An endpoint is an exact
square/cube only when both coefficient numerator and denominator are exact
integer squares/cubes and every symbol exponent is divisible by two/three.
These checks use exact integer-root arithmetic; they never construct radicals
or approximate roots. For example, `4*x**4-y**2/9` matches a difference of
squares with endpoints `2*x**2` and `y/3`, but `2*x**2-y**2` does not. Negative
terms are permitted in the difference identities and the perfect-square cross
term; a sum of two negative cubes is outside the current scope.

Identity rules inspect the entire selected sum, not arbitrary subsets of a
larger sum. Composite endpoints such as `(x+y)**2-z**2` are outside the matching
scope. Perfect squares require the remaining term to be exactly `+2*a*b` or
`-2*a*b` as a canonical monomial. The squared binomial's sign is oriented using
SymPy's syntactic minus-sign convention, so `x**2-2*x+1` produces `(x-1)**2`.
Expressions with collapsed or combined terms may no longer have the required
two/three-term shape. No general `factor()` or unrestricted simplification is
used to implement these rules. Reverse expansion is already available through
the original expansion rules.

### Selective-action policy

All indices refer to canonical `.args` in the action's original subexpression:

- `distribute_term`: `selection=(factor_index, summand_index)`. Enumerate
  additive factors in index order and one summand at a time in index order.
  The additive factor must have at least three summands, leaving a remainder
  of at least two. With two summands this would just duplicate full expansion,
  so use `expand_mul` instead. A product with several additive factors offers
  choices for each factor. There are at most the sum of their arities in
  candidates, rather than all subsets of a sum.
- `factor_pair`: `selection=(first_term_index, second_term_index)` with strictly
  increasing distinct indices. At sums with at least three summands, enumerate
  all pairs in lexicographic order, at most `n*(n-1)/2` candidates. Each pair
  uses the unchanged `factor_common` operation and must produce a nontrivial,
  representable factorization. There must be an unselected remainder; use
  `factor_common` for a two-term sum. No arbitrary subsets or grouping-only
  actions are offered.

For manual actions, include `expected` as the original selected node, or obtain
actions through enumeration:

```python
from mathai import Action, apply_action, enumerate_actions, parse_expression, State

expression = parse_expression("x*(a+b+c)")
initial = State(expression, parse_expression("a*x+x*(b+c)"))
action = next(action for action in enumerate_actions(initial, rules=("distribute_term",))
              if action.selection == (1, 0))
final, step = apply_action(initial, action)
assert action == Action("distribute_term", (), (1, 0), expression)
assert step.verified
```

Normal constructors can combine products into powers, distribute rational
scalars or minus signs, and flatten sums while rebuilding parents. For example,
`2*(x+y+z)` is already fully distributed before enumeration; factoring just a
rational coefficient from a pair is automatically undone. Such no-ops are
rejected. Factoring a symbol along with rational content can remain meaningful.
There is no action whose sole effect is parentheses that SymPy immediately
removes. Duplicate successors from different selections or rules are emitted
only once, with the documented first-action preference.

These are local algebra operations, not necessarily single textbook elementary
steps. In particular, a power expansion may produce many terms, and normal
parent reconstruction can combine or cancel them. No unrestricted `simplify()`
action is provided. Factoring and expansion may reverse each other; this
environment itself does not choose a proof strategy or prevent cycles; the
search module tracks visited states.

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
original spelling. Large degrees and expression trees can be expensive. Search
has admission limits, but individual SymPy operations have no resource sandbox.
The CLI's restricted text parser accepts only the polynomial syntax described
below; parsing and construction are not governed by the search resource limits.

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

## Bounded breadth-first search

```python
from mathai import SearchLimits, SearchStatus, breadth_first_search

result = breadth_first_search(state, SearchLimits(
    max_depth=6, max_states=1000, max_expression_size=200,
    timeout_seconds=5.0,
))
if result.status == SearchStatus.SOLVED:
    for step in result.proof:
        assert step.verified
    assert result.final_state.expression == state.target
```

`breadth_first_search`, `iter_actions`, and `enumerate_actions` also accept a
`rules` subset of identifiers. `None` enables all rules; an empty API subset
creates a graph with no edges. Unknown identifiers are rejected, even for an
initially solved task. Caller ordering and repeated identifiers do not change
the canonical rule order. Filtering happens before successor deduplication.
The BFS counters and resource-limit semantics below are unchanged; richer
rule sets change the explored graph and therefore the resulting counter values.

Search uses a FIFO queue and the environment's deterministic action order.
Each action is applied through `apply_action`, including independent exact
verification. The visited dictionary uses the immutable SymPy expression itself
as its structural key: native expression equality distinguishes factored and
expanded forms. It does not use polynomial equality, expansion, printed strings,
or iteration over hash sets to choose traversal order. The target is fixed for
the run. States are marked visited when admitted and enqueued, preventing
duplicate queue entries and repeated expansion through reversible cycles.

Each admitted node stores its state, depth, parent expression key, and exact
incoming proof step. The proof is reconstructed by following parent references
after finding a goal; paths are not copied into queue entries. Consequently,
returned action locations still refer to their corresponding parent states.
The input is never changed. Search does not catch environment exceptions:
verification failures and unexpected errors remain visible to callers.

With unit-cost actions, BFS returns a shortest proof in the graph restricted to
expressions that pass the size bound; deterministic action order breaks ties.
Pruning large expressions can remove shorter paths, or every path, from the
full transformation graph. Other limits can terminate the run before any proof
is found. Mathematical equivalence alone never satisfies the goal.

### Limits and exact boundaries

`SearchLimits` is frozen and validated on construction. Depth is an integer
at least zero; state and size bounds are integers at least one. Booleans and
noninteger values are rejected for these fields. Timeout is `None` or a finite
nonnegative Python integer/float; booleans, NaN, and infinity are rejected.

| Setting | Default | Boundary behavior |
| --- | --- | --- |
| `max_depth` | 6 | Inclusive maximum admitted depth. The initial state is depth 0. At the boundary, actions are inspected but successors are not admitted. A depth restriction is recorded only if a new, size-eligible successor would require exceeding the depth. Dead ends and edges to visited states do not record a depth restriction. |
| `max_states` | 1000 | Inclusive maximum unique admitted expressions, including the initial and goal states. Filling the budget alone does not stop search: existing queue entries can still be processed. Search stops when another new, size- and depth-eligible successor needs admission. Even a goal must fit this budget. |
| `max_expression_size` | 200 | Inclusive maximum current-expression tree-node count, using `expression_size`. Every operator, atom, and power exponent counts as one node; repeated subtrees count at every occurrence. For example, `(x+y)**2` has five nodes. Target size and integer digit count are excluded. Oversized successors are skipped; the remaining actions are still explored. |
| `timeout_seconds` | `None` | Wall-clock duration measured with `time.monotonic`. Expiration is `elapsed >= timeout`. Checks surround queue processing, lazy iterator advances, transitions, and successor size counting. Zero expires before any unsolved input is expanded. |

Initial size admission is checked first. If it fails, search returns incomplete
with zero admitted states and maximum admitted depth `None`, even if the input
already matches the target. Otherwise the initial state is admitted and checked
for a goal before exploration limits: an initially solved input returns an empty
proof even at depth zero or timeout zero. Initial validation/size work counts
toward elapsed time. No early equivalence rejection or target expansion occurs.

For a generated successor, deadline checks precede classification; otherwise
the processing order is visited check, size check, depth check, state budget,
then admission and structural goal check. A successor fitting a bound exactly
is allowed. Only restrictions actually encountered are recorded. An oversized
successor is classified as size-pruned rather than also depth-pruned. If a
deadline expires during an operation, its computed result is not admitted.

These limits bound admitted work, not the allocation or time spent inside an
individual algebra operation. A large power expansion can temporarily create an
oversized expression before pruning. Timeout checks cannot interrupt SymPy,
size traversal, or one iterator advance, which may test several inapplicable
or duplicate candidate actions. Elapsed time may therefore exceed the timeout.
Lazy enumeration avoids constructing the entire action list before processing
the first action; the environment retains per-parent successor keys for its own
deduplication. It verifies candidates while enumerating and search verifies
each selected transition again; enumeration's internal work is not counted as
separate search transitions.

### Results and statistics

`SearchResult` is frozen. `proof` is a tuple of actual `ProofStep` objects and is
empty on every unsolved result. `final_state` is populated only when solved.
`restricted_by` is a frozenset of `SearchLimit` values, retaining all restrictions
encountered, even if search later solves the task or stops for another limit.

| Status | Meaning |
| --- | --- |
| `solved` | The target's structural form was reached. |
| `exhausted` | The queue emptied with no recorded restriction. No proof is reachable with the current rules; this is not a statement that the identity is false. |
| `incomplete` | The initial expression was oversized, or the queue emptied after size/depth restrictions prevented complete exploration. |
| `state_limit` | Another eligible unique successor could not be admitted within the state budget. |
| `timeout` | The monotonic deadline was reached during exploration. |

The counters describe search-layer events:

- `admitted_states`: unique expressions stored in the visited dictionary,
  including the initial state when it fits and the goal when found.
- `expanded_states`: queue entries for which action iteration was started,
  including dead ends, depth-boundary inspections, and partially processed
  entries interrupted by a limit. Each admitted expression is expanded at most
  once. A goal detected on admission is not expanded.
- `attempted_transitions`: calls search makes to `apply_action` for yielded
  actions. Includes duplicate, pruned, and budget-blocked successors and calls
  that complete after the timeout. Excludes environment enumeration's internal
  candidate tests. No result is returned if such a call raises an exception.
- `duplicate_successors`: attempted successors already admitted when checked.
  The environment already suppresses same-parent duplicate successors.
- `size_pruned_successors`: nonvisited successor occurrences rejected for size,
  not a count of distinct pruned expressions. The initial size rejection is
  reported through `restricted_by`, not this counter.
- `max_admitted_depth`: deepest admitted node, zero for the initial state alone,
  or `None` if no state was admitted.
- `elapsed_seconds`: monotonic elapsed time from search entry through result
  construction, including proof reconstruction. This is the only expected
  nondeterministic statistic when timeouts are disabled.

### Automatic demonstration

The original `python -m mathai.demo` remains scripted. The separate automatic
demo gets its proof entirely from BFS and accepts all resource limits:

```console
python -m mathai.search_demo
python -m mathai.search_demo --max-depth 4 --max-states 100 --max-expression-size 100 --timeout-seconds 5
python -m mathai.search_demo --max-depth 0
python -m pytest -q
```

Supply both `--start` and `--target` to search a custom problem. Omitting both
keeps the default identity; supplying just one is an argparse error. Quote each
expression so your terminal passes it as one argument:

```console
python -m mathai.search_demo --start "(x + y)**2" --target "x**2 + 2*x*y + y**2" --max-depth 4 --max-states 100 --timeout-seconds 5
python -m mathai.search_demo --start "(x+y)**2/3" --target "x**2/3+2*x*y/3+y**2/3"
python -m mathai.search_demo --start "(x+y)**2" --target "x**2+2*x*y+y**2" --max-depth 0
```

The first command discovers a verified one-step square expansion; the second
uses exact rational coefficients. The third reports incomplete exploration
because of the depth limit. All existing limit options apply to custom tasks.

`parse_expression` reads an AST and explicitly constructs allowed operations;
it never evaluates source code or uses SymPy's eval-based text parsers. It
supports identifier symbol names, integer literals, parentheses, unary `+`/`-`,
addition, subtraction, multiplication, `/` for rational coefficients, and `**`
for nonnegative integer powers. Names such as `pi` and `I` are ordinary symbols,
not predefined constants. Multiplication must be explicit (`2*x`, not `2x`).
Denominators must be nonzero rational constants, and exponents must be
nonnegative integer constants; constant arithmetic such as `**(1+1)` is allowed.
Symbol-containing denominators or exponents are rejected even if symbols would
cancel. Calls (including `Rational(...)`), attributes, subscripts, floats,
inequalities, and other unsupported syntax produce an argparse error identifying
`--start` or `--target`. In this text syntax, `1/3` is exact; in Python API code,
continue to use `sp.Rational(1, 3)` instead of Python's floating-point `1/3`.
Normal SymPy canonicalization still applies after parsing.

An actual default run with SymPy 1.14.0 produced the following (the elapsed-time
line is omitted because it varies):

```text
Start: -(x - y)**2 + (x + y)**2
Target: 4*x*y
Search status: solved
1. expand_power at (0,): x**2 + 2*x*y + y**2 - (x - y)**2
   Verified: True
2. expand_power at (2, 1): 4*x*y
   Verified: True
Proof length: 2
Admitted states: 6
Expanded states: 2
Attempted transitions: 5
Duplicate successors: 0
Size-pruned successors: 0
Maximum admitted depth: 2
Target reached: True
```

Only the documented local rules and matching scope are available; some genuine
identities still have no path to a requested structural form. Expression growth can
make BFS expensive in both time and memory, and SymPy canonicalization can
combine several algebraic changes during a single action. Bounded failure is
not evidence that no proof exists outside the explored graph.

### Rule-filtered experiments and actual traces

`--rules RULE_ID RULE_ID ...` restricts automatic search to the selected rules.
When omitted, all ten rules are enabled. `python -m mathai.search_demo --help`
lists every accepted identifier, and unknown identifiers are argparse errors.
The filter deliberately excludes `collect` in the pair-factoring example so it
cannot group several pairs in a single action:

```console
python -m mathai.search_demo --start "x**2-y**2" --target "(x-y)*(x+y)" --rules factor_difference_squares
python -m mathai.search_demo --start "x**2+2*x*y+y**2" --target "(x+y)**2" --rules factor_perfect_square
python -m mathai.search_demo --start "x**3-y**3" --target "(x-y)*(x**2+x*y+y**2)" --rules factor_difference_cubes
python -m mathai.search_demo --start "x**3+y**3" --target "(x+y)*(x**2-x*y+y**2)" --rules factor_sum_cubes
python -m mathai.search_demo --start "x*(a+b+c)" --target "a*x+x*(b+c)" --rules distribute_term
python -m mathai.search_demo --start "a*x+a*y+b*x+b*y" --target "(a+b)*(x+y)" --rules factor_pair factor_common --max-depth 4 --max-states 100
```

Actual automatically discovered proof lines with SymPy 1.14.0 (each step
verified, each search reported `solved`):

```text
1. factor_difference_squares at (): (x - y)*(x + y)

1. factor_perfect_square at (): (x + y)**2

1. factor_difference_cubes at (): (x - y)*(x**2 + x*y + y**2)

1. factor_sum_cubes at (): (x + y)*(x**2 - x*y + y**2)

1. distribute_term at () selection=(1, 0): a*x + x*(b + c)

1. factor_pair at () selection=(0, 1): a*(x + y) + b*x + b*y
2. factor_pair at () selection=(1, 2): a*(x + y) + b*(x + y)
3. factor_common at (): (a + b)*(x + y)
```

The filtered pair-factoring run admitted 8 states, expanded 6, attempted 9
transitions, skipped 2 duplicates, pruned none for size, and reached depth 3.
All metadata is retained in the reconstructed proof for exact replay. Enabling
`collect` can yield a shorter proof because that existing rule can group both
pairs at once; these are different experiments, not different BFS semantics.

BFS will remain useful as a deterministic baseline and a source of shortest
verified trajectories on small tasks when learned guidance is added. The next
milestone is generating training examples from these search results and
training a small Transformer to rank available actions. This implementation
does not yet generate a dataset or train a model.
