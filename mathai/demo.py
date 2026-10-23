"""Run a scripted proof, without symbolic search."""

import sympy as sp

from .environment import Action, State, apply_action, equivalent, is_solved


def _locate(expression, selected, path=()):
    """Look up a prescribed subtree in the current canonical tree."""
    if expression == selected:
        return path
    for index, child in enumerate(expression.args):
        found = _locate(child, selected, path + (index,))
        if found is not None:
            return found
    return None


def run_demo():
    """Return the initial state, final state, and actual verified proof steps."""
    x, y = sp.symbols("x y")
    initial = State((x + y)**2 - (x - y)**2, 4*x*y)
    state = initial
    steps = []
    # Subtrees are prescribed; locations are resolved again after each rebuild.
    script = (
        ("expand_power", (x + y)**2),
        ("expand_power", (x - y)**2),
    )
    for rule, selected in script:
        path = _locate(state.expression, selected)
        if path is None:
            raise RuntimeError(f"Scripted subtree is missing: {selected}")
        state, step = apply_action(state, Action(rule, path))
        steps.append(step)
    return initial, state, tuple(steps)


def main():
    initial, final, steps = run_demo()
    print(f"Start: {initial.expression}")
    print(f"Target: {initial.target}")
    print(f"Initially equivalent: {equivalent(initial.expression, initial.target)}")
    print(f"Initially reached target: {is_solved(initial)}")
    for number, step in enumerate(steps, 1):
        print(f"{number}. {step.action.rule} at {step.action.path}: {step.after}")
        print(f"   Verified: {step.verified}")
    print(f"All steps verified: {all(step.verified for step in steps)}")
    print(f"Target reached: {is_solved(final)}")


if __name__ == "__main__":
    main()
