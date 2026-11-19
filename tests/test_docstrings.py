"""Check the rubric's documentation requirements for every learner-written Python file.

The rubric requires that every file has a docstring naming its purpose, author and
creation date, and that every function docstring states its purpose, its input
parameters and its output. These tests parse the source with ``ast`` (nothing is
imported or executed) and fail with a list of every function that is incomplete.

Author: Colin Wuest
Date created: 2026-11-21
"""

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEARNER_FILES = [
    "churn_library.py",
    "churn_script_logging_and_tests.py",
    "constants.py",
    "conftest.py",
    "tests/conftest.py",
    "tests/test_churn_library.py",
    "tests/test_docstrings.py",
]

SECTION_RE = r"^\s*{}:\s*$"


def _parse(relative_path):
    """Parse one project file into an abstract syntax tree.

    Args:
        relative_path: str path of the file relative to the project root

    Returns:
        tree: ``ast.Module`` of the file's source code
    """
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source, filename=relative_path)


def _functions(tree):
    """Collect every function and method defined in a module.

    Args:
        tree: ``ast.Module`` returned by ``_parse``

    Returns:
        functions: list of ``ast.FunctionDef`` nodes, including methods and nested functions
    """
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _section(docstring, name):
    """Return the body of a Google-style docstring section such as ``Args``.

    Args:
        docstring: cleaned docstring text
        name: section header without the colon, e.g. ``"Args"`` or ``"Returns"``

    Returns:
        section: str with the section's lines, or None when the section is missing
    """
    lines = docstring.splitlines()
    for index, line in enumerate(lines):
        if re.match(SECTION_RE.format(name), line):
            body = []
            for following in lines[index + 1 :]:
                if following.strip() and not following.startswith(" "):
                    break
                body.append(following)
            return "\n".join(body)
    return None


def _function_problems(node):
    """List what is missing from one function's docstring.

    Args:
        node: ``ast.FunctionDef`` of the function to check

    Returns:
        problems: list of str descriptions; empty when the docstring is complete
    """
    docstring = ast.get_docstring(node, clean=True)
    if not docstring:
        return ["no docstring"]

    problems = []
    if not docstring.splitlines()[0].strip():
        problems.append("no purpose summary line")

    params = [arg.arg for arg in node.args.args + node.args.kwonlyargs if arg.arg != "self"]
    args_section = _section(docstring, "Args")
    if args_section is None:
        problems.append("no Args section (use 'None' when there are no inputs)")
    elif not params and "None" not in args_section:
        problems.append("Args section should say 'None' for a function without inputs")
    else:
        for param in params:
            if not re.search(rf"^\s*{param}\s*:", args_section, re.MULTILINE):
                problems.append(f"parameter '{param}' not documented")

    if _section(docstring, "Returns") is None and _section(docstring, "Yields") is None:
        problems.append("no Returns/Yields section (use 'None' when nothing is returned)")
    return problems


@pytest.mark.parametrize("relative_path", LEARNER_FILES)
def test_module_docstring(relative_path):
    """Each file has a docstring with a purpose, the author and the creation date.

    Args:
        relative_path: str path of the file relative to the project root
            (parametrized over ``LEARNER_FILES``)

    Returns:
        None (the test fails with an ``AssertionError`` if a check does not hold)
    """
    docstring = ast.get_docstring(_parse(relative_path))
    assert docstring, f"{relative_path}: missing module docstring"
    assert docstring.splitlines()[0].strip(), f"{relative_path}: missing purpose line"
    assert re.search(r"^Author: \S+", docstring, re.MULTILINE), f"{relative_path}: no author"
    assert re.search(r"^Date created: \d{4}-\d{2}-\d{2}$", docstring, re.MULTILINE), (
        f"{relative_path}: no 'Date created: YYYY-MM-DD'"
    )


@pytest.mark.parametrize("relative_path", LEARNER_FILES)
def test_function_docstrings(relative_path):
    """Every function documents its purpose, each input (or None) and its output (or None).

    Args:
        relative_path: str path of the file relative to the project root
            (parametrized over ``LEARNER_FILES``)

    Returns:
        None (the test fails with an ``AssertionError`` listing every incomplete function)
    """
    problems = [
        f"{relative_path}:{node.lineno} {node.name}: {problem}"
        for node in _functions(_parse(relative_path))
        for problem in _function_problems(node)
    ]
    assert not problems, "incomplete docstrings:\n" + "\n".join(problems)


def test_checker_detects_incomplete_docstring():
    """The checker itself flags a docstring that lacks Args and Returns sections.

    Args:
        None

    Returns:
        None (the test fails with an ``AssertionError`` if a check does not hold)
    """
    node = ast.parse('def f(a):\n    """Only a summary."""\n').body[0]
    problems = _function_problems(node)
    assert any("Args" in problem for problem in problems)
    assert any("Returns" in problem for problem in problems)
