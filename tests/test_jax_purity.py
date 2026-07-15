"""JAX-purity CI gate for the methods + sim-core.

Standing project constraint: the *methods* (gmphd, nonmyopic, infomax,
coverage_control) and the *sim-core compute* files must be genuine JAX, i.e.
they must

  1. import jax / jax.numpy, AND
  2. use at least one of jit / vmap / scan somewhere in real code.

This test parses each required file to an AST and checks for those markers as
real identifiers: imports must be genuine Import/ImportFrom nodes and
``jit``/``vmap``/``scan`` must appear as a Name or Attribute in executable code.
Docstring/comment mentions, substring collisions (``rescan_foo``), and
unparseable files can never satisfy the gate (a broken file FAILS it loudly).

KNOWN LIMIT (by design): this is a static gate. It proves the markers exist in
real code; it cannot prove the jitted function is the hot path actually called
by the sim. Wiring-level checks belong to each package's own tests.

Some files satisfy the requirement *as a group* rather than individually. For
example ``gmphd/models.py`` is, by design, the standalone NumPy density contract
(``cv_model``/``measurement_matrix``/``mvn_pdf`` keep their original NumPy
signatures); the genuine-JAX numeric kernels for the gmphd package live in
``gmphd/gmphd.py`` and ``gmphd/kernels.py``. So the gmphd requirement is checked
across that compute group: the group passes if jax+jnp+(jit|vmap|scan) are all
present *somewhere in the group*.

EXEMPT files (must never be flagged): evaluation/ (intentionally offline NumPy
per project decision), modtrack/ (intentionally untouched/unused), all test
files, the GUI files (app.py / visualize.py), and pure-orchestration modules.
Those are simply not in the REQUIRED set, so they are never checked here.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Project root = parent of this tests/ directory.
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# REQUIRED-JAX set.
#
# Each entry is a *requirement group*: a logical unit (usually one file, but a
# group of files for packages whose JAX numerics are split across modules). The
# group passes iff `import jax`+`jnp` appear AND at least one of jit/vmap/scan
# appears, considering all files in the group together. Listing more than one
# file means "these collectively must satisfy the constraint".
#
# Paths are relative to the project root.
# ---------------------------------------------------------------------------
REQUIRED_JAX_GROUPS: dict[str, list[str]] = {
    # gmphd methods: gmphd.py is the orchestrating filter; its numeric kernels
    # live in kernels.py; models.py is the standalone NumPy density contract.
    # Treated as one compute group.
    "gmphd": [
        "gmphd/gmphd.py",
        "gmphd/models.py",
        "gmphd/kernels.py",
    ],
    # nonmyopic planner. riccati.py provides the pure-jnp Kalman/Riccati kernels
    # (predict_j / riccati_step_j / kalman_update_j); those kernels are composed
    # under jax.vmap/jax.jit in tree.py (minimax_value_vectorized). HONESTY NOTE:
    # that jitted path is exercised only when greedy_assignment is given r_params
    # (the paper's Eqs 4-5 isotropic noise); the sim's camera-noise adapter uses
    # the NumPy pruned planner (pruning.py) because the camera R_fn is not
    # jit-traceable yet — JAX-ifying that live path is a tracked backlog item,
    # and this static gate cannot see the difference (see module docstring).
    "nonmyopic": [
        "nonmyopic/riccati.py",
        "nonmyopic/tree.py",
    ],
    # infomax controller objective.
    "infomax/objective.py": ["infomax/objective.py"],
    # coverage control Voronoi/Lloyd kernels.
    "coverage_control/voronoi.py": ["coverage_control/voronoi.py"],
    # sim-core compute files.
    "netcomm/tracking/sensors.py": ["netcomm/tracking/sensors.py"],
    "netcomm/tracking/coverage.py": ["netcomm/tracking/coverage.py"],
    # differentiable core: soft sensor/motion model + jax.grad-able episode rollout.
    "diffsim": ["diffsim/model.py", "diffsim/rollout.py"],
}

# ---------------------------------------------------------------------------
# EXEMPT set — documented here for clarity / future maintenance. These are NOT
# part of REQUIRED_JAX_GROUPS, so they are never checked. Listed explicitly so
# the intent is obvious and a future reader does not "accidentally" promote one
# into the required set without a deliberate decision.
# ---------------------------------------------------------------------------
EXEMPT = [
    "evaluation/",                 # intentionally offline NumPy (project decision)
    "modtrack/",                   # intentionally untouched / unused
    "**/tests/**",                 # all test files
    "tests/**",                    # this directory
    "netcomm/tracking/app.py",     # GUI
    "netcomm/tracking/visualize.py",  # GUI
    # pure orchestration (runner, repositioner/tracker factories, testing harness,
    # config/types/dataclasses, package __init__ files) — not numeric compute.
]

# Marker identifier sets (matched as AST identifiers, never substrings — a helper
# named `rescan_foo` or a docstring mentioning "vmap" can NOT satisfy the gate).
_KERNEL_NAMES = frozenset({"jit", "vmap", "scan"})  # at least one hot-kernel transform


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(rel: str, src: str) -> ast.AST:
    """Parse the file, failing the gate LOUDLY on a syntactically-broken file
    (an unparseable file is unimportable, so it cannot be genuine JAX either)."""
    try:
        return ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{rel}: source does not parse ({e}); "
                    "the JAX-purity gate requires importable code")


def _has_import_jax(tree: ast.AST) -> bool:
    """A real `import jax[...]` / `from jax[...] import ...` statement."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "jax" or a.name.startswith("jax.") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            m = node.module or ""
            if m == "jax" or m.startswith("jax."):
                return True
    return False


def _has_jnp(tree: ast.AST) -> bool:
    """Real-code use of the jnp alias or the jax.numpy attribute chain."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "jnp":
            return True
        if (isinstance(node, ast.Attribute) and node.attr == "numpy"
                and isinstance(node.value, ast.Name) and node.value.id == "jax"):
            return True
    return False


def _has_kernel_transform(tree: ast.AST) -> tuple[bool, list[str]]:
    """jit/vmap/scan used as a real identifier: bare name (`from jax import jit`,
    decorators) or attribute (`jax.jit`, `jax.vmap`, `lax.scan`). Docstrings and
    comments are invisible to the AST, so they can never count."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _KERNEL_NAMES:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _KERNEL_NAMES:
            found.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name in _KERNEL_NAMES or (a.asname or "") in _KERNEL_NAMES:
                    found.add(a.name if a.name in _KERNEL_NAMES else a.asname)
    return (len(found) > 0, sorted(found))


def test_required_files_exist():
    """Every file in every required group must actually be present on disk."""
    missing = []
    for group, files in REQUIRED_JAX_GROUPS.items():
        for rel in files:
            if not (ROOT / rel).is_file():
                missing.append(f"{group}: {rel}")
    assert not missing, (
        "REQUIRED-JAX files are missing on disk (update REQUIRED_JAX_GROUPS or "
        f"restore the files): {missing}"
    )


@pytest.mark.parametrize(
    "group,files",
    list(REQUIRED_JAX_GROUPS.items()),
    ids=list(REQUIRED_JAX_GROUPS.keys()),
)
def test_group_is_genuine_jax(group: str, files: list[str]):
    """Each required group imports jax + uses jnp + uses jit/vmap/scan.

    Evaluated across all files in the group collectively. Failures name the
    group, the files inspected, and exactly which marker is missing.
    """
    trees = {}
    for rel in files:
        path = ROOT / rel
        assert path.is_file(), f"{group}: required file {rel} not found"
        trees[rel] = _parse(rel, _read(path))

    imports_jax = any(_has_import_jax(t) for t in trees.values())
    uses_jnp = any(_has_jnp(t) for t in trees.values())
    kernel_hits: dict[str, list[str]] = {}
    uses_kernel = False
    for rel, t in trees.items():
        ok, found = _has_kernel_transform(t)
        if found:
            kernel_hits[rel] = found
        uses_kernel = uses_kernel or ok

    problems = []
    if not imports_jax:
        problems.append("does not `import jax` / `from jax ...`")
    if not uses_jnp:
        problems.append("does not use jnp / jax.numpy")
    if not uses_kernel:
        problems.append(
            "does not use any of jit/vmap/scan in real code "
            "(docstring/comment mentions do not count)"
        )

    assert not problems, (
        f"REQUIRED-JAX group '{group}' is not genuine JAX.\n"
        f"  files inspected: {list(trees.keys())}\n"
        f"  jit/vmap/scan found per file: {kernel_hits or 'NONE'}\n"
        f"  missing: {problems}"
    )


def test_exempt_files_are_not_in_required_set():
    """Sanity guard: nothing in the EXEMPT list is also a required file.

    Keeps the two lists from silently contradicting each other if someone edits
    one without the other. Only checks the concrete (non-glob) exempt paths.
    """
    required_files = {f for files in REQUIRED_JAX_GROUPS.values() for f in files}
    conflicts = []
    for ex in EXEMPT:
        if "*" in ex:
            continue
        ex_norm = ex.rstrip("/")
        for rf in required_files:
            if rf == ex_norm or rf.startswith(ex_norm + "/"):
                conflicts.append((ex, rf))
    assert not conflicts, (
        f"EXEMPT and REQUIRED_JAX_GROUPS conflict: {conflicts}"
    )
