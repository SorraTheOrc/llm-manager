"""Static guard: every proxy ``/slots`` URL construction carries ``?model=``.

Regression fence for **AC3(a)** of LP-0MVP7HV7Q001AJF8 (router ``/slots``
HTTP 400s from a model-less query). llama-server's router answers a model-less
``GET /slots`` with HTTP 400 and logs no query string, so a model-less call
site is invisible in ``llama-server.log``; the only durable defence is a
source-level fence. The inventory this fence mirrors (every site, its target
port and the ``?model=`` value it sends) is the AC1 audit comment
``LP-C0MVVWU5PS004W1YR`` on the parent epic.

The rule
--------

A ``/slots`` URL construction is compliant when the URL itself carries
``?model=`` **or** the variable holding the base URL is unconditionally
extended with ``?model=`` in the same scope. An appender behind an ``if``
guard (``if model: url = f"{url}?model={model}"``) is **not** compliant: a
model-less request stays reachable whenever the guard is false, which is
exactly the live defect the audit classified as #1-#3.

Construction sites are found structurally (``ast``): a string or f-string
whose literal segments contain a ``/slots`` path segment, whether inline in a
request call or as the endpoint argument of a URL builder such as
``_build_llama_url(...)``. Docstrings, comments and the slot save/restore
path ``/slots/{slot_id}?action=`` (a POST with the model in the body, a
different request class) are not construction sites and are ignored.

The fence is deliberately **fail-closed**: a model-less ``/slots`` string that
is only used in a log message is flagged too, because a log-only literal is
cheap to fix and any attempt to reason about dataflow would let a real request
slip through.

The allowlist
-------------

``ALLOWLIST`` holds exactly the offending sites the AC1 audit identified, so
the fence is green on landing (slice 2, LP-0MVU7ZYWS002RDYM) and each later
slice provably closes its own entry:

* slice 3 (LP-0MVU7ZZ7M002D9GT) threaded a resolved model through the four
  model-less probes (its entries are removed: the URLs now carry `?model=`
  inline, or the site skips the query when no model is known);
* slice 4 (LP-0MVU7ZZIA0092AXW) guarantees a resolved model id everywhere,
  which removes the three conditional-appender constructors and empties the
  allowlist.

The allowlist cannot rot: an entry whose site no longer offends (or no longer
exists) is a *stale entry* and fails the suite just like an unlisted violation.

Runs offline: the module only reads ``proxy/proxy/**/*.py`` and parses it,
performing no imports of the proxy package and no I/O beyond the file reads.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass
from pathlib import Path

#: The scanned package: ``<repo>/proxy/proxy`` (this file lives in
#: ``<repo>/proxy/tests``).
PROXY_DIR = Path(__file__).resolve().parents[1] / "proxy"

#: A ``/slots`` *path segment*: followed by the end of the string or by a query
#: string. ``/slots/{slot_id}`` (save/restore) and prose such as
#: ``"/slots single-flight"`` therefore never match.
_SLOTS_SEGMENT = re.compile(r"(?<![\w/])/slots(?:$|\?)")

#: The query component that makes a router ``/slots`` request resolvable.
_MODEL_PARAM = re.compile(r"[?&]model=")

#: URL builders whose endpoint argument is a URL construction (``_build_llama_url``).
_URL_BUILDER = re.compile(r"build.*url$", re.IGNORECASE)

_NO_MODEL_INLINE = "carries '?model=' inline"
_APPENDER_UNCONDITIONAL = "'{var}' is unconditionally extended with '?model='"
_NONE_REASON = "no '?model=' component on the URL and no model appender in scope"

#: Offending ``/slots`` constructions accepted until their slice removes them.
#: Keyed by ``(path relative to proxy/proxy, enclosing symbol)``; every value
#: names the follow-up slice that deletes the entry.
ALLOWLIST: dict[tuple[str, str], str] = {
    # ── slice 4 (LP-0MVU7ZZIA0092AXW): the model id is resolved from /models
    # for every caller, so these three constructors can drop their `if model:`
    # guard. The two debug-log URL strings inside the detail/progress helpers
    # (observability.py:734 and :802) are flagged under the same symbols.
    ("observability.py", "_query_slots"): (
        "LP-0MVU7ZZIA0092AXW (slice 4): '?model=' is appended only behind "
        "`if model:`, so a model-less request is reachable when the caller "
        "passes model=None. Slice 4 guarantees a resolved model id."
    ),
    ("observability.py", "_query_slots_detail"): (
        "LP-0MVU7ZZIA0092AXW (slice 4): same conditional appender as "
        "_query_slots (AC1 audit site #2)."
    ),
    ("observability.py", "_sample_slot_occupancy"): (
        "LP-0MVU7ZZIA0092AXW (slice 4): the slot-occupancy sampler "
        "(LP-0MVT1O0VG00905EL) appends '?model=' only behind `if model:`; "
        "slice 4 guarantees a resolved model id for every /slots caller."
    ),
    ("observability.py", "_query_slots_progress"): (
        "LP-0MVU7ZZIA0092AXW (slice 4): same conditional appender as "
        "_query_slots (AC1 audit site #3); _query_prefill_progress defaults "
        "model_name to None."
    ),
}

#: The follow-up slices allowed to own an allowlist entry.
_REMOVING_SLICES = ("LP-0MVU7ZZ7M002D9GT", "LP-0MVU7ZZIA0092AXW")
_SLICE_ID = re.compile(r"LP-[0-9A-Z]+")


@dataclass(frozen=True)
class Site:
    """One ``/slots`` URL construction found by the scan."""

    path: str
    """Module path relative to :data:`PROXY_DIR` (``backends/llama.py``)."""
    symbol: str
    """Enclosing symbol: ``func``, ``Class.method`` or ``<module>``."""
    lineno: int
    carries_model: bool
    """True when the URL carries ``?model=`` (inline or unconditional appender)."""
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.symbol)

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno} in {self.symbol}() — {self.reason}"


@dataclass(frozen=True)
class ScanResult:
    """Outcome of scanning a tree: every site found, plus the files read."""

    sites: tuple[Site, ...] = ()
    scanned: tuple[str, ...] = ()

    @property
    def violations(self) -> tuple[Site, ...]:
        return tuple(site for site in self.sites if not site.carries_model)

    def unallowlisted(self, allowlist=ALLOWLIST) -> list[Site]:
        """Offending sites with no allowlist entry — the fence's failure list."""
        return [site for site in self.violations if site.key not in allowlist]

    def stale_allowlist_entries(self, allowlist=ALLOWLIST) -> set[tuple[str, str]]:
        """Allowlist keys that match no current violation — rot in the list."""
        present = {site.key for site in self.violations}
        return {key for key in allowlist if key not in present}


def _string_segments(node: ast.AST) -> list[str]:
    """Literal string segments of a str/f-string/``+``-concatenation node."""
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else []
    if isinstance(node, ast.JoinedStr):
        return [
            value.value
            for value in node.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string_segments(node.left) + _string_segments(node.right)
    return []


def _has_slots_segment(node: ast.AST) -> bool:
    return any(_SLOTS_SEGMENT.search(text) for text in _string_segments(node))


def _has_model_param(node: ast.AST) -> bool:
    return any(_MODEL_PARAM.search(text) for text in _string_segments(node))


def _is_url_builder(func: ast.AST) -> bool:
    name = (
        func.attr
        if isinstance(func, ast.Attribute)
        else func.id
        if isinstance(func, ast.Name)
        else None
    )
    return bool(name and _URL_BUILDER.search(name))


def _url_variable(site: ast.AST, ancestors: tuple[ast.AST, ...]) -> str | None:
    """Name of the variable holding the URL *site* builds, if it is one.

    Only an assignment whose value the site flows into directly (optionally
    through a URL builder such as ``_build_llama_url``) counts. A URL handed
    straight to ``client.get(...)`` is not held by a variable, and the
    assignment its caller makes holds a *response*, not a URL.
    """
    for node in reversed(ancestors):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return None
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if getattr(node, "value", None) is None:
                return None
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            return target.id if isinstance(target, ast.Name) else None
        if isinstance(node, ast.Call):
            if _is_url_builder(node.func):
                continue
            return None
        if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Return, ast.Compare, ast.IfExp)):
            return None
    return None


def _walk_statements(
    node: ast.AST, conditional: bool = False
) -> list[tuple[ast.AST, bool]]:
    """Flatten *node*'s subtree, flagging statements nested inside an ``if``."""
    found: list[tuple[ast.AST, bool]] = []
    for child in ast.iter_child_nodes(node):
        child_conditional = conditional or isinstance(node, ast.If)
        found.append((child, child_conditional))
        found.extend(_walk_statements(child, child_conditional))
    return found


def _model_appender(scope: ast.AST | None, variable: str) -> str:
    """``"unconditional"``, ``"conditional"`` or ``"none"`` for *variable*."""
    if scope is None:
        return "none"
    verdict = "none"
    for node, conditional in _walk_statements(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        if getattr(node, "value", None) is None:
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not (isinstance(target, ast.Name) and target.id == variable):
            continue
        if not _has_model_param(node.value):
            continue
        if not conditional:
            return "unconditional"
        verdict = "conditional"
    return verdict


def _docstring(node: ast.AST) -> ast.Constant | None:
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value
    return None


class _Scanner:
    """Collect ``/slots`` construction sites from one module's AST."""

    def __init__(self, path: str):
        self.path = path
        self.sites: list[Site] = []

    def scan(self, tree: ast.AST) -> list[Site]:
        self._visit(tree, symbol=(), scope=None, doc=None, ancestors=())
        return self.sites

    # -- traversal ---------------------------------------------------------
    def _visit(
        self,
        node: ast.AST,
        symbol: tuple[str, ...],
        scope: ast.AST | None,
        doc: ast.Constant | None,
        ancestors: tuple[ast.AST, ...],
    ) -> None:
        if isinstance(node, ast.Module):
            module_doc = _docstring(node)
            for child in node.body:
                self._visit(child, symbol, node, module_doc, ancestors + (node,))
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{'.'.join(symbol)}.{node.name}" if symbol else node.name
            inner_doc = _docstring(node)
            for child in node.body:
                self._visit(child, (name,), node, inner_doc, ancestors + (node,))
            return

        is_string = isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        if is_string and node is not doc:
            # Constants inside an f-string are part of that f-string's site.
            nested_in_fstring = isinstance(node, ast.Constant) and any(
                isinstance(ancestor, ast.JoinedStr) for ancestor in ancestors
            )
            if not nested_in_fstring:
                self._record(node, symbol, scope, ancestors)
        for child in ast.iter_child_nodes(node):
            self._visit(child, symbol, scope, doc, ancestors + (node,))

    # -- classification ----------------------------------------------------
    def _record(
        self,
        node: ast.AST,
        symbol: tuple[str, ...],
        scope: ast.AST | None,
        ancestors: tuple[ast.AST, ...],
    ) -> None:
        if not _has_slots_segment(node):
            return
        name = ".".join(symbol) if symbol else "<module>"
        variable = _url_variable(node, ancestors)
        if _has_model_param(node):
            carries_model, reason = True, _NO_MODEL_INLINE
        else:
            appender = _model_appender(scope, variable) if variable else "none"
            if appender == "unconditional":
                carries_model = True
                reason = _APPENDER_UNCONDITIONAL.format(var=variable)
            elif appender == "conditional":
                carries_model = False
                reason = (
                    "model-less '/slots' URL: the only '?model=' appender for "
                    f"'{variable}' sits behind an `if` guard, so a model-less "
                    "request is reachable"
                )
            else:
                carries_model = False
                reason = _NONE_REASON
        self.sites.append(
            Site(
                path=self.path,
                symbol=name,
                lineno=node.lineno,
                carries_model=carries_model,
                reason=reason,
            )
        )


def scan_source(source: str, path: str = "<synthetic>") -> ScanResult:
    """Scan one module's *source*; *path* labels the sites in messages."""
    sites = _Scanner(path).scan(ast.parse(source, filename=path))
    return ScanResult(sites=tuple(sites), scanned=(path,))


def scan_tree(root: Path | str = PROXY_DIR) -> ScanResult:
    """Scan every ``**/*.py`` under *root* (default: the proxy package)."""
    root_path = Path(root)
    sites: list[Site] = []
    scanned: list[str] = []
    for file in sorted(root_path.rglob("*.py")):
        relative = file.relative_to(root_path).as_posix()
        sites.extend(scan_source(file.read_text(encoding="utf-8"), relative).sites)
        scanned.append(relative)
    return ScanResult(sites=tuple(sites), scanned=tuple(scanned))


#: The live tree, parsed once at import so the fence costs one AST pass.
_TREE = scan_tree()


class TestScanCoverage:
    """The scan must cover the whole package — and only real constructions."""

    def test_scans_every_proxy_module(self):
        expected = {
            file.relative_to(PROXY_DIR).as_posix()
            for file in PROXY_DIR.rglob("*.py")
        }
        assert set(_TREE.scanned) == expected
        assert len(_TREE.scanned) >= 40, "the proxy package scan looks truncated"
        assert "router_helpers.py" in _TREE.scanned
        assert "backends/llama.py" in _TREE.scanned

    def test_recognises_the_model_bearing_call_sites(self):
        """Inline ``?model=`` sites are found and pass — the scan is not blind."""
        compliant = {
            site.key
            for site in _TREE.sites
            if site.carries_model
        }
        assert ("handlers.py", "poll_slots_for_model") in compliant
        assert ("router_helpers.py", "_check_slot_availability") in compliant

    def test_docstrings_and_prose_are_not_construction_sites(self):
        source = '''
def helper(port):
    """Documented diagnostic: ``GET /slots?model=...`` and a bare /slots."""
    return "no url here"
'''
        assert scan_source(source, "synthetic_module.py").sites == ()

    def test_a_new_model_less_site_in_the_scanned_tree_fails_the_fence(self, tmp_path):
        """A temporary model-less ``/slots`` in a scanned file breaks the fence."""
        (tmp_path / "new_probe.py").write_text(
            "def probe(port):\n"
            '    return f"http://localhost:{port}/slots"\n',
            encoding="utf-8",
        )
        offenders = scan_tree(tmp_path).unallowlisted()
        assert len(offenders) == 1
        assert "new_probe.py:2" in str(offenders[0])
        assert "?model=" in str(offenders[0])

    def test_slot_save_restore_path_is_a_different_request_class(self):
        source = '''
def save(port, slot_id, action):
    url = f"http://localhost:{port}/slots/{slot_id}?action={action}"
    return url
'''
        assert scan_source(source, "synthetic_module.py").sites == ()


class TestLiveTreeFence:
    """The fence itself: nothing model-less beyond the documented allowlist."""

    def test_no_unallowlisted_model_less_slots_url(self):
        offenders = _TREE.unallowlisted()
        assert offenders == [], (
            "every /slots URL construction must carry '?model=' (add it, or "
            "document the site in ALLOWLIST with the slice that removes it):\n"
            + "\n".join(f"  - {site}" for site in offenders)
        )

    def test_allowlist_has_no_stale_entries(self):
        stale = _TREE.stale_allowlist_entries()
        assert stale == set(), (
            "ALLOWLIST entries with no matching model-less site (fixed or "
            f"renamed — delete them): {sorted(stale)}"
        )

    def test_allowlist_entries_name_the_slice_that_removes_them(self):
        for key, note in ALLOWLIST.items():
            slices = set(_SLICE_ID.findall(note))
            assert slices & set(_REMOVING_SLICES), (
                f"ALLOWLIST[{key}] must name the follow-up slice that removes "
                f"it, one of {_REMOVING_SLICES}: {note!r}"
            )


class TestSyntheticViolations:
    """A new model-less construction must fail with file:line and the reason."""

    def test_model_less_url_reports_file_line_and_reason(self):
        source = (
            "def probe(port):\n"
            '    url = f"http://localhost:{port}/slots"\n'
            "    return url\n"
        )
        result = scan_source(source, "synthetic_module.py")
        assert len(result.violations) == 1
        site = result.violations[0]
        assert (site.path, site.symbol, site.lineno) == (
            "synthetic_module.py",
            "probe",
            2,
        )
        message = str(site)
        assert "synthetic_module.py:2" in message
        assert "?model=" in message

    def test_model_less_url_passed_straight_to_a_client_is_detected(self):
        source = (
            "async def probe(client, host, port):\n"
            '    return await client.get(f"http://{host}:{port}/slots")\n'
        )
        result = scan_source(source, "synthetic_module.py")
        assert [site.symbol for site in result.violations] == ["probe"]
        assert "?model=" in result.violations[0].reason

    def test_conditional_appender_is_still_a_violation(self):
        source = (
            "def slots_url(port, model):\n"
            '    url = f"http://localhost:{port}/slots"\n'
            "    if model:\n"
            '        url = f"{url}?model={model}"\n'
            "    return url\n"
        )
        result = scan_source(source, "synthetic_module.py")
        assert len(result.violations) == 1
        assert "`if` guard" in result.violations[0].reason

    def test_inline_model_param_is_accepted(self):
        source = (
            "def slots_url(port, model):\n"
            '    return f"http://localhost:{port}/slots?model={model}"\n'
        )
        result = scan_source(source, "synthetic_module.py")
        assert len(result.sites) == 1
        assert result.violations == ()

    def test_unconditional_appender_is_accepted(self):
        source = (
            "def slots_url(port, model):\n"
            '    url = _build_llama_url(port, "/slots")\n'
            '    url = f"{url}?model={model}"\n'
            "    return url\n"
        )
        result = scan_source(source, "synthetic_module.py")
        assert len(result.sites) == 1
        assert result.violations == ()
        assert "unconditionally" in result.sites[0].reason


class TestAllowlistSelfTest:
    """The allowlist machinery cannot silently rot or over-allow."""

    def test_stale_allowlist_entry_is_detected(self):
        result = scan_source(
            "def slots_url(port, model):\n"
            '    return f"http://localhost:{port}/slots?model={model}"\n',
            "synthetic_module.py",
        )
        allowlist = {("synthetic_module.py", "probe"): "LP-0MVU7ZZ7M002D9GT"}
        assert result.unallowlisted(allowlist) == []
        assert result.stale_allowlist_entries(allowlist) == {
            ("synthetic_module.py", "probe")
        }

    def test_allowlisted_violation_is_excused_but_still_reported(self):
        source = 'def probe(port):\n    return f"http://localhost:{port}/slots"\n'
        result = scan_source(source, "synthetic_module.py")
        allowlist = {("synthetic_module.py", "probe"): "LP-0MVU7ZZ7M002D9GT"}
        assert len(result.violations) == 1
        assert result.unallowlisted(allowlist) == []
        assert result.stale_allowlist_entries(allowlist) == set()


class TestScanCost:
    """The fence is cheap enough to run on every changed-scope gate."""

    def test_scan_is_offline_and_under_one_second(self):
        started = time.perf_counter()
        result = scan_tree()
        elapsed = time.perf_counter() - started
        assert result.scanned, "the scan read no files"
        assert elapsed < 1.0, f"scan took {elapsed:.2f}s (budget 1.0s)"
