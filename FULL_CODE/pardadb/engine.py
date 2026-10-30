"""Query-time masking engine.

1. parse the SQL (sqlglot) in the backend's dialect, expand SELECT * against the live schema
2. inference guard: refuse writes, unsafe functions, and filters/sorts/joins on columns the role cannot see
3. add LIMIT when missing, execute read-only
4. trace each output column to its source columns (lineage) and mask it with the strongest strategy
5. write a hash-chained audit record
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from typing import Any, TypedDict

import sqlglot
from sqlglot import exp
from sqlglot.lineage import Node, lineage
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope
from sqlglot.schema import MappingSchema

from . import audit, backends, config, detectors, masks, ml, scanner

log = logging.getLogger(__name__)
DB_ERRORS = backends.driver_errors()

ROLES = ["admin", "support", "analyst"]
PLAN_CACHE_SIZE = 256  # (SQL text, role) plans remembered per engine
ORDER = [
    "REDACT",
    "BUCKET",
    "GENERALIZE",
    "HASH",
    "JSON_REDACT",
    "SPANS_TYPED",
    "JSON_PARTIAL",
    "SPANS_PARTIAL",
    "PARTIAL",
    "FULL",
]
ORACLE_NODES = (exp.Case, exp.If, exp.Nullif, exp.Div, exp.Mod)
BANNED_FUNCS = {
    "LOAD_FILE",
    "SLEEP",
    "BENCHMARK",
    "LOAD_EXTENSION",
    "READFILE",
    "WRITEFILE",
    "SYS_EXEC",
    "SYS_EVAL",
    "PG_READ_FILE",
    "PG_SLEEP",
    "LO_IMPORT",
    "LO_EXPORT",
    "DBLINK",
    # PostgreSQL: whole-table / query export, server file and setting reads, timing
    "TABLE_TO_XML",
    "TABLE_TO_XMLSCHEMA",
    "TABLE_TO_XML_AND_XMLSCHEMA",
    "QUERY_TO_XML",
    "QUERY_TO_XMLSCHEMA",
    "QUERY_TO_XML_AND_XMLSCHEMA",
    "CURSOR_TO_XML",
    "SCHEMA_TO_XML",
    "DATABASE_TO_XML",
    "PG_READ_BINARY_FILE",
    "PG_LS_DIR",
    "PG_STAT_FILE",
    "PG_LS_LOGDIR",
    "PG_LS_WALDIR",
    "PG_LS_TMPDIR",
    "PG_LS_ARCHIVE_STATUSDIR",
    "LO_GET",
    "CURRENT_SETTING",
    "PG_SLEEP_FOR",
    "PG_SLEEP_UNTIL",
}


ColumnInfo = dict[str, Any]  # one scanner catalog entry: table, column, tag, entity, json_keys, ...
SourceColumn = tuple[str, str]  # (table, column)


class PlanEntry(TypedDict):
    """Masking decision for one output column of a query."""

    name: str
    strategy: str
    info: ColumnInfo | None
    source: list[str]


class Blocked(Exception):
    """The query was refused (write, banned function or inference-guard violation)."""


WRITE_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter)
READ_ONLY_MSG = (
    "write statements are blocked on bank data: PardaDB is read-only by design "
    "(create your own test tables in the Playground database)"
)


def parse_single(sql: str, dialect: str) -> exp.Expression:
    """Parse exactly one SQL statement. Trailing semicolons and comment-only fragments are ignored.

    Raises ValueError on a syntax error or empty input, and Blocked when the input holds more
    than one statement: stacked queries ("SELECT ...; DROP TABLE ...") are refused whole.
    """
    try:
        parsed = sqlglot.parse(sql, read=dialect)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as e:
        raise ValueError(f"SQL syntax error: {str(e).splitlines()[0]}") from None
    statements = [s for s in parsed if s is not None and not isinstance(s, exp.Semicolon)]
    if len(statements) > 1:
        raise Blocked(f"only one statement per query: found {len(statements)}; run them one at a time")
    if not statements:
        raise ValueError("SQL syntax error: no statement to run")
    return statements[0]


def audit_sql(sql: str, dialect: str) -> str:
    """The statement as written to the audit log: same shape, every literal replaced by ``?``.

    Values typed into a query (``WHERE pan_no = 'ABCPK0000F'``) are personal data themselves, so the
    log keeps what was asked, not the values it was asked about.
    """
    try:
        statements = [t for t in sqlglot.parse(sql, read=dialect) if t is not None]
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
        return "<statement that did not parse>"
    return "; ".join(
        t.transform(lambda n: exp.Placeholder() if isinstance(n, exp.Literal) else n).sql(dialect=dialect)
        for t in statements
    )


def is_write(sql: str, dialect: str) -> bool:
    """True when `sql` is one statement that changes data or schema.

    Input that does not parse, or holds several statements, is not called a write here: the engine
    refuses it with a proper reason when it runs.
    """
    try:
        tree = parse_single(sql, dialect)
    except (ValueError, Blocked):
        return False
    return isinstance(tree, WRITE_NODES) or tree.find(*WRITE_NODES, exp.Command) is not None


def match_identifier_case(tree: exp.Expression, schema: backends.Schema) -> exp.Expression:
    """Write unquoted table and column names with the schema's exact spelling, quoted.

    SQLite accepts ``card_no`` for a ``CARD_NO`` column, PostgreSQL folds unquoted names to lower case,
    and sqlglot resolves names case-sensitively; quoting the real spelling lets one query work on every
    backend. A name is rewritten only when it matches exactly, or when exactly one table (or exactly one
    column among the query's tables) matches case-insensitively; anything ambiguous is left for
    qualify() to reject.
    """
    tables_ci: dict[str, list[str]] = {}
    for table in schema:
        tables_ci.setdefault(table.lower(), []).append(table)
    aliases: dict[str, str] = {}
    for node in tree.find_all(exp.Table):
        ident = node.this
        if isinstance(ident, exp.Identifier) and not ident.quoted:
            match = [ident.name] if ident.name in schema else tables_ci.get(ident.name.lower(), [])
            if len(match) == 1:
                node.set("this", exp.to_identifier(match[0], quoted=True))
        if node.name in schema:
            aliases[node.alias_or_name.lower()] = node.name
    in_query = set(aliases.values())
    for col in tree.find_all(exp.Column):
        ident = col.this
        if not isinstance(ident, exp.Identifier) or ident.quoted:
            continue
        scope = [aliases[col.table.lower()]] if col.table and col.table.lower() in aliases else in_query
        names = {c for t in scope for c in schema[t]}
        match = {ident.name} if ident.name in names else {c for c in names if c.lower() == ident.name.lower()}
        if len(match) == 1:
            col.set("this", exp.to_identifier(match.pop(), quoted=True))
    return tree


def _check_read_only(tree: exp.Expression) -> None:
    """Refuse writes, commands, SELECT ... INTO and banned functions anywhere in the tree."""
    if isinstance(tree, WRITE_NODES) or tree.find(*WRITE_NODES) or tree.find(exp.Command):
        raise Blocked(READ_ONLY_MSG)
    if not (isinstance(tree, (exp.Select, exp.SetOperation)) or tree.find(exp.Select)):
        raise Blocked("only read queries are allowed through PardaDB")
    if tree.find(exp.Into):
        raise Blocked("SELECT ... INTO is blocked")
    for f in tree.find_all(exp.Func):
        name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).upper()
        if name in BANNED_FUNCS:
            raise Blocked(f"function {name} is blocked")


def _sqlite_json_call(node: exp.Expression) -> exp.Expression:
    """SQLite before 3.38 has no -> / ->> operators: spell JSON extraction as json_extract()."""
    if isinstance(node, (exp.JSONExtract, exp.JSONExtractScalar)):
        return exp.Anonymous(this="json_extract", expressions=[node.this, node.expression, *node.expressions])
    return node


def render(tree: exp.Expression, dialect: str) -> str:
    """SQL text for the backend: the tree in its dialect, with constructs older engines reject rewritten."""
    if dialect == "sqlite":
        tree = tree.transform(_sqlite_json_call)
    return tree.sql(dialect=dialect)


def load_policy() -> dict[str, Any]:
    """Role x tag masking policy from policy.json."""
    return json.loads(config.POLICY_PATH.read_text())


def _source_columns(node: Node) -> set[SourceColumn]:
    out: set[SourceColumn] = set()
    for n in node.walk():
        if not n.downstream and isinstance(n.source, exp.Table):
            out.add((n.source.name, n.name.split(".")[-1].strip('"`')))
    return out


def _is_direct(node: Node) -> bool:
    """True when every hop from output to source is a bare column or a rename."""
    for n in node.walk():
        e = n.expression
        inner = e.this if isinstance(e, exp.Alias) else e
        if not isinstance(inner, (exp.Column, exp.Table)):
            return False
    return True


class Engine:
    """Plans, guards, executes and masks queries against one backend."""

    def __init__(
        self,
        backend: backends.Backend,
        catalog: dict[str, Any] | None,
        policy: dict[str, Any] | None = None,
        names: detectors.NameIndex | None = None,
    ) -> None:
        self.b = backend
        self.catalog = catalog or {"columns": []}
        self.idx = scanner.index(self.catalog)
        self.policy = policy or load_policy()
        self.names = names
        self.schema = {t: {c: "TEXT" for c in cols} for t, cols in backend.schema().items()}
        # PostgreSQL keeps quoted upper-case names (cards."CARD_NO"); never fold their case
        self.pg = backend.dialect == "postgres"
        # Built once per engine: sqlglot would otherwise re-parse every table and column name on
        # each qualify() / lineage() call. A schema change installs a new Engine (Runtime.install).
        self.sg_schema = MappingSchema(self.schema, dialect=backend.dialect, normalize=not self.pg)
        self._plans: collections.OrderedDict[tuple[str, str], tuple[str, list[PlanEntry]]] = collections.OrderedDict()
        self._plans_lock = threading.Lock()  # the HTTP server runs queries on many threads

    def _lineage(self, name: str, expression: exp.Expression, scope: Scope | None = None) -> Node:
        col = exp.to_identifier(name, quoted=True) if self.pg else name
        # trim_selects=False: skip sqlglot's per-column copy of the SELECT (only used for its graph labels)
        return lineage(col, expression, schema=self.sg_schema, dialect=self.b.dialect, scope=scope, trim_selects=False)

    def _output_lineage(self, tree: exp.Expression) -> list[tuple[str, Node]]:
        """Lineage of every output column of `tree`, qualifying it and building its scope only once.

        Same result as one lineage() call per column, which would re-qualify the whole tree each time.
        """
        names = tree.named_selects
        if not names:
            return []
        tree = qualify(
            tree, schema=self.sg_schema, dialect=self.b.dialect, validate_qualify_columns=False, identify=False
        )
        scope = build_scope(tree)
        return [(name, self._lineage(name, tree, scope)) for name in names]

    # ----------------------------------------------------------- policy lookup
    def info(self, table: str, col: str) -> ColumnInfo | None:
        """Catalog entry for a tagged column, or None when it holds no PII."""
        return self.idx.get((table.lower(), col.lower()))

    def strategy(self, role: str, info: ColumnInfo | None) -> str:
        """Masking strategy `role` gets for a column: entity overrides win over the tag default."""
        if not info:
            return "FULL"
        over = self.policy.get("entity_overrides", {}).get(info.get("entity") or "", {})
        return over.get(role) or self.policy["tags"][info["tag"]][role]

    # ----------------------------------------------------------- inference guard
    def _resolve(self, col: exp.Column, scope: Scope) -> set[SourceColumn]:
        if not col.table and col.name in scope.expression.named_selects:
            # ORDER BY / GROUP BY an output alias, e.g. ORDER BY n where n = COUNT(*)
            return _source_columns(self._lineage(col.name, scope.expression))
        src = scope.sources.get(col.table)
        if isinstance(src, exp.Table):
            return {(src.name, col.name)}
        if isinstance(src, Scope):
            try:
                return _source_columns(self._lineage(col.name, src.expression))
            except (sqlglot.errors.SqlglotError, KeyError, ValueError) as e:
                log.info("lineage failed for %s: %s", col.sql(), e)  # fall through: fail closed
        raise Blocked(f"cannot resolve column {col.sql()} for the inference guard; query refused")

    def _base_hash(self, col: exp.Column, scope: Scope, role: str) -> bool:
        src = scope.sources.get(col.table)
        if not isinstance(src, exp.Table):
            return False
        return self.strategy(role, self.info(src.name, col.name)) == "HASH"

    def _hash_safe(self, col: exp.Column, clause: exp.Expression, scope: Scope, role: str) -> bool:
        if not self._base_hash(col, scope, role):
            return False
        if isinstance(clause, exp.Group) and col.parent is clause:
            return True
        if isinstance(clause, exp.Join) and isinstance(col.parent, exp.EQ):
            other = col.parent.expression if col.parent.this is col else col.parent.this
            return isinstance(other, exp.Column) and self._base_hash(other, scope, role)
        return False

    def _guard(self, tree: exp.Expression, role: str) -> None:
        for setop in tree.find_all(exp.SetOperation):
            order = setop.args.get("order")
            outputs = setop.named_selects
            for item in order.expressions if order else []:
                e = item.this if isinstance(item, exp.Ordered) else item
                if isinstance(e, exp.Literal) and e.is_int and 1 <= int(e.this) <= len(outputs):
                    name = outputs[int(e.this) - 1]
                elif isinstance(e, exp.Column) and not e.table and e.name in outputs:
                    name = e.name
                else:
                    raise Blocked(f"ORDER BY {e.sql()} on a set operation must name an output column; query refused")
                for t, c in _source_columns(self._lineage(name, setop)):
                    i = self.info(t, c)
                    if self.strategy(role, i) != "FULL":
                        raise Blocked(f"inference guard: role '{role}' cannot sort on {t}.{c} ({i['tag']})")
        scopes = list(build_scope(tree).traverse())
        covered = {id(sc.expression) for sc in scopes}
        if any(id(sel) not in covered for sel in tree.find_all(exp.Select)):
            raise Blocked("query contains a SELECT the inference guard cannot analyse; query refused")
        for scope in scopes:
            for clause in (exp.Where, exp.Having, exp.Order, exp.Group, exp.Join, exp.Window, exp.Qualify):
                for node in scope.expression.find_all(clause):
                    for c in node.find_all(exp.Column):
                        owner = c.find_ancestor(exp.Select)
                        if owner is None or owner is not scope.expression:
                            continue
                        if isinstance(c.parent, exp.Is) and isinstance(c.parent.expression, exp.Null):
                            continue  # IS [NOT] NULL reveals only whether a value exists
                        for t, cname in self._resolve(c, scope):
                            i = self.info(t, cname)
                            strat = self.strategy(role, i)
                            if strat == "FULL" or (strat == "HASH" and self._hash_safe(c, node, scope, role)):
                                continue
                            verb = {
                                exp.Where: "filter",
                                exp.Having: "filter",
                                exp.Order: "sort",
                                exp.Group: "group",
                                exp.Join: "join",
                            }.get(clause, "use")
                            raise Blocked(
                                f"inference guard: role '{role}' cannot {verb} on {t}.{cname} "
                                f"({i['tag']}); that would reveal masked values"
                            )

    # ----------------------------------------------------------- planning
    def plan(self, sql: str, role: str) -> tuple[exp.Expression, list[PlanEntry]]:
        """Validate `sql` for `role`; return the qualified, LIMITed tree and one masking decision per output."""
        tree = parse_single(sql, self.b.dialect)
        _check_read_only(tree)
        tree = match_identifier_case(tree, self.b.schema())
        try:
            tree = qualify(tree, schema=self.sg_schema, dialect=self.b.dialect)
        except (sqlglot.errors.SqlglotError, KeyError, ValueError) as e:
            raise ValueError(f"cannot resolve query against the schema: {str(e).splitlines()[0]}") from None
        for node in [tree, *tree.find_all(exp.Select, exp.SetOperation)]:
            names = [n.lower() for n in node.named_selects]
            if len(names) != len(set(names)):
                raise Blocked("duplicate output column names; give every output a unique alias")
        if role != "admin":
            self._guard(tree, role)
            if any(not isinstance(lim.expression, exp.Literal) for lim in tree.find_all(exp.Limit, exp.Offset)):
                raise Blocked("inference guard: LIMIT and OFFSET must be plain numbers")
        if isinstance(tree, (exp.Select, exp.SetOperation)) and not tree.args.get("limit"):
            tree = tree.limit(config.MAX_ROWS)
        plan: list[PlanEntry] = []
        for name, node in self._output_lineage(tree):
            srcs = _source_columns(node)
            infos = [i for i in (self.info(t, c) for t, c in srcs) if i]
            if not infos:
                if role != "admin" and node.expression.find(exp.Anonymous):
                    raise Blocked(f"output '{name}' calls a function PardaDB cannot trace to a column; query refused")
                plan.append(
                    {"name": name, "strategy": "FULL", "info": None, "source": sorted(f"{t}.{c}" for t, c in srcs)}
                )
                continue
            strat = min((self.strategy(role, i) for i in infos), key=ORDER.index)
            info = min(infos, key=lambda i: ORDER.index(self.strategy(role, i)))
            if strat != "FULL" and not (len(srcs) == 1 and _is_direct(node)):
                if node.expression.find(*ORACLE_NODES):
                    # CASE / IF / NULLIF / division turn a masked value into NULL-or-not or an error: a 1-bit oracle
                    raise Blocked(
                        f"inference guard: role '{role}' cannot branch or divide on masked column(s) "
                        f"{', '.join(sorted(f'{t}.{c}' for t, c in srcs))}"
                    )
                strat = "REDACT"
            plan.append({"name": name, "strategy": strat, "info": info, "source": sorted(f"{t}.{c}" for t, c in srcs)})
        return tree, plan

    def planned(self, sql: str, role: str) -> tuple[str, list[PlanEntry]]:
        """plan() rendered as backend SQL, remembered per (SQL text, role) in a bounded LRU.

        A plan depends only on the SQL, the role and this engine's schema, catalog and policy. Those
        are fixed for the engine's life: a schema change or re-scan installs a new Engine
        (Runtime.install), which starts with an empty cache. Blocked and invalid queries are not cached.
        """
        key = (sql, role)
        with self._plans_lock:
            hit = self._plans.get(key)
            if hit is not None:
                self._plans.move_to_end(key)
                return hit
        tree, plan = self.plan(sql, role)
        hit = (render(tree, self.b.dialect), plan)
        with self._plans_lock:
            self._plans[key] = hit
            if len(self._plans) > PLAN_CACHE_SIZE:
                self._plans.popitem(last=False)
        return hit

    # ----------------------------------------------------------- execution
    def run(self, sql: str, role: str, user: str | None = None, record: bool = True) -> dict[str, Any]:
        """Execute `sql` as `role` and return the masked result, or why it was blocked or failed."""
        t0 = time.perf_counter()
        base = {"user": user, "role": role, "backend": self.b.name, "sql": sql}
        if getattr(self.b, "writable", False):
            res = self.write(sql, base, t0, record)
            if res is not None:
                return res
        try:
            query, plan = self.planned(sql, role)
        except Blocked as e:
            if record:
                self._audit(base, status="BLOCKED", reason=str(e))
            return {
                **base,
                "ok": False,
                "blocked": True,
                "reason": str(e),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        except ValueError as e:
            return {
                **base,
                "ok": False,
                "blocked": False,
                "error": str(e),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        try:
            cols, rows = self.b.execute(query, limit=config.MAX_ROWS)
        except DB_ERRORS as e:
            log.warning("query failed on %s: %s", self.b.name, e)
            return {**base, "ok": False, "blocked": False, "error": f"database error: {str(e).splitlines()[0]}"}
        ctx = {"names": self.names}
        masked_cells = 0
        out, marks = [], {}
        text_cols = [ci for ci, p in enumerate(plan) if (p["info"] or {}).get("tag") == "PII_FREE_TEXT"]
        ai = {}
        if text_cols and ml.client.ready:
            ai = ml.client.spans_for({r[ci] for r in rows for ci in text_cols if isinstance(r[ci], str) and r[ci]})
        for ri, r in enumerate(rows):
            row = []
            for ci, (v, p) in enumerate(zip(r, plan, strict=False)):
                text_col = (p["info"] or {}).get("tag") == "PII_FREE_TEXT" and isinstance(v, str)
                if text_col and p["strategy"] in ("SPANS_PARTIAL", "SPANS_TYPED"):
                    mv, m = masks.mask_text(
                        v, p["strategy"] == "SPANS_TYPED", self.names, with_marks=True, extra=ai.get(v)
                    )
                    if m:
                        marks[f"{ri}:{ci}"] = [[a, b, ent, 1] for a, b, ent in m]
                else:
                    mv = masks.apply(p["strategy"], v, p["info"], ctx)
                    if text_col and p["strategy"] == "FULL":
                        found = detectors.find_spans(v, self.names, ai.get(v))
                        if found:  # Admin: show what would be masked
                            marks[f"{ri}:{ci}"] = [[a, b, ent, 0] for a, b, ent in found]
                masked_cells += mv != v
                row.append(mv)
            out.append(row)
        masked = {p["name"]: p["strategy"] for p in plan if p["strategy"] != "FULL"}
        if record:
            self._audit(base, status="OK", rows=len(out), masked=masked)
        return {
            **base,
            "ok": True,
            "columns": [
                {
                    "name": p["name"],
                    "strategy": masks.DISPLAY.get(p["strategy"], p["strategy"]),
                    "tag": (p["info"] or {}).get("tag"),
                    "entity": (p["info"] or {}).get("entity"),
                    "source": list(p["source"]),  # a copy: the plan itself may be cached
                }
                for p in plan
            ],
            "rows": out,
            "marks": marks,
            "ai": bool(ai),
            "row_count": len(out),
            "truncated": len(out) >= config.MAX_ROWS,
            "masked_cells": masked_cells,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _audit(self, base: dict[str, Any], **fields: Any) -> None:
        """Append one audit record; the SQL is stored with its literal values removed."""
        audit.append({**base, "sql": audit_sql(base["sql"], self.b.dialect), **fields})

    def write(self, sql: str, base: dict[str, Any], t0: float, record: bool) -> dict[str, Any] | None:
        """Playground only: run CREATE / INSERT / UPDATE / DELETE / ALTER / DROP. Returns None for reads."""
        try:
            tree = parse_single(sql, self.b.dialect)
        except (Blocked, ValueError):
            return None  # plan() reports the syntax error or the stacked statements, and audits the block
        if not isinstance(tree, WRITE_NODES):
            return None
        try:
            n = self.b.write(sql)
        except DB_ERRORS as e:
            log.warning("playground write failed: %s", e)
            return {**base, "ok": False, "blocked": False, "error": f"database error: {str(e).splitlines()[0]}"}
        if record:
            self._audit(base, status="WRITE", rows=n)
        msg = (
            f"{tree.key.upper()} done on the playground: {max(n, 0)} row(s) affected. "
            "PardaDB is re-scanning the playground, so new tables and columns get tagged."
        )
        return {
            **base,
            "ok": True,
            "write": True,
            "affected": n,
            "columns": [{"name": "result", "strategy": "FULL", "tag": None, "entity": None, "source": []}],
            "rows": [[msg]],
            "marks": {},
            "row_count": 1,
            "masked_cells": 0,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def browse(
        self,
        table: str,
        role: str,
        user: str | None = None,
        offset: int = 0,
        limit: int = 25,
        filter_col: str | None = None,
        filter_val: str | None = None,
    ) -> dict[str, Any]:
        """Page through a table as `role`. Filters go through the same inference guard."""
        if table not in self.schema:
            return {"ok": False, "blocked": False, "error": f"unknown table {table}"}
        q = exp.select("*").from_(exp.to_table(table, quoted=True))
        if filter_col:
            if filter_col not in self.schema[table]:
                return {"ok": False, "blocked": False, "error": f"unknown column {filter_col}"}
            col = exp.column(filter_col, quoted=True)
            val = exp.Literal.string(str(filter_val or ""))
            q = q.where(
                exp.Like(this=col, expression=val) if "%" in str(filter_val) else exp.EQ(this=col, expression=val)
            )
        count_sql = exp.select(exp.Count(this=exp.Star())).from_(q.subquery("t")).sql(dialect=self.b.dialect)
        res = self.run(q.limit(int(limit)).offset(int(offset)).sql(dialect=self.b.dialect), role, user)
        if res.get("ok"):
            try:
                res["total"] = self.b.execute(count_sql)[1][0][0]
            except DB_ERRORS as e:
                log.warning("row count failed for %s: %s", table, e)
                res["total"] = None
        res.update(offset=int(offset), limit=int(limit), table=table)
        return res
