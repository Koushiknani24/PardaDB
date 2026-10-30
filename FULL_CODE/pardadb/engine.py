"""Query-time masking engine.

1. parse the SQL (sqlglot) in the backend's dialect, expand SELECT * against the live schema
2. inference guard: refuse writes, unsafe functions, and filters/sorts/joins on columns the role cannot see
3. add LIMIT when missing, execute read-only
4. trace each output column to its source columns (lineage) and mask it with the strongest strategy
5. write a hash-chained audit record
"""

from __future__ import annotations

import json
import logging
import time

import sqlglot
from sqlglot import exp
from sqlglot.lineage import lineage
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope
from sqlglot.schema import MappingSchema

from . import audit, backends, config, detectors, masks, ml, scanner

log = logging.getLogger(__name__)
DB_ERRORS = backends.driver_errors()

ROLES = ["admin", "support", "analyst"]
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
}


class Blocked(Exception):
    pass


def load_policy() -> dict:
    return json.loads(config.POLICY_PATH.read_text())


def _source_columns(node):
    out = set()
    for n in node.walk():
        if not n.downstream and isinstance(n.source, exp.Table):
            out.add((n.source.name, n.name.split(".")[-1].strip('"`')))
    return out


def _is_direct(node):
    """True when every hop from output to source is a bare column or a rename."""
    for n in node.walk():
        e = n.expression
        inner = e.this if isinstance(e, exp.Alias) else e
        if not isinstance(inner, (exp.Column, exp.Table)):
            return False
    return True


class Engine:
    def __init__(self, backend, catalog: dict | None, policy: dict | None = None, names=None):
        self.b = backend
        self.catalog = catalog or {"columns": []}
        self.idx = scanner.index(self.catalog)
        self.policy = policy or load_policy()
        self.names = names
        self.schema = {t: {c: "TEXT" for c in cols} for t, cols in backend.schema().items()}
        # PostgreSQL keeps quoted upper-case names (cards."CARD_NO"); never fold their case
        self.pg = backend.dialect == "postgres"
        self.sg_schema = (
            MappingSchema(self.schema, dialect=backend.dialect, normalize=False) if self.pg else self.schema
        )

    def _lineage(self, name, expression):
        col = exp.to_identifier(name, quoted=True) if self.pg else name
        return lineage(col, expression, schema=self.sg_schema, dialect=self.b.dialect)

    # ----------------------------------------------------------- policy lookup
    def info(self, table, col):
        return self.idx.get((table.lower(), col.lower()))

    def strategy(self, role, info):
        if not info:
            return "FULL"
        over = self.policy.get("entity_overrides", {}).get(info.get("entity") or "", {})
        return over.get(role) or self.policy["tags"][info["tag"]][role]

    # ----------------------------------------------------------- inference guard
    def _resolve(self, col, scope):
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

    def _base_hash(self, col, scope, role):
        src = scope.sources.get(col.table)
        if not isinstance(src, exp.Table):
            return False
        return self.strategy(role, self.info(src.name, col.name)) == "HASH"

    def _hash_safe(self, col, clause, scope, role):
        if not self._base_hash(col, scope, role):
            return False
        if isinstance(clause, exp.Group) and col.parent is clause:
            return True
        if isinstance(clause, exp.Join) and isinstance(col.parent, exp.EQ):
            other = col.parent.expression if col.parent.this is col else col.parent.this
            return isinstance(other, exp.Column) and self._base_hash(other, scope, role)
        return False

    def _guard(self, tree, role):
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
    def plan(self, sql: str, role: str):
        try:
            tree = sqlglot.parse_one(sql, read=self.b.dialect)
        except sqlglot.errors.ParseError as e:
            raise ValueError(f"SQL syntax error: {str(e).splitlines()[0]}") from None
        if tree is not None and (
            isinstance(tree, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter))
            or tree.find(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter)
        ):
            raise Blocked(
                "write statements are blocked on bank data: PardaDB is read-only by design "
                "(create your own test tables in the Playground database)"
            )
        if tree is None or not (isinstance(tree, (exp.Select, exp.SetOperation)) or tree.find(exp.Select)):
            raise Blocked("only read queries are allowed through PardaDB")
        if tree.find(exp.Command):
            raise Blocked(
                "write statements are blocked on bank data: PardaDB is read-only by design "
                "(create your own test tables in the Playground database)"
            )
        if tree.find(exp.Into):
            raise Blocked("SELECT ... INTO is blocked")
        for f in tree.find_all(exp.Func):
            name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).upper()
            if name in BANNED_FUNCS:
                raise Blocked(f"function {name} is blocked")
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
        if isinstance(tree, (exp.Select, exp.SetOperation)) and not tree.args.get("limit"):
            tree = tree.limit(config.MAX_ROWS)
        plan = []
        for name in tree.named_selects:
            node = self._lineage(name, tree)
            srcs = _source_columns(node)
            infos = [i for i in (self.info(t, c) for t, c in srcs) if i]
            if not infos:
                plan.append(
                    {"name": name, "strategy": "FULL", "info": None, "source": sorted(f"{t}.{c}" for t, c in srcs)}
                )
                continue
            strat = min((self.strategy(role, i) for i in infos), key=ORDER.index)
            info = min(infos, key=lambda i: ORDER.index(self.strategy(role, i)))
            if strat != "FULL" and not (len(srcs) == 1 and _is_direct(node)):
                strat = "REDACT"
            plan.append({"name": name, "strategy": strat, "info": info, "source": sorted(f"{t}.{c}" for t, c in srcs)})
        return tree, plan

    # ----------------------------------------------------------- execution
    def run(self, sql: str, role: str, user: str | None = None, record: bool = True) -> dict:
        t0 = time.perf_counter()
        base = {"user": user, "role": role, "backend": self.b.name, "sql": sql}
        if getattr(self.b, "writable", False):
            res = self.write(sql, base, t0, record)
            if res is not None:
                return res
        try:
            tree, plan = self.plan(sql, role)
        except Blocked as e:
            if record:
                audit.append({**base, "status": "BLOCKED", "reason": str(e)})
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
            cols, rows = self.b.execute(tree.sql(dialect=self.b.dialect), limit=config.MAX_ROWS)
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
            audit.append({**base, "status": "OK", "rows": len(out), "masked": masked})
        return {
            **base,
            "ok": True,
            "columns": [
                {
                    "name": p["name"],
                    "strategy": masks.DISPLAY.get(p["strategy"], p["strategy"]),
                    "tag": (p["info"] or {}).get("tag"),
                    "entity": (p["info"] or {}).get("entity"),
                    "source": p["source"],
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

    def write(self, sql, base, t0, record):
        """Playground only: run CREATE / INSERT / UPDATE / DELETE / ALTER / DROP. Returns None for reads."""
        try:
            tree = sqlglot.parse_one(sql, read=self.b.dialect)
        except sqlglot.errors.ParseError as e:
            return {**base, "ok": False, "blocked": False, "error": f"SQL syntax error: {str(e).splitlines()[0]}"}
        if not isinstance(tree, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Alter, exp.Drop)):
            return None
        try:
            n = self.b.write(sql)
        except DB_ERRORS as e:
            log.warning("playground write failed: %s", e)
            return {**base, "ok": False, "blocked": False, "error": f"database error: {str(e).splitlines()[0]}"}
        if record:
            audit.append({**base, "status": "WRITE", "rows": n})
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

    def browse(self, table: str, role: str, user=None, offset=0, limit=25, filter_col=None, filter_val=None) -> dict:
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
