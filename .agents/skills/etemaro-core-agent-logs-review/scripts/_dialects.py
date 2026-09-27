#!/usr/bin/env python3
"""Dialect-aware readers for the Etemaro **core agent** log streams.

Scope: this module reads only the core-agent family produced by
``packages/core`` and written under ``<instance>/logs/``. The copy-trader CLI
bot (``apps/bot-copy-trader``) is a different family with different dialects and
is covered by the sibling skill ``etemaro-copy-trader-logs-review`` — its
streams must not be parsed here.

Dialects
--------
======================== ================================================ ==========================================================
dialect                  location                                          shape
======================== ================================================ ==========================================================
``coreagent_runtime``    ``<inst>/logs/agent-<agentId>-<date>.log``        text ``[ts] [component] [agentId] msg``
``coreagent_actions``    ``<inst>/logs/actions-<agentId>-<date>.jsonl``    JSONL ``{ts,agentId,dryRun,tool,args,result,success?}``
``coreagent_structured`` ``<inst>/logs/structured-<agentId>-<date>.jsonl``  JSONL ``{ts,category,agentId,dryRun,message,metadata}``
``jsonl_unclassified``   anywhere                                          JSONL with ``ts`` but no known signature
======================== ================================================ ==========================================================

Why every reader must come through here
---------------------------------------
Reading the wrong stream with the right probe fails *silently*: a swallowed
``json.loads`` plus a ``"ts"`` regex that returns ``None`` yields an empty
result that is indistinguishable from "nothing happened". The three core-agent
streams carry the same conceptual data under three incompatible shapes, and the
error signal lives in exactly one of them:

* ``agent-*.log`` — human narrative. Its bracket is a **component tag, not a
  severity level**; it contains almost no error information.
* ``structured-*.jsonl`` — where ``api_error`` / ``swap_error`` / ``tx_error``
  actually are.
* ``actions-*.jsonl`` — per-tool audit trail; ``success: false`` is the failure
  signal.

An unrecognised JSONL record stays visible as ``jsonl_unclassified`` rather than
being mapped onto a dialect whose field names will not resolve (that would yield
``None`` keys and quietly corrupt every count built from them). Lines matching
no dialect (multi-line LLM output, pretty-printed JSON, stack traces) attach to
the preceding event as ``Event.continuation`` and are counted in
``coverage()["continuations"]``, never dropped.

Stdlib only.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

# [2026-09-12T07:53:58.883Z] [startup] [agent-default] DLMM LP Agent starting...
CORE_RUNTIME_RE = re.compile(
    r'^\[(?P<ts>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z?)\]\s*'
    r'\[(?P<component>[^\]]+)\]\s*'
    r'\[(?P<agent>[^\]]+)\]\s*'
    r'(?P<msg>.*)$'
)

#: log file names this family owns, glob-relative to ``<instance>/logs/``
LOG_GLOBS = ('agent-*.log', 'actions-*.jsonl', 'structured-*.jsonl')

DIALECTS = (
    'coreagent_runtime',
    'coreagent_actions',
    'coreagent_structured',
    'jsonl_unclassified',
)


@dataclass
class Event:
    """One normalized log event. ``fields`` keeps the raw record, losslessly.

    ``component`` holds the runtime component tag, the structured ``category``
    or the action ``tool`` — one uniform axis for grouping. ``level`` is only
    ever set when the stream actually carries a severity (i.e. a failed action);
    it is never inferred from the runtime bracket.
    """

    ts: str | None
    dialect: str
    source: str
    line_no: int
    level: str | None = None
    component: str | None = None
    agent_id: str | None = None
    kind: str | None = None
    msg: str | None = None
    fields: dict = field(default_factory=dict)
    continuation: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# timestamps
# --------------------------------------------------------------------------- #
def ts_key(ts):
    """Return a timezone-aware sortable key for an ISO-8601 timestamp, or None.

    Never compare raw timestamp strings: the streams disagree on
    fractional-second precision, so lexicographic order is not time order.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def _try_json(line: str):
    try:
        rec = json.loads(line)
    except ValueError:
        return None
    return rec if isinstance(rec, dict) else None


def classify_line(line: str):
    """Return the dialect name for one physical line, or None if it is a continuation.

    Detection is by explicit key signature, never by a bare ``"ts"`` catch-all.
    """
    line = line.rstrip('\n')
    if not line.strip():
        return None
    if CORE_RUNTIME_RE.match(line):
        return 'coreagent_runtime'
    rec = _try_json(line)
    if rec is None:
        return None
    if 'tool' in rec and ('agentId' in rec or 'success' in rec):
        return 'coreagent_actions'
    if 'category' in rec and 'message' in rec:
        return 'coreagent_structured'
    if 'ts' in rec:
        return 'jsonl_unclassified'
    return None


def detect_dialect(path: str):
    """Sniff a file's dialect from its first non-blank line."""
    try:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if line.strip():
                    return classify_line(line)
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- #
# file discovery
# --------------------------------------------------------------------------- #
def iter_files(target: str):
    """Yield this family's log files under *target* (a file or an instance directory)."""
    if os.path.isfile(target):
        yield target
        return
    seen = set()
    for pattern in LOG_GLOBS:
        for path in sorted(glob.glob(os.path.join(target, 'logs', pattern))):
            if path not in seen:
                seen.add(path)
                yield path


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _normalize(rec: dict, dialect: str, source: str, line_no: int) -> Event:
    if dialect == 'coreagent_actions':
        success = rec.get('success')
        return Event(
            ts=rec.get('ts'),
            dialect=dialect,
            source=source,
            line_no=line_no,
            # `success` is optional in this repo's writer: only an explicit
            # false is a failure, an absent key is not evidence of success.
            level='error' if success is False else 'info',
            component=rec.get('tool'),
            agent_id=rec.get('agentId'),
            kind=rec.get('tool'),
            msg=rec.get('tool'),
            fields=rec,
        )
    if dialect == 'coreagent_structured':
        return Event(
            ts=rec.get('ts'),
            dialect=dialect,
            source=source,
            line_no=line_no,
            component=rec.get('category'),
            agent_id=rec.get('agentId'),
            kind=rec.get('category'),
            msg=rec.get('message'),
            fields=rec,
        )
    if dialect == 'jsonl_unclassified':
        return Event(
            ts=rec.get('ts'),
            dialect=dialect,
            source=source,
            line_no=line_no,
            level=rec.get('level'),
            component=rec.get('context') or rec.get('category') or rec.get('tool'),
            agent_id=rec.get('agentId'),
            kind=rec.get('type'),
            msg=rec.get('msg') or rec.get('message'),
            fields=rec,
        )
    raise ValueError(f'no record normalizer for dialect {dialect!r}')


def _in_window(ev, lo, hi, use_window: bool) -> bool:
    if not use_window:
        return True
    when = ts_key(ev.ts)
    if when is None:
        return False
    if lo and when < lo:
        return False
    if hi and when > hi:
        return False
    return True


def iter_events(target: str, window=None, stats=None):
    """Yield normalized :class:`Event` objects from *target*.

    Parameters
    ----------
    target
        A log file or an instance directory.
    window
        Optional ``(from, to)`` ISO-8601 bounds. Events outside the window are
        counted in *stats* but not yielded.
    stats
        Optional dict, mutated in place with parse accounting. Pass one to get
        :func:`coverage`-equivalent numbers without a second pass over the data.
    """
    if stats is not None:
        stats.setdefault('files', 0)
        stats.setdefault('total_lines', 0)
        stats.setdefault('parsed', 0)
        stats.setdefault('continuations', 0)
        stats.setdefault('skipped', 0)
        stats.setdefault('dialects', collections.Counter())

    lo = ts_key(window[0]) if window else None
    hi = ts_key(window[1]) if window else None

    for path in iter_files(target):
        source = os.path.basename(path)
        dialect = detect_dialect(path)
        if stats is not None:
            stats['files'] += 1
            if dialect:
                stats['dialects'][dialect] += 1
        pending = None
        try:
            fh = open(path, encoding='utf-8', errors='ignore')
        except OSError:
            continue
        with fh:
            for line_no, line in enumerate(fh, 1):
                if stats is not None:
                    stats['total_lines'] += 1
                line = line.rstrip('\n')
                line_dialect = classify_line(line)
                if line_dialect is None:
                    # continuation of a multi-line record, e.g. LLM markdown
                    if pending is not None and line.strip():
                        pending.continuation.append(line)
                        if stats is not None:
                            stats['continuations'] += 1
                    elif line.strip() and stats is not None:
                        stats['skipped'] += 1
                    continue
                if line_dialect == 'coreagent_runtime':
                    m = CORE_RUNTIME_RE.match(line)
                    ev = Event(
                        ts=m.group('ts'),
                        dialect=line_dialect,
                        source=source,
                        line_no=line_no,
                        component=m.group('component'),
                        agent_id=m.group('agent'),
                        kind=m.group('component'),
                        msg=m.group('msg'),
                        fields={'raw': line},
                    )
                else:
                    ev = _normalize(_try_json(line), line_dialect, source, line_no)
                if stats is not None:
                    stats['parsed'] += 1
                # A record is only complete once a *new* record starts, so hold it
                # back one step: Event.continuation must be final before the caller
                # receives the event.
                if pending is not None and _in_window(pending, lo, hi, bool(window)):
                    yield pending
                pending = ev
        if pending is not None and _in_window(pending, lo, hi, bool(window)):
            yield pending


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #
def coverage(target: str, window=None) -> dict:
    """Parse accounting for *target*: how much was understood, and as what."""
    stats = {}
    for _ in iter_events(target, window=window, stats=stats):
        pass
    stats['dialects'] = dict(stats.get('dialects', {}))
    return stats


def log_bounds(instance_dir: str):
    """Return ``(first_ts, last_ts, total_lines)`` over the instance's log files.

    ``total_lines`` counts every physical line (blank lines included).
    """
    stats = {}
    first = last = None
    first_key = last_key = None
    for ev in iter_events(instance_dir, stats=stats):
        key = ts_key(ev.ts)
        if key is None:
            continue
        if first_key is None or key < first_key:
            first, first_key = ev.ts, key
        if last_key is None or key > last_key:
            last, last_key = ev.ts, key
    return first, last, stats.get('total_lines', 0)


# --------------------------------------------------------------------------- #
# shared readers used by the numbered tools
# --------------------------------------------------------------------------- #
def agent_ids(instance_dir: str):
    """Distinct ``agentId`` values seen anywhere in the instance's logs."""
    found = set()
    for ev in iter_events(instance_dir):
        if ev.agent_id:
            found.add(ev.agent_id)
    return sorted(found)


def component_census(instance_dir: str, window=None) -> dict:
    """``{dialect: Counter(component -> n)}`` — the one grouping axis that matters."""
    census = collections.defaultdict(collections.Counter)
    for ev in iter_events(instance_dir, window=window):
        census[ev.dialect][ev.component or '(none)'] += 1
    return {dialect: dict(counter) for dialect, counter in census.items()}


def structured_categories(instance_dir: str, window=None) -> dict:
    """``Counter(category -> n)`` over ``structured-*.jsonl``."""
    counts = collections.Counter()
    for ev in iter_events(instance_dir, window=window):
        if ev.dialect == 'coreagent_structured':
            counts[ev.component or '(none)'] += 1
    return dict(counts)


def action_calls(instance_dir: str, window=None):
    """Distinct-timestamp tool call counts plus the raw per-record list.

    Returns ``(by_tool, distinct_by_tool, records)`` where ``records`` is a list
    of ``(ts, tool, success, error)``.

    **Some tools are logged twice under two spellings** — ``sweepUnsoldTokens``
    (summary) and ``sweep_unsold_tokens`` (per-token detail). They are not
    aliases: counting either alone undercounts, summing both double-counts the
    overlap. Always report ``distinct_by_tool`` (distinct timestamps), never a
    raw record count, when talking about tool volume.
    """
    by_tool = collections.Counter()
    stamps = collections.defaultdict(set)
    records = []
    for ev in iter_events(instance_dir, window=window):
        if ev.dialect != 'coreagent_actions':
            continue
        tool = ev.component or '(none)'
        by_tool[tool] += 1
        if ev.ts:
            stamps[tool].add(ev.ts)
        records.append((ev.ts, tool, ev.fields.get('success'), ev.fields.get('error')))
    distinct = {tool: len(values) for tool, values in stamps.items()}
    return dict(by_tool), distinct, records


def failed_actions(instance_dir: str, window=None):
    """Action records with an explicit ``success: false``."""
    out = []
    for ev in iter_events(instance_dir, window=window):
        if ev.dialect == 'coreagent_actions' and ev.fields.get('success') is False:
            out.append(
                {
                    'ts': ev.ts,
                    'tool': ev.component,
                    'error': ev.fields.get('error'),
                    'args': ev.fields.get('args'),
                }
            )
    return out


def structured_errors(instance_dir: str, window=None):
    """Error-category records from ``structured-*.jsonl``.

    This is where the run's failures actually live: ``agent-*.log`` carries
    almost none. Categories observed in this repo: ``api_error``,
    ``swap_error``, ``tx_error``.
    """
    out = []
    for ev in iter_events(instance_dir, window=window):
        if ev.dialect != 'coreagent_structured':
            continue
        if ev.component in ('api_error', 'swap_error', 'tx_error'):
            out.append(
                {
                    'ts': ev.ts,
                    'category': ev.component,
                    'message': ev.msg,
                    'metadata': (ev.fields.get('metadata') or {}),
                }
            )
    return out


def cron_cycles(instance_dir: str, gap_seconds: int = 30, window=None):
    """Cluster ``[cron]`` runtime events into management cycles.

    The ``cron`` component emits **many lines per cycle** — on the reference
    instance 11,219 cron lines spanned 3,663 cycles (median inter-line gap 0.4 s,
    median inter-cycle gap 2.0 min). Measuring raw consecutive-cron gaps
    therefore reports 0.4 s and is meaningless. Cluster first, then measure.

    Returns ``(cycle_start_ts_list, cycle_gap_minutes_list)``.
    """
    stamps = []
    for ev in iter_events(instance_dir, window=window):
        if ev.dialect == 'coreagent_runtime' and ev.component == 'cron':
            key = ts_key(ev.ts)
            if key is not None:
                stamps.append(key)
    stamps.sort()
    if not stamps:
        return [], []
    starts = [stamps[0]]
    for i in range(1, len(stamps)):
        if (stamps[i] - stamps[i - 1]).total_seconds() > gap_seconds:
            starts.append(stamps[i])
    gaps = [round((starts[i + 1] - starts[i]).total_seconds() / 60.0, 2) for i in range(len(starts) - 1)]
    return [s.isoformat().replace('+00:00', 'Z') for s in starts], gaps
