"""Rendering preflight's results for the terminal: states, source excerpts with carets, hints."""

import ast
import json
import linecache
import os

from .. import _dispatch, _preflight as P
from cimo.cli.term import Terminal, visible_len, wrap

KEEP_HINT = ('a bounds check runs on every iteration; index with the loop variable over `range(len(a))`, '
             'or use `@scram(boundscheck=False)` once verified')


def _mark(t, ok):
    return t.style(t.symbol('ok'), 'ok') if ok else t.style(t.symbol('fail'), 'error')


def _partly(t):
    return t.style('~', 'warning')


def _excerpt(t, lines, p, indent):
    """The source line of a problem, with carets under the construct and the reason beside them."""
    bar = t.style(t.symbol('bar'), 'muted')
    num = str(p.line)
    pad = ' ' * indent
    text = lines[p.line - 1].expandtabs(4) if 1 <= p.line <= len(lines) else ''
    t.write(f'{pad}{t.style(num, "muted")} {bar} {text.rstrip()}')
    gutter = pad + ' ' * len(num) + ' ' + bar + ' '
    reason = t.style(p.reason, 'bold')
    if p.col is not None and 1 <= p.line <= len(lines):
        col = len(lines[p.line - 1][:p.col].expandtabs(4))
        room = max(1, min(p.width, max(len(text) - col, 1)))
        t.write(f'{gutter}{" " * col}{t.style("^" * room, "caret")} {reason}')
    else:
        t.write(f'{gutter}{reason}')
    if p.hint:
        hint = wrap(f'{t.style("hint:", "hint")} {t.code(p.hint)}', max(40, t.width - visible_len(gutter)))
        t.write(gutter + hint[0])
        for more in hint[1:]:
            t.write(f'{gutter}      {more}')


def static(t, reports, quiet=False):
    """The report for scanned files. Returns (compiled, partly compiled, python, broken)."""
    good = partly = bad = broken = 0
    for r in reports:
        if r.error is not None:
            broken += 1
            where = f'{r.path}:{r.error.lineno}' if r.error.lineno else r.path
            t.write(f'{t.style("error:", "error")} {t.style(r.error.msg, "bold")}')
            t.write(f'  {t.style(t.symbol("arrow"), "muted")} {t.style(where, "path")}')
            continue
        failing = [f for f in r.functions if f[1].compiler is None]
        good += len(r.functions) - len(failing)
        partly += sum(a.state == 'partly' for _, a in failing)
        bad += sum(a.state == 'python' for _, a in failing)
        shown = [] if quiet else r.candidates
        if not failing and not shown and quiet:
            continue
        t.write(t.style(r.path, 'path'))
        lines = r.source.splitlines()
        width = max([len(f.name) for f, _ in r.functions] + [len(f.name) for f, _ in shown])
        called = {n.func.id for f, _ in r.functions for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for f, a in r.functions:
            if a.compiler is not None:
                if not quiet:
                    t.write(f'  {_mark(t, True)} {f.name:<{width}}  {t.style("compiled", "muted")}')
                    if not P.has_loop(f) and f.name not in called:      # from compiled code a call has no such cost
                        t.write(f'      {t.style("note:", "note")} no loop: a compiled call costs more than a small Python function')
                    for line, text in sorted(a.compiler.gotchas.items()):
                        t.write(f'      {t.style("note:", "warning")} line {line} {text}')
                continue
            if a.state == 'partly':
                spans = ', '.join(str(x) if x == y else f'{x}-{y}' for x, y in a.loops)
                t.write(f'  {_partly(t)} {t.style(f"{f.name:<{width}}", "bold")}  {t.style("partly compiled", "warning")} '
                        f'{t.style(f"(native: lines {spans})", "muted")}')
            else:
                t.write(f'  {_mark(t, False)} {t.style(f"{f.name:<{width}}", "bold")}  runs as Python')
            for p in a.problems:
                _excerpt(t, lines, p, 6)
        for f, _ in shown:
            arrow = t.style(t.symbol('arrow'), 'cyan')
            t.write(f'  {arrow} {f.name:<{width}}  {t.style("would compile with", "muted")} {t.style("@scram", "code")} '
                    f'{t.style(f"(line {f.lineno})", "muted")}')
    return good, partly, bad, broken


def summary(t, good, partly, bad, broken=0, candidates=0):
    parts = [t.style(f'{good} compiled', 'ok') if good else '0 compiled']
    if partly:
        parts.append(t.style(f'{partly} partly compiled', 'warning'))
    if bad:
        parts.append(t.style(f'{bad} {"runs" if bad == 1 else "run"} as Python', 'error'))
    if broken:
        parts.append(t.style(f'{broken} {"file" if broken == 1 else "files"} not readable', 'error'))
    if candidates:
        parts.append(t.style(f'{candidates} more would compile', 'cyan'))
    t.write(', '.join(parts))
    if candidates:
        t.write(t.style('Suggestions assume the arguments are numbers or arrays; check with --run once decorated.', 'muted'))


def static_json(stream, reports):
    files = []
    good = partly = bad = 0
    for r in reports:
        if r.error is not None:
            files.append({'path': r.path, 'error': r.error.msg, 'line': r.error.lineno, 'functions': []})
            continue
        fs = []
        for f, a in r.functions:
            good += a.compiler is not None
            partly += a.state == 'partly'
            bad += a.state == 'python'
            fs.append({'name': f.name, 'line': f.lineno, 'state': a.state, 'signature': [x.name for x in a.signature],
                       'native_loops': [{'first': x, 'last': y} for x, y in a.loops],
                       'problems': [p.as_dict() for p in a.problems],
                       'differences': [{'line': k, 'text': v} for k, v in sorted(a.compiler.gotchas.items())] if a.compiler else []})
        files.append({'path': r.path, 'functions': fs,
                      'would_compile': [{'name': f.name, 'line': f.lineno, 'signature': [x.name for x in a.signature]}
                                        for f, a in r.candidates]})
    json.dump({'files': files, 'compiled': good, 'partly': partly, 'python': bad,
               'would_compile': sum(len(r.candidates) for r in reports)}, stream, indent=2)
    stream.write('\n')


def _count_note(line, runs, heads):
    """What a line's count means for what it holds: how long its loop runs, how often its branch is taken."""
    kind, body = heads.get(line, (None, 0))
    n = runs.get(line, 0)
    if kind == 'loop' and n:
        return f'{body / n:,.1f} per entry'
    if kind == 'if' and n:
        return f'taken {100 * body / n:.0f}%'
    return ''


def lines_table(t, lines, first, fdef, offset, title, status, bounds, problems, types, gotchas=None, loops=(), counts=None):
    """One function, line by line. loops: (first line, last line) of what is native in a partly compiled one.
    counts: (runs, heads) from Function.line_counts, for a column of how often each line ran."""
    runs, heads = counts or ({}, {})
    top = max(runs.values(), default=0)
    across = max((len(f'{v:,}') for v in runs.values()), default=0)
    t.write(t.style(title, 'bold'))
    t.write(status)
    rule = t.style(t.symbol('dash') * min(t.width, 72), 'muted')
    t.write(rule)
    kept, hoisted, checked = bounds or ({}, (), True)
    indexed = P.subscript_lines(fdef, offset)
    in_loop = P.loop_lines(fdef, offset)
    bad = {}
    for p in problems:
        bad.setdefault(p.line, p)
    body = fdef.body
    if len(body) > 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]                                 # a docstring isn't code
    start = body[0].lineno + offset
    width = max((len(s) for s in lines), default=0)
    hints = []
    bar = t.style(t.symbol('bar'), 'muted')
    for i, text in enumerate(lines):
        n = first + i
        note = ''
        if n < start or not text.strip():
            mark = ' '
        elif any(x <= n <= y for x, y in loops):
            mark = _mark(t, True)
            if any(n == x for x, _ in loops):
                note = t.style('compiled on its own', 'ok')
        elif problems or loops:
            mark = _mark(t, False)
            if n in bad:
                note = t.style(bad[n].reason, 'bold')
        else:
            mark = _mark(t, True)
            if n in indexed and not checked:
                note = t.style('bounds checks off', 'muted')
            elif n in indexed:
                k = kept.get(n, 0)
                if k:
                    note = t.style(f'{k} bounds check{"s" if k != 1 else ""} kept', 'yellow')
                    if n in in_loop:
                        hints.append(f'line {n}: {KEEP_HINT}')
                else:
                    note = t.style('bounds hoisted out of the loop' if n in hoisted else 'bounds proven', 'muted')
            if gotchas and n in gotchas:
                note = t.style(gotchas[n], 'yellow')
        column = ''
        if counts:
            v = runs.get(n)
            shown = f'{v:,}' if v is not None else ''
            column = t.style(f'{shown:>{across}}', 'yellow' if v and v * 2 >= top else 'muted') + ' '
            extra = _count_note(n, runs, heads)
            if extra:
                note = f'{note}  {t.style(extra, "muted")}' if note else t.style(extra, 'muted')
        pad = ' ' * (width - len(text)) if note else ''
        t.write(f'{t.style(f"{n:>4}", "muted")} {bar} {mark} {column}{text}{pad}  {note}'.rstrip())
    t.write(rule)
    if types:
        t.write(f'{t.style("types:", "heading")} ' + ', '.join(f'{k}: {t.style(v, "code")}' for k, v in types.items()))
    hints += [f'line {p.line}: {p.hint}' for p in problems if p.hint]
    for h in dict.fromkeys(hints):
        text = wrap(f'{t.style("hint:", "hint")} {t.code(h)}', t.width)
        t.write(text[0])
        for more in text[1:]:
            t.write(f'      {more}')


def _state_line(t, ok, rest):
    if ok == 'partly':
        return f'{_partly(t)} {t.style("partly compiled", "warning")} {t.style(rest, "muted")}'
    return f'{_mark(t, ok)} {t.style("compiled", "ok") if ok else t.style("runs as Python", "error")} {t.style(rest, "muted")}'


def one_static(t, fdef, a, source, path, boundscheck):
    """FILE:FUNCTION, one assumed signature."""
    start = fdef.decorator_list[0].lineno if fdef.decorator_list else fdef.lineno
    text = source.splitlines()[start - 1:fdef.end_lineno]
    c = a.compiler
    lines_table(t, text, start, fdef, 0, f'{fdef.name}({", ".join(x.name for x in a.signature)})',
                _state_line(t, 'partly' if a.state == 'partly' else c is not None, f'· {os.path.basename(path)} · assumed signature'),
                (c.checks, c.hoisted_lines, boundscheck) if c else None, a.problems,
                c.shown_types() if c else {}, c.gotchas if c else None, a.loops)


def function_report(f, signature, stream):
    """f.preflight(): every signature it has seen (or the one given), else one assumed signature."""
    import inspect
    import textwrap
    t = Terminal(stream)
    fdef = f._tree
    where = os.path.basename(f._filename)
    if fdef is None:
        t.write(t.style(f.__name__, 'bold'))
        t.write(_state_line(t, False, f'· {where}'))
        t.write(f._why_not)
        return
    try:
        lines = textwrap.dedent(inspect.getsource(f._fn)).splitlines()
    except (OSError, TypeError):
        lines = []
    offset = f._first - 1
    counts = f.line_counts()
    counts = counts if counts[0] else None
    keys = [tuple(signature)] if signature is not None else list(f._cache)
    if not keys:
        a = P.best_analysis(fdef, f._fn.__globals__, f._filename, offset, {'boundscheck': f._boundscheck, 'parity': f._parity})
        c = a.compiler
        lines_table(t, lines, f._first, fdef, offset, f'{f.__name__}({", ".join(x.name for x in a.signature)})',
                    _state_line(t, 'partly' if a.state == 'partly' else c is not None, f'· {where} · assumed signature, not called yet'),
                    (c.checks, c.hoisted_lines, f._boundscheck) if c else None, a.problems,
                    c.shown_types() if c else {}, c.gotchas if c else None, a.loops)
        return
    for i, key in enumerate(keys):
        if i:
            t.write('')
        entry = f._cache.get(key)
        names = ', '.join(x.name if x is not None else '?' for x in key[:len(key) - len(f._hidden or ())])
        if isinstance(entry, _dispatch.Compiled):
            s = f._compile_s.get(key)
            when = f' · compiled in {s * 1000:.0f} ms' if s is not None else ''
            lines_table(t, lines, f._first, fdef, offset, f'{f.__name__}({names})', _state_line(t, True, f'· {where}{when}'),
                        (entry.checks, entry.hoisted, f._boundscheck), [], entry.locals, entry.gotchas, counts=counts)
        else:
            line, why = f._why.get(key, (f._first, 'not compiled'))
            loops = [(x._tree.lineno + offset, x._tree.end_lineno + offset) for x in f._native_loops()]
            if loops:
                problems = [] if any(a <= line <= b for a, b in loops) else [P.Problem(line, None, 1, why)]
                lines_table(t, lines, f._first, fdef, offset, f'{f.__name__}({names})',
                            _state_line(t, 'partly', f'· {where}'), None, problems, {}, None, loops, counts)
                continue
            lines_table(t, lines, f._first, fdef, offset, f'{f.__name__}({names})', _state_line(t, False, f'· {where}'),
                        None, [P.Problem(line, None, 1, why)], {})


def _bar(t, share, width=20):
    n = round(share * width)
    full, empty = ('█', '░') if t.unicode else ('#', '-')
    return t.style(full * n, 'green') + t.style(empty * (width - n), 'muted')


def run_rows():
    rows = []
    for f in sorted(_dispatch.functions(), key=lambda f: f.__name__):
        st = f._stats
        if st:
            loops = f._all_loops()
            done = [x for x in loops if x._stats]
            native = sum(s[1] for x in done for s in x._stats.values())          # lifted loops: inside its Python time
            building = sum(v for x in done for v in x._compile_s.values())
            compiled = sum(isinstance(f._cache.get(k), _dispatch.Compiled) for k in st)
            rows.append({'function': f, 'name': f.__name__, 'calls': sum(s[0] for s in st.values()),
                         'compile': sum(f._compile_s.get(k, 0.0) for k in st) + building,
                         'native': sum(s[1] for s in st.values()) + native,
                         'python': max(sum(s[2] for s in st.values()) - native - building, 0.0), 'signatures': len(st),
                         'compiled': compiled,
                         'state': 'compiled' if compiled == len(st) else 'partly' if f._native_loops() else 'python'})
    return rows


def run_hints(rows):
    out = []
    for r in rows:
        f = r['function']
        for k in f._stats:
            if k in f._why:
                line, why = f._why[k]
                out.append((f'{r["name"]}, line {line}', why, P.hint_for(why)))
        if r['signatures'] > 1:
            per = [{i: t.name for i, t in enumerate(k) if t is not None} for k in f._stats]
            for i in sorted(set().union(*per)):
                seen = sorted({p[i] for p in per if i in p})
                if len(seen) > 1:
                    name = f._fn.__code__.co_varnames[i]
                    out.append((r['name'], f'{name} arrives as {" and ".join(seen)}, so it compiles once per type',
                                f'annotate it, such as `{name}: {seen[0]}`, or convert once before calling'))
    return out


def run(t, total, status, allow_partial=False):
    """The table for a run. Returns how many functions weren't compiled, as far as the exit status goes."""
    rows = run_rows()
    native = sum(r['native'] for r in rows)
    share = min(native / total, 1.0) if total else 0.0
    t.write('')
    t.write(f'{t.style("scramjit", "command")} {t.style(f"{total:.2f} s run", "muted")}  {_bar(t, share)} '
            f'{t.style(f"{share * 100:.0f}%", "number")} native')
    if not rows:
        t.write(f'  {t.style("note:", "note")} no @scram function was called')
        return 0
    secs = lambda v: f'{v:.2f} s' if v else '–'  # noqa: E731
    names = [r['name'] + (f' ×{r["signatures"]}' if r['signatures'] > 1 else '') for r in rows]
    width = max(len(n) for n in names + ['function'])
    t.write(t.style(f'  {"  "}{"function":<{width}}  {"calls":>8}  {"compile":>7}  {"native":>7}  {"python":>7}', 'heading'))
    bad = 0
    for r, n in zip(rows, names):
        ok = r['compiled'] == r['signatures']
        bad += r['state'] == 'python' or (r['state'] == 'partly' and not allow_partial)
        mark = _partly(t) if r['state'] == 'partly' else _mark(t, ok)
        t.write(f'  {mark} {n:<{width}}  {r["calls"]:>8,}  {secs(r["compile"]):>7}  {secs(r["native"]):>7}  {secs(r["python"]):>7}')
    hints = run_hints(rows)
    if hints:
        t.write('')
        for where, why, hint in hints:
            t.write(f'{t.style(where + ":", "bold")} {why}')
            if hint:
                for k, text in enumerate(wrap(f'{t.style("hint:", "hint")} {t.code(hint)}', t.width - 2)):
                    t.write(f'  {"      " if k else ""}{text}')
    some = [r['name'] for r in rows if r['state'] == 'partly']
    if some:
        t.write('')
        t.write(t.style(f'{", ".join(some)} {"is" if len(some) == 1 else "are"} partly compiled', 'warning'))
    failed = [r['name'] for r in rows if r['state'] == 'python']
    if failed:
        t.write('')
        t.write(t.style(f'{", ".join(failed)} {"runs" if len(failed) == 1 else "run"} as Python', 'error'))
    return bad


def hottest(limit=10):
    """The lines that ran most, over every counted function: (times, function name, file, line)."""
    out = []
    for f in _dispatch.functions():
        out += [(v, f.__name__, f._filename, n) for n, v in f.line_counts()[0].items() if v]
    return sorted(out, key=lambda x: (-x[0], x[2], x[3]))[:limit]


def hot_lines(t):
    hot = hottest()
    if not hot:
        return
    t.write('')
    t.write(t.style('hottest lines', 'heading') + t.style(', exact counts; counting slowed this run, so the times above read high', 'muted'))
    across = max(len(f'{v:,}') for v, *_ in hot)
    for v, name, path, n in hot:
        where = f'{os.path.basename(path)}:{n}'
        t.write(f'  {t.style(f"{v:>{across},}", "number")}  {where}  {linecache.getline(path, n).strip()}')


def counts_json(f):
    runs, heads = f.line_counts()
    return {'lines': {str(n): v for n, v in sorted(runs.items())},
            'branches_taken': {str(n): v for n, (k, v) in sorted(heads.items()) if k == 'if'},
            'loop_iterations': {str(n): v for n, (k, v) in sorted(heads.items()) if k == 'loop'}}


def run_json(stream, total, status, allow_partial=False, one_line=False):
    rows = run_rows()
    functions = []
    for r in rows:
        row = {k: v for k, v in r.items() if k != 'function'}
        c = counts_json(r['function'])
        if c['lines']:
            row['counts'] = c
        functions.append(row)
    out = {'seconds': total, 'exit_status': status, 'native_share': min(sum(r['native'] for r in rows) / total, 1.0) if total else 0.0,
           'functions': functions,
           'hints': [{'where': w, 'reason': why, 'hint': h} for w, why, h in run_hints(rows)]}
    if one_line:
        out['type'] = 'report'
    json.dump(out, stream, indent=None if one_line else 2)
    stream.write('\n')
    return sum(r['state'] == 'python' or (r['state'] == 'partly' and not allow_partial) for r in rows)
