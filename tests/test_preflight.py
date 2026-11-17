"""The preflight report, the command line, the disk cache across processes, and the log."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from scramjit import scram

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
CIMO = os.path.join(ROOT, '..', 'cimo', 'python')

SIM = '''
import math
from scramjit import scram

@scram
def ema(x, y, a: float):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]

@scram
def grid(h):
    rows, cols = h.shape
    s = 0.0
    for i in range(rows):
        for j in range(cols):
            s += math.sqrt(h[i, j])
    return s

@scram
def sloppy(a, k: int):
    s = 0.0
    for i in range(3):
        s += a[k]
    return s

@scram(boundscheck=False)
def fast(a, k: int):
    return a[k]

@scram
def summarize(a):
    names = {}
    for i in range(len(a)):
        names[i] = str(a[i])
    return len(names)

def plain():
    return {}
'''
APP = '''
import sys
import numpy as np
import sim
x = np.ones(50)
y = np.zeros(50)
sim.ema(x, y, 0.5)
sim.ema(x, y, 1)
sim.grid(np.ones((4, 4)))
sim.summarize(x)
print('args', sys.argv[1:])
'''


def run(args, cwd, **env):
    e = {k: v for k, v in os.environ.items() if k not in ('FORCE_COLOR', 'NO_COLOR', 'SCRAMJIT_COLOR')}
    e.update(PYTHONPATH=os.pathsep.join([ROOT, CIMO]), PYTHONWARNINGS='ignore', PYTHONIOENCODING='utf-8', **env)
    return subprocess.run([sys.executable, *args], cwd=cwd, env=e, capture_output=True, text=True, timeout=120, encoding='utf-8')


@pytest.fixture
def proj(tmp_path):
    (tmp_path / 'sim.py').write_text(SIM)
    (tmp_path / 'app.py').write_text(APP)
    return tmp_path


def test_static_report(proj):
    r = run(['-m', 'scramjit', 'preflight', 'sim.py'], proj)
    assert r.returncode == 1, r.stderr
    assert '✓ ema' in r.stdout and '✓ grid' in r.stdout and '✗ summarize' in r.stdout
    assert "dicts aren't compiled" in r.stdout and 'calls str()' in r.stdout
    assert 'plain' not in r.stdout
    assert '4 compiled, 1 runs as Python' in r.stdout
    # the line that only fails because an earlier one did isn't blamed
    assert 'len() of something' not in r.stdout


def test_carets_and_hints(proj):
    out = run(['-m', 'scramjit', 'preflight', 'sim.py'], proj).stdout.splitlines()
    i = next(k for k, ln in enumerate(out) if 'names[i] = str(a[i])' in ln)
    src, under = out[i], out[i + 1]
    assert src.index('str(a[i])') == under.index('^') and under.count('^') == len('str(a[i])')
    assert 'hint:' in out[i + 2] and 'return numbers' in out[i + 2]


def test_static_imports_nothing(tmp_path):
    (tmp_path / 'boom.py').write_text('raise SystemExit("imported!")\nfrom scramjit import scram\n@scram\ndef f(a: int):\n    return a + 1\n')
    r = run(['-m', 'scramjit', 'preflight', str(tmp_path)], tmp_path)
    assert r.returncode == 0 and '✓ f' in r.stdout, r.stdout + r.stderr


def test_exit_zero_when_all_compiled(tmp_path):
    (tmp_path / 'ok.py').write_text('from scramjit import scram\n@scram\ndef f(a: int):\n    return a + 1\n')
    assert run(['-m', 'scramjit', 'preflight', 'ok.py'], tmp_path).returncode == 0
    assert run(['-m', 'scramjit', 'preflight', '--allow-partial', 'ok.py'], tmp_path).returncode == 0


def test_help_and_version(proj):
    r = run(['-m', 'scramjit'], proj)
    assert r.returncode == 0 and 'Usage: scramjit <command>' in r.stdout and 'preflight' in r.stdout and 'Examples:' in r.stdout
    assert 'scramjit preflight --run' in run(['-m', 'scramjit', 'help', 'preflight'], proj).stdout
    assert '--quiet' in run(['-m', 'scramjit', 'preflight', '--help'], proj).stdout
    assert 'Delete cached files' in run(['-m', 'scramjit', 'cache', '--help'], proj).stdout
    v = run(['-m', 'scramjit', '--version'], proj).stdout
    assert v.startswith('scramjit ') and 'cimo' in v and 'Python' in v


@pytest.mark.parametrize('args, needle, also', [
    (['preflght'], 'Did you mean', 'preflight'),
    (['preflight', '--jsn'], 'Did you mean', '--json'),
    (['sim.py'], 'hint:', 'scramjit preflight sim.py'),
    (['preflight', 'sm.py'], 'Did you mean', 'sim.py'),
    (['preflight', 'sim.py:eam'], 'Did you mean', 'sim.py:ema'),
    (['preflight', 'nosuchdir/x.py:f'], 'no such file', 'nosuchdir'),
    (['preflight', '--run'], 'needs a program', '--run -m pytest'),
    (['preflight', '--run', 'ap.py'], 'Did you mean', 'app.py'),
    (['c', 'sim.py'], 'hint:', 'FILE:FUNCTION'),
    (['completion', 'bsh'], 'Did you mean', 'bash'),
    (['help', 'preflght'], 'Did you mean', 'preflight'),
])
def test_mistakes_are_explained(proj, args, needle, also):
    r = run(['-m', 'scramjit', *args], proj)
    assert r.returncode == 2 and needle in r.stderr and also in r.stderr, r.stderr
    assert 'Traceback' not in r.stderr


def test_the_cli_is_cimos():
    import cimo.cli.parse
    from scramjit.cli import commands
    root = commands.build_tree()
    assert type(root) is cimo.cli.parse.Command and commands.UsageError is cimo.cli.parse.UsageError
    assert root.unknown_hint(root, 'sim.py') == 'to check it: `scramjit preflight sim.py`'


def test_default_path_and_no_functions(proj, tmp_path_factory):
    assert '✓ ema' in run(['-m', 'scramjit', 'preflight'], proj).stdout
    empty = tmp_path_factory.mktemp('empty')
    (empty / 'x.py').write_text('print(1)\n')
    r = run(['-m', 'scramjit', 'preflight'], empty)
    assert r.returncode == 0 and 'no @scram functions' in r.stderr


def test_a_syntax_error_is_reported_not_skipped(tmp_path):
    (tmp_path / 'bad.py').write_text('def f(:\n')
    r = run(['-m', 'scramjit', 'preflight', '.'], tmp_path)
    assert r.returncode == 1 and 'bad.py:1' in r.stdout


CANDIDATES = """
import math


def smooth(a, out):
    for i in range(1, len(a)):
        out[i] = 0.5 * (a[i] + a[i - 1])


def count_hits(a, limit: float):
    return sum(1 for v in a if v > limit)


def add(x: float, y: float):
    return x + y                                # no loop: nothing to gain


def uses_dict(a):
    seen = {}
    for v in a:
        seen[v] = 1
    return seen


def fetch(path: str):
    for line in open(path):
        print(line)


def no_arguments():
    for i in range(3):
        pass


def test_something(a):
    for v in a:
        pass


class Thing:
    def method(self, a):
        s = 0.0
        for v in a:
            s += v
        return s


def outer(a):
    def inner(b):
        return sum(v for v in b)
    return inner(a)


@staticmethod
def wrapped(a):
    for v in a:
        pass


def uses_global(a):
    return len(a) + UNKNOWN
"""


def test_suggestions(tmp_path):
    (tmp_path / 'lib.py').write_text(CANDIDATES)
    r = run(['-m', 'scramjit', 'preflight', 'lib.py'], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr       # advice, not a failure
    assert 'smooth' in r.stdout and 'count_hits' in r.stdout and 'would compile' in r.stdout
    for name in ('add', 'uses_dict', 'fetch', 'no_arguments', 'test_something', 'method', 'inner', 'outer', 'wrapped', 'uses_global'):
        assert name not in r.stdout, name
    assert '2 more would compile' in r.stdout and 'numbers or arrays' in r.stdout


def test_suggestions_can_be_turned_off(tmp_path):
    (tmp_path / 'lib.py').write_text(CANDIDATES)
    assert 'smooth' not in run(['-m', 'scramjit', 'preflight', '--no-suggest', 'lib.py'], tmp_path).stdout
    assert 'smooth' not in run(['-m', 'scramjit', 'preflight', '-q', 'lib.py'], tmp_path).stdout
    d = json.loads(run(['-m', 'scramjit', 'preflight', '--json', 'lib.py'], tmp_path).stdout)
    assert d['would_compile'] == 2 and d['files'][0]['would_compile'][0]['name'] == 'smooth'


def test_a_suggestion_really_compiles(tmp_path):
    """Decorate what preflight suggests, and it compiles and matches."""
    (tmp_path / 'lib.py').write_text('from scramjit import scram\n' + CANDIDATES.replace('def smooth', '@scram(fallback=False)\ndef smooth'))
    (tmp_path / 'a.py').write_text('import numpy as np\nimport lib\na = np.arange(4.0); o = np.zeros(4)\nlib.smooth(a, o)\nprint(o.tolist())\n')
    r = run(['a.py'], tmp_path)
    assert r.stdout.strip() == '[0.0, 0.5, 1.5, 2.5]', r.stderr


def test_a_complex_power_is_flagged_only_when_possible(tmp_path):
    (tmp_path / 'p.py').write_text('from scramjit import scram\n@scram\ndef f(x: float):\n    return x ** 0.5\n'
                                   '@scram\ndef g(x: float):\n    return x ** 2 + 2.0 ** x\n')
    out = run(['-m', 'scramjit', 'preflight', 'p.py'], tmp_path).stdout
    assert 'line 4 runs as Python if the base is negative' in out
    assert 'line 7' not in out and '2 compiled' in out


def test_a_function_with_no_loop_is_noted(tmp_path):
    (tmp_path / 'p.py').write_text('from scramjit import scram\n@scram\ndef f(x: float):\n    return x + 1\n'
                                   '@scram\ndef g(a):\n    for v in a:\n        pass\n')
    out = run(['-m', 'scramjit', 'preflight', 'p.py'], tmp_path).stdout
    assert out.count('no loop') == 1 and out.index('no loop') < out.index('✓ g')


def test_differences_from_python_are_flagged(tmp_path):
    (tmp_path / 'p.py').write_text('from scramjit import scram\n@scram\ndef f(n: int, k: int, out):\n    out[0] = 1\n    return n ** k\n')
    r = run(['-m', 'scramjit', 'preflight', 'p.py'], tmp_path)
    assert r.returncode == 0 and 'note: line 5 raises ValueError if the exponent is negative' in r.stdout
    assert 'raises ValueError' in run(['-m', 'scramjit', 'preflight', 'p.py:f'], tmp_path).stdout
    d = json.loads(run(['-m', 'scramjit', 'preflight', '--json', 'p.py'], tmp_path).stdout)
    assert d['files'][0]['functions'][0]['differences'][0]['line'] == 5


def test_quiet(proj):
    out = run(['-m', 'scramjit', 'preflight', '-q', 'sim.py'], proj).stdout
    assert '✗ summarize' in out and '✓' not in out


def test_json(proj):
    r = run(['-m', 'scramjit', 'preflight', '--json', 'sim.py'], proj)
    d = json.loads(r.stdout)
    assert d['compiled'] == 4 and d['python'] == 1 and r.returncode == 1
    bad = next(f for f in d['files'][0]['functions'] if f['name'] == 'summarize')
    assert bad['state'] == 'python' and bad['problems'][0]['line'] == SIM.splitlines().index('    names = {}') + 1 and bad['problems'][0]['hint']
    r = run(['-m', 'scramjit', 'preflight', '--json', '-o', 'out.json', '--run', 'app.py'], proj)
    assert 'args []' in r.stdout
    d = json.loads((proj / 'out.json').read_text())
    assert {f['name'] for f in d['functions']} == {'ema', 'grid', 'summarize'} and d['exit_status'] == 0


def test_json_to_a_file_without_running(proj):
    r = run(['-m', 'scramjit', 'preflight', '--json', '-o', 'out.json', 'sim.py'], proj)
    assert r.stdout == '' and json.loads((proj / 'out.json').read_text())['compiled'] == 4


def test_color_rules(proj):
    esc = '\x1b['
    args = ['-m', 'scramjit', 'preflight', 'sim.py']
    assert esc not in run(args, proj).stdout                               # a pipe
    assert esc in run(args, proj, FORCE_COLOR='1').stdout
    assert esc not in run(args, proj, NO_COLOR='1').stdout
    assert esc in run(['-m', 'scramjit', '--color', 'always', *args[2:]], proj).stdout
    assert esc in run(args, proj, SCRAMJIT_COLOR='always').stdout
    assert esc not in run(['-m', 'scramjit', '--color=never', *args[2:]], proj, FORCE_COLOR='1').stdout


def test_terminals_without_unicode(proj):
    e = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, CIMO]), PYTHONIOENCODING='ascii', PYTHONWARNINGS='ignore')
    for args in (['preflight', 'sim.py'], ['preflight', 'sim.py:ema'], ['preflight', '--run', 'app.py']):
        r = subprocess.run([sys.executable, '-m', 'scramjit', *args], cwd=proj, env=e, capture_output=True, timeout=120)
        assert b'Traceback' not in r.stderr, r.stderr
        assert r.returncode == 1 or args[1] == 'sim.py:ema'


def test_c(proj):
    r = run(['-m', 'scramjit', 'c', 'sim.py:ema'], proj)
    assert r.returncode == 0 and 'int sj_kernel' in r.stdout and '#line' in r.stdout and 'sj_floordiv' not in r.stdout
    assert 'sj_floordiv' in run(['-m', 'scramjit', 'c', 'sim.py:ema', '--runtime'], proj).stdout
    r = run(['-m', 'scramjit', 'c', 'sim.py:summarize'], proj)
    assert r.returncode == 1 and 'has no C' in r.stderr and "dicts aren't compiled" in r.stderr


def test_completion(proj):
    assert 'preflight' in run(['-m', 'scramjit', '__complete', 'pre'], proj).stdout
    assert '--json' in run(['-m', 'scramjit', '__complete', 'preflight', '--'], proj).stdout
    for shell in ('bash', 'zsh', 'fish'):
        assert 'scramjit' in run(['-m', 'scramjit', 'completion', shell], proj).stdout


def test_cache_commands(proj, tmp_path):
    cc = str(tmp_path / 'cc')
    assert run(['-m', 'scramjit', 'cache', 'path'], proj, CIMO_CACHE_DIR=cc).stdout.strip() == cc
    assert run(['-m', 'scramjit', 'cache', 'info'], proj, CIMO_CACHE_DIR=cc).returncode == 0


def test_one_function_bounds(proj):
    r = run(['-m', 'scramjit', 'preflight', 'sim.py:ema'], proj)
    assert r.returncode == 0
    lines = {ln.split()[0]: ln for ln in r.stdout.splitlines() if ln[:4].strip().isdigit()}
    # y[0] is checked; the loop's accesses are versioned: one test before it, no check inside
    assert '2 bounds checks kept' in next(v for v in lines.values() if 'y[0] = x[0]' in v)
    assert 'bounds hoisted out of the loop' in next(v for v in lines.values() if 'y[i - 1]' in v)
    assert 'hint' not in r.stdout


def test_kept_check_in_a_loop_gets_a_hint(proj):
    r = run(['-m', 'scramjit', 'preflight', 'sim.py:sloppy'], proj)
    assert '1 bounds check kept' in r.stdout and 'runs on every iteration' in r.stdout


def test_checks_off(proj):
    assert 'bounds checks off' in run(['-m', 'scramjit', 'preflight', 'sim.py:fast'], proj).stdout


def test_one_function_python(proj):
    r = run(['-m', 'scramjit', 'preflight', 'sim.py:summarize'], proj)
    assert r.returncode == 1 and '✗' in r.stdout and 'calls str()' in r.stdout


def test_run_report(proj):
    r = run(['-m', 'scramjit', 'preflight', '--run', 'app.py', 'one', 'two'], proj)
    assert r.stdout == "args ['one', 'two']\n", r.stdout       # the program keeps stdout
    assert '% native' in r.stderr and '✓ ema ×2' in r.stderr and '✗ summarize' in r.stderr
    assert 'a arrives as float and int' in r.stderr
    assert 'summarize runs as Python' in r.stderr
    assert r.returncode == 1


def test_run_module(proj):
    (proj / 'okmod.py').write_text('import numpy as np\nimport sim\nsim.grid(np.ones((2, 2)))\n')
    r = run(['-m', 'scramjit', 'preflight', '--run', '-m', 'okmod'], proj)
    assert r.returncode == 0 and '✓ grid' in r.stderr, r.stdout + r.stderr


def test_run_passes_on_the_programs_failure(proj):
    (proj / 'bad.py').write_text('import sys\nsys.exit(7)\n')
    assert run(['-m', 'scramjit', 'preflight', '--run', 'bad.py'], proj).returncode == 7


def test_method_on_a_function(capsys):
    f = scram(_probe)
    f.preflight()
    assert 'assumed signature' in capsys.readouterr().out
    f(np.ones(3))
    f.preflight()
    out = capsys.readouterr().out
    assert 'compiled in' in out and 'types' in out
    f.preflight(f.signatures[0])
    assert 'compiled in' in capsys.readouterr().out
    assert 'int sj_kernel' in f.c()


def _probe(a):
    return a[0]


def test_log(proj):
    r = run(['app.py'], proj, SCRAMJIT_LOG='1')
    assert 'scramjit: compiled grid(float64[:, :]) in' in r.stderr
    assert 'scramjit: summarize(float64[:]) runs as Python: line' in r.stderr
    assert 'scramjit:' not in run(['app.py'], proj).stderr


def _images(d):
    return [f for f in os.listdir(d)] if os.path.isdir(d) else []


def test_a_second_process_reuses_the_disk_cache(tmp_path):
    (tmp_path / 'p.py').write_text(textwrap.dedent('''
        import sys, time
        import numpy as np
        from scramjit import scram
        @scram(fallback=False)
        def f(a, n: int):
            s = 0.0
            for k in range(n):
                for i in range(len(a)):
                    s += a[i] * %d
            return s
        t = time.perf_counter()
        print(f(np.ones(4), 2))
        print(f'{time.perf_counter() - t:.3f}', file=sys.stderr)
    ''' % (os.getpid() % 1000 + 3)))
    cc = tmp_path / 'cc'
    first = run(['p.py'], tmp_path, CIMO_CACHE_DIR=str(cc))
    assert first.returncode == 0, first.stderr
    made = _images(cc)
    assert made
    before = {f: os.path.getmtime(cc / f) for f in made}
    second = run(['p.py'], tmp_path, CIMO_CACHE_DIR=str(cc))
    assert second.stdout == first.stdout
    assert {f: os.path.getmtime(cc / f) for f in _images(cc)} == before     # read, not rebuilt


def test_cache_false_writes_nothing(tmp_path):
    (tmp_path / 'p.py').write_text(textwrap.dedent('''
        import numpy as np
        from scramjit import scram
        @scram(fallback=False, cache=False)
        def f(a):
            return a[0] * 91.5
        print(f(np.ones(2)))
    '''))
    cc = tmp_path / 'cc'
    r = run(['p.py'], tmp_path, CIMO_CACHE_DIR=str(cc))
    assert r.returncode == 0 and '91.5' in r.stdout, r.stderr
    assert not [f for f in _images(cc) if f.endswith('.img')]


TALLY = '''
import numpy as np
from scramjit import scram

@scram
def tally(px, hist):
    labels = {}
    for i in range(len(px)):
        hist[px[i]] += 1
    for k in range(len(hist)):
        if hist[k]:
            labels[k] = f"{k}: {hist[k]}"
    return labels

@scram
def nothing_fits(names):
    out = {}
    for n in names:
        out[n] = str(n)
    return out

@scram
def whole(x):
    s = 0.0
    for v in x:
        s += v
    return s

if __name__ == '__main__':
    px = np.array([1, 2, 2], dtype=np.int64)
    print(tally(px, np.zeros(4, dtype=np.int64)))
    print(whole(np.ones(3)))
'''


@pytest.fixture
def tally_file(tmp_path):
    p = tmp_path / 'tally.py'
    p.write_text(TALLY)
    return p


def _cli(*args, cwd=None):
    env = dict(os.environ, NO_COLOR='1', PYTHONWARNINGS='ignore', PYTHONIOENCODING='utf-8')
    env['PYTHONPATH'] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run([sys.executable, '-m', 'scramjit', *map(str, args)], capture_output=True, encoding='utf-8', env=env, cwd=cwd)


def test_static_shows_all_three_states(tally_file):
    r = _cli('preflight', tally_file, '--no-suggest')
    assert 'partly compiled (native: lines 8-9)' in r.stdout
    assert '1 compiled, 1 partly compiled, 1 runs as Python' in r.stdout
    assert r.returncode == 1


def test_static_json(tally_file):
    r = _cli('preflight', tally_file, '--json', '--no-suggest')
    d = json.loads(r.stdout)
    states = {f['name']: f for f in d['files'][0]['functions']}
    assert {k: v['state'] for k, v in states.items()} == {'tally': 'partly', 'nothing_fits': 'python', 'whole': 'compiled'}
    assert states['tally']['native_loops'] == [{'first': 8, 'last': 9}]
    assert (d['compiled'], d['partly'], d['python']) == (1, 1, 1)


def test_allow_partial_accepts_only_partly(tally_file):
    assert _cli('preflight', f'{tally_file}:tally').returncode == 1
    assert _cli('preflight', f'{tally_file}:tally', '--allow-partial').returncode == 0
    assert _cli('preflight', f'{tally_file}:nothing_fits', '--allow-partial').returncode == 1


def test_one_function_marks_each_line(tally_file):
    out = _cli('preflight', f'{tally_file}:tally').stdout
    lines = out.splitlines()
    assert any('partly compiled' in x for x in lines[:3])
    assert 'compiled on its own' in next(x for x in lines if 'for i in range' in x)
    native = next(x for x in lines if 'hist[px[i]] += 1' in x)
    python = next(x for x in lines if 'labels = {}' in x)
    assert '✓' in native or '+' in native
    assert '✗' in python or ' x ' in python


def test_run_reports_partly(tally_file, tmp_path):
    r = _cli('preflight', '--json', '-o', tmp_path / 'out.json', '--allow-partial', '--run', tally_file)
    d = json.loads((tmp_path / 'out.json').read_text())
    states = {f['name']: f for f in d['functions']}
    assert states['tally']['state'] == 'partly' and states['whole']['state'] == 'compiled'
    assert states['tally']['native'] > 0
    assert r.returncode == 0
    assert _cli('preflight', '--run', tally_file).returncode == 1


def test_f_preflight_after_a_call():
    import contextlib
    import io
    import warnings
    import numpy as np
    from scramjit import scram

    def tally(px, hist):
        labels = {}
        for i in range(len(px)):
            hist[px[i]] += 1
        return labels

    g = scram(cache=False)(tally)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g(np.array([1], dtype=np.int64), np.zeros(2, dtype=np.int64))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g.preflight()
    assert 'partly compiled' in out.getvalue() and 'compiled on its own' in out.getvalue()


DICTS = '''
from scramjit import scram

@scram
def step(p, x):
    s = 0.0
    for i in range(p['n']):
        s += x[i] * p['k']
    p['s'] = s
    return s

@scram
def table(p):
    s = 0.0
    for i in range(len(p['a'])):
        s += p['a'][i] * p['k']
    return s

@scram
def not_fixed(p):
    p['new'] = 1.0
    return len(p)
'''


def test_static_understands_dict_arguments(tmp_path):
    f = tmp_path / 'dicts.py'
    f.write_text(DICTS)
    d = json.loads(_cli('preflight', f, '--json', '--no-suggest').stdout)
    states = {x['name']: x['state'] for x in d['files'][0]['functions']}
    assert states == {'step': 'compiled', 'table': 'compiled', 'not_fixed': 'python'}


OBJECTS = '''
import numpy as np
from scramjit import scram

class Spring:
    @scram
    def advance(self, dt, n):
        for i in range(n):
            self.x += self.v * dt
            self.log[i] = self.x

    @scram
    def calls_a_method(self):
        return self.norm() + 1.0

    def push(self, dv):
        self.v += dv * self.k

    @scram
    def uses_push(self, n):
        for i in range(n):
            self.push(0.5)
            self.log[i] = self.v

@scram
def reads(p, x):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * p.k
    return s

@scram
def is_an_array(x):
    return x.shape[0] + x.sum()

@scram
def passes_it_on(p):
    return str(p.a)
'''


def test_static_understands_instances(tmp_path):
    f = tmp_path / 'objects.py'
    f.write_text(OBJECTS)
    d = json.loads(_cli('preflight', f, '--json', '--no-suggest').stdout)
    states = {x['name']: x['state'] for x in d['files'][0]['functions']}
    assert states == {'advance': 'compiled', 'calls_a_method': 'python', 'uses_push': 'compiled', 'reads': 'compiled', 'is_an_array': 'compiled',
                      'passes_it_on': 'python'}, states


VIEWS = '''
from scramjit import scram

@scram
def by_row(m):
    s = 0.0
    for i in range(len(m)):
        row = m[i]
        for j in range(len(row)):
            s += row[j]
    return s

@scram
def by_iteration(m):
    s = 0.0
    for row in m:
        for j in range(len(row)):
            s += row[j]
    return s

@scram
def chained(m, i: int, j: int):
    return m[i][j]

@scram
def sliced(x):
    s = 0.0
    for v in x[1:]:
        s += v
    return s

@scram
def gives_back_a_view(x):
    return x[1:]
'''


def test_static_understands_views(tmp_path):
    f = tmp_path / 'views.py'
    f.write_text(VIEWS)
    d = json.loads(_cli('preflight', f, '--json', '--no-suggest').stdout)
    states = {x['name']: x['state'] for x in d['files'][0]['functions']}
    assert states == {'by_row': 'compiled', 'by_iteration': 'compiled', 'chained': 'compiled', 'sliced': 'compiled', 'gives_back_a_view': 'compiled'}


REDUCES = '''
import numpy as np
from scramjit import scram

@scram
def total(x):
    return x.sum()

@scram
def biggest(x):
    return np.max(x)

@scram
def by_row(m):
    s = 0.0
    for i in range(len(m)):
        row = m[i]
        s += row.sum()
    return s

@scram
def by_axis(m):
    return m.sum(axis=0)
'''


def test_static_understands_reductions(tmp_path):
    f = tmp_path / 'r.py'
    f.write_text(REDUCES)
    d = json.loads(_cli('preflight', f, '--json', '--no-suggest').stdout)
    states = {x['name']: x['state'] for x in d['files'][0]['functions']}
    assert states['total'] == states['biggest'] == 'compiled'
    assert states['by_row'] == 'compiled'


GLOBALS = '''
import numpy as np
from scramjit import scram

K = 3
RATE = 0.5
TABLE = (1.0, 2.0)
WEIGHTS = np.ones(4)
SETTINGS = {'dt': 0.5, 'steps': 4}
NAME = 'text'

@scram
def with_number(x):
    s = 0.0
    for i in range(K):
        s += x[i] * RATE
    return s

@scram
def with_tuple(x):
    return x[0] * TABLE[0]

@scram
def with_array(x):
    return x[0] * WEIGHTS[1]

@scram
def with_dict(x):
    s = 0.0
    for i in range(SETTINGS['steps']):
        s += x[i] * SETTINGS['dt']
    return s

@scram
def with_text(x):
    return x[0] if NAME == 'text' else x[1]
'''


def test_static_understands_module_level_constants(tmp_path):
    f = tmp_path / 'g.py'
    f.write_text(GLOBALS)
    d = json.loads(_cli('preflight', f, '--json', '--no-suggest').stdout)
    states = {x['name']: x['state'] for x in d['files'][0]['functions']}
    assert states == {'with_number': 'compiled', 'with_tuple': 'compiled', 'with_array': 'compiled', 'with_dict': 'compiled', 'with_text': 'python'}


NEW_OBJECTS = textwrap.dedent('''
    import dataclasses
    import math
    from scramjit import scram

    @dataclasses.dataclass
    class Vec:
        x: float
        y: float

        def __add__(self, o):
            return Vec(self.x + o.x, self.y + o.y)

        def __mul__(self, k):
            return Vec(self.x * k, self.y * k)

        def norm(self):
            return math.sqrt(self.x * self.x + self.y * self.y)

    @dataclasses.dataclass
    class Node:
        key: int
        left: object = None
        right: object = None

    @scram
    def drift(p, v, n, dt):
        for i in range(n):
            p = p + v * dt
        return (p + v).norm()

    @scram
    def depth(root, key):
        node = root
        d = 0
        while node is not None and node.key != key:
            node = node.left if key < node.key else node.right
            d += 1
        return d
''')


def test_static_report_follows_operators_and_references(tmp_path):
    (tmp_path / 'objs.py').write_text(NEW_OBJECTS)
    r = run(['-m', 'scramjit', 'preflight', 'objs.py'], tmp_path)
    assert '✓ drift' in r.stdout and '✓ depth' in r.stdout and '2 compiled' in r.stdout, r.stdout
