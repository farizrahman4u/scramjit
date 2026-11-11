"""The live demo's driver, run in Pyodide: each demo's state and work, and each pipeline stage's output."""

import ast
import json
import linecache
import math
import time
import warnings

import numpy as np

import scramjit

N = 128
PRESETS = {
    'erosion': {'entry': 'erode', 'demo': 'terrain'},
    'beacon': {'entry': 'viewshed', 'demo': 'terrain'},
    'landslides': {'entry': 'slide', 'demo': 'terrain'},
    'fractal': {'entry': 'mandelbrot', 'demo': 'fractal'},
    'sand': {'entry': 'sand', 'demo': 'sand'},
    'flock': {'entry': 'flock', 'demo': 'flock'},
}
S = {'k': 1, 'row': 0, 'angle': 0.0, 'preset': 'erosion', 'demo': None}
UPLIFT_S = 2.0       # the land rises back toward its first shape with this time constant, in seconds


def terrain(seed):
    """Fractal value noise (numpy, vectorized: this part isn't the demo)."""
    rng = np.random.default_rng(seed)
    h = np.zeros((N, N))
    amp = 1.0
    for octave in range(5):           # the finest octave is 4 cells across, not 1
        k = 2 ** (octave + 2)
        g = rng.random((k + 1, k + 1))
        t = np.linspace(0, k, N)
        i = np.minimum(t.astype(int), k - 1)
        f = t - i
        rows = g[i] * (1 - f)[:, None] + g[i + 1] * f[:, None]
        h += (rows[:, i] * (1 - f) + rows[:, i + 1] * f) * amp
        amp *= 0.45
    h = (h - h.min()) / (h.max() - h.min())
    return h ** 1.8                                    # peaks stand out, valleys open up


def reset(seed=7, preset=None):
    """Fresh state for a preset's demo."""
    preset = preset or S['preset']
    demo = PRESETS[preset]['demo']
    if preset != S['preset']:
        S['fn'] = None                                # the old kernel doesn't fit: no work until set_code
    S['preset'], S['demo'], S['row'], S['k'] = preset, demo, 0, 1
    S['rng'] = np.random.default_rng(seed)
    S['t'] = None
    if demo == 'terrain':
        S['h'] = terrain(seed)
        S['h0'] = S['h'].copy()
        S['vis'] = np.zeros((N, N), np.uint8)
    elif demo == 'fractal':
        S['img'] = np.zeros((FRACTAL_N, FRACTAL_N, 4), np.uint8)
        S['img'][..., 3] = 255
        S['pal'] = palette()
        S['dive'] = (seed % len(DIVES)) if seed != 7 else 0
        S['scale'] = 3.2 / FRACTAL_N
    elif demo == 'sand':
        S['cells'] = sand_world()
        S['sand_rng'] = np.array([seed], np.int64)
        S['grain'] = (S['rng'].random((SAND_H, SAND_W)) * 18).astype(np.int16)
    elif demo == 'flock':
        S['pos'] = S['rng'].random((BOIDS, 2)) * [FLOCK_W, FLOCK_H]
        a = S['rng'].random(BOIDS) * 2 * np.pi
        S['vel'] = np.stack([np.cos(a), np.sin(a)], 1) * 2.0
        S['pointer'] = None


# ---- the fractal: a dive into the Mandelbrot set, a frame at a time
FRACTAL_N = 240
DIVES = [(-0.743643887037158704752191506114774, 0.131825904205311970493132056385139),   # seahorse valley
         (-0.10109636384562, 0.95628651080914),
         (-1.7497591451303665, -0.0000000000000001),
         (0.360240443437614363236125244449545, -0.641313061064803174860375015179302)]


def palette(n=256):
    """A cyclic gradient: deep blue, scramjit's orange and red, and back."""
    stops = np.array([[8, 14, 40], [37, 99, 235], [147, 197, 253], [255, 247, 237],
                      [249, 115, 22], [185, 28, 28], [40, 8, 20], [8, 14, 40]], float)
    t = np.linspace(0, len(stops) - 1, n, endpoint=False)
    i = t.astype(int)
    f = (t - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def fractal_maxit():
    depth = math.log2(3.2 / FRACTAL_N / S['scale'])
    return int(min(400, 100 + depth * 22))


def fractal_target(fx, fy):
    """Steer the dive: the clicked point (fractions of the view) becomes the centre."""
    cx, cy = DIVES[S['dive']]
    nx = cx + (fx - 0.5) * FRACTAL_N * S['scale']
    ny = cy + (fy - 0.5) * FRACTAL_N * S['scale']
    DIVES.append((nx, ny))
    S['dive'] = len(DIVES) - 1


# ---- falling sand
SAND_W, SAND_H = 180, 210
SAND_COLORS = np.array([[10, 12, 22], [70, 76, 96], [232, 178, 92], [56, 132, 236], [24, 26, 38]], np.int16)


def sand_world():
    c = np.zeros((SAND_H, SAND_W), np.uint8)
    c[-1, :] = 1
    for x0 in (8, 78, 148):                           # drains in the floor
        c[-1, x0:x0 + 24] = 4
    for k, (x0, y0, x1) in enumerate([(12, 55, 50), (100, 70, 138), (52, 105, 92), (126, 130, 164), (22, 150, 60)]):
        for x in range(x0, x1):                        # slanted ledges, two cells thick
            y = y0 + (x - x0) // (4 if k % 2 == 0 else -4)
            c[y:y + 2, x] = 1
    return c


def paint(x, y, r, kind):
    """Paint a disc of sand, water or wall, or erase, from the page's pointer."""
    if S.get('demo') != 'sand':
        return
    yy, xx = np.ogrid[:SAND_H, :SAND_W]
    disc = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    disc[-1, :] = False                               # keep the floor and its drains
    if kind in (2, 3):
        disc &= S['rng'].random(disc.shape) < 0.5      # a spray, not a block
        disc &= S['cells'] == 0
    S['cells'][disc] = kind


# ---- the flock
BOIDS, FLOCK_W, FLOCK_H = 1200, 1200.0, 760.0
FLOCK_ARGS = (FLOCK_W, FLOCK_H, 42.0, 11.0, 0.06, 0.0035, 2.2)


def pointer(x, y):
    S['pointer'] = None if x is None or x < 0 else (x * FLOCK_W, y * FLOCK_H)


def flock_move():
    """After every boid has steered: scatter from the pointer, then move (numpy: not the demo)."""
    if S.get('pointer'):
        d = S['pos'] - S['pointer']
        d2 = (d ** 2).sum(1, keepdims=True)
        near = d2 < 140 ** 2
        S['vel'] += np.where(near, d / np.sqrt(d2 + 1) * 1.2, 0)
    S['pos'] = (S['pos'] + S['vel']) % [FLOCK_W, FLOCK_H]


def outline(node, depth=0, out=None, limit=60):
    """A compact syntax tree: node types with names, one per line."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    label = type(node).__name__
    for attr in ('name', 'id', 'arg', 'attr'):
        if isinstance(getattr(node, attr, None), str):
            label += f' {getattr(node, attr)}'
            break
    if isinstance(node, ast.Constant):
        label += f' {node.value!r}'
    if hasattr(node, 'op'):
        label += f' {type(node.op).__name__}'
    if not isinstance(node, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
        out.append('  ' * depth + label)
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
                outline(child, depth + 1, out, limit)
    return out


def short(node, n=72):
    text = ast.unparse(node).split('\n')[0]
    return text if len(text) <= n else text[:n - 1] + '…'


def names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [x for e in target.elts for x in names(e)]
    return []


def expr_tree(node, types):
    """An expression as nested nodes; names carry their inferred types."""
    kind = type(node).__name__
    if hasattr(node, 'op'):
        kind += ' ' + type(node.op).__name__
    out = {'k': kind, 't': short(node, 48)}
    if isinstance(node, ast.Name) and node.id in types:
        out['ty'] = types[node.id]
    elif isinstance(node, ast.Constant):
        out['ty'] = type(node.value).__name__
    kids = [expr_tree(c, types) for c in ast.iter_child_nodes(node)
            if isinstance(c, ast.expr) and not isinstance(c, ast.expr_context)]
    if kids:
        out['c'] = kids
    return out


def stmt_tree(node, types):
    """A statement: its kind, its source, the names it binds with their types, its expressions
    (shown on demand) and the statements it holds."""
    kind = type(node).__name__
    out = {'k': kind}
    bound = []
    if isinstance(node, ast.FunctionDef):
        out['t'] = f'def {node.name}'                 # the arguments follow, with their types
        bound = [a.arg for a in node.args.args]
    elif isinstance(node, (ast.For, ast.While, ast.If)):
        head = {'For': lambda: f'for {ast.unparse(node.target)} in {ast.unparse(node.iter)}',
                'While': lambda: f'while {ast.unparse(node.test)}',
                'If': lambda: f'if {ast.unparse(node.test)}'}[kind]()
        out['t'] = head if len(head) <= 72 else head[:71] + '…'
        if kind == 'For':
            bound = names(node.target)
    else:
        out['t'] = short(node)
        for t in getattr(node, 'targets', []) + [getattr(node, 'target', None)]:
            if t is not None:
                bound += names(t)
    out['b'] = [[n, types[n]] for n in dict.fromkeys(bound) if n in types]
    exprs = []
    for field, value in ast.iter_fields(node):
        if field in ('body', 'orelse', 'args', 'decorator_list', 'returns', 'type_comment'):
            continue
        for v in (value if isinstance(value, list) else [value]):
            if isinstance(v, ast.expr):
                exprs.append(expr_tree(v, types))
    if exprs:
        out['e'] = exprs
    kids = [stmt_tree(c, types) for c in getattr(node, 'body', [])
            if not (isinstance(c, ast.Expr) and isinstance(c.value, ast.Constant))]     # docstrings
    orelse = [stmt_tree(c, types) for c in getattr(node, 'orelse', [])]
    if orelse:
        kids.append({'k': 'Else', 't': 'else', 'b': [], 'c': orelse})
    if kids:
        out['c'] = kids
    return out


def set_code(preset, src, compiled):
    """Load the editor's code, as Python or through scramjit; return what each stage shows."""
    if PRESETS[preset]['demo'] != S.get('demo'):
        reset(preset=preset)
    S['preset'] = preset
    S['k'] = 1
    name = f'<{preset}.py>'
    linecache.cache[name] = (len(src), None, src.splitlines(True), name)
    info = {'state': 'python'}
    try:
        tree = ast.parse(src, name)
        ns = {}
        exec(compile(tree, name, 'exec'), ns)
        fn = ns[PRESETS[preset]['entry']]
    except Exception as e:  # noqa: BLE001
        S['fn'] = None
        return json.dumps({'state': 'error', 'error': f'{type(e).__name__}: {e}'})
    lines = outline(tree.body[-1])
    info['ast'] = '\n'.join(lines) + ('\n  …' if len(lines) >= 60 else '')
    S['fn'] = fn
    if not compiled:
        return json.dumps(info)
    f = scramjit.scram(fn)
    t = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            run(f, 1, warm=True)          # compiles on the first call
    except Exception as e:  # noqa: BLE001
        S['fn'] = None
        return json.dumps({'state': 'error', 'error': f'{type(e).__name__}: {e}', 'ast': info['ast']})
    info['compile_ms'] = round((time.perf_counter() - t) * 1000)
    fell = [str(x.message) for x in w if issubclass(x.category, scramjit.FallbackWarning)]
    S['fn'] = f
    if fell:
        info.update(state='fallback', reason=fell[0])
        return json.dumps(info)
    entry = next(v for v in f._cache.values() if hasattr(v, 'source'))
    fdef = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == PRESETS[preset]['entry'])
    info.update(state='compiled', c=entry.source, tree=stmt_tree(fdef, dict(entry.locals)))
    return json.dumps(info)


def wasm_text():
    """The compiled function as WebAssembly text: cimo's disassembler prints scramjit's kernel from
    the module the page already compiled."""
    f = S.get('fn')
    entry = next((v for v in getattr(f, '_cache', {}).values() if hasattr(v, 'source')), None)
    if entry is None:
        return 'Turn scramjit on to compile.'
    from cimo import wat
    mod = entry.module.wasm
    text, sections, others = wat.disassemble(mod, {'sj_kernel'})
    head = [f';; scramjit\'s kernel as WebAssembly text, disassembled from the {len(mod) / 1024:.1f} KB module',
            ';; cimo compiled in this page (sections: ' + ', '.join(f'{n.replace("custom ", "")} {z} B' for n, z in sections) + ')',
            f';; the module also holds {len(others)} runtime functions', '']
    return '\n'.join(head) + text


def run(f, k, warm=False):
    """Do about k units of the current preset's work with f; return the units done. warm: a token
    call on copies, which makes scramjit compile."""
    p = S['preset']
    if p == 'erosion':
        seeds = S['rng'].random(2 * k)
        f(S['h'] if not warm else S['h'].copy(), seeds, 0.05, 1.0, 0.3, 0.01, 0.02, 40)
        return k
    if p == 'beacon':
        a = S['angle']
        oy, ox = int(N / 2 + np.sin(a) * N * 0.3), int(N / 2 + np.cos(a) * N * 0.3)
        r0 = S['row'] if not warm else 0
        r1 = min(N, r0 + k)
        f(S['h'], S['vis'] if not warm else S['vis'].copy(), oy, ox, 0.08, r0, r1 if not warm else 1)
        if not warm:
            S['row'] = 0 if r1 >= N else r1
            if r1 >= N:
                S['angle'] = a + 0.12
        return (r1 - r0) * N                 # cells: one sightline each
    if p == 'landslides':
        r0 = S['row'] if not warm else 0
        r1 = min(N, r0 + k)
        f(S['h'] if not warm else S['h'].copy(), 0.012, 0.25, r0, r1 if not warm else 2)
        if not warm:
            S['row'] = 0 if r1 >= N else r1
        return (r1 - r0) * N                 # cells
    if p == 'fractal':                       # rows of the frame; a finished frame zooms in
        cx, cy = DIVES[S['dive']]
        r0 = S['row'] if not warm else 0
        r1 = min(FRACTAL_N, r0 + k) if not warm else 1
        its = f(S['img'] if not warm else S['img'].copy(), S['pal'], cx, cy, S['scale'], fractal_maxit(), r0, r1)
        if not warm:
            S['row'] = r1
            if r1 >= FRACTAL_N:
                S['row'] = 0
                S['frame_done'] = True
                S['scale'] *= 0.955
                # deep enough, or too deep for maxit to find the edge (mostly black): the next dive
                if S['scale'] < 1e-9 or (S['img'][..., :3].max(axis=2) == 0).mean() > 0.55:
                    S['dive'] = (S['dive'] + 1) % 4
                    S['scale'] = 3.2 / FRACTAL_N
        return int(its)                      # iterations
    if p == 'sand':                          # whole steps of the world
        k = min(k, 6 - S.get('sand_steps', 0)) if not warm else 1
        if warm:
            f(S['cells'].copy(), S['sand_rng'].copy(), 1)
        else:
            f(S['cells'], S['sand_rng'], k)
            S['sand_steps'] = S.get('sand_steps', 0) + k
        return k * SAND_H * SAND_W           # cells
    # flock: a band of boids steers by all the others; a finished pass moves them
    i0 = S['row'] if not warm else 0
    i1 = min(BOIDS, i0 + k) if not warm else 1
    f(S['pos'] if not warm else S['pos'].copy(), S['vel'] if not warm else S['vel'].copy(), i0, i1, *FLOCK_ARGS)
    if not warm:
        S['row'] = i1
        if i1 >= BOIDS:
            S['row'] = 0
            S['frame_done'] = True
            flock_move()
    return (i1 - i0) * (BOIDS - 1)            # pairs


def step(budget_ms):
    """Work for up to budget_ms; return units done per second. The batch size adapts. A demo whose
    motion has a natural pace stops early: a frame of the fractal or the flock, six sand steps."""
    if S.get('fn') is None:
        return json.dumps({'rate': 0})
    t = time.perf_counter()
    done = 0
    S['frame_done'] = False
    S['sand_steps'] = 0
    while (time.perf_counter() - t) * 1000 < budget_ms:
        k = S['k']
        t1 = time.perf_counter()
        try:
            n = run(S['fn'], k)
            done += n
        except Exception as e:  # noqa: BLE001
            S['fn'] = None
            return json.dumps({'rate': 0, 'error': f'{type(e).__name__}: {e}'})
        dt = max(time.perf_counter() - t1, 1e-5) * 1000
        S['k'] = max(1, min(int(k * min(2.0, (budget_ms / 3) / dt)) or 1, 2_000_000))
        if S['frame_done'] or S['sand_steps'] >= 6:
            break
    rate = done / max(time.perf_counter() - t, 1e-6)
    uplift()
    return json.dumps({'rate': rate, 'paced': S['demo'] in ('fractal', 'sand', 'flock')})


def uplift():
    """Erosion's steady state is a flat plain, and compiled code gets there in a fraction of a
    second. So the land also rises back toward its first shape, at a rate set by the clock, not by
    the work done: what you see is the balance, and more work per second carves deeper."""
    now = time.perf_counter()
    dt = min(now - S['t'], 0.1) if S['t'] is not None else 0.0
    S['t'] = now
    if dt > 0 and S['preset'] in ('erosion', 'landslides'):
        S['h'] += (S['h0'] - S['h']) * (1 - math.exp(-dt / UPLIFT_S))


def views():
    """What the page draws, by name: JSON names; view(name) gives each array."""
    d = S['demo']
    if d == 'terrain':
        return json.dumps(['h', 'vis'] if S['preset'] == 'beacon' else ['h'])
    return json.dumps({'fractal': ['rgba'], 'sand': ['rgba'], 'flock': ['pos', 'vel']}[d])


def view(name):
    if name == 'h':
        return S['h'].astype(np.float32)
    if name == 'vis':
        return S['vis']
    if name == 'rgba' and S['demo'] == 'fractal':
        return S['img']
    if name == 'rgba':                                  # sand: colours, with a little grain
        rgb = SAND_COLORS[S['cells']]
        rgb[..., :3] += np.where((S['cells'] == 2)[..., None], S['grain'][..., None], 0)
        out = np.empty((SAND_H, SAND_W, 4), np.uint8)
        out[..., :3] = np.clip(rgb, 0, 255)
        out[..., 3] = 255
        return out
    return S[name].astype(np.float32)


reset()
