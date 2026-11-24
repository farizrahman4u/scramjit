"""The page's Python side: weights, tokenizer, compiling the editor's source, generating."""
import ast
import hashlib
import importlib.util
import json
import os
import re
import struct
import sys
import time

import cimo
import lab  # the site's demos: the syntax tree with its types, and the WebAssembly text
import numpy as np
from scramjit import scram

BOS, EOS = 1, 2
# Fixed names: a compiled module's cache key comes from its source, and two workers share compiled modules.
_dir = '/tmp/forward_src'
os.makedirs(_dir, exist_ok=True)
_memo = {}
_fresh = []             # the cache entries of what was compiled last, for take_entries


class State:
    pass


S = State()


def setup(weights_path, tokenizer_path):
    raw = np.fromfile(weights_path, dtype=np.float32).astype(np.float64)
    with open(weights_path, 'rb') as f:
        dim, hidden, layers, heads, _, vocab, seq_len = struct.unpack('7i', f.read(28))
    vocab = abs(vocab)
    sizes = [vocab * dim, layers * dim] + [layers * dim * dim] * 4 + [layers * dim] + \
            [layers * dim * hidden] * 3 + [dim]
    S.weights, p = [], 7
    for n in sizes:
        S.weights.append(raw[p:p + n])
        p += n
    S.shape = (dim, hidden, layers, heads, seq_len, vocab)
    S.vocab, S.seq_len = vocab, seq_len
    S.caches = [np.zeros(layers * seq_len * dim) for _ in range(2)]
    S.logits = np.zeros(vocab)
    S.work = [np.zeros(n) for n in (dim, dim, dim, hidden, hidden, dim, dim, dim, seq_len)]
    _tokenizer(tokenizer_path)
    S.fn = S.plain = S.previous = None
    S.use_plain = False
    S.rng = np.random.default_rng()
    os.remove(weights_path)
    return json.dumps({'params': int(raw.size - 7), 'layers': layers, 'dim': dim, 'heads': heads})


def _tokenizer(path):
    with open(path, 'rb') as f:
        data = f.read()
    S.pieces, S.index, S.scores = [], {}, []
    p = 4
    for i in range(S.vocab):
        score, n = struct.unpack_from('fi', data, p)
        piece = data[p + 8:p + 8 + n].decode('utf8', 'replace')
        S.pieces.append(piece)
        S.scores.append(score)
        S.index.setdefault(piece, i)
        p += 8 + n


def encode(text):
    """llama2.c's tokenizer: one token per character, then merge the best-scoring pair until none is left."""
    out = []
    for ch in ' ' + text:
        if ch in S.index:
            out.append(S.index[ch])
        else:
            out.extend(S.index[f'<0x{b:02X}>'] for b in ch.encode('utf8'))
    while True:
        best, at = -1e10, -1
        for i in range(len(out) - 1):
            t = S.index.get(S.pieces[out[i]] + S.pieces[out[i + 1]])
            if t is not None and S.scores[t] > best:
                best, at = S.scores[t], i
        if at < 0:
            return out
        out[at:at + 2] = [S.index[S.pieces[out[at]] + S.pieces[out[at + 1]]]]


def token_of(word):
    """The id of a word that is exactly one token, or -1."""
    t = encode(word)
    return t[0] if len(t) == 1 else -1


def _decode(token):
    piece = S.pieces[token]
    m = re.fullmatch('<0x([0-9A-F]{2})>', piece)
    return bytes([int(m.group(1), 16)]).decode('latin1') if m else piece.replace('▁', ' ')


def _args(token, pos, caches=None, work=None):
    return (token, pos, *S.shape, *S.weights, *(caches or S.caches), S.logits, *(work or S.work))


def _load(code):
    name = 'forward_' + hashlib.sha1(code.encode()).hexdigest()[:16]
    path = os.path.join(_dir, name + '.py')
    with open(path, 'w') as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, 'forward', None)):
        raise ValueError('the code must define a function called forward')
    return mod.forward


def _message(e):
    text = f'{type(e).__name__}: {e}'
    text = re.sub(r'[\w/.-]*forward_[0-9a-f]+\.py', 'forward.py', text)
    text = re.sub(r'forward\((?:int|float64\[:\]|, )+\)', 'forward(...)', text)
    return text[:1500]


def _stages(fn, code):
    """What the tabs show: the generated C, and the syntax tree with the types scramjit inferred."""
    entry = next(v for v in fn._cache.values() if hasattr(v, 'source'))
    fdef = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    return {'c': entry.source, 'tree': lab.stmt_tree(fdef, dict(entry.locals))}


def compile_forward(code, compiled=True):
    """Swaps in the editor's source, compiled or as plain Python, or reports why not and leaves the running
    code alone. Generation state stays as it is: a trial call uses its own caches, so the story carries on."""
    t = time.perf_counter()
    try:
        out = {}
        if compiled:
            cached = code in _memo
            if cached:
                fn, plain = _memo[code]
            else:
                plain = _load(code)
                fn = scram(fallback=False)(plain)
                keep = S.logits.copy()
                fn(*_args(BOS, 0, [c.copy() for c in S.caches], [w.copy() for w in S.work]))
                ok = np.isfinite(S.logits).all()
                S.logits[:] = keep
                if not ok:
                    raise ValueError('this forward pass produces NaN or infinite logits')
                _memo[code] = fn, plain
                out = _stages(fn, code)
            _keep(fn)
            out['cached'] = cached
        else:
            fn, plain = None, _load(code)
        S.previous = S.fn, S.plain, S.use_plain
        S.fn, S.plain, S.use_plain = fn, plain, not compiled
        S.code, S.counted = code, None
        return json.dumps({'seconds': time.perf_counter() - t, **out})
    except BaseException as e:  # a compile error, or the trial call failing
        return json.dumps({'error': _message(e)})


def wasm_text(code):
    """The running compiled function as WebAssembly text."""
    lab.S['fn'] = _memo[code][0]
    return lab.wasm_text()


def cache_stamp():
    """What compiled files depend on, apart from the source they came from: other than this, they're dropped."""
    h = hashlib.sha256(sys.version.encode())
    for name in sorted(os.listdir('/home/pyodide/scramjit')) + ['../cimo.py']:
        path = os.path.join('/home/pyodide/scramjit', name)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()


def _keep(fn):
    """fn's compiled modules as cimo cache entries, each named by its content, for take_entries."""
    global _fresh
    _fresh = []
    for entry in fn._cache.values():
        if getattr(entry, 'module', None) is not None:
            data = entry.module.cache_entry()
            _fresh.append((hashlib.sha256(data).hexdigest()[:24], data))


def take_entries():
    """[(name, bytes)]: the cache entries of what was compiled last. Another worker, or a later visit, adds them
    with add_entry and then compiles the same code without compiling anything."""
    global _fresh
    out, _fresh = _fresh, []
    return out


def add_entry(data):
    """Puts a cache entry from take_entries into cimo's cache; False if cimo won't take it."""
    try:
        cimo.add_cache_entry(data.to_bytes())
        return True
    except ValueError:
        return False


def precompile(code, counts=False):
    """For the worker with no weights: compiles against arrays of the right types and keeps the cache entries
    (take_entries), so the worker that's writing the story only has to load them. Empty shapes: nothing runs."""
    t = time.perf_counter()
    error = fn = None
    try:
        fn = scram(fallback=False, counts=counts)(_load(code))
        fn(BOS, 0, 0, 0, 0, 6, 0, 0, *[np.zeros(1) for _ in range(23)])
    except BaseException as e:
        error = _message(e)
    # the call only exists to compile, and on these arrays it may well raise: that says nothing about the code
    if error and not (fn is not None and fn.signatures):
        return json.dumps({'error': error})
    _keep(fn)
    return json.dumps({'seconds': time.perf_counter() - t})


def count_lines(code, on):
    """Writes with a copy of the running code that counts its lines, or goes back to the one that doesn't."""
    if not on:
        S.counted = None
        return json.dumps({})
    if code != getattr(S, 'code', None) or S.use_plain:
        return json.dumps({'stale': True})
    try:
        fn = scram(fallback=False, counts=True)(S.plain)
        keep = S.logits.copy()
        fn(*_args(BOS, 0, [c.copy() for c in S.caches], [w.copy() for w in S.work]))
        S.logits[:] = keep
        S.counted = fn
        _keep(fn)
        return json.dumps({})
    except BaseException as e:
        return json.dumps({'error': _message(e)})


def start(prompt):
    S.prompt = encode(prompt) if prompt.strip() else []
    S.prompt_text = prompt
    _begin()


def _begin():
    S.token, S.pos, S.pending = BOS, 0, list(S.prompt)


def step(ms):
    """Generates for about `ms`, at least one token. Returns the text and the time in forward passes."""
    out, seconds, n = [], 0.0, 0
    fn = S.plain if S.use_plain else getattr(S, 'counted', None) or S.fn
    end = time.perf_counter() + ms / 1000
    try:
        while True:
            args = _args(S.token, S.pos)
            t = time.perf_counter()
            fn(*args)
            seconds += time.perf_counter() - t
            n += 1
            S.pos += 1
            if S.pending:
                token = S.pending.pop(0)
            else:
                p = S.logits - S.logits.max()
                if not np.isfinite(p).all():
                    raise ValueError('this forward pass produced NaN or infinite logits')
                p = np.exp(p)
                token = int(S.rng.choice(S.vocab, p=p / p.sum()))
            if token in (BOS, EOS) or S.pos >= S.seq_len:
                out.append('\n\n')
                _begin()
            else:
                out.append(_decode(token))
                S.token = token
            if time.perf_counter() >= end:
                break
    except BaseException as e:
        S.counted = None
        if S.previous and S.previous[1]:
            S.fn, S.plain, S.use_plain = S.previous      # the edit failed further into the story than its trial call
        _begin()
        return json.dumps({'error': _message(e), 'text': out, 'tokens': n, 'seconds': seconds})
    done = {'text': out, 'tokens': n, 'seconds': seconds}
    if fn is getattr(S, 'counted', None):
        done['lines'] = fn.line_counts()[0]
    return json.dumps(done)
