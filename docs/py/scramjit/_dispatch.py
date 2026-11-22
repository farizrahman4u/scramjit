"""The @scram decorator: signatures, the compile cache, the call boundary and fallback."""

import array
import ast
import collections
import itertools
import copy
import enum
import math
import functools
import operator
import inspect
import struct
import sys
import textwrap
import ctypes
import os
import threading
import time
import types
import warnings
import weakref

from . import _types as T
from ._compiler import OPERATORS, Callee, Compiler, Unit, array_record_slots, cimo_diagnostic, leaf_types, ret_arrays, ret_pairs, returns_made
from ._errors import CompileError, FallbackWarning, Unsupported
from . import _interrupt
from . import _lift
from ._runtime import LIBM, POOL_FULL, PRELUDE, RERUN, RETURNS_INT, exception

_cimo = None
_cimo_lock = threading.Lock()


def _get_cimo():
    """cimo, imported with fused multiply-add contraction off (Python never contracts)."""
    global _cimo
    if _cimo is None:
        with _cimo_lock:
            if _cimo is None:
                import cimo
                _cimo = cimo
    return _cimo


_libm = None
_libm_np = False


def _libm_table():
    """Addresses of the C library math functions CPython itself calls (LIBM's order), packed for
    compiled code, which calls them through this table: the same functions, the same results. Then numpy's own
    loops, two slots each in _nploops.LOOPS' order (0 where there's none), for the functions whose results the
    numpy present computes otherwise; read once numpy is imported."""
    global _libm, _libm_np
    np_loaded = 'numpy' in sys.modules
    if _libm is not None and _libm_np != np_loaded:
        _libm = None
    if _libm is None:
        _libm_np = np_loaded
        import ctypes
        import ctypes.util
        import struct as st
        if sys.platform == 'emscripten':         # Pyodide: libm is in the main module; addresses are table indices
            lib = ctypes.CDLL(None)
        elif sys.platform == 'win32':
            lib = ctypes.CDLL('ucrtbase')
        elif sys.platform == 'darwin':
            lib = ctypes.CDLL('libSystem.dylib')
        else:
            lib = ctypes.CDLL(ctypes.util.find_library('m') or 'libm.so.6')
        _libm = st.pack(f'<{len(LIBM)}Q', *(ctypes.cast(getattr(lib, n), ctypes.c_void_p).value for n in LIBM))
        from . import _nploops
        loops = [_nploops.loop(n, s) if np_loaded else None for n, s in _nploops.LOOPS]
        _libm += st.pack(f'<{2 * len(loops)}Q', *(v for lp in loops for v in (lp or (0, 0))))
    return _libm


def libm_address(name):
    """The address compiled code calls for the C library's name."""
    return struct.unpack_from('<Q', _libm_table(), 8 * LIBM.index(name))[0]


def _arg_type(x):
    """The compiled type of an argument, or None if it isn't compiled."""
    t = type(x)
    if issubclass(t, Function) or T.is_c_module(x) or T.is_c_function(x):
        return T.function_of(x)
    if t is int:                             # past int64 but below 2**64: held unsigned, passed as its bits
        return T.INT if -(1 << 63) <= x < (1 << 63) else T.U64 if 0 <= x < 1 << 64 else None
    if t is float:
        return T.FLOAT
    if t is bool:
        return T.BOOL
    if t is list and x and type(x[0]) is list:   # a list of lists: rectangular, every list its own object
        shape = _nested_shape(x)
        if shape is None:
            return None
        flat = _flatten(x, len(shape))
        elem = _arg_type(flat)
        return None if elem is None else T.array(elem.elem, True, len(shape))
    if t is list:                            # copied into an array at the call, and back if written
        kinds = set(map(type, x))                 # in C: a list of 200,000 in about 2 ms
        if len(kinds) != 1 or kinds.pop() not in (int, float, bool):
            return None
        if type(x[0]) is int and not (-(1 << 63) <= min(x) and max(x) < 1 << 63):
            return None
        return T.list_of({int: T.INT, float: T.FLOAT, bool: T.BOOL}[type(x[0])])
    if t is set:
        return _set_arg_type(x)
    if t is tuple:
        elems = [_arg_type(v) for v in x]
        if not elems or any(e is None or not T.is_number_tree(e) or e is T.U64 or (e.kind == 'tuple' and T.U64 in T.leaves(e)) for e in elems):
            return None
        return T.tuple_of(elems)
    if isinstance(x, enum.Enum) and T.enum_ok(t):       # a member: passed as its value
        return T.enum_member_of(t)
    if isinstance(x, type) and issubclass(x, enum.Enum):
        return T.enum_class_of(x) if T.enum_ok(x) else None
    if isinstance(x, type):
        return T.classref_of(x)
    np = sys.modules.get('numpy')
    if np is not None:
        if t is np.ndarray:
            if x.ndim >= 1 and x.dtype.isnative:
                name = x.dtype.name
                elem = T.NP_FLOAT64 if name == 'float64' else T.NP_INT64 if name == 'int64' else T.NP_UINT64 if name == 'uint64' else T.NARROW.get(name)
                if elem is not None:
                    if x.flags.c_contiguous:
                        return T.array(elem, x.flags.writeable, x.ndim)
                    if x.flags.aligned and all(s % x.itemsize == 0 for s in x.strides):      # a view, as it is
                        return T.array(elem, x.flags.writeable, x.ndim, True)
            return None
        if t is np.float64:
            return T.NP_FLOAT64
        if t is np.int64:
            return T.NP_INT64
        if t is np.bool_:
            return T.NP_BOOL
        if t is np.uint64:
            return T.NP_UINT64
        if issubclass(t, np.generic) and x.dtype.isnative and x.dtype.name in T.NARROW:     # np.intc too
            return T.NARROW[x.dtype.name]
    return None


def _describe(x, t):
    if t is not None:
        return t.name
    np = sys.modules.get('numpy')
    if np is not None and isinstance(x, np.ndarray):
        return f'{x.dtype}{"[:]" * x.ndim}' + ('' if x.flags.c_contiguous else ' (strided)')
    return type(x).__name__


_ANNOTATIONS = {int: 'int', float: 'float', bool: 'bool'}


def _contract_ok(x, ann):
    np = sys.modules.get('numpy')
    if ann is bool:
        return isinstance(x, bool) or (np is not None and isinstance(x, np.bool_))
    ints = (int,) + ((np.integer,) if np is not None else ())
    if ann is int:
        return isinstance(x, ints)
    return isinstance(x, ints + (float,) + ((np.floating,) if np is not None else ()))


def _raise_at(exc, filename, line, name):
    """Raise exc so the traceback's last frame is the user's file and line."""
    if not name.isidentifier():
        name = 'compiled'
    ns = {'__sj_exc__': exc}
    if line and line >= 2:
        exec(compile('\n' * (line - 2) + f'def {name}():\n raise __sj_exc__\n', filename, 'exec'), ns)
        ns[name]()
    raise exc


def _raise_chain(exc, frames):
    """Raise exc through a frame for each of [(file, line, function name)], outermost first, so the traceback reads as Python's would."""
    def run(i):
        if i == len(frames) - 1:
            _raise_at(exc, *frames[i])
        filename, line, name = frames[i]
        if not name.isidentifier():
            name = 'compiled'
        ns = {'__sj_next__': lambda: run(i + 1)}
        if line and line >= 2:
            exec(compile('\n' * (line - 2) + f'def {name}():\n __sj_next__()\n', filename, 'exec'), ns)
            ns[name]()
        run(i + 1)
    run(0)


def _mangle(fdef, qualname):
    """A function defined in a class: __x in it means _Class__x, as Python compiles it. Done on the tree once, so
    everything after sees the names Python uses."""
    parts = qualname.split('.')
    if len(parts) < 2 or parts[-2] == '<locals>' or not parts[-2].lstrip('_'):
        return
    cls = parts[-2].lstrip('_')

    def mangled(name):
        return f'_{cls}{name}' if name.startswith('__') and not name.endswith('__') else name
    for n in ast.walk(fdef):
        if isinstance(n, ast.Attribute):
            n.attr = mangled(n.attr)
        elif isinstance(n, ast.Name):
            n.id = mangled(n.id)
        elif isinstance(n, ast.arg):
            n.arg = mangled(n.arg)


def _dict_keys(fdef):
    """For each parameter subscripted by a constant string or int: the keys, if it turns out to be a dict."""
    if fdef is None:
        return {}
    index = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
    found = {}
    for n in _facts(fdef).subscripts:
        if isinstance(n.value, ast.Name) and n.value.id in index \
                and isinstance(n.slice, ast.Constant) and type(n.slice.value) in (str, int):
            found.setdefault(index[n.value.id], set()).add(n.slice.value)
    return {i: sorted(keys, key=repr) for i, keys in found.items()}


def _attrs_used(fdef):
    """For each parameter used as p.a: (the attributes read or assigned, sorted; those assigned; the methods called, as
    p.m(...)), if it turns out to be an instance. Not p.__a: in a class that means another name, which the source doesn't say."""
    direct = _path_uses(fdef)[0]
    return {i: (sorted(r[0]), set(r[1]), set(r[2])) for i, r in direct.items() if any(r[:3])}


def _made_locals(fdef, classes):
    """{name: class's name} for v = C(...) statements, C one of the names given."""
    out = {}
    facts = _facts(fdef)
    for n in facts.assigns + facts.returns:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call) \
                and isinstance(n.value.func, ast.Name) and n.value.func.id in classes:
            out[n.targets[0].id] = n.value.func.id
        elif isinstance(n, ast.Return) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id in classes:
            out[f'sjret__{n.lineno}_{n.col_offset}'] = n.value.func.id      # return C(...): made, and handed back
    return out


_DATACLASS_INIT_FILES = ('<string>',)


def _init_recipe(cls):
    """How cls(...) fills a new instance, if it only stores into it: a dataclass's own __init__, an __init__ of
    nothing but self.a = expression of its parameters, or none. A reason, as a string, if not."""
    import dataclasses
    if type(cls).__call__ is not type.__call__:
        return f'{cls.__name__} has a metaclass with its own __call__'
    if _in_class(cls, '__new__') is not object.__dict__['__new__']:
        return f'{cls.__name__} defines __new__'
    if _in_class(cls, '__getattribute__') is not object.__dict__['__getattribute__'] or _in_class(cls, '__getattr__') is not _MISSING:
        return f'{cls.__name__} defines __getattribute__ or __getattr__'
    init = _in_class(cls, '__init__')
    params, assigns = [], []
    frozen = dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
    if dataclasses.is_dataclass(cls) and cls.__dataclass_params__.init and getattr(getattr(init, '__code__', None), 'co_filename', '') in _DATACLASS_INIT_FILES:
        for f in cls.__dataclass_fields__.values():
            if f._field_type is not dataclasses._FIELD:
                if f._field_type is dataclasses._FIELD_INITVAR:
                    return f'{cls.__name__} has an InitVar'
                continue
            if f.default_factory is not dataclasses.MISSING:
                return f'{cls.__name__}.{f.name} has a default_factory'
            default = None if f.default is dataclasses.MISSING else ('const', f.default)
            if f.init:
                params.append((f.name, default, bool(getattr(f, 'kw_only', False))))
                assigns.append((f.name, ('param', f.name)))
            elif default is not None:
                assigns.append((f.name, default))
    elif init is object.__dict__['__init__']:
        pass
    elif type(init) is types.FunctionType:
        w = _method(init)
        tree = w._tree
        if tree is None or not tree.args.args or tree.args.vararg or tree.args.kwarg or tree.args.posonlyargs:
            return f'{cls.__name__}.__init__ takes what can\'t be bound here'
        me = tree.args.args[0].arg
        names = [a.arg for a in tree.args.args[1:w._npos]]
        defaults = list(init.__defaults__ or ())
        defaults = [None] * (len(names) - len(defaults)) + [('const', d) for d in defaults]
        kwd = init.__kwdefaults__ or {}
        params = [(n, d, False) for n, d in zip(names, defaults)]
        params += [(a.arg, ('const', kwd[a.arg]) if a.arg in kwd else None, True) for a in w._kwonly_args]
        allowed = set(names) | {a.arg for a in w._kwonly_args}
        for k, s in enumerate(tree.body):
            if k == 0 and isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                continue
            if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Attribute)
                    and isinstance(s.targets[0].value, ast.Name) and s.targets[0].value.id == me):
                return f'{cls.__name__}.__init__ does more than assign attributes of {me}'
            used = {n.id for n in ast.walk(s.value) if isinstance(n, ast.Name)}
            if not used <= allowed or any(isinstance(n, (ast.Call, ast.Lambda, ast.NamedExpr)) for n in ast.walk(s.value)):
                return f'{cls.__name__}.__init__ assigns something other than an expression of its parameters'
            assigns.append((s.targets[0].attr, ('expr', s.value)))
    else:
        return f'{cls.__name__}.__init__ is not a plain function'
    for a, _ in assigns:
        c = _in_class(cls, a)
        if c is not _MISSING and (type(c) is not types.MemberDescriptorType and any(hasattr(type(c), m) for m in ('__get__', '__set__', '__delete__'))):
            return f'{cls.__name__}.{a} is a descriptor'
    if not frozen and _in_class(cls, '__setattr__') is not _DEFAULT_SET:
        return f'{cls.__name__} defines __setattr__'
    return {'params': params, 'assigns': assigns, 'post_init': _in_class(cls, '__post_init__') is not _MISSING, 'frozen': bool(frozen)}


def _local_classes(f, owners):
    """{name: class} for the names in f holding an instance it makes: v = C(...), and v = g(...), v = x.m(...) or v = x + y
    when what's called returns a new instance. owners: the classes of names known already (its instance parameters)."""
    tree = f._ctree or f._tree
    g = f._fn.__globals__
    known, out = dict(owners or {}), {}
    for _ in range(8):
        grew = False
        for n in _facts(tree).assigns:
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id not in out:
                cls = _made_by(n.value, g, known)
                if cls is not None:
                    out[n.targets[0].id] = known[n.targets[0].id] = cls
                    grew = True
        if not grew:
            break
    return out


def _made_by(e, g, known):
    """The class of the new instance an expression makes: C(...), or a call of a function or method returning one."""
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
        x = g.get(e.func.id)
        if isinstance(x, type) and not issubclass(x, (enum.Enum, BaseException)):
            return x
        if isinstance(x, Function) and x._tree is not None:
            return returns_made(x._tree, x._fn.__globals__)
        return None
    w = None
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute):
        c = _class_of(e.func.value, g, known)
        w = _in_class(c, e.func.attr) if c is not None else None
    elif isinstance(e, (ast.BinOp, ast.UnaryOp)) and type(e.op) in OPERATORS:
        c = _class_of(e.left if isinstance(e, ast.BinOp) else e.operand, g, known)
        w = _in_class(c, OPERATORS[type(e.op)]) if c is not None else None
    if isinstance(w, Function) or type(w) is types.FunctionType:
        w = _method(w)
        if w._tree is not None:
            return returns_made(w._tree, w._fn.__globals__)
    return None


def _class_of(e, g, known):
    """The class of the instance an expression is: a name whose class is known, or a new instance."""
    if isinstance(e, ast.Name):
        return known.get(e.id)
    return _made_by(e, g, known)


def _unnamed_uses(tree, g, known, cls):
    """(attribute, 'load' or 'call') used directly on a new instance of cls that has no name: (a - b).norm(), (a + b).x."""
    facts = _facts(tree)
    called = facts.called
    for n in facts.attributes + facts.operators:
        if isinstance(n, ast.Attribute) and not isinstance(n.value, ast.Name) and _made_by(n.value, g, known) is cls:
            yield n.attr, 'call' if id(n) in called else 'load'
        elif isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPERATORS:
            x = n.left if isinstance(n, ast.BinOp) else n.operand
            if not isinstance(x, ast.Name) and _made_by(x, g, known) is cls:
                yield OPERATORS[type(n.op)], 'call'


def _callees_of(f, known):
    """[(compiled function, {its parameter: class})] for what f calls: functions at module level by name, and methods of
    names whose class is known, operators too."""
    tree = f._ctree or f._tree
    g = f._fn.__globals__
    out = []
    facts = _facts(tree)
    for n in facts.calls + facts.operators:
        w, me = None, None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and isinstance(g.get(n.func.id), Function):
            out.append((g[n.func.id], {}))
            continue
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            me = _class_of(n.func.value, g, known)
            w = _in_class(me, n.func.attr) if me is not None else None
        elif isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPERATORS:
            me = _class_of(n.left if isinstance(n, ast.BinOp) else n.operand, g, known)
            w = _in_class(me, OPERATORS[type(n.op)]) if me is not None else None
        if isinstance(w, Function) or type(w) is types.FunctionType:
            w = _method(w)
            ps = w._tree.args.posonlyargs + w._tree.args.args if w._tree is not None else []
            if ps:
                out.append((w, {ps[0].arg: me}))
    return out


def _made_uses(f, cls, owners, seen=None):
    """(attributes used, assigned, methods called) of the instances of cls that f, or anything it calls, makes; None if
    nothing does. owners: {name: class} of f's instance parameters."""
    seen = set() if seen is None else seen
    if id(f) in seen or f._tree is None:
        return None
    seen.add(id(f))
    tree = f._ctree or f._tree
    g = f._fn.__globals__
    mine = _local_classes(f, owners)
    known = dict(owners or {})
    known.update(mine)
    local = {v for v, c in mine.items() if c is cls}
    facts = _facts(tree)
    handed = any(n.value is not None and _made_by(n.value, g, known) is cls for n in facts.returns) \
        or any(_made_by(n, g, known) is cls for n in facts.calls + facts.operators)
    uses = None
    if local or handed:
        uses = (set(), set(), set())
        for v in local:
            u = facts.first_uses(v)
            for k in range(3):
                uses[k].update(u[k])
        for attr, how in _unnamed_uses(tree, g, known, cls):
            _add_use(uses, attr, how)
        for n in facts.calls:                    # passed on: what the callee makes of it
            if isinstance(n.func, ast.Name):
                h = g.get(n.func.id)
                for pos, a in enumerate(n.args):
                    if isinstance(a, ast.Name) and a.id in local and isinstance(h, Function) and h._tree is not None:
                        u = _param_uses(h, pos, cls, h._attrs.get(pos, ((), set(), set())))
                        for k in range(3):
                            uses[k].update(u[k])
    for h, theirs in _callees_of(f, known):      # what they make and hand back, or make for themselves
        u = _made_uses(h, cls, theirs, seen)
        if u is not None:
            uses = uses or (set(), set(), set())
            for k in range(3):
                uses[k].update(u[k])
    return uses


def _made_class_type(f, cls, name, owners=None):
    """The type of a class the function, or what it calls, makes instances of, under this name; None if they can't be
    compiled, or nothing makes any."""
    found = _made_uses(f, cls, owners)
    if found is None:
        return None
    recipe = _init_recipe(cls)
    if type(recipe) is str:
        f._made_why = recipe
        return None
    recipe = dict(recipe)
    uses = found
    if recipe['post_init']:
        uses[2].add('__post_init__')
    assigned = {a for a, _ in recipe['assigns']}
    found = _methods_called(None, uses[2], cls) if uses[2] else {}
    if type(found) is str:
        f._made_why = found
        return None
    names = set(uses[0]).union(*(n for n, _, _ in found.values()))
    writes = set(uses[1]).union(*(w for _, w, _ in found.values()))
    if not names <= assigned:
        f._made_why = f'{cls.__name__}.__init__ doesn\'t assign {sorted(names - assigned)[0]}'
        return None
    if writes and recipe['frozen']:
        f._made_why = f'{cls.__name__} is frozen'
        return None
    methods = {m: (n, w) for m, (n, w, _) in found.items()}
    return T.madeclass_of(cls, recipe, methods, writes, [(m, T.function_of(w)) for m, (_, _, w) in sorted(found.items())])


def _merge_nested(a, b):
    """Two {attribute: [used, assigned, called, deeper]} as one."""
    if not b:
        return a
    out = {k: [set(v[0]), set(v[1]), set(v[2]), dict(v[3])] for k, v in (a or {}).items()}
    for k, v in b.items():
        e = out.setdefault(k, [set(), set(), set(), {}])
        e[0] |= v[0]
        e[1] |= v[1]
        e[2] |= v[2]
        e[3] = _merge_nested(e[3], v[3])
    return out


def _chain(n):
    """(name, [attributes]) for name.a.b..., else (None, None)."""
    path = []
    while isinstance(n, ast.Attribute):
        path.append(n.attr)
        n = n.value
    return (n.id, path[::-1]) if isinstance(n, ast.Name) else (None, None)


def _nested_uses(fdef):
    """For each parameter used as p.a.b: {a: [attributes of p.a used, assigned, methods called, {deeper}]}."""
    direct = _path_uses(fdef)[0]
    return {i: copy.deepcopy(r[3]) for i, r in direct.items() if r[3]}


def _object_aliases(fdef):
    """{q: p} for q = p at the top of the body, p a parameter: the compiler takes q as p where that's safe."""
    params = {p.arg for p in fdef.args.posonlyargs + fdef.args.args}
    return {s.targets[0].id: s.value.id for s in fdef.body if isinstance(s, ast.Assign) and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Name) and s.value.id in params and s.targets[0].id not in params}


def _returned(fdef):
    """The parameters returned as they are, by name or through q = p, or only asked isinstance() of."""
    if fdef is None:
        return set()
    index = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
    index.update({q: index[p] for q, p in _object_aliases(fdef).items()})
    facts = _facts(fdef)
    out = {index[n.value.id] for n in facts.returns if isinstance(n.value, ast.Name) and n.value.id in index}
    return out | {index[n.args[0].id] for n in facts.calls if isinstance(n.func, ast.Name)       # isinstance(p, C)
                  and n.func.id == 'isinstance' and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id in index}


def _element_nested(fdef):
    """For each parameter that may be a list of instances: {attribute: [used, assigned, called, {deeper}]} of the objects in
    its elements' attributes, through the names _element_attrs follows: b.pos.x for b in bs."""
    elems = _path_uses(fdef)[1]
    return {i: copy.deepcopy(r[3]) for i, r in elems.items() if r[3]}


def _element_alias(fdef, index):
    """{name: list parameter} for names bound to an element: for p in ps, for i, p in enumerate(ps), p = ps[i]."""
    return _facts(fdef).element_alias(index)


def _element_attrs(fdef):
    """For each parameter that may be a list of instances: the attributes its elements are used with, as _attrs_used
    gives them, through for p in ps, for i, p in enumerate(ps), p = ps[i], and ps[i].a."""
    elems = _path_uses(fdef)[1]
    return {i: (sorted(r[0]), set(r[1]), set(r[2])) for i, r in elems.items() if any(r[:3])}


def _path_uses(fdef):
    """Every parameter's use tree: ({index: tree} for the parameter itself, through q = p at the top; {index: tree} for its
    elements, through for p in ps, enumerate, p = ps[i] and ps[i]; {index: [(target, position[, path])]} for what it, an
    element of it or an object in its attribute is given to). Read from _facts."""
    if fdef is None:
        return {}, {}, {}
    return _facts(fdef).params


class _Facts:
    """What one walk of a function finds, for every analysis that asks about its names: each name's use tree by path
    (names; subs for ps[i]), the operators applied to it (ops), what it's given to (given, subs_given), and the nodes of each
    kind those analyses look at (calls, operators, returns, assigns, compares, loops), with how often each name is stored
    and which are stored inside a loop or an if (stores, looped)."""

    def __init__(self, fdef):
        self.fdef = fdef
        self.names, self.subs, self.ops = {}, {}, {}
        self.given, self.subs_given = {}, {}
        self.calls, self.operators, self.returns, self.assigns, self.compares, self.loops, self.attributes = [], [], [], [], [], [], []
        self.stores, self.looped = {}, set()
        self.called, self.subscripts, self.seen = set(), [], set()
        self._walk(fdef.body, False)
        self.params = self._params()
        self.ref_candidates = self._refs()

    def _walk(self, nodes, inside):
        """One pass, keeping whether each node is inside a loop's or an if's body."""
        todo = [(n, inside) for n in reversed(nodes)]
        while todo:
            n, inside = todo.pop()
            if isinstance(n, (ast.For, ast.While, ast.If)):
                if isinstance(n, ast.For):
                    self.loops.append(n)
                head = [n.test] if not isinstance(n, ast.For) else [n.target, n.iter]
                todo += [(x, True) for x in reversed(n.body + n.orelse)] + [(x, inside) for x in reversed(head)]
                continue
            self._see(n, inside)
            todo += [(x, inside) for x in reversed(list(ast.iter_child_nodes(n)))]

    def _see(self, n, inside):
        if isinstance(n, ast.Name):
            self.seen.add(n.id)
        if isinstance(n, ast.Subscript):
            self.subscripts.append(n)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            self.stores[n.id] = self.stores.get(n.id, 0) + 1
            if inside:
                self.looped.add(n.id)
        elif isinstance(n, ast.Call):
            self.calls.append(n)
            self.called.add(id(n.func))
            f = n.func
            target = ('f', f.id) if isinstance(f, ast.Name) else ('m', f.value.id, f.attr) \
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ('e', f.value, f.attr) \
                if isinstance(f, ast.Attribute) else None   # (a - b).dot(c): a method of a new instance, its class found later
            if target is not None:
                self._given(target, n.args)
        elif isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPERATORS:
            self.operators.append(n)
            x = n.left if isinstance(n, ast.BinOp) else n.operand
            if isinstance(x, ast.Name):
                self.ops.setdefault(x.id, set()).add(OPERATORS[type(n.op)])
            if isinstance(n, ast.BinOp):     # a + b gives b to a.__add__
                self._given(('m', x.id, OPERATORS[type(n.op)]) if isinstance(x, ast.Name) else ('e', x, OPERATORS[type(n.op)]), [n.right])
        elif isinstance(n, ast.Return):
            self.returns.append(n)
        elif isinstance(n, ast.Assign):
            self.assigns.append(n)
        elif isinstance(n, ast.Compare):
            self.compares.append(n)
        elif isinstance(n, ast.comprehension):
            self.loops.append(n)
        elif isinstance(n, ast.Attribute):
            self.attributes.append(n)

    def _given(self, target, args):
        for pos, a in enumerate(args):
            if isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name):
                self.subs_given.setdefault(a.value.id, []).append((target, pos))
            elif isinstance(a, ast.Name):
                self.given.setdefault(a.id, []).append((target, pos))
            elif isinstance(a, ast.Attribute):
                base, path = _chain(a)
                if base is not None:
                    self.given.setdefault(base, []).append((target, pos, tuple(path)))

    def uses(self):
        """Each name's use tree (and each list's, through ps[i]), [used, assigned, called, {attribute: tree}] by path. Not
        p.__a: in a class that means another name, which the source doesn't say."""
        if getattr(self, '_trees', None) is not None:
            return self._trees
        def private(a):
            return a.startswith('__') and not a.endswith('__')
        names, subs = {}, {}
        for n in self.attributes:
            path, e = [], n
            while isinstance(e, ast.Attribute):
                path.append(e.attr)
                e = e.value
            path.reverse()
            if isinstance(e, ast.Name):
                root = names.setdefault(e.id, [set(), set(), set(), {}])
            elif isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name):
                root = subs.setdefault(e.value.id, [set(), set(), set(), {}])
            else:
                continue
            how = 'call' if id(n) in self.called else 'load' if isinstance(n.ctx, ast.Load) else 'store'
            if len(path) == 1:
                if not private(path[0]):
                    _add_use(root, path[0], how)
                continue
            if any(private(a) for a in path):
                continue
            node = root[3]
            for a in path[:-2]:
                node = node.setdefault(a, [set(), set(), set(), {}])[3]
            _add_use(node.setdefault(path[-2], [set(), set(), set(), {}]), path[-1], how)
        self._trees = (names, subs)
        return self._trees

    def first_uses(self, name):
        """(attributes used, assigned, methods called) of a name, operators as the methods they call."""
        names, _ = self.uses()
        r = names.get(name)
        return (set(r[0]) if r else set(), set(r[1]) if r else set(), (set(r[2]) if r else set()) | self.ops.get(name, set()))

    def element_alias(self, index):
        """{name: list parameter} for names bound to an element: for p in ps, for i, p in enumerate(ps), p = ps[i]."""
        alias = {}
        for n in self.loops:
            target, it = n.target, n.iter
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'enumerate' and it.args \
                    and isinstance(target, ast.Tuple) and len(target.elts) == 2:
                it, target = it.args[0], target.elts[1]
            if isinstance(target, ast.Name) and isinstance(it, ast.Name) and it.id in index:
                alias[target.id] = it.id
        for n in self.assigns:
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Subscript) \
                    and isinstance(n.value.value, ast.Name) and n.value.value.id in index:
                alias[n.targets[0].id] = n.value.value.id
        return alias

    def _params(self):
        fdef = self.fdef
        plain = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
        index = dict(plain)
        index.update({q: index[p] for q, p in _object_aliases(fdef).items()})
        alias = self.element_alias(plain)
        given_alias = self.element_alias(index)
        names, subs = self.uses()
        direct, elems, passes = {}, {}, {}
        for name, tree in names.items():
            if name in index:
                _node_merge(direct.setdefault(index[name], [set(), set(), set(), {}]), tree)
            elif name in alias:
                _node_merge(elems.setdefault(plain[alias[name]], [set(), set(), set(), {}]), tree)
        for name, tree in subs.items():
            if name in plain:
                _node_merge(elems.setdefault(plain[name], [set(), set(), set(), {}]), tree)
        for name, ops in self.ops.items():
            if name in index:
                direct.setdefault(index[name], [set(), set(), set(), {}])[2].update(ops)
        for name, found in self.given.items():
            if name in index or name in given_alias:
                passes.setdefault(index[given_alias.get(name, name)], []).extend(found)
        for name, found in self.subs_given.items():
            if name in index:
                passes.setdefault(index[name], []).extend(found)
        return direct, elems, passes

    def _refs(self):
        """(attributes, names) that may hold references chosen by data: compared with None, given None or an object, or
        read into a name that's rebound (n = n.next). The values at the call decide which really are."""
        params = {p.arg for p in self.fdef.args.posonlyargs + self.fdef.args.args}

        def none(e):
            return isinstance(e, ast.Constant) and e.value is None

        def parts(e):                            # what a value may be: both sides of a if c else b
            return parts(e.body) + parts(e.orelse) if isinstance(e, ast.IfExp) else [e]
        attrs, names = set(), set()
        for n in self.compares:
            sides = [n.left] + n.comparators
            for k, op in enumerate(n.ops):
                if isinstance(op, (ast.Is, ast.IsNot)):
                    for x in (sides[k], sides[k + 1]):
                        if isinstance(x, ast.Attribute):
                            attrs.add(x.attr)
                        elif isinstance(x, ast.Name):
                            names.add(x.id)
                        elif isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name):
                            names.add(x.value.id)    # heads[i] is None: a list of references
        rebound = lambda t: self.stores.get(t, 0) > 1 or t in self.looped or t in params
        for n in self.assigns:
            if len(n.targets) != 1:
                continue
            t, v = n.targets[0], n.value
            vs = parts(v)
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and any(isinstance(x, ast.Attribute) or none(x) for x in vs) \
                    and all(isinstance(x, (ast.Name, ast.Attribute)) or none(x) for x in vs):
                names.add(t.value.id)                # heads[i] = heads[i].next
                attrs.update(x.attr for x in vs if isinstance(x, ast.Attribute))
            if isinstance(t, ast.Attribute) and all(isinstance(x, (ast.Name, ast.Attribute)) or none(x) for x in vs):
                attrs.add(t.attr)
            elif isinstance(t, ast.Name) and all(none(x) for x in vs):
                names.add(t.id)
            elif isinstance(t, ast.Name) and all(isinstance(x, ast.Attribute) or none(x) for x in vs) and rebound(t.id):
                attrs.update(x.attr for x in vs if isinstance(x, ast.Attribute))
                names.add(t.id)
            elif isinstance(t, ast.Name) and isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id[:1].isupper() and \
                    rebound(t.id) and any(isinstance(a, ast.Name) and a.id == t.id for a in v.args):
                names.add(t.id)                      # head = Node(x, head): a new instance holding the last one
        for _ in range(8):
            grew = False
            for n in self.assigns:
                if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    t, v = n.targets[0].id, n.value
                    if isinstance(v, ast.Name) and (v.id in names) != (t in names):    # cur = head: both name one
                        names |= {t, v.id}
                        grew = True
                    for x in parts(v):
                        if t in names and isinstance(x, ast.Attribute) and x.attr not in attrs:
                            attrs.add(x.attr)
                            grew = True
            if not grew:
                break
        return attrs, names


def _facts(fdef):
    """The facts of a function's tree, found once: kept on the tree with its parameter count, since a copy given more
    parameters (the globals it reads) is walked again."""
    count = len(fdef.args.posonlyargs + fdef.args.args)
    cached = getattr(fdef, '_sj_facts', None)
    if cached is not None and cached[0] == count:
        return cached[1]
    f = _Facts(fdef)
    fdef._sj_facts = (count, f)
    return f


def _objlist_type(xs, attrs, seen, dicts=False, nested=None):
    """A list of instances, all one class whose lookups run no code, holding numbers in every attribute used, or objects
    of one class each doing the same (each its own object), and not passed otherwise. Its numbers are copied into a list
    each, at the call, by path (pos.x)."""
    if not xs:
        return None
    cls = type(xs[0])
    if set(map(type, xs)) != {cls}:
        return None
    methods = {}
    fields = _object_fields(xs[0], *attrs, methods=methods, nested=nested)
    if type(fields) is not list:
        return None
    ids = list(map(id, xs))
    shared = len(set(ids)) != len(ids)           # an instance twice in it: each once in a pool, and an index list
    if shared:
        xs = list({id(x): x for x in xs}.values())
        ids = list(map(id, xs))
    if seen.intersection(ids):
        return None                              # an instance here and in another argument: not yet
    seen.update(ids)
    written = set(attrs[1]).union(*(w for _, w in methods.values()))
    has_dict = hasattr(xs[0], '__dict__')
    levels, pooled = {}, set()
    if not _inner_ok(xs, fields, levels, pooled):
        return None
    inner = [id(o) for objs in levels.values() for o in objs]
    if len(set(inner)) != len(inner) or seen.intersection(inner) or set(inner) & set(ids):
        return None                              # an object at two paths, or here and elsewhere: not yet
    seen.update(inner)
    if shared:
        pooled.add('')
    for k, ft in fields:
        if ft.kind == 'function':
            if has_dict and any(k in x.__dict__ for x in xs):
                return None
            continue
        if ft.kind == 'object':
            continue
        if ft not in (T.INT, T.FLOAT, T.BOOL):
            return None
        want = {T.INT: int, T.FLOAT: float, T.BOOL: bool}[ft]
        try:
            values = list(map(operator.attrgetter(k), xs))      # in C: the class was checked, so this runs none of its code
        except AttributeError:
            return None
        if set(map(type, values)) != {want} or (want is int and not (-(1 << 63) <= min(values) and max(values) < 1 << 63)):
            return None
        if k in written and has_dict and type(_in_class(cls, k)) is not types.MemberDescriptorType and not all(k in d for d in map(vars, xs)):
            return None                          # assigning would add it to an instance that only had the class's
    if dicts and has_dict:                       # an instance's own __dict__, passed as a dict too, is one thing under two names
        held = list(map(id, map(vars, xs)))
        if seen.intersection(held):
            return None
        seen.update(held)
    return T.objlist_of(T.object_of(cls.__qualname__, fields, methods, cls, written), pooled)


def _inner_ok(xs, fields, levels, pooled, prefix=''):
    """Every element's objects in attributes are of the class the first's are, holding the same types. levels: path -> the
    objects there, each once; pooled gets the paths where one object is at several places."""
    for k, ft in fields:
        if ft.kind != 'object':
            continue
        try:
            objs = list(map(operator.attrgetter(k), xs))
        except AttributeError:
            return False
        if set(map(type, objs)) != {ft.cls}:
            return False
        once = list({id(o): o for o in objs}.values())
        if len(once) != len(objs):
            pooled.add(prefix + k)
        objs = levels[prefix + k] = once
        has_dict = hasattr(objs[0], '__dict__')
        for k2, ft2 in ft.elem:
            if ft2.kind == 'function':
                if has_dict and any(k2 in o.__dict__ for o in objs):
                    return False
            elif ft2.kind != 'object':
                want = {T.INT: int, T.FLOAT: float, T.BOOL: bool}.get(ft2)
                vals = list(map(operator.attrgetter(k2), objs))
                if want is None or set(map(type, vals)) != {want} or (want is int and not (-(1 << 63) <= min(vals) and max(vals) < 1 << 63)):
                    return False
        if not _inner_ok(objs, ft.elem, levels, pooled, prefix + k + '.'):
            return False
    return True


def _pool_levels(pool, t):
    """For a list of instances' type: [(path, the objects there each once, or at each place if not pooled; the index list
    or None)], breadth first as its parameters come, the list itself first."""
    out = [('', pool, None)]
    objs_at = {'': pool}
    todo = [('', t.elem.elem)]
    while todo:
        prefix, fields = todo.pop(0)
        for key, ft in fields:
            if ft.kind != 'object':
                continue
            path = prefix + key
            objs = list(map(operator.attrgetter(key), objs_at[prefix[:-1]]))
            ix = None
            if path in t.strided:
                where = {}
                ix = array.array('q', (where.setdefault(id(o), len(where)) for o in objs))
                objs = list({id(o): o for o in objs}.values())
            objs_at[path] = objs
            out.append((path, objs, ix))
            todo.append((path + '.', ft.elem))
    return out


def _changed(a, before):
    """The places where the kernel's list a holds other bits than were copied in: what this call wrote. -0.0 differs from
    0.0; a NaN it didn't write doesn't differ from itself. Compared a block at a time as bytes, then element by element
    only in the blocks that differ."""
    raw, old = a.tobytes(), before.tobytes()
    if raw == old:
        return ()
    size = a.itemsize
    step = 4096 * size
    out = []
    for at in range(0, len(raw), step):
        if raw[at:at + step] != old[at:at + step]:
            x, y = array.array('q' if size == 8 else 'i'), array.array('q' if size == 8 else 'i')
            x.frombytes(raw[at:at + step])
            y.frombytes(old[at:at + step])
            first = at // size
            out.extend(first + k for k in itertools.compress(range(len(x)), map(operator.ne, x, y)))
    return out


def _write_numbers(x, new, et, before=None, now=None):
    """A list of plain numbers passed in, held by reference, as the kernel left it (new): compared bit for bit in C, the
    elements that didn't change kept as the very objects they were. before, now: their bytes, if known."""
    m = min(len(x), len(new))
    if before is not None and now is not None:
        a, b = array.array('q', before[:8 * m]), array.array('q', now[:8 * m])
    else:
        code = 'd' if et is T.FLOAT else 'q'
        a, b = array.array(code, x if len(x) == m else x[:m]), array.array(code, new if len(new) == m else new[:m])
        if code == 'd':                          # bits: -0.0 isn't 0.0, and a NaN is itself
            a, b = array.array('q', a.tobytes()), array.array('q', b.tobytes())
    if a != b:
        raw, old, step = a.tobytes(), b.tobytes(), 4096 * 8
        blocks = sum(raw[at:at + step] != old[at:at + step] for at in range(0, len(raw), step))
        if blocks * 4096 * 8 < m:                # a few places: found and assigned one by one
            for k in _changed(b, a):
                x[k] = new[k]
        elif blocks * 4096 * 2 < m:
            x[:m] = [o if same else v for o, v, same in zip(x, new, map(operator.eq, a, b))]
        else:                                    # most of it (as after a sort): refilled, in C
            x[:m] = new if len(new) == m else new[:m]
    if len(new) > m:
        x.extend(new[m:])
    elif len(x) > m:
        del x[m:]


def _write_changed(lst, a, et, before):
    """lst[k] = a[k] where this call changed it, as Python's values; all of them in one go when it changed every one."""
    ks = _changed(a, before)
    if len(ks) == len(lst) and len(a) == len(lst):
        lst[:] = [bool(v) for v in a] if et is T.BOOL else a.tolist()
    else:
        for k in ks:
            lst[k] = bool(a[k]) if et is T.BOOL else a[k]


def _copy_back_attr(xs, key, a, et, before):
    """Into the instances, the attribute's values this call changed: the others stay the very objects they were, even if
    another thread set them meanwhile. key may be a path (pos.x)."""
    if '.' in key:
        *path, key = key.split('.')
        xs = [_member(x, '.'.join(path)) for x in xs]
    for k in _changed(a, before):
        v = a[k]
        setattr(xs[k], key, bool(v) if et is T.BOOL else v)


def _passes(fdef):
    """{parameter index: [(target, position)]}: a parameter, or an element of one (for p in ps, p = ps[i]), passed as a
    positional argument to a function at module level, ('f', name), or to a method, ('m', its owner's name, method), or to a
    method of a new instance ('e', the expression, method); with a path for an object in an attribute (p.pos)."""
    return {i: list(v) for i, v in _path_uses(fdef)[2].items()}


def _unused(fdef):
    """The parameters the function never names: anything can be passed there."""
    if fdef is None:
        return set()
    named = _facts(fdef).seen
    return {i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args) if p.arg not in named}


def _owner_classes(f, args):
    """{name: class} for the parameters that are instances, and for names bound to an element of a list of them."""
    names = [p.arg for p in f._tree.args.posonlyargs + f._tree.args.args]
    out = {n: type(x) for n, x in zip(names, args) if type(x) is not list}
    out.update({q: out[p] for q, p in _object_aliases(f._tree).items() if p in out})
    made = getattr(f, '_made', None)
    if made:                                     # v = C(...): v's class is the global C
        out.update({v: f._fn.__globals__[c] for v, c in _made_locals(f._ctree or f._tree, set(made)).items() if c in f._fn.__globals__})
    lists = {n: type(x[0]) for n, x in zip(names, args) if type(x) is list and x}
    for v, ps in _facts(f._tree).element_alias(lists).items():     # for p in ps, p = ps[i]: p is of the elements' class
        out[v] = lists[ps]
    for v, c in _local_classes(f, out).items():         # v = a + b, v = a.add(b): new instances, of the class the call returns
        out.setdefault(v, c)
    return out


def _param_uses(f, idx, cls, base, owners=None, stack=(), nested_out=None):
    """(attributes used, those assigned, methods called) of an instance of cls given to f as parameter idx, through the
    functions and methods f passes it to. base: f's own uses of it. A target that can't be found adds nothing: the call
    to it won't compile either."""
    names, writes, calls = set(base[0]), set(base[1]), set(base[2])
    if (id(f), idx) in stack:
        return names, writes, calls
    stack = stack + ((id(f), idx),)
    for entry in f._passes.get(idx, ()):
        target, pos = entry[:2]
        path = entry[2] if len(entry) > 2 else ()
        g, at = None, pos
        if target[0] == 'f':
            g = f._fn.__globals__.get(target[1])
        elif target[0] == 'e' and owners is not None:
            cls_of = _class_of(target[1], f._fn.__globals__, owners)
            c = _in_class(cls_of, target[2]) if cls_of is not None else None
            g, at = (c if isinstance(c, Function) or type(c) is types.FunctionType else None), pos + 1
            g = None if g is None else _method(g)
        elif target[0] == 'm' and owners is not None and target[1] in owners:
            c = _in_class(owners[target[1]], target[2])
            g, at = (c if isinstance(c, Function) or type(c) is types.FunctionType else None), pos + 1
            g = None if g is None else _method(g)
        if not isinstance(g, Function) or g._tree is None:
            continue
        deeper = {}
        inner = _param_uses(g, at, cls, g._attrs.get(at, ((), set(), set())), None, stack, deeper)
        deeper = _merge_nested(deeper, g._nested.get(at, {}))
        if not path:
            names |= inner[0]
            writes |= inner[1]
            calls |= inner[2]
            if nested_out is not None:
                nested_out.update(_merge_nested(dict(nested_out), deeper))
        elif nested_out is not None:         # given p.pos: what it does with that is p's pos's
            node = nested_out
            for a in path[:-1]:
                node = node.setdefault(a, [set(), set(), set(), {}])[3]
            e = node.setdefault(path[-1], [set(), set(), set(), {}])
            e[0] |= inner[0]
            e[1] |= inner[1]
            e[2] |= inner[2]
            e[3] = _merge_nested(e[3], deeper)
            names.add(path[0])
    return names, writes, calls


def _add_use(entry, attr, how):
    names, stores, calls = entry[:3]
    if how == 'call':
        calls.add(attr)
    else:
        names.add(attr)
        if how == 'store':
            stores.add(attr)


def _attribute_uses(fdef):
    """(name, attribute, 'load', 'store' or 'call') for each name.attribute; name + x calls name.__add__, if it's an instance."""
    facts = _facts(fdef)
    for name in set(facts.uses()[0]) | set(facts.ops):
        used, stores, calls = facts.first_uses(name)
        for a in used:
            yield name, a, 'store' if a in stores else 'load'
        for a in calls:
            yield name, a, 'call'


_methods = weakref.WeakKeyDictionary()   # a plain function on a class -> the Function that compiles it as a callee


def _method(fn):
    """The Function a method on a class compiles as, when compiled code calls it: itself if it's @scram."""
    if isinstance(fn, Function):
        return fn
    w = _methods.get(fn)
    if w is None:
        w = _methods[fn] = Function(fn, cache=True)
        _functions.discard(w)               # not a decorated function, for preflight
    return w


def _in_class(cls, name):
    """What the class holds under a name, as Python's attribute lookup finds it. _MISSING if nothing."""
    for k in cls.__mro__:
        if name in k.__dict__:
            return k.__dict__[name]
    return _MISSING


# Python's own lookup: object's, and tuple's (namedtuples), which 3.14 dropped for object's
_DEFAULT_GET = tuple({object.__dict__['__getattribute__'], tuple.__dict__.get('__getattribute__', object.__dict__['__getattribute__'])})
_DEFAULT_SET = object.__dict__['__setattr__']
_TUPLE_FIELD = type(collections.namedtuple('_', 'a').a)     # a namedtuple's field; property if its C version is missing


def _in_slots(cls, name):
    for k in cls.__mro__:
        slots = k.__dict__.get('__slots__', ())
        if name == slots or (not isinstance(slots, str) and name in slots):
            return True
    return False


def _object_fields(x, names, written, calls=(), methods=None, nested=None, ids=None, inner=None, refs=None):
    """[(attribute, type)] if reading these attributes runs no code of the object's, and assigning those written only
    stores; if not, the reason as a string. This holds only for now, since a class can change: it's asked at every call.
    A method called is a plain function on the class, or a @scram one; what it uses joins these, and methods (a dict, if
    given) gets what each call assigns. inner(object, attribute, its uses): the type of an object in an attribute, or a
    reason; by default it's found by these same rules."""
    cls = type(x)
    if calls:
        found = _methods_called(x, calls)
        if type(found) is str:
            return found
        names = sorted(set(names).union(*(n for n, _, _ in found.values())))
        written = set(written).union(*(w for _, w, _ in found.values()))
        for _, _, w in found.values():          # what the methods do with objects in attributes
            nested = _merge_nested(nested, w._nested.get(0, {}))
            more = {}
            _param_uses(w, 0, cls, ((), set(), set()), None, (), more)
            nested = _merge_nested(nested, more)
        if methods is not None:
            methods.update({m: (n, w) for m, (n, w, _) in found.items()})
    if not any(_in_class(cls, '__getattribute__') is d for d in _DEFAULT_GET):
        return f'{cls.__name__} defines __getattribute__'
    if _in_class(cls, '__getattr__') is not _MISSING:
        return f'{cls.__name__} defines __getattr__'
    if written and _in_class(cls, '__setattr__') is not _DEFAULT_SET:
        return f'{cls.__name__} defines __setattr__ (a frozen dataclass does), and {sorted(written)[0]} is assigned'
    held = getattr(x, '__dict__', None)
    if type(held) is not dict:
        held = {}
    fields = []
    for name in names:
        c = _in_class(cls, name)
        if c is _MISSING:
            if name not in held:
                return f'{cls.__name__} has no attribute {name}'     # assigning a new one, or Python's AttributeError
        elif type(c) is _TUPLE_FIELD and _TUPLE_FIELD is not property and isinstance(x, tuple):
            if name in written:
                return f'{name} is a field of a namedtuple, and is assigned'
        elif type(c) is types.MemberDescriptorType and _in_slots(cls, name):
            pass
        elif any(hasattr(type(c), m) for m in ('__get__', '__set__', '__delete__')):
            return f'{cls.__name__}.{name} is a method, property or other descriptor, which runs Python'
        elif name in written and name not in held:
            return f'{name} is on the class and not yet on the instance, and is assigned'
        v = getattr(x, name, _MISSING)
        if v is _MISSING:
            return f'{cls.__name__} has no attribute {name}'
        if refs is not None and name in refs:     # a reference chosen by data: its index in its class's pool
            fields.append((name, T.ref_of(refs[name])))
            continue
        if nested and name in nested and _arg_type(v) is None:     # an object in an attribute: its own attributes, by the same rules
            if name in written:
                return f'{name}, an object, is assigned'
            sub = nested[name]
            if inner is not None:
                got = inner(v, name, sub)
                if type(got) is str:
                    return f'{name}: {got}'
                fields.append((name, got))
                continue
            inner_methods = {}
            got = _object_fields(v, sorted(sub[0]), sub[1], sub[2], inner_methods, sub[3], ids)
            if type(got) is str:
                return f'{name}: {got}'
            if ids is not None:
                ids.append(id(v))
            w = set(sub[1]).union(*(w for _, w in inner_methods.values()))
            fields.append((name, T.object_of(type(v).__qualname__, got, inner_methods, type(v), w)))
            continue
        t = _arg_type(v)
        if t is None or t is T.U64 or t.kind in ('list', 'function', 'enumclass', 'classref') or T.is_pylist(t):
            return f'{name} is a {type(v).__name__}'
        fields.append((name, t))
    if calls:
        fields += [(m, T.function_of(w)) for m, (_, _, w) in sorted(found.items())]
    return fields


def _methods_called(x, calls, cls=None):
    """{method: (attributes it uses, those a call assigns, its Function)}, through the methods it calls; or a reason.
    x None: for instances of cls the function makes, which hold nothing of their own but what __init__ assigns."""
    cls = type(x) if x is not None else cls
    held = getattr(x, '__dict__', None) if x is not None and type(x).__getattribute__ is object.__getattribute__ else None
    found, todo, called = {}, list(calls), {}
    while todo:
        m = todo.pop()
        if m in found:
            continue
        c = _in_class(cls, m)
        if not isinstance(c, Function) and type(c) is not types.FunctionType:
            return f'{cls.__name__}.{m} is {"not there" if c is _MISSING else "not a plain method"}'
        if type(held) is dict and m in held:
            return f'{m} is on the instance, so calling it calls what is held there'
        w = _method(c)
        if w._tree is None:
            return f'{cls.__name__}.{m}: {w._why_not}'
        if not (w._tree.args.posonlyargs + w._tree.args.args):
            return f'{cls.__name__}.{m} takes no self'
        names, stores, inner = _param_uses(w, 0, cls, w._attrs.get(0, ((), set(), set())))
        found[m] = (set(names), set(stores), w)
        called[m] = set(inner)
        todo += sorted(inner)
    for _ in found:                      # what a call assigns, through the methods it calls
        for m in found:
            for k in called[m]:
                found[m][0].update(found[k][0])
                found[m][1].update(found[k][1])
    return found


def _construction_uses(f):
    """{parameter index: use tree} for parameters given to v = C(...), C a class at module level whose __init__ only stores:
    what f does through the attribute each becomes (v.pos.x is the parameter's x), and what v's methods and the functions v
    is given to do with it."""
    tree = f._ctree or f._tree
    facts = _facts(tree)
    index = {p.arg: i for i, p in enumerate(tree.args.posonlyargs + tree.args.args)}
    g = f._fn.__globals__
    trees = facts.uses()[0]
    out = {}
    for n in facts.assigns:
        if not (len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and isinstance(g.get(n.value.func.id), type)):
            continue
        given = {}
        recipe = _init_recipe(g[n.value.func.id])
        if type(recipe) is str:
            continue
        positional = [p for p, _, kw in recipe['params'] if not kw]
        for p, a in list(zip(positional, n.value.args)) + [(k.arg, k.value) for k in n.value.keywords]:
            if isinstance(a, ast.Name) and a.id in index:
                given[p] = index[a.id]
        attrs = {a: given[src[1]] for a, src in recipe['assigns'] if src[0] == 'param' and src[1] in given}
        if not attrs:
            continue
        v = n.targets[0].id
        cls = g[n.value.func.id]
        mine = trees.get(v)
        for a, sub in (mine[3] if mine else {}).items():      # v.pos.x: the parameter's x
            if a in attrs:
                _node_merge(out.setdefault(attrs[a], [set(), set(), set(), {}]), sub)
        targets = [(_method(w), 0) for w in (_in_class(cls, m) for m in (mine[2] if mine else ())) if type(w) in (Function, types.FunctionType)]
        targets += [(g[t[1]], pos) for t, pos, *path in facts.given.get(v, ()) if t[0] == 'f' and not path and isinstance(g.get(t[1]), Function)]
        for w, at in targets:                    # through v's methods, and what v is given to
            if w._tree is None:
                continue
            more = {}
            _param_uses(w, at, cls, w._attrs.get(at, ((), set(), set())), None, (), more)
            more = _merge_nested(more, w._nested.get(at, {}))
            for a, sub in more.items():
                if a in attrs:
                    _node_merge(out.setdefault(attrs[a], [set(), set(), set(), {}]), [sub[0], sub[1], sub[2], sub[3]])
    return out


_POOLDICT = {'sjpool': object()}     # the pool argument's place among the globals: a value no type has, typed by _pool_plan


def _ref_candidates(fdef):
    """(attributes, names) that may hold references chosen by data, from the tree alone (see _Facts._refs)."""
    if fdef is None:
        return set(), set()
    attrs, names = _facts(fdef).ref_candidates
    return set(attrs), set(names)


def _plain_get(x, a):
    """x.a if reading it runs none of x's code: a default lookup, and a an instance attribute or slot. Else _MISSING."""
    cls = type(x)
    if not any(_in_class(cls, '__getattribute__') is d for d in _DEFAULT_GET) or _in_class(cls, '__getattr__') is not _MISSING:
        return _MISSING
    c = _in_class(cls, a)
    if c is not _MISSING and not (type(c) is types.MemberDescriptorType and _in_slots(cls, a)) \
            and any(hasattr(type(c), m) for m in ('__get__', '__set__', '__delete__')):
        return _MISSING                          # a method, property or other descriptor: its code would run
    return getattr(x, a, _MISSING)


def _is_instance(x):
    return x is not None and _arg_type(x) is None and type(x).__module__ != 'builtins' and not isinstance(x, (type, enum.Enum))


def _pool_plan(f, args):
    """For a function whose references are chosen by data: ({parameter index: ref type}, the pools' type, {attribute in a
    fixed place: its class}), the pools being every object reached from the arguments through references, each once, a
    pool per class; or (None, a reason) if they can't be held so, or ({}, NOPOOL, {}) if nothing is a reference."""
    tree = f._ctree or f._tree
    attrs, names = _ref_candidates(tree)
    attrs = set(attrs)
    g = f._fn.__globals__
    facts = _facts(tree)
    for n in facts.calls:                        # what the classes it makes are given may be a reference too
        c = g.get(n.func.id) if isinstance(n.func, ast.Name) else None
        if isinstance(c, type) and not issubclass(c, (enum.Enum, BaseException)) and (attrs or names):
            recipe = _init_recipe(c)
            if type(recipe) is dict:
                attrs |= {a for a, _ in recipe['assigns']}
    names = set(names)
    for n in facts.calls:                        # given where a compiled callee takes a reference: a reference here too
        h = g.get(n.func.id) if isinstance(n.func, ast.Name) else None
        if isinstance(h, Function) and h._tree is not None:
            theirs = _ref_candidates(h._tree)[1]
            ps = [q.arg for q in h._tree.args.posonlyargs + h._tree.args.args]
            for q, a in zip(ps, n.args):
                if q in theirs and isinstance(a, ast.Name):
                    names.add(a.id)
                elif q in theirs and isinstance(a, ast.Attribute):
                    attrs.add(a.attr)
    params = [p.arg for p in tree.args.posonlyargs + tree.args.args]
    pooled = [i for i, n in enumerate(params) if n in names and (args[i] is None or _is_instance(args[i]))]
    held = _ref_lists(tree, names)               # lists whose elements are references: for n in heads, n = heads[i]
    lists = [i for i, n in enumerate(params) if n in held and type(args[i]) is list and args[i]
             and all(v is None or _is_instance(v) for v in args[i])]
    refnames = names | {params[i] for i in pooled}
    extra_used = set()
    for _ in range(4):                           # what callees given references follow and read is the pool's too
        plan = _pool_walk(args, pooled, attrs, lists)
        more_attrs, more_used = _callee_ref_uses(f, tree, refnames, set(map(type, plan[2])))
        extra_used |= more_used
        if more_attrs <= attrs:
            break
        attrs |= more_attrs
    roots, fixed, objs = plan
    if not roots:
        return {}, T.NOPOOL, {}
    found = _pool_types(f, tree, args, pooled, attrs, refnames, roots, fixed, objs, extra_used)
    if found[0] is not None:
        for i in lists:
            cs = {type(v) for v in args[i] if v is not None}
            if len(cs) > 1:
                return None, f'{params[i]} holds objects of {len(cs)} classes'
            found[0][i] = T.reflist_of(cs.pop() if cs else None)
    return found


def _guess_fields(f, tree, args, cls, refnames):
    """For a class whose instances the function keeps in a pool that has none of them yet: ([(number attribute, type)],
    [(reference attribute, its class)]) from what every C(...) is given, or a reason. A wrong guess can't give a wrong result:
    the compiler refuses to store a value of another type, and the call runs as Python."""
    recipe = _init_recipe(cls)
    if type(recipe) is str:
        return recipe
    params = [p.arg for p in tree.args.posonlyargs + tree.args.args]
    loops = {}                                   # for x in xs, xs an argument: x's type is its elements'
    facts = _facts(tree)
    for n in facts.loops:
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and isinstance(n.iter, ast.Name) and n.iter.id in params:
            xs = args[params.index(n.iter.id)]
            kinds = set(map(type, xs)) if type(xs) is list else set()
            if len(kinds) == 1:
                loops[n.target.id] = kinds.pop()

    def kind(e):
        if isinstance(e, ast.Constant):
            return 'ref' if e.value is None else type(e.value)
        if isinstance(e, ast.Name) and e.id in refnames:
            return 'ref'
        if isinstance(e, ast.Name) and e.id in params:
            v = args[params.index(e.id)]
            return 'ref' if v is None or _is_instance(v) else type(v)
        if isinstance(e, ast.Name) and e.id in loops:
            return loops[e.id]
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in ('float', 'int', 'bool'):
            return {'float': float, 'int': int, 'bool': bool}[e.func.id]
        return None
    found = {}
    for n in facts.calls:
        if not (isinstance(n.func, ast.Name) and f._fn.__globals__.get(n.func.id) is cls):
            continue
        positional = [p for p, _, kw in recipe['params'] if not kw]
        given = dict(zip(positional, n.args))
        given.update({k.arg: k.value for k in n.keywords})
        for a, src in recipe['assigns']:
            if src[0] == 'param':
                k = kind(given[src[1]]) if src[1] in given else None
                if k is None and src[1] not in given:
                    d = next((d for p, d, _ in recipe['params'] if p == src[1]), None)
                    k = 'ref' if d is not None and d[1] is None else type(d[1]) if d is not None else None
            elif src[0] == 'const':
                k = 'ref' if src[1] is None else type(src[1])
            else:
                k = None
            if k not in ('ref', int, float, bool) or found.setdefault(a, k) != k:
                return f'can\'t tell what {cls.__name__}.{a} holds from what {cls.__name__}(...) is given here'
    numbers = [(a, {int: T.INT, float: T.FLOAT, bool: T.BOOL}[k]) for a, k in found.items() if k != 'ref']
    return numbers, [(a, cls) for a, k in found.items() if k == 'ref']


_POOL_CLASSES = weakref.WeakKeyDictionary()    # a pooled class -> (its __mro__, what its dicts held under the names read)


def _pool_class_ok(cls, names):
    """Reading these attributes of cls's instances runs none of its code: checked once, then only that it hasn't changed."""
    seen = _POOL_CLASSES.get(cls)
    if seen is not None and seen[0] is cls.__mro__ and seen[1] >= names and all(
            k.__dict__.get(n, _MISSING) is v for (k, n), v in seen[2].items()):
        return True
    if not any(_in_class(cls, '__getattribute__') is d for d in _DEFAULT_GET) or _in_class(cls, '__getattr__') is not _MISSING:
        return False
    for n in names:
        c = _in_class(cls, n)
        if c is not _MISSING and not (type(c) is types.MemberDescriptorType and _in_slots(cls, n)) \
                and any(hasattr(type(c), m) for m in ('__get__', '__set__', '__delete__')):
            return False
    held = {(k, n): k.__dict__.get(n, _MISSING) for k in cls.__mro__ if k.__flags__ & (1 << 9)
            for n in list(names) + ['__getattribute__', '__getattr__']}
    _POOL_CLASSES[cls] = (cls.__mro__, frozenset(names), held)
    return True


_KIND = {T.INT: int, T.FLOAT: float, T.BOOL: bool}


def _pool_build(t, args, room=None):
    """The pool for a call: every object references reach, each once, a pool per class, as _pool_plan typed it, and the
    lists the kernel takes. None if anything isn't as the type says (another class reached, a number of another type, a class
    changed), and the call takes the general path. (by class, id -> index, C arguments, [(slot, list, type, objects,
    attribute)] for numbers, [(slot, list, objects, attribute, the objects it indexes)] for references, made, room)"""
    pools, roots, made = t.elem
    follow = {cls: [a for a, _ in refs] for cls, _, refs in pools}
    for cls, numbers, refs in pools:
        if cls is not None and not _pool_class_ok(cls, frozenset([a for a, _ in numbers] + [a for a, _ in refs])):
            return None
    by_cls = {cls: [] for cls, _, _ in pools}
    where, todo = {}, []
    for i, path in roots:
        v = args[i] if not path or path == '[]' else _member(args[i], path)
        todo += [y for y in v if y is not None] if path == '[]' else [v] if v is not None else []
    while todo:
        o = todo.pop()
        if id(o) in where:
            continue
        mine = by_cls.get(type(o))
        if mine is None:
            return None
        where[id(o)] = len(mine)
        mine.append(o)
        for a in follow[type(o)]:
            v = getattr(o, a)
            if v is not None:
                todo.append(v)
    cargs, columns, refcols = [], [], []
    made_k = dict(made)
    for k, (cls, numbers, refs) in enumerate(pools):
        objs = by_cls.get(cls, [])
        spare = (room if room is not None else max(16, len(objs))) if k in made_k else 0     # room for what it makes
        if k in made_k:
            room = spare
        for key, ft in numbers:
            vals = [getattr(o, key) for o in objs]
            want = _KIND.get(ft)
            if want is None or not all(type(v) is want for v in vals):
                return None
            try:
                a = array.array(_LIST_CODES[ft.name], vals)
            except OverflowError:                # an int past 64 bits: the general path, which runs it as Python
                return None
            a.extend(array.array(_LIST_CODES[ft.name], bytes(a.itemsize * spare)))
            columns.append((f'{k}.{key}', a, ft, objs, key))
            cargs += [a.buffer_info()[0], len(a)]
        for key, target in refs:
            targets = by_cls.setdefault(target, []) if target is not None else []
            vals = [getattr(o, key) for o in objs]
            if not all(v is None or type(v) is target for v in vals):
                return None
            a = array.array('q', (-1 if v is None else where[id(v)] for v in vals))
            a.extend([-1] * spare)
            refcols.append((f'{k}.{key}', a, objs, key, targets))
            cargs += [a.buffer_info()[0], len(a)]
        if k in made_k:
            cargs.append(len(objs))             # how many are in use
    return by_cls, where, cargs, columns, refcols, made_k, room


def _ref_lists(tree, names):
    """Names of lists whose elements go into reference names: for n in heads, n = heads[i], heads[i] = n."""
    out = set(names)                             # heads[i] is None, heads[i] = ...: the list itself is a candidate
    facts = _facts(tree)
    for n in facts.loops + facts.assigns:
        if isinstance(n, (ast.For, ast.comprehension)) and isinstance(n.target, ast.Name) and n.target.id in names and isinstance(n.iter, ast.Name):
            out.add(n.iter.id)
        elif isinstance(n, ast.Assign) and len(n.targets) == 1:
            t, v = n.targets[0], n.value
            if isinstance(t, ast.Name) and t.id in names and isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name):
                out.add(v.value.id)
            elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and isinstance(v, (ast.Name, ast.Attribute)) and \
                    (isinstance(v, ast.Attribute) or v.id in names):
                out.add(t.value.id)
    return out


def _callee_ref_uses(f, tree, refnames, classes):
    """(reference candidates, attributes used) of the methods called on references and the functions given them."""
    g = f._fn.__globals__
    attrs, used = set(), set()
    for n in _facts(tree).calls:
        targets = []
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, (ast.Name, ast.Attribute)):
            for cls in classes:
                w = _in_class(cls, n.func.attr)
                if type(w) in (Function, types.FunctionType):
                    targets.append((_method(w), 0))
        elif isinstance(n.func, ast.Name) and isinstance(g.get(n.func.id), Function):
            targets += [(g[n.func.id], k) for k, a in enumerate(n.args) if isinstance(a, ast.Name) and a.id in refnames]
        for w, at in targets:
            if w._tree is None:
                continue
            ca, cn = _ref_candidates(w._tree)
            ps = [p.arg for p in w._tree.args.posonlyargs + w._tree.args.args]
            attrs |= ca
            mine = cn | ({ps[at]} if at < len(ps) else set())
            for m in _facts(w._tree).attributes:
                if isinstance(m.value, ast.Name) and m.value.id in mine:
                    used.add(m.attr)
                elif isinstance(m, ast.Attribute) and isinstance(m.value, ast.Attribute):
                    used.add(m.attr)
    return attrs, used


def _pool_walk(args, pooled, attrs, lists=()):
    """(roots, {fixed reference attribute: classes}, the objects reached, each once)."""
    roots, todo, fixed = [], [], {}             # fixed: references in a fixed place (a.pos) -> the classes they reach
    for i, x in enumerate(args):
        if i in lists:                           # each element
            roots.append((i, '[]'))
            todo += [v for v in x if v is not None]
        elif i in pooled:
            roots.append((i, ''))
            if x is not None:
                todo.append(x)
        elif _is_instance(x):
            for a in sorted(attrs):
                v = _plain_get(x, a)
                if v is None or _is_instance(v):
                    fixed.setdefault(a, set()).add(type(v) if v is not None else None)
                    roots.append((i, a))
                    if v is not None:
                        todo.append(v)
    seen, objs = set(), []
    while todo:
        o = todo.pop(0)
        if id(o) in seen:
            continue
        seen.add(id(o))
        objs.append(o)
        for a in sorted(attrs):
            v = _plain_get(o, a)
            if v is not None and _is_instance(v):
                todo.append(v)
    return roots, fixed, objs


def _pool_types(f, tree, args, pooled, attrs, refnames, roots, fixed, objs, extra_used):
    classes = list(dict.fromkeys(map(type, objs)))
    specs = []
    for cls in classes:                          # per class: its reference attributes, each reaching one class or None
        mine = [o for o in objs if type(o) is cls]
        refs = []
        for a in sorted(attrs):
            vals = [_plain_get(o, a) for o in mine]
            if all(v is None or _is_instance(v) for v in vals):
                targets = {type(v) for v in vals if v is not None}
                if len(targets) > 1:
                    return None, f'{cls.__name__}.{a} reaches objects of {len(targets)} classes'
                refs.append((a, targets.pop() if targets else None))
        if _in_class(cls, '__bool__') is not _MISSING or _in_class(cls, '__len__') is not _MISSING:
            return None, f'{cls.__name__} defines __bool__ or __len__, so a reference\'s truth is its code\'s'
        specs.append((cls, mine, refs))
    refattrs = {a for _, _, refs in specs for a, _ in refs} | set(fixed)
    used = set()
    facts = _facts(tree)
    for n in facts.attributes:
        if n.attr not in refattrs:
            v = n.value
            if isinstance(v, ast.Name) and v.id in refnames or isinstance(v, ast.Attribute) and v.attr in refattrs:
                used.add(n.attr)
    used |= extra_used - refattrs
    pools = []
    for cls, mine, refs in specs:
        numbers = []
        for a in sorted(used):
            vals = [_plain_get(o, a) for o in mine]
            kinds = set(map(type, vals))
            if len(kinds) == 1 and kinds.pop() in (int, float, bool) and not (type(vals[0]) is int and not all(-(1 << 63) <= v < 1 << 63 for v in vals)):
                numbers.append((a, {int: T.INT, float: T.FLOAT, bool: T.BOOL}[type(vals[0])]))
            elif any(v is not _MISSING and type(v) not in (int, float, bool) for v in vals) and _in_class(cls, a) is _MISSING:
                return None, f'{cls.__name__}.{a} isn\'t a number of one type in every {cls.__name__} references reach'
        pools.append((cls, numbers, refs))
    made = []                                    # classes the function makes instances of, kept in the pool: every attribute
    g = f._fn.__globals__                        # __init__ assigns is a list, so a new one can be made after the call
    built = {g[n.func.id] for n in facts.calls if isinstance(n.func, ast.Name) and isinstance(g.get(n.func.id), type)
             and not issubclass(g[n.func.id], (enum.Enum, BaseException))}
    have_cls = {c for c, _, _ in pools}
    for cls in sorted(built - have_cls, key=lambda c: c.__qualname__):     # none of them yet: types from what it's given
        guessed = _guess_fields(f, tree, args, cls, refnames)
        if type(guessed) is str:
            return None, guessed
        pools.append((cls, guessed[0], guessed[1]))
    for k, (cls, numbers, refs) in enumerate(pools):
        if cls not in built:
            continue
        recipe = _init_recipe(cls)
        if type(recipe) is str or recipe['post_init'] or recipe['frozen']:
            return None, f'{cls.__name__} is made and kept in a linked structure, and ' + (recipe if type(recipe) is str else 'has __post_init__ or is frozen')
        order = list(dict.fromkeys(a for a, _ in recipe['assigns']))
        have = {a for a, _ in numbers} | {a for a, _ in refs}
        mine = [o for o in objs if type(o) is cls]
        for a in order:
            if a in have:
                continue
            vals = [_plain_get(o, a) for o in mine]
            kinds = set(map(type, vals))
            if len(kinds) != 1 or next(iter(kinds)) not in (int, float, bool):
                return None, f'{cls.__name__}.{a} isn\'t a number or a reference in every {cls.__name__}, so a new one can\'t be kept'
            numbers.append((a, {int: T.INT, float: T.FLOAT, bool: T.BOOL}[type(vals[0])]))
        made.append((k, tuple(order)))
    if not pools:                                # only None reached: nothing says otherwise, and nothing is read
        prefs = sorted(a for a in attrs & used if a not in fixed)
        pools = [(None, [(a, T.FLOAT) for a in sorted(used) if a not in prefs], [(a, None) for a in prefs])]
    fixed_cls = {}
    for a, cs in fixed.items():
        known = cs - {None}
        if len(known) > 1:
            return None, f'.{a} reaches objects of {len(known)} classes'
        fixed_cls[a] = known.pop() if known else None
    return {i: T.ref_of(type(args[i]) if args[i] is not None else None) for i in pooled}, T.pool_of(pools, roots, made), fixed_cls


def _node_merge(into, node):
    """Add a use tree [attributes used, assigned, methods called, {attribute: tree}] into another. True if it grew."""
    grew = False
    for k in range(3):
        if not set(node[k]) <= into[k]:
            into[k] |= set(node[k])
            grew = True
    for a, sub in node[3].items():
        if a not in into[3]:
            into[3][a] = [set(), set(), set(), {}]
        grew = _node_merge(into[3][a], sub) or grew
    return grew


def _object_graph(roots, refs=None):
    """The types of instance arguments, roots [(index, object, use tree)], with their alias pattern: every path the function
    names is walked from the arguments, and an object met a second time is same_as where it was first met, its type made
    for what all its names use. ({index: type, or None if it can't compile}, the ids of the objects and their __dict__s.)
    Objects in attributes are walked only through attributes the function names: the cost is the code's, not the data's."""
    merged = {}
    for _ in range(16):                  # each round, an object's tree only grows: it settles
        first, grew, held, why = {}, [False], set(), {}

        def walk(x, node, at):
            k = id(x)
            m = merged.get(k)
            if m is None:
                merged[k] = m = [set(), set(), set(), {}]
            if _node_merge(m, node) and k in first:
                grew[0] = True          # what this name uses wasn't in the type made at the first
            if k in first:
                i0, p0 = first[k]
                return T.same_as(None, p0) if i0 == at[0] else T.same_as(i0, p0)
            first[k] = at
            if not any(m[:3]) and not m[3] and not at[1]:     # nothing of it reaches the C
                return T.object_of(type(x).__qualname__, [], None, type(x))
            methods = {}
            fields = _object_fields(x, sorted(m[0]), set(m[1]), set(m[2]), methods, m[3],
                                    inner=lambda v, a, sub: walk(v, sub, (at[0], at[1] + (a,))), refs=refs)
            if type(fields) is not list:
                return fields
            d = getattr(x, '__dict__', None)
            if d is not None:
                held.add(id(d))
            written = set(m[1]).union(*(w for _, w in methods.values()))
            return T.object_of(type(x).__qualname__, fields, methods, type(x), written)
        out = {}
        for i, x, node in roots:
            t = walk(x, node, (i, ()))
            out[i] = t if isinstance(t, T.Type) else None
            if type(t) is str:
                why[i] = t
        if not grew[0]:
            break
    if set(first) & held:                # an instance and its own __dict__ are two names for one thing as well
        return {i: None for i, _, _ in roots}, set(first) | held
    return out, set(first) | held


def _field_type(t, path):
    """The type at a path in an instance's type, or a type of kind 'none' if it isn't one."""
    for a in path.split('.') if isinstance(path, str) else [path]:
        if t is None or t.kind != 'object':
            return T.NONE
        t = dict(t.elem).get(a)
    return t if t is not None else T.NONE


def _member(x, key):
    """What a dict holds under a key, or an instance under an attribute, or a path of them (p.pos.x is 'pos.x')."""
    if type(x) is dict:
        return x[key]
    for a in key.split('.'):
        x = getattr(x, a)
    return x


def _set_member(x, key, value):
    if type(x) is dict:
        x[key] = value
        return
    *path, last = key.split('.')
    for a in path:
        x = getattr(x, a)
    setattr(x, last, value)


def _push(cargs, x, t, where=None):
    """The C arguments for one Python argument. where: id -> index in the pool, for references."""
    if t.kind in ('function', 'enumclass', 'classref', 'madeclass', 'same', 'nopool'):
        return
    if t.kind == 'inlist':
        raise AssertionError('an element of another argument is passed by _run, which knows where it is')
    if t.kind == 'ref':
        cargs.append(-1 if x is None else where[id(x)])
        return
    if t.kind == 'pool':
        raise AssertionError('a pool is passed by _run, which writes it back')
    if t.kind == 'enum':
        cargs.append(x._value_)
        return
    if t.kind in ('list', 'objlist'):
        raise AssertionError('lists are passed by _run, which copies them back')
    if t.kind == 'array' and t.strided:
        cargs.append(x.ctypes.data)
        cargs.extend(x.shape)
        cargs.extend(s // x.itemsize for s in x.strides)
    elif t.kind == 'array':
        cargs.append(x)
        cargs.extend(x.shape)
    elif t.kind == 'tuple':
        for v, et in zip(x, t.elem):
            _push(cargs, v, et, where)
    elif t.kind == 'dict':
        for key, ft in t.elem:
            _push(cargs, x[key], ft)
    elif t.kind == 'object':              # in the order its parameters come: breadth first through inner objects
        for path, ft in leaf_types(t):
            _push(cargs, _member(x, path), ft, where)
    elif t.np:
        cargs.append(x.item())
    elif t is T.U64:
        cargs.append(x - (1 << 64))
    else:
        cargs.append(x)


class _Interface:
    """Memory numpy can wrap as an array without copying: it keeps the array it belongs to alive."""

    __slots__ = ('base', '__array_interface__')

    def __init__(self, base, address, shape, strides, typestr=None):
        self.base = base
        self.__array_interface__ = {'data': (address, base is not None and not base.flags.writeable), 'shape': tuple(shape),
                                    'strides': None if strides is None else tuple(strides),
                                    'typestr': typestr or base.dtype.str, 'version': 3}


_MISSING = object()               # a global that isn't there: nothing compiled can read it, so that call runs as Python


class _Python:
    """Marker: this signature runs as the original Python function."""


PYTHON = _Python()


_LIST_CODES = {'int': 'q', 'float': 'd', 'bool': 'i'}      # a list's element type -> its array, as the kernel's C type


def _nested_shape(x):
    """The shape of a rectangular list of lists, or None: ragged, empty, or one list in two places (a write would show
    through both, which a copy can't do)."""
    shape, level, seen = [], [x], set()
    while level and type(level[0]) is list:
        n = len(level[0])
        if n == 0 or any(type(r) is not list or len(r) != n for r in level):
            return None
        for r in level:
            if id(r) in seen:
                return None
            seen.add(id(r))
        shape.append(n)
        level = list(itertools.chain.from_iterable(level))
    return shape                               # lists among the numbers fail the element check that follows


def _flatten(x, depth):
    for _ in range(depth - 1):
        x = list(itertools.chain.from_iterable(x))
    return x


def _lists_from(address, shape, et):
    """Nested Python lists of the numbers at address, of this shape, read in C."""
    import ctypes
    code = _LIST_CODES[et.name]
    n = 1
    for s in shape:
        n *= s
    flat = memoryview((ctypes.c_char * max(1, n * struct.calcsize(code))).from_address(address)).cast('B')[:n * struct.calcsize(code)].cast(code).tolist() if n else []
    if et is T.BOOL:
        flat = [bool(v) for v in flat]

    def nest(values, dims):
        if len(dims) == 1:
            return values
        size = 1
        for s in dims[1:]:
            size *= s
        return [nest(values[k * size:(k + 1) * size], dims[1:]) for k in range(dims[0])]
    return nest(flat, list(shape))


def _copy_back_nested(x, a, et, depth, before):
    """Into a list of lists passed in, the rows this call changed, each replaced in one go, in C: rows it didn't change
    stay as they are, even if another thread changed them meanwhile."""
    if a.tobytes() == before.tobytes():
        return
    rows = [x]
    for _ in range(depth - 1):
        rows = [r for row in rows for r in row]
    n = len(rows[0])
    for k, row in enumerate(rows):
        new = a[k * n:(k + 1) * n]
        if new.tobytes() != before[k * n:(k + 1) * n].tobytes():
            row[:] = [bool(v) for v in new] if et is T.BOOL else new.tolist()


def _copy_back(lst, a, et, before):
    """Into a list passed in, the elements this call changed: the others stay the very objects they were, even if another
    thread set them meanwhile."""
    _write_changed(lst, a, et, before)


class Compiled:
    __slots__ = ('gl_types', 'module', 'kernel', 'ret', 'types', 'source', 'libm', 'locals', 'checks', 'hoisted', 'gotchas', 'stop', 'rerun', 'outs', 'lists',
                 'counts', 'count_lines', 'count_heads', 'count_rest', 'array_args', 'records', 'io_size', 'free', 'frames', 'columns', 'instance_ret', 'tb_at')

    def __init__(self, kernel, ret, types_, source, libm, locals_=None, checks=None, hoisted=(), gotchas=None, stop=False, rerun=False):
        self.rerun = rerun
        self.outs = []                        # written dict keys: (argument index, key, byte in io, type)
        self.lists = set()                    # the arguments that are lists the kernel writes
        self.columns = set()                  # (argument, attribute) of lists of instances the kernel writes
        self.instance_ret = None              # returns an instance: see Compiler.instance_ret
        self.counts = None                    # preflight --counts: the int64 counters the kernel adds to
        self.count_lines, self.count_heads, self.count_rest = (), {}, {}
        self.checks, self.hoisted, self.gotchas, self.stop = checks or {}, hoisted, gotchas or {}, stop
        self.locals = locals_ or {}           # variable -> inferred type name
        self.kernel = kernel
        self.ret = ret
        self.types = types_
        self.source = source
        self.libm = libm

    def read_counts(self):
        """(line -> times it ran, line -> ('if' or 'loop', times its body ran)). A kernel adds to its counters
        with the GIL released, so this works while one runs, from any thread."""
        values = list(struct.unpack(f'<{len(self.counts) // 8}q', self.counts))
        for slot in sorted(self.count_rest):      # an if comes before its else, so a chain of elifs resolves in order
            whole, taken = self.count_rest[slot]
            values[slot] = max(values[whole] - values[taken], 0)
        runs = {}
        for lines, v in zip(self.count_lines, values):
            for n in lines:
                runs[n] = max(runs.get(n, 0), v)    # a line can hold several statements
        return runs, {n: (kind, values[slot]) for n, (kind, slot) in self.count_heads.items()}


_functions = weakref.WeakSet()    # every decorated function, for preflight
_profile = False                  # preflight --run: time each call
_counts = None                    # preflight --counts: True, or the names of the functions to count
_running = None                   # preflight --live: thread id -> (the function, native or not, when it started)
_log = bool(os.environ.get('SCRAMJIT_LOG'))


def enable_profile():
    global _profile
    _profile = True


def enable_counts(names=None):
    """Compile a copy of each function, or of those named, that counts how often each line runs."""
    global _counts
    enable_profile()
    _counts = set(names) if names else True


def enable_live():
    global _running
    enable_profile()
    _running = {}


def running():
    return list(_running.values()) if _running else []


def functions():
    return list(_functions)


def _class_guards(env, i, cls, names, written, again):
    """That cls still looks up these names as it did, for a plan: a class can change, and only those written in Python can."""
    env[f'kcls{i}'], env[f'kmro{i}'], env['MISSING'] = cls, cls.__mro__, _MISSING
    out = [f'    if kcls{i}.__mro__ is not kmro{i}:\n        return slow({again})']
    asked = list(names) + ['__getattribute__', '__getattr__'] + (['__setattr__'] if written else [])
    for c, k in enumerate(k for k in cls.__mro__ if k.__flags__ & (1 << 9)):
        env[f'kheld{i}_{c}'] = k.__dict__
        for a, name in enumerate(asked):
            env[f'kwas{i}_{c}_{a}'] = k.__dict__.get(name, _MISSING)
            out.append(f'    if kheld{i}_{c}.get({name!r}, MISSING) is not kwas{i}_{c}_{a}:\n        return slow({again})')
    return out


def _object_guards(env, i, x, t, n, again):
    """What a plan checks of an instance: that its class looks up these names as it did (a class can change, and only those
    written in Python can), then what _object_fields checks of the object. The first fetched value of each attribute is
    f{i}_{k}. Nothing here runs the object's code, once the class has been checked."""
    written = t.writes
    cls = type(x)
    env[f'cls{i}'], env[f'mro{i}'], env['MISSING'] = cls, cls.__mro__, _MISSING
    out = [f'    if cls{i}.__mro__ is not mro{i}:\n        return slow({again})']
    asked = [k for k, _ in t.elem] + ['__getattribute__', '__getattr__'] + (['__setattr__'] if written else [])
    for c, k in enumerate(k for k in cls.__mro__ if k.__flags__ & (1 << 9)):     # 1 << 9: a class made at run time
        env[f'held{i}_{c}'] = k.__dict__
        for a, name in enumerate(asked):
            env[f'was{i}_{c}_{a}'] = k.__dict__.get(name, _MISSING)
            out.append(f'    if held{i}_{c}.get({name!r}, MISSING) is not was{i}_{c}_{a}:\n        return slow({again})')
    for k, (key, ft) in enumerate(t.elem):
        if ft.kind == 'function':           # a method: the class's, checked above, unless the instance holds its own
            if hasattr(x, '__dict__'):
                out.append(f'    if {key!r} in {n}.__dict__:\n        return slow({again})')
            continue
        env[f'type{i}_{k}'] = type(getattr(x, key))
        out.append(f'    f{i}_{k} = getattr({n}, {key!r}, MISSING)\n    if type(f{i}_{k}) is not type{i}_{k}:\n        return slow({again})')
        if ft is T.INT:
            out.append(f'    if not -9223372036854775808 <= f{i}_{k} <= 9223372036854775807:\n        return slow({again})')
        if key in written and type(_in_class(cls, key)) is not types.MemberDescriptorType:      # not on the class only: assigning would add it, whatever ran
            out.append(f'    if {key!r} not in {n}.__dict__:\n        return slow({again})')
    return out


def _pooled_plan(f, entry, args):
    """A call plan for a function holding references in a pool: each call checks the arguments are as typed, then builds
    the pool in one walk that checks every object reached (_pool_build); anything else takes the general path."""
    real = len(args) - len(f._hidden or ())
    pool_t = next(t for t in entry.types if t.kind == 'pool')
    checks = []
    for i, (x, t) in enumerate(zip(args, entry.types)):
        if t.kind in ('pool', 'nopool'):
            continue
        if t.kind in ('ref', 'reflist'):
            checks.append((i, 'list' if t.kind == 'reflist' else 'ref', None))
        elif t in _KIND and not t.np:
            checks.append((i, 'number', _KIND[t]))
        elif i >= real and t.kind in ('function', 'enumclass', 'classref'):
            checks.append((i, 'same', x))
        else:
            return f._slow_call                  # instances, arrays and the rest: the general path checks them
    classes = tuple(c for c, _, _ in pool_t.elem[0] if c is not None)

    def plan(*given):
        full = f._with_globals(given)
        for i, how, want in checks:
            x = full[i]
            if how == 'number':
                if type(x) is not want or (want is int and not -(1 << 63) <= x < (1 << 63)):
                    return f._slow(given)
            elif how == 'ref':
                if x is not None and type(x) not in classes:
                    return f._slow(given)
            elif how == 'list':
                if type(x) is not list:
                    return f._slow(given)
            elif x is not want:
                return f._slow(given)
        built = _pool_build(pool_t, full)
        if built is None:
            return f._slow(given)
        return f._run(entry, full, prebuilt=built)
    return plan


def _all_fields(t):
    """An instance type's fields, through the objects in its attributes."""
    out = []
    for key, ft in t.elem:
        out.append((key, ft))
        if ft.kind == 'object':
            out += _all_fields(ft)
    return out


def _deep_guards(env, i, tag, x, t, n, path, again, leaves, places):
    """What a plan checks of an instance holding objects: each object's class and fields as _object_guards checks them,
    the variable each object is fetched into, and that another name for an object is still that object. Its numbers go to
    D{i}[path], the order leaf_types gives them."""
    places[i, path] = n
    out = _object_guards(env, tag, x, T.object_of(t.py, [(k, ft) for k, ft in t.elem if ft.kind not in ('object', 'same')],
                                                  t.methods, t.cls, t.writes), n, again)
    data = [k for k, ft in t.elem if ft.kind not in ('object', 'same')]
    for k, (key, ft) in enumerate([kt for kt in t.elem if kt[1].kind not in ('object', 'same')]):
        if ft.kind != 'function':
            out.append(f'    D{i}[{".".join(path + (key,))!r}] = f{tag}_{k}' + ('.item()' if ft.np else ''))
    for key, ft in t.elem:
        sub = f'{tag}_{key}'
        if ft.kind == 'object':
            v = getattr(x, key)
            env[f'cls{sub}'] = type(v)
            out.append(f'    {sub} = getattr({n}, {key!r}, MISSING)\n    if type({sub}) is not cls{sub}:\n        return slow({again})')
            out += _deep_guards(env, i, sub, v, ft, sub, path + (key,), again, leaves, places)
        elif ft.kind == 'same':
            j, to = ft.elem
            target = places.get((i if j is None else j, tuple(to)))
            if target is None:
                out.append(f'    return slow({again})')
                continue
            out.append(f'    if getattr({n}, {key!r}, MISSING) is not {target}:\n        return slow({again})')
    return out


def _make_plan(f, entry, args):
    """A small function for calls with these argument types, or f._slow_call if they need the general
    path. It's generated because a call is where the time goes: what's known from the types is decided
    here, once. Anything it can't vouch for at call time (an int past 64 bits, an array of another
    dtype, dimension or layout, an empty one) goes to f._slow, so plans change only speed."""
    if any(t.kind == 'pool' for t in entry.types) and entry.counts is None:
        return _pooled_plan(f, entry, args)
    if any(t.kind in ('glist', 'record', 'gset') for t in entry.types) or entry.counts is not None or entry.ret.kind in ('list', 'glist', 'record', 'gset') or (entry.ret.kind == 'tuple' and any(t.kind in ('glist', 'record', 'gset') for t in entry.ret.elem)) or entry.instance_ret or any(t.kind in ('pool', 'ref') for t in entry.types):
        return f._slow_call
    env = {'kernel': entry.kernel, 'slow': f._slow, 'failed': f._failed, 'box': f._box, 'ret': entry.ret, 'pool': [], 'compiled': entry,
           'addressof': ctypes.addressof, 'from_buffer': ctypes.c_char.from_buffer, 'bytearray': bytearray, 'memoryview': memoryview}
    names = [f'a{i}' for i in range(len(args))]
    real = names[:len(args) - len(f._hidden or ())]
    again = '(' + ''.join(n + ', ' for n in real) + ')'
    hidden = []                        # globals the function reads: read now, and turned down if their type has changed
    for k, (g, gname) in enumerate(f._hidden_from or ()):
        i = len(real) + k
        env[f'gname{k}'], env[f'gtype{k}'], env[f'G{k}'] = gname, type(args[i]), g
        hidden.append(f'    a{i} = G{k}.get(gname{k}, MISSING)\n    if type(a{i}) is not gtype{k}:\n        return slow({again})')
        if entry.types[i].kind in ('function', 'enumclass', 'classref', 'madeclass'):     # a new function or class is a new signature
            env[f'gval{k}'] = args[i]
            hidden.append(f'    if a{i} is not gval{k}:\n        return slow({again})')
    if hidden:
        env.update(MISSING=_MISSING)
    cargs = []
    late, deep, places = [], {}, {}         # same_as through attributes; objects walked deep; (argument, path) -> its variable
    for i, (x, t, n) in enumerate(zip(args, entry.types, names)):
        if t.kind == 'enum':
            cargs.append(f'{n}._value_')
            continue
        if t.kind in ('function', 'enumclass', 'classref', 'nopool'):
            continue
        if t.kind == 'same':                # one object under two names: the plan checks it still is
            if t.elem[0] is None:
                return f._slow_call
            if t.elem[1]:                       # an object in another argument's attribute: checked once that's fetched
                late.append((n, (t.elem[0], tuple(t.elem[1]))))
            else:
                hidden.append(f'    if {n} is not a{t.elem[0]}:\n        return slow({again})')
            continue
        if t.kind == 'madeclass':           # what its instances need of it: as for an instance's class
            hidden.extend(_class_guards(env, i, t.cls, [a for a, _ in t.recipe['assigns']] + [m for m, _ in t.recipe['methods']]
                                            + ['__init__', '__new__', '__post_init__'], t.writes, again))
            continue
        if t.kind == 'array' and (t.strided or T.is_pylist(t)):
            return f._slow_call
        if t.kind == 'array':
            env[f'd{i}'] = x.dtype
            cargs.append(f'p{i}' if x.flags.writeable else n)
            cargs.extend(f's{i}[{k}]' for k in range(x.ndim))
        elif t.kind in ('tuple', 'list', 'objlist'):
            return f._slow_call
        elif t.kind == 'object':
            if any(y.kind == 'dict' for y in entry.types):     # x, vars(x)
                return f._slow_call
            if not all(ft.is_number or ft.kind == 'function' for _, ft in t.elem):    # objects in attributes: their numbers, by path
                if not all(ft.is_number or ft.kind in ('function', 'object', 'same') for _, ft in _all_fields(t)):
                    return f._slow_call
                deep[i] = {}
                cargs.extend(f'D{i}[{path!r}]' for path, ft in leaf_types(t))
                continue
            for k, (key, ft) in enumerate(t.elem):
                if ft.kind != 'function':
                    cargs.append(f'f{i}_{k}.item()' if ft.np else f'f{i}_{k}')
        elif t.kind == 'dict':
            if not all(ft.is_number for _, ft in t.elem):
                return f._slow_call
            for k, (key, ft) in enumerate(t.elem):
                cargs.append(f'f{i}_{k}.item()' if ft.np else f'f{i}_{k}')
        elif t.np:
            cargs.append(f'{n}.item()')
        elif t is T.U64:
            cargs.append(f'({n} - 18446744073709551616)')
        else:
            cargs.append(n)
    if entry.libm is not None:
        pointer = ctypes.c_char_p(entry.libm)
        env['libm_keep'] = (entry.libm, pointer)        # alive as long as the plan
        env['libm'] = ctypes.cast(pointer, ctypes.c_void_p).value
        cargs.append('libm')
    if entry.stop:
        env.update(get_ident=threading.get_ident, FLAG=_interrupt.FLAG, NEVER=_interrupt.NEVER, interrupt=_interrupt)
        cargs.append('(FLAG if get_ident() == interrupt._main else NEVER)')
    # an array's checks go one at a time, each read after the last passed, then its address and shape
    arrays = [i for i, t in enumerate(entry.types) if t.kind == 'array']
    src_guards = hidden
    for i, (t, n) in enumerate(zip(entry.types, names)):
        if t.kind == 'array':
            x = args[i]
            src_guards.append(f'    fl = {n}.flags\n    if {n}.dtype is not d{i} or {n}.ndim != {x.ndim} or not fl.c_contiguous or fl.writeable is not {bool(x.flags.writeable)}:\n        return slow({again})\n    s{i} = {n}.shape')
        elif t.kind == 'dict':
            # only what this function reads is checked: a missing key or another type goes the general way
            for k, (key, ft) in enumerate(t.elem):
                env[f'key{i}_{k}'], env[f'type{i}_{k}'] = key, type(args[i][key])
                src_guards.append(f'    f{i}_{k} = {n}.get(key{i}_{k})\n    if type(f{i}_{k}) is not type{i}_{k}:\n        '
                                  f'return slow({again})')
                if ft is T.INT:
                    src_guards.append(f'    if not -9223372036854775808 <= f{i}_{k} <= 9223372036854775807:\n        '
                                      f'return slow({again})')
        elif t.kind == 'object' and i in deep:   # through the objects in its attributes, as the general path walks them
            src_guards.append(f'    D{i} = {{}}')
            src_guards.extend(_deep_guards(env, i, f'o{i}', args[i], t, n, (), again, deep[i], places))
        elif t.kind == 'object':
            src_guards.extend(_object_guards(env, i, args[i], t, n, again))
        elif t is T.U64:
            src_guards.append(f'    if not 9223372036854775808 <= {n} <= 18446744073709551615:\n        return slow({again})')
        elif t.kind == 'int' and not t.np:
            src_guards.append(f'    if not -9223372036854775808 <= {n} <= 9223372036854775807:\n        return slow({again})')
    for n, at in late:
        if at not in places:
            return f._slow_call
        src_guards.append(f'    if {n} is not {places[at]}:\n        return slow({again})')
    addresses = ''.join(f'        p{i} = addressof(from_buffer({names[i]}))\n' for i in arrays if args[i].flags.writeable)
    take = (f'    try:\n{addresses}    except (ValueError, TypeError, BufferError):      # empty: no buffer to take the address of\n'
            f'        return slow({again})\n') if addresses else ''
    size = entry.io_size
    r = entry.ret
    if entry.records:
        env.update(arrays=f._array_result)
        result = f"arrays(compiled, io, ({''.join(n + ', ' for n in names)}))"
    elif r is T.NONE:
        result = 'None'
    elif r is T.INT:
        result = 'q[0]'
    elif r is T.FLOAT:
        result = 'd[0]'
    elif r.kind == 'tuple':
        env['box_tuple'] = f._box_tuple
        result = 'box_tuple(io, ret)'
    else:
        result = 'box(io, 0, 16, ret)'
    rerun = ''
    if entry.rerun:
        env.update(f=f, RERUN=RERUN)
        rerun = f"    if status == RERUN:\n        pool.append(entry)\n        return f._py({', '.join(real)})\n"
    saves = ''
    for n, (i, key, at, ot) in enumerate(entry.outs):
        env[f'okey{n}'], env[f'otype{n}'] = key, ot
        if entry.types[i].kind == 'object':
            saves += f'    {names[i]}.{key} = box(io, {at}, {at + 8}, otype{n})\n'
        else:
            saves += f'    {names[i]}[okey{n}] = box(io, {at}, {at + 8}, otype{n})\n'
    dicts = [names[i] for i, t in enumerate(entry.types) if t.kind == 'dict']
    if len(dicts) > 1:                 # two names for one dict: a write must show through both
        pairs = ' or '.join(f'{a} is {b}' for k, a in enumerate(dicts) for b in dicts[k + 1:])
        src_guards.append(f'    if {pairs}:\n        return slow({again})')
    objects = [names[i] for i, t in enumerate(entry.types) if t.kind == 'object']
    if len(objects) > 1:
        pairs = ' or '.join(f'{a} is {b}' for k, a in enumerate(objects) for b in objects[k + 1:])
        src_guards.append(f'    if {pairs}:\n        return slow({again})')
    src = f"""def plan({', '.join(real)}):
{chr(10).join(src_guards)}
{take}    if pool:
        io, ioaddr, q, d = entry = pool.pop()
    else:
        io = bytearray({size})
        ioaddr = addressof(from_buffer(io))
        view = memoryview(io)
        q, d = view.cast('q'), view.cast('d')
        entry = (io, ioaddr, q, d)
    status = kernel({''.join(c + ', ' for c in cargs)}ioaddr)
{rerun}{saves}    if status:
        pool.append(entry)
        failed(compiled, status, io)
    value = {result}
    pool.append(entry)
    return value
"""
    exec(src, env)  # noqa: S102
    return env['plan']


class _Marshalled:
    """The C arguments of a call, and what the kernel may have written that goes back into the Python objects after it."""
    __slots__ = ('cargs', 'columns', 'refcols', 'copies', 'by_cls', 'where', 'made_k', 'room', 'pool_t', '_pool_cargs', '_pool_columns',
                 'before', 'gl')

    def __init__(self):
        self.cargs, self.columns, self.refcols, self.copies = [], [], [], []
        self.by_cls, self.where, self.made_k, self.room, self.pool_t = {}, None, {}, None, None
        self.before = {}          # id of a list the kernel may write -> its values as copied in (_snapshot)
        self.gl = None            # lists held by reference (_GLBuild), built for the call and written back after it


def _pack_pool(m, i, x, t, args):
    """Per class: a list per number, then an index list per reference (built by _pool_build, before the others)."""
    m.cargs += m._pool_cargs
    m.columns += [(i,) + c for c in m._pool_columns]
    m.refcols = [(i,) + c for c in m.refcols]


def _pack_reflist(m, i, x, t, args):
    """Objects kept in the pool: their indexes, written back as the objects."""
    a = array.array('q', (-1 if y is None else m.where[id(y)] for y in x))
    m.copies.append((('ref', i, t.cls), a))
    m.cargs += [a.buffer_info()[0], len(a)]


def _pack_inlist(m, i, x, t, args):
    """An element of another argument: its place there."""
    m.cargs.append(next(k for k, y in enumerate(args[t.elem]) if y is x))


def _pack_objlist(m, i, x, t, args):
    """A list of numbers per attribute, by path, written back into the instances; pooled where objects repeat."""
    pool, ixs = x, []
    if '' in t.strided:                          # repeats: each instance once, and which is at each place
        place = {}
        ixs.append(array.array('q', (place.setdefault(id(y), len(place)) for y in x)))
        pool = list({id(y): y for y in x}.values())
    levels = _pool_levels(pool, t)
    at = {path: objs for path, objs, _ in levels}
    ixs += [ix for _, _, ix in levels if ix is not None]
    for key, ft in leaf_types(t.elem):           # breadth first through objects in attributes, as flattened
        path, _, last = key.rpartition('.')
        a = array.array(_LIST_CODES[ft.name], map(operator.attrgetter(last), at[path]))
        m.columns.append((i, key, a, ft, at[path], last))
        m.cargs += [a.buffer_info()[0], len(a)]
    for ix in ixs:
        m.copies.append((None, ix))             # kept alive for the call
        m.cargs += [ix.buffer_info()[0], len(ix)]


def _pack_list(m, i, x, t, args):
    """A list of numbers: a copy, written back where it changed."""
    a = array.array(_LIST_CODES[t.elem.name], x)
    m.copies.append((i, a))
    m.cargs += [a.buffer_info()[0], len(a)]


def _pack_pylist(m, i, x, t, args):
    """A list of lists: one array, row after row, and its extents."""
    a = array.array(_LIST_CODES[t.elem.name], _flatten(x, t.ndim))
    m.copies.append((i, a))
    m.cargs.append(a.buffer_info()[0])
    level = x
    for _ in range(t.ndim):
        m.cargs.append(len(level))
        level = level[0]


def _pack_glist(m, i, x, t, args):
    """A list held by reference (_glist): its header and elements built in memory Python owns, one header per list
    object, so a list in two places is one list; written back after the call."""
    if m.gl is None:
        m.gl = _GLBuild()
    h = m.gl.build(x, t) if t.kind == 'glist' else m.gl.build_set(x, t) if t.kind == 'gset' else m.gl.build_record(x, t)
    m.gl.top.append((i, h))
    m.cargs.append(h)


_PACKERS = {'pool': _pack_pool, 'reflist': _pack_reflist, 'inlist': _pack_inlist, 'objlist': _pack_objlist, 'list': _pack_list,
            'glist': _pack_glist, 'record': _pack_glist, 'gset': _pack_glist}


def _marshal(entry, args, room=None, prebuilt=None):
    """One call's C arguments, by each argument's kind (_PACKERS; the rest by _push), and what goes back after it. None if
    the pool isn't as the signature says."""
    m = _Marshalled()
    m.pool_t = next((t for t in entry.types if t.kind == 'pool'), None)
    if m.pool_t is not None:                     # every object references reach, each once, a pool per class
        built = prebuilt if prebuilt is not None else _pool_build(m.pool_t, args, room)
        if built is None:
            return None
        m.by_cls, m.where, m._pool_cargs, m._pool_columns, m.refcols, m.made_k, room = built
    m.room = room
    for i, (x, t) in enumerate(zip(args, entry.types)):
        pack = _pack_pylist if T.is_pylist(t) else _PACKERS.get(t.kind)
        if pack is not None:
            pack(m, i, x, t, args)
        else:
            _push(m.cargs, x, t, m.where)
    if entry.libm is not None:
        m.cargs.append(entry.libm)
    if entry.stop:
        m.cargs.append(_interrupt.address())
    if entry.counts is not None:
        m.cargs.append(entry.counts)
    _snapshot(entry, m)
    return m


def _snapshot(entry, m):
    """The values of each list the kernel may write, as copied in: what goes back is only what this call changed, so a call
    on another thread that changed other elements of the same list or objects meanwhile keeps its writes."""
    written = [c[2] for c in m.columns if c[:2] in entry.columns] + [c[2] for c in m.refcols if c[:2] in entry.columns]
    for i, a in m.copies:
        if i is not None and (i[1] if type(i) is tuple else i) in entry.lists:
            written.append(a)
    for a in written:
        m.before[id(a)] = a[:]


def _write_back(entry, m, args):
    """What the kernel wrote that Python would have, even before an exception: attributes' lists into their objects (the
    values that didn't change keep their identity), references as the objects at their indexes, lists copied back."""
    if m.gl is not None:
        m.gl.write_back(entry, args)
    before = m.before
    for i, key, a, ft, held, last in m.columns:
        if (i, key) in entry.columns:
            b = before[id(a)]
            if len(a) > len(held):
                a, b = a[:len(held)], b[:len(held)]
            _copy_back_attr(held, last, a, ft, b)
    for i, slot, a, objs, key, targets in m.refcols:
        if (i, slot) in entry.columns:
            for o, k, k0 in zip(objs, a, before[id(a)]):
                if k != k0:
                    setattr(o, key, None if k < 0 else targets[k])
    for i, a in m.copies:
        if type(i) is tuple:                     # a list of references: the objects at those indexes
            _, i, cls = i
            if i in entry.lists:
                objs = m.by_cls.get(cls, [])
                for k, (j, j0) in enumerate(zip(a, before[id(a)])):
                    if j != j0:
                        args[i][k] = None if j < 0 else objs[j]
            continue
        if i is not None and i in entry.lists:
            t = entry.types[i]
            if T.is_pylist(t):
                _copy_back_nested(args[i], a, t.elem, t.ndim, before[id(a)])
            else:
                _copy_back(args[i], a, t.elem, before[id(a)])


class _Never:
    """No value has this type: an empty fast slot matches no call."""


class _Absent:
    """An argument a front's call left out."""


_ABSENT = _Absent()
_FRONTS = {}
_FRONT_MAX = 8              # positional parameters up to which a Function gets a __call__ of its own arity


def _front(n):
    """A subclass of Function whose __call__ takes n arguments by position and, when their types are those of the
    last plan made (_fast: the types, then the plan), calls that plan directly: no *args tuple, no dict lookup, no
    tuple of types. Anything else (a keyword, a default, another type) takes Function.__call__, as before."""
    cls = _FRONTS.get(n)
    if cls is not None:
        return cls
    ps = [f'_sj_a{i}' for i in range(n)]
    test = ''.join(f'type({a}) is f[{i}] and ' for i, a in enumerate(ps)) or 'f[0] is not None and '
    src = (f"def __call__(_sj_self, {''.join(a + '=ABSENT, ' for a in ps)}*_sj_rest, **_sj_kw):\n"
           f"    f = _sj_self._fast\n"
           f"    if {test}not _sj_rest and not _sj_kw:\n"
           f"        return f[{n}]({', '.join(ps)})\n"
           f"    return general(_sj_self, *[a for a in ({''.join(a + ', ' for a in ps)}) if a is not ABSENT], *_sj_rest, **_sj_kw)\n")
    env = {'ABSENT': _ABSENT, 'general': Function.__call__}
    exec(src, env)  # noqa: S102
    cls = _FRONTS[n] = type('Function', (Function,), {'__call__': env['__call__'], '__module__': Function.__module__,
                                                     '__qualname__': 'Function'})
    return cls


class Function:
    def __init__(self, fn, boundscheck=True, cache=True, fallback=True, parity=True, _loop=None, counts=False):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._py = fn             # what runs when a signature doesn't compile: fn, or fn with its loops lifted
        self._lifted = None       # the Functions of its lifted loops, once that has been tried
        self._loop = _loop        # a lifted loop: (its function's tree, the first line, the file)
        self._owner = fn.__name__ # the decorated function this is, or that this loop was lifted from
        self._counting = counts   # @scram(counts=True)
        self._boundscheck = boundscheck
        self._fallback = fallback
        self._parity = parity
        self._cache = {}
        self._use_disk = cache
        self._plans = {}          # the arguments' Python types -> a function that makes the call (see _make_plan)
        self._gl_held = None      # positions of the arguments a list passed in must be held by reference for (_glist_key)
        self._compile_s = {}      # signature -> seconds spent compiling it
        self._why = {}            # signature -> (line, reason) it runs as Python
        self._stats = {}          # signature -> [calls, native seconds, Python seconds], under preflight --run
        self._lock = threading.Lock()
        self._hidden = None       # the globals this function reads, found at the first call: they may not exist before it
        self._hidden_from = None  # for each of those: (the dict it's in, its name there), which a called function's may differ in
        self._ctree = None        # the tree with those as parameters, which is what's compiled
        code = fn.__code__
        if _loop is None:
            _functions.add(self)
            self._first = code.co_firstlineno
        else:
            self._first = _loop[1]
        self._filename = code.co_filename
        self._contracts = [(i, p, a) for i, (p, a) in enumerate(
            (p, fn.__annotations__.get(p)) for p in code.co_varnames[:code.co_argcount]) if a in _ANNOTATIONS]
        self._tree, self._why_not = self._parse(fn) if _loop is None else (_loop[0], None)
        self._source_tree = self._tree      # as written: for lifting loops, which makes Python of it
        self._kwonly_args = list(self._tree.args.kwonlyargs) if self._tree is not None else []
        self._kwonly = [a.arg for a in self._kwonly_args]
        self._npos = len(self._tree.args.posonlyargs + self._tree.args.args) if self._tree is not None else 0
        if self._kwonly:                    # compiled as parameters after the others, bound by keyword (_bind)
            self._tree = copy.deepcopy(self._tree)
            self._tree.args.args = self._tree.args.args + self._tree.args.kwonlyargs
            self._tree.args.kwonlyargs, self._tree.args.kw_defaults = [], []
            self._py = self._by_keyword(fn)
        self._dict_keys = _dict_keys(self._tree)
        self._attrs = _attrs_used(self._tree)
        self._elems = _element_attrs(self._tree)
        self._passes = _passes(self._tree)
        self._unused = _unused(self._tree)
        self._returned = _returned(self._tree)
        self._nested = _nested_uses(self._tree)
        self._replans = 8         # how many times more a dict's new value types get a call plan of their own
        self._np = None
        n = code.co_argcount
        self._fast = (_Never,) * n + (None,)        # the last plan's argument types and the plan (see _front)
        if not self._kwonly and not code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS) and n <= _FRONT_MAX:
            self.__class__ = _front(n)

    @staticmethod
    def _parse(fn):
        if hasattr(fn, '__wrapped__'):
            return None, 'the function is already wrapped by another decorator'
        if fn.__code__.co_freevars:
            return None, 'closures (the function reads variables of an enclosing function)'
        try:
            src = textwrap.dedent(inspect.getsource(fn))
            tree = ast.parse(src)
        except (OSError, TypeError, SyntaxError, IndentationError):
            return None, 'the source code is not available'
        fdef = tree.body[0] if tree.body else None
        if not isinstance(fdef, ast.FunctionDef):
            return None, 'not a plain def function'
        _mangle(fdef, fn.__qualname__)
        return fdef, None

    def _uses_of(self, idx, cls):
        """What this function makes of an instance of cls given as parameter idx: for a caller passing one."""
        return _param_uses(self, idx, cls, self._attrs.get(idx, ((), set(), set())))

    # descriptor: work as a method too (self is then an argument)
    def __get__(self, obj, cls=None):
        return self if obj is None else types.MethodType(self, obj)

    @property
    def signatures(self):
        """The argument types of each compiled specialization."""
        return [k for k, v in self._cache.items() if isinstance(v, Compiled)]

    def c(self, signature=None):
        """The C scramjit generated for a signature (the most recent one by default)."""
        compiled = [v for v in self._cache.values() if isinstance(v, Compiled)]
        if signature is None:
            if not compiled:
                raise ValueError(f'{self.__name__} has no compiled signatures yet')
            return compiled[-1].source
        v = self._cache.get(tuple(signature))
        if not isinstance(v, Compiled):
            raise KeyError(signature)
        return v.source

    def _by_keyword(self, fn):
        """fn, called with its keyword-only parameters' values after the others, as this function's calls carry them."""
        npos, names = self._npos, self._kwonly

        def py(*full):
            return fn(*full[:npos], **dict(zip(names, full[npos:])))
        return py

    def _bind(self, args, kwargs):
        """args and kwargs as one positional tuple, with the defaults filled in, as Python binds them; None for what
        isn't a plain call of plain parameters (Python then binds it, or raises its TypeError)."""
        code = self._fn.__code__
        if code.co_flags & 0x0C:      # *args or **kwargs
            return None
        n = code.co_argcount
        if code.co_kwonlyargcount:
            kw = {k: v for k, v in kwargs.items() if k in self._kwonly}
            rest = {k: v for k, v in kwargs.items() if k not in self._kwonly}
            head = self._bind_positional(args, rest, code, n)
            if head is None:
                return None
            defaults = self._fn.__kwdefaults__ or {}
            if any(k not in kw and k not in defaults for k in self._kwonly):
                return None                       # Python's TypeError
            return head + tuple(kw[k] if k in kw else defaults[k] for k in self._kwonly)
        return self._bind_positional(args, kwargs, code, n)

    def _bind_positional(self, args, kwargs, code, n):
        if len(args) > n:
            return None
        names = code.co_varnames[:n]
        bound = list(args) + [_MISSING] * (n - len(args))
        for k, v in kwargs.items():
            if k not in names or names.index(k) < code.co_posonlyargcount or bound[names.index(k)] is not _MISSING:
                return None
            bound[names.index(k)] = v
        defaults = self._fn.__defaults__ or ()
        for i in range(n):
            if bound[i] is _MISSING:
                if i < n - len(defaults):
                    return None
                bound[i] = defaults[i - (n - len(defaults))]
        return tuple(bound)

    def _set_plan(self, key, plan):
        """A plan for calls with these argument types; the last one made is also what the front tries first."""
        self._plans[key] = plan
        if len(key) == len(self._fast) - 1:
            self._fast = key + (plan,)

    def __call__(self, /, *args, **kwargs):
        if kwargs or len(args) != self._fn.__code__.co_argcount or self._kwonly:     # a default or a keyword: bound as Python binds it
            full = self._bind(args, kwargs)
            if full is None:
                return self._fn(*args, **kwargs)
            args, kwargs = full, {}
        key = tuple(map(type, args))
        plan = self._plans.get(key)
        if plan is not None and not kwargs:             # these types were checked when the plan was made
            if len(key) == len(self._fast) - 1:
                self._fast = key + (plan,)              # the front tries these types first from now on
            return plan(*args)
        for i, p, ann in self._contracts:
            if i < len(args) and not _contract_ok(args[i], ann):
                raise TypeError(f'{self.__name__}() argument {p!r} must be {_ANNOTATIONS[ann]}, '
                                f'not {type(args[i]).__name__}')
        real = args
        args, key = self._args_and_key(args)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._specialize(key, args)
        if _profile:
            return self._timed(key, entry, args)
        if entry is not PYTHON:
            self._set_plan(tuple(map(type, real)), _make_plan(self, entry, args))
        elif None not in key and not any(t.kind in ('array', 'object', 'objlist') for t in key) and dict not in map(type, args) and not self._hidden:
            self._set_plan(tuple(map(type, real)), self._py)   # these types always run as Python
        if entry is PYTHON:
            return self._py(*real)
        return self._run(entry, args)

    def _args_and_key(self, real):
        """The arguments with the globals after them, and the call's signature. A method found in it may read globals
        this function doesn't know of yet: they become more of its globals, and the arguments are taken again."""
        args = self._with_globals(real)
        key = self._key(args)
        if self._tree is not None and self._add_method_globals(key):
            args = self._with_globals(real)
            key = self._key(args)
        return args, key

    def _add_method_globals(self, key):
        """Take the globals of the methods in these types, the instances' and the lists'. True if any were new."""
        if self._hidden is None:
            self._find_globals()
        new = []
        for t in key:
            if t is None or t.kind not in ('object', 'objlist', 'madeclass', 'elemref'):
                continue
            for _, ft in (t.elem.elem if t.kind in ('objlist', 'elemref') else t.recipe['methods'] if t.kind == 'madeclass' else t.elem):
                w = ft.elem if ft.kind == 'function' else None
                if not isinstance(w, Function) or w._tree is None:
                    continue
                if w._hidden is None:
                    w._find_globals()
                for og, on in w._hidden_from:
                    if og is _POOLDICT:
                        continue
                    if not any(og is a and on == b for a, b in self._hidden_from + new):
                        new.append((og, on))
        if not new:
            return False
        tree = self._ctree or self._tree
        for og, on in new:
            name = f'sjg{len(self._hidden)}_{on}'
            self._hidden.append(name)
            self._hidden_from.append((og, on))
        self._ctree = copy.deepcopy(self._tree)
        self._ctree.args.args = self._ctree.args.args + [ast.copy_location(ast.arg(n), tree) for n in self._hidden]
        self._dict_keys, self._attrs, self._elems = _dict_keys(self._ctree), _attrs_used(self._ctree), _element_attrs(self._ctree)
        self._passes = _passes(self._ctree)
        self._plans.clear()             # a plan reads the globals it was made with
        self._fast = (_Never,) * len(self._fast[:-1]) + (None,)
        return True

    def _with_globals(self, args):
        """The arguments, and after them the globals the function reads, as they are now."""
        if self._hidden is None:
            self._find_globals()
        if not self._hidden:
            return args
        return args + tuple(g.get(n, _MISSING) for g, n in self._hidden_from)

    def _find_globals(self):
        """A number, tuple, array or dict at module level is one more input: it's read at each call and passed like the others, so
        loops can use it, a new value needs no recompile, and a new type is a new signature. Anything else stays a name."""
        self._hidden, self._hidden_from = [], []
        tree = self._tree
        if tree is None:
            return
        local = _lift.local_names(tree)
        loads = set()
        for st in tree.body:
            loads |= _lift._split(st)[0]
        called = {n.func.id for n in _facts(tree).calls if isinstance(n.func, ast.Name)}
        g = self._fn.__globals__
        attrs = {}
        for name, attr, how in _attribute_uses(tree):
            _add_use(attrs.setdefault(name, (set(), set(), set())), attr, how)
        self._hidden = sorted(n for n in loads - local - called if n in g and (
            _arg_type(g[n]) is not None or type(g[n]) is dict
            or (n in attrs and type(_object_fields(g[n], sorted(attrs[n][0]), attrs[n][1], attrs[n][2])) is list)))
        callees = sorted(n for n in called - local if isinstance(g.get(n), Function))
        self._hidden += callees + sorted(n for n in called - local if T.is_c_function(g.get(n)))
        self._made = sorted(n for n in called - local if isinstance(g.get(n), type) and not issubclass(g[n], (enum.Enum, BaseException)))      # classes it makes instances of: in the signature, by identity
        self._hidden += self._made
        self._hidden_from = [(g, n) for n in self._hidden]
        for n in callees:                # what they read is read at each call too, and may be in another module
            f = g[n]
            if f._hidden is None:
                f._find_globals()        # recursion finds this one's list unfinished, and won't compile anyway
            for og, on in f._hidden_from:
                if og is _POOLDICT:              # its own: a caller doesn't make it
                    continue
                if not any(og is a and on == b for a, b in self._hidden_from):
                    self._hidden.append(on if og is g and on not in self._hidden else f'sjg{len(self._hidden)}_{on}')
                    self._hidden_from.append((og, on))
        if any(_ref_candidates(tree)):           # references chosen by data: their pool, made at each call
            self._hidden.append('sjpool')
            self._hidden_from.append((_POOLDICT, 'sjpool'))
        if self._hidden:
            self._ctree = copy.deepcopy(tree)
            self._ctree.args.args = self._ctree.args.args + [ast.copy_location(ast.arg(n), tree) for n in self._hidden]
            self._dict_keys = _dict_keys(self._ctree)
            self._attrs = _attrs_used(self._ctree)
            self._elems = _element_attrs(self._ctree)
            self._passes = _passes(self._ctree)
            self._nested = _nested_uses(self._ctree)
            self._unused = _unused(self._tree)

    def _hidden_names_read(self):
        """The globals this function itself reads, not those it reads only through what it calls."""
        return {n for n in self._hidden if not n.startswith('sjg')}

    def _real(self, args):
        """The caller's arguments, without the globals added to them."""
        return args[:len(args) - len(self._hidden)] if self._hidden else args

    def _slow(self, args):
        """A call a plan turned down: an int past 64 bits, an array of another shape of type."""
        real = args
        args, key = self._args_and_key(args)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._specialize(key, args)
        if entry is PYTHON:
            return self._py(*real)
        if self._replans and any(t.kind in ('dict', 'object') for t in entry.types):    # writing p['s'] can change what the plan checks for
            self._replans -= 1
            self._set_plan(tuple(map(type, real)), _make_plan(self, entry, args))
        return self._run(entry, args)

    def _failed(self, entry, status, io):
        slots = [int.from_bytes(io[i:i + 8], 'little', signed=True) for i in range(0, 40, 8)]
        first = self._first - 1                    # the kernel's lines count from the def
        frames = [(self._filename, slots[1] + first if slots[1] else 0, self.__name__)]
        if entry.frames:                 # those a call passed through, innermost first
            n = int.from_bytes(io[entry.tb_at:entry.tb_at + 8], 'little', signed=True)
            for k in range(n, 0, -1):
                v = int.from_bytes(io[entry.tb_at + 8 * k:entry.tb_at + 8 * k + 8], 'little', signed=True)
                filename, name, offset = entry.frames[v >> 32]
                line = v & 0xffffffff
                frames.append((filename, line + offset if line else 0, name))
        _raise_chain(exception(status, slots), frames)

    def _callee(self):
        """What a compiled function needs to call this one."""
        if self._tree is None:
            raise Unsupported(None, self._why_not)
        if self._hidden is None:
            self._find_globals()
        return Callee(self, self._ctree or self._tree, self._fn.__globals__, self._filename, self._first - 1, self._boundscheck, self._parity,
                      [(id(g), n) for g, n in self._hidden_from], self._hidden_map(), self.__name__, npos=self._npos)

    def _hidden_map(self):
        return {(id(g), on): p for (g, on), p in zip(self._hidden_from, self._hidden)}

    def _slow_call(self, *args):
        return self._slow(args)

    def _timed(self, key, entry, args):
        s = self._stats.setdefault(key, [0, 0.0, 0.0])
        s[0] += 1
        t = time.perf_counter()
        if _running is not None:
            me = threading.get_ident()
            outer = _running.get(me)
            _running[me] = (self, entry is not PYTHON, t)
        try:
            return self._py(*self._real(args)) if entry is PYTHON else self._run(entry, args)
        finally:
            s[2 if entry is PYTHON else 1] += time.perf_counter() - t
            if _running is not None:
                if outer is None:
                    del _running[me]
                else:
                    _running[me] = outer

    def line_counts(self):
        """With counting on, over all signatures and lifted loops: (line -> times it ran, line -> (kind, times
        its body ran)), kind being 'if' or 'loop'. Lines are the file's. A kernel adds to the counts with the GIL
        released, so another thread can read them during a call."""
        runs, heads = {}, {}
        for f in [self] + self._all_loops():
            for entry in list(f._cache.values()):
                if isinstance(entry, Compiled) and entry.counts is not None:
                    r, h = entry.read_counts()
                    for n, v in r.items():
                        runs[n] = runs.get(n, 0) + v
                    for n, (kind, v) in h.items():
                        heads[n] = (kind, heads.get(n, (kind, 0))[1] + v)
        return runs, heads

    def preflight(self, signature=None):
        """Print this function line by line: native or Python and why, types, bounds checks."""
        from .cli.report import function_report
        function_report(self, signature, sys.stdout)

    def _specialize(self, key, args):
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                return entry
            sig = ', '.join(_describe(x, t) for x, t in zip(self._real(args), key))
            t0 = time.perf_counter()
            try:
                entry = self._compile(key, args)
                self._compile_s[key] = time.perf_counter() - t0
                if _log:
                    print(f'scramjit: compiled {self.__name__}({sig}) in {self._compile_s[key]:.2f} s', file=sys.stderr)
            except Unsupported as u:
                line = (u.lineno + self._first - 1) if u.lineno else self._first
                self._why[key] = (line, u.reason)
                if _log:
                    print(f'scramjit: {self.__name__}({sig}) runs as Python: line {line}: {u.reason}', file=sys.stderr)
                if not self._fallback:
                    raise CompileError(f'{self._filename}:{line}: {self.__name__}({sig}) is not compiled: '
                                       f'{u.reason}') from None
                self._lift()
                if self._loop is None:
                    apart = ', apart from the loops that compile' if self._lifted else ''
                    warnings.warn_explicit(f'{self.__name__}({sig}) runs as Python{apart}: {u.reason}',
                                           FallbackWarning, self._filename, line)
                entry = PYTHON
            self._cache[key] = entry
            return entry

    def _native_loops(self):
        """The lifted loops that have compiled for some signature, however deep."""
        out = []
        for f in self._lifted or []:
            if any(isinstance(v, Compiled) for v in f._cache.values()):
                out.append(f)
            out += f._native_loops()
        return out

    def _all_loops(self):
        out = []
        for f in self._lifted or []:
            out.append(f)
            out += f._all_loops()
        return out

    def _lift(self):
        """Python runs from now on for some signature: lift the loops that may still compile."""
        if self._lifted is not None or self._tree is None:
            return
        self._lifted = []

        def native(python, tree, loop):
            f = Function(python, self._boundscheck, self._use_disk, True, self._parity, (tree, self._first), self._counting)
            f._owner = self._owner
            return f

        source = self._source_tree if self._source_tree is not None else self._tree
        self._py, self._lifted = _lift.lift(source, self._fn, self._filename, self._first, native,
                                            source.body[0] if self._loop else None)
        if self._kwonly:
            self._py = self._by_keyword(self._py)

    def _compile(self, key, args):
        if self._tree is None:
            raise Unsupported(None, self._why_not)
        real = len(args) - len(self._hidden or ())
        if self._hidden and 'sjpool' in self._hidden and key[real + self._hidden.index('sjpool')] is None:
            raise Unsupported(None, getattr(self, '_pool_why', 'its references can\'t be pooled'))
        for i, (x, t) in enumerate(zip(args, key)):
            if t is None and i >= real and self._hidden[i - real] == 'sjpool':
                raise Unsupported(None, getattr(self, '_pool_why', 'its references can\'t be pooled'))
            if t is None and i >= real and self._hidden[i - real] not in self._hidden_names_read():
                raise Unsupported(None, f'a function it calls reads the global {self._hidden_from[i - real][1]}, which is {_describe(x, t)}')
            if t is None and i >= real and getattr(self, '_made', None) and self._hidden[i - real] in self._made:
                why = getattr(self, '_made_why', "it can't be compiled")
                raise Unsupported(None, f'makes instances of {self._hidden[i - real]}: {why}')
            if t is None:
                fields = _object_fields(x, *self._attrs[i], nested=self._nested.get(i)) if i in self._attrs and type(x) not in (dict, list) else None
                if type(fields) is str:
                    raise Unsupported(None, f'an argument of type {_describe(x, t)} is not compiled: {fields}')
                raise Unsupported(None, f'an argument of type {_describe(x, t)} is not compiled')
            if t.kind == 'function' and i < real:
                raise Unsupported(None, 'a function passed as an argument')
        counts = self._counting or _counts is True or bool(_counts and self._owner in _counts)
        c = Compiler(self._ctree or self._tree, list(key), self._fn.__globals__, self._filename, self._first - 1,
                     self._boundscheck, self._parity, counts, Unit(self), self._hidden_map() if self._hidden else None)
        source = c.generate()
        cimo = _get_cimo()
        try:
            if c.unit.modules:           # as those modules were compiled, or their results would change
                module = cimo.compile(PRELUDE + source, cache=self._use_disk, fma=c.unit.fma, wrapv=c.unit.wrapv)
            else:
                module = cimo.compile(PRELUDE + source, cache=self._use_disk, fma=not self._parity)   # Python never contracts
            kernel = module.sj_kernel
        except cimo.CompileError as e:
            e = cimo_diagnostic(str(e), self._filename, self._first - 1, c.unit.frames)
            if c.unit.modules:
                raise Unsupported(None, f'the C it calls doesn\'t compile in one file with the C generated here: {e}')
            raise Unsupported(None, f'internal error: cimo rejected the generated C ({e}); please report it')
        if c.uses_stop:
            _interrupt.ensure()
        if any(t.np for t in key) or c.ret.np:
            import numpy
            self._np = numpy
        entry = Compiled(kernel, c.ret, key, source, _libm_table() if c.uses_libm else None,
                         c.shown_types(), c.checks, c.hoisted_lines, c.gotchas, c.uses_stop, c.uses_rerun)
        entry.module = module            # cimo's module: in Pyodide, module.wasm is its WebAssembly (the site shows it)
        entry.outs = [(i, k, 8 * c.out_slot(n), c.env[flat]) for n, (flat, i, k) in enumerate(c.outs)]
        entry.lists = c.written_lists()
        entry.columns = c.written_columns()
        entry.instance_ret = c.instance_ret
        entry.array_args = c.array_args
        entry.records = [(8 * c.record_slot(k), struct.Struct(f'<{array_record_slots(t)}q'), t.elem.name[3:] if not T.is_pylist(t) else t)
                         for k, t in enumerate(ret_arrays(c.ret))]
        if any(not isinstance(name, T.Type) for _, _, name in entry.records):
            import numpy
            self._np = numpy
            entry.records = [(at, layout, name if isinstance(name, T.Type) else numpy.dtype(name).str) for at, layout, name in entry.records]
        entry.io_size = 8 * c.record_slot(len(entry.records))
        entry.frames = c.unit.frames if c.calls else None
        if c.calls:                      # room for the trace of a failed call
            entry.tb_at = entry.io_size
            entry.io_size += 8 * (1 + len(c.unit.frames))
        entry.free = getattr(module, 'sj_release', None) if c.makes or c.ret.kind == 'list' else None
        entry.gl_types = {i: c.env[p] for i, p in enumerate(c.params) if c.env[p].kind in ('glist', 'record', 'gset')}   # as appends made them
        if counts:
            entry.counts = bytearray(8 * max(len(c.count_lines), 1))
            entry.count_lines, entry.count_heads, entry.count_rest = c.count_lines, c.count_heads, c.count_rest
        return entry

    def _key(self, args):
        """The signature of a call. A dict's type is the keys this function uses: nothing else in it matters. A dict
        holding lists, or held by reference here, is a record (_glist)."""
        key = self._key_of(args)
        if any(type(x) is dict for x in args):
            self._held()
            out = list(key)
            for i, x in enumerate(args):
                if type(x) is dict and (i in self._gl_held or (out[i] is None and any(
                        type(v) in (list, dict) or hasattr(v, '__array_interface__') for v in x.values()))) \
                        and not any(getattr(a, '__dict__', None) is x for a in args):   # an object's own dict stays with it
                    t = _glist_elem_type(x, 0)
                    if t is not None and t.kind == 'record':
                        out[i] = t
            key = tuple(out)
        return key

    def _held(self):
        if self._gl_held is None:
            from ._glist import held_names
            code = self._fn.__code__
            names = held_names(self._tree) if self._tree is not None else set()
            self._gl_held = {i for i, n in enumerate(code.co_varnames[:code.co_argcount]) if n in names}
        return self._gl_held

    def _key_of(self, args):
        key = tuple(map(_arg_type, args))
        flat = [type(x) is list and t is not None and t.kind == 'list' for x, t in zip(args, key)]   # numbers only: no rows
        if any(type(x) is list for x in args):
            key = self._glist_key(args, key)
        lists = [id(y) for x, f in zip(args, flat) if type(x) is list for y in [x] + ([] if f else [r for r in x if type(r) is list])]
        if len(set(lists)) != len(lists):        # one list under two names: one copy, the second name the first's
            first, rows = {}, [id(r) for x, f in zip(args, flat) if type(x) is list and not f for r in x if type(r) is list]
            out = list(key)
            for i, x in enumerate(args):
                if type(x) is not list or (out[i] is not None and out[i].kind == 'glist'):   # held by reference: shared as it is
                    continue
                if len(set(rows)) != len(rows) or id(x) in rows or (out[i] is not None and T.is_pylist(out[i])):
                    out[i] = None                # a row in two places, or a list that's a row too: not yet
                elif id(x) in first:
                    out[i] = T.same_as(first[id(x)]) if out[first[id(x)]] is not None and out[i] is not None else None
                else:
                    first[id(x)] = i
            key = tuple(out)
        if self._hidden and any(t is not None and t.kind == 'classref' for t in key):
            out = list(key)
            real = len(args) - len(self._hidden)
            owners = None
            for k, name in enumerate(self._hidden):
                x = args[real + k]
                if out[real + k] is not None and out[real + k].kind == 'classref' and not issubclass(x, enum.Enum):
                    if owners is None:
                        owners = _owner_classes(self, args)
                    t = _made_class_type(self, x, name, owners)
                    if t is not None or name in getattr(self, '_made', ()):
                        out[real + k] = t          # made here, or by what it calls: how, in the signature
            key = tuple(out)
        refs = None
        if self._hidden and 'sjpool' in self._hidden:         # references chosen by data: one pool of their objects
            out = list(key)
            at = len(args) - len(self._hidden) + self._hidden.index('sjpool')
            plan = _pool_plan(self, args)
            if plan[0] is None:
                out[at], self._pool_why = None, plan[1]
            else:
                out[at] = plan[1]
                for i, t in plan[0].items():
                    out[i] = t
                refs = plan[2] if plan[1].kind == 'pool' else None
            key = tuple(out)
        if not ((self._attrs or self._elems or self._passes or self._unused or self._returned) and None in key) \
                and (not self._dict_keys or dict not in map(type, args)):
            return key
        seen, out = set(), list(key)
        owners = _owner_classes(self, args) if self._passes else None
        empty = ((), set(), set())
        roots = []                               # instances: their types together, as one graph with its alias pattern
        places = {}                              # id of an element of a list of instances argument -> (that argument, its place)
        for k, x in enumerate(args):
            if type(x) in (list, dict) and out[k] is None and k in self._unused:
                out[k] = T.object_of(type(x).__qualname__, [], None, type(x))     # never named: nothing of it reaches the C
            elif type(x) is list and x and out[k] is None and _is_instance(x[0]):
                for at, y in enumerate(x):
                    places.setdefault(id(y), (k, at))
        joined = {}                              # an argument that is an element of another: its uses join that list's
        for i, x in enumerate(args):
            if out[i] is not None or type(x) in (dict, list) or not (i in self._attrs or i in self._passes or i in self._unused
                                                                      or i in self._returned):
                continue
            if id(x) in places and (i in self._attrs or i in self._passes):
                joined[i] = places[id(x)]
                continue
            uses, nested = self._attrs.get(i, empty), self._nested.get(i)
            if i in self._passes:
                more = {}
                uses = _param_uses(self, i, type(x), uses, owners, (), more)
                nested = _merge_nested(nested, more)
            node = [set(uses[0]), set(uses[1]), set(uses[2]), {}]
            _node_merge(node, [(), (), (), nested or {}])
            name = (self._ctree or self._tree).args.args[i].arg if i < len((self._ctree or self._tree).args.args) else None
            if name is not None and name in _facts(self._ctree or self._tree).stores:
                recipe = _init_recipe(type(x))   # given a new instance: it starts as a copy of all __init__ assigns
                if type(recipe) is dict:
                    node[0] |= {a for a, _ in recipe['assigns']}
            built = _construction_uses(self).get(i)
            if built is not None:                # given to C(...): what's done through the attribute it becomes
                _node_merge(node, built)
            roots.append((i, x, node))
        if roots:
            found, ids = _object_graph(roots, refs)
            for i, t in found.items():
                out[i] = t
            seen |= ids
        for i, x in enumerate(args):
            if type(x) is dict:
                out[i] = None if id(x) in seen else self._dict_type(x, self._dict_keys.get(i))
                seen.add(id(x))          # two names for one dict: a write must show through both
            elif out[i] is None and type(x) is list and (i in self._elems or i in self._passes) and x:
                uses = self._elems.get(i, empty)
                tree = self._ctree or self._tree
                if getattr(self, '_enested_for', None) is not tree:
                    self._enested, self._enested_for = _element_nested(tree), tree
                nested = self._enested.get(i)
                if i in self._passes:
                    more = {}
                    uses = _param_uses(self, i, type(x[0]), uses, owners, (), more)
                    nested = _merge_nested(nested, more)
                for q, (k, _) in joined.items():     # an argument that is one of its elements: what's done through it
                    if k != i:
                        continue
                    qu, qn = self._attrs.get(q, empty), self._nested.get(q)
                    if q in self._passes:
                        more = {}
                        qu = _param_uses(self, q, type(x[0]), qu, owners, (), more)
                        qn = _merge_nested(qn, more)
                    uses = (set(uses[0]) | set(qu[0]), set(uses[1]) | set(qu[1]), set(uses[2]) | set(qu[2]))
                    nested = _merge_nested(nested, qn)
                out[i] = _objlist_type(x, (sorted(uses[0]),) + tuple(uses[1:]), seen, dict in map(type, args), nested)
        for q, (k, _) in joined.items():
            out[q] = T.inlist_of(k) if out[k] is not None and out[k].kind == 'objlist' else None
        return tuple(out)

    @staticmethod
    def _dict_type(d, keys):
        if not keys:
            return None
        fields = []
        for k in keys:
            if k not in d:
                return None              # Python raises KeyError, at the right moment
            t = _arg_type(d[k])
            if t is None or t is T.U64 or t.kind == 'list' or T.is_pylist(t):     # a list in a dict: no copy back yet
                return None
            fields.append((k, t))
        return T.dict_of(fields)

    def _run(self, entry, args, room=None, prebuilt=None):
        m = _marshal(entry, args, room, prebuilt)
        if m is None:                            # not as the signature said: can only be a class changed since
            return self._py(*self._real(args))
        io = bytearray(entry.io_size)
        status = entry.kernel(*m.cargs, io)
        if status == POOL_FULL:                  # nothing has been written back: the same call, with twice the room
            return self._run(entry, args, 2 * m.room)
        if m.made_k:
            self._made_into_pool(entry, io, args, m.pool_t, m.by_cls, m.columns, m.refcols, m.made_k)
        _write_back(entry, m, args)
        by_cls = m.by_cls
        if status == RERUN and entry.rerun:
            return self._py(*self._real(args))
        for i, key, at, ot in entry.outs:
            if isinstance(key, str) and key.endswith('.#count'):
                continue                         # how many a pool holds: taken by _made_into_pool
            v = self._box(io, at, at + 8, ot)
            ft = _field_type(entry.types[i], key)
            if ft.kind == 'ref':                 # an index in its class's pool: the object, or None
                v = None if v < 0 else by_cls[ft.cls][v]
            _set_member(args[i], key, v)
        if status:
            self._failed(entry, status, io)
        r = entry.ret
        if entry.instance_ret and entry.instance_ret[0] == 'made':     # a new instance, filled as its __init__ fills it
            _, cls, attrs = entry.instance_ret
            values = iter(self._box_tuple(io, r) if not entry.records else self._array_result(entry, io, args))

            def build(cls, attrs):
                obj = cls.__new__(cls)
                for a in attrs:
                    if type(a) is tuple:         # an object it holds: made too, or an argument's, as it is
                        a, held = a
                        v = build(held[1], held[2]) if held[0] == 'made' else _member(args[held[1]], held[2]) if held[2] else args[held[1]]
                    else:
                        v = next(values)
                    object.__setattr__(obj, a, v)
                return obj
            return build(cls, attrs)
        if r is T.NONE:
            return None
        if r.kind in ('glist', 'record', 'gset'):
            return self._glist_result(entry, io, args, m.gl)
        if r.kind == 'tuple' and any(t.kind in ('glist', 'record', 'gset') for t in r.elem):
            return self._glist_tuple(entry, io, r, args, m.gl)
        if entry.records:
            return self._array_result(entry, io, args)
        if r.kind == 'tuple':
            return self._box_tuple(io, r)
        if r.kind == 'list':
            return self._list_result(entry, io, args)
        if entry.instance_ret and entry.instance_ret[0] == 'pool':     # a reference: the object at that index, or None
            k = self._box(io, 0, 16, r)
            return None if k < 0 else by_cls[entry.instance_ret[1]][k]
        if entry.instance_ret:                  # the instance itself: an argument, an element of one, or an object in one
            k = self._box(io, 0, 16, r)
            if entry.instance_ret[0] == 'path':
                return _member(args[entry.instance_ret[1]], entry.instance_ret[2])
            return args[k] if entry.instance_ret[0] == 'arg' else args[entry.instance_ret[1]][k]
        return self._box(io, 0, 16, r)

    def _made_into_pool(self, entry, io, args, pool_t, by_cls, columns, refcols, made_k):
        """The instances the kernel made and kept in the pool: made as __init__ would, after the existing ones, so references
        to them find them; the existing ones are written back by the caller."""
        counts = {}
        for i, key, at, ot in entry.outs:
            if isinstance(key, str) and key.endswith('.#count'):
                counts[int(key.split('.')[0])] = self._box(io, at, at + 8, ot)
        new = {}
        for k, order in made_k.items():
            cls = pool_t.elem[0][k][0]
            objs = by_cls.setdefault(cls, [])
            start = len(objs)
            new[k] = (start, [cls.__new__(cls) for _ in range(counts.get(k, start) - start)])
        for k, (start, made) in new.items():   # every one exists first: a reference may be to any of them
            by_cls[pool_t.elem[0][k][0]].extend(made)
        for k, (start, made) in new.items():
            cls = pool_t.elem[0][k][0]
            for i, slot, a, ft, held, last in columns:
                if slot.startswith(f'{k}.'):
                    for n, o in enumerate(made):
                        v = a[start + n]
                        object.__setattr__(o, last, bool(v) if ft is T.BOOL else v)
            for i, slot, a, held, last, targets in refcols:
                if slot.startswith(f'{k}.'):
                    for n, o in enumerate(made):
                        j = a[start + n]
                        object.__setattr__(o, last, None if j < 0 else targets[j])
            order = made_k[k]                    # the attributes in the order __init__ assigns them, as a new one has them
            for o in made:
                d = getattr(o, '__dict__', None)
                if d is not None:
                    o.__dict__ = {a: d[a] for a in order if a in d}
        for idx, (i, slot, a, ft, held, last) in enumerate(columns):     # the existing ones: written back as usual
            k = int(slot.split('.')[0])
            if k in new:
                columns[idx] = (i, slot, a, ft, held[:new[k][0]], last)
        for idx, (i, slot, a, held, last, targets) in enumerate(refcols):
            k = int(slot.split('.')[0])
            if k in new:
                refcols[idx] = (i, slot, a, held[:new[k][0]], last, targets)

    def _glist_key(self, args, key):
        """Lists held by reference (_glist): those the other kinds can't hold (ragged, deeper, of tuples or arrays,
        empty), those this function grows or gives another name, and lists shared between arguments."""
        self._held()
        ids = collections.Counter()
        for x, t in zip(args, key):
            if type(x) is list:
                if t is not None and t.kind == 'list':       # numbers only (_arg_type looked): nothing inside to count
                    ids[id(x)] += 1
                else:
                    _count_lists(x, ids)
        shared = {k for k, c in ids.items() if c > 1}
        out = list(key)
        for i, x in enumerate(args):
            if type(x) is not list:
                continue
            if out[i] is None or i in self._gl_held or (shared and _reaches(x, shared)):
                if i not in self._gl_held and x and _gl_is_object(x[0]) and not (shared and _reaches(x, shared)):
                    continue                     # a list of instances: the object path (no need to look at them all)
                t = _glist_arg_type(x)
                if t is not None and (i in self._gl_held or not _has_instances(t)):   # instances: the object path, unless grown
                    out[i] = t
        return tuple(out)

    def _glist_result(self, entry, io, args, gl=None):
        """A returned list held by reference (_glist): read into Python lists, one list object per header so a list in two
        places is one list in two places (and a list passed in is itself), then every block it used released."""
        blocks = []
        gl = gl or _GLBuild()
        try:
            return _read_held(struct.unpack_from('<q', io, 0)[0], entry.ret, getattr(gl, 'memo', {}), blocks, _array_reader_for(entry, args, gl))
        finally:
            gl.release(entry, blocks)

    def _glist_tuple(self, entry, io, r, args, gl=None):
        """A returned tuple holding lists: each read as _glist_result reads one, sharing kept across them."""
        gl = gl or _GLBuild()
        blocks, memo, out = [], getattr(gl, 'memo', {}), []
        reader = _array_reader_for(entry, args, gl)
        try:
            for i, et in enumerate(r.elem):
                if et.kind in ('glist', 'record', 'gset'):
                    out.append(_read_held(struct.unpack_from('<q', io, 32 + 16 * i)[0], et, memo, blocks, reader))
                else:
                    out.append(self._box(io, 32 + 16 * i, 40 + 16 * i, et))
            return tuple(out)
        finally:
            gl.release(entry, blocks)

    def _array_reader(self, entry, args):
        """For arrays in a returned list: record values -> the numpy array, as _arrays makes one (a copy of what the kernel
        made, once per block; the argument itself or a view of it)."""
        made = {}

        def read(value, et, blocks):
            numpy = self._np or _numpy()
            address, which, same, block, data, ndim = value[:6]
            typestr = numpy.dtype(et.elem.name[3:]).str
            if block:
                blocks.append(block)
                root = made.get(block)
                if root is None:
                    root = made[block] = numpy.array(_Interface(None, data, value[6:6 + ndim], None, typestr))
                if not same:
                    address += root.__array_interface__['data'][0] - data
            else:
                i, key = entry.array_args[which]
                root = args[i] if key is None else _member(args[i], key)
            if same:
                return root
            return numpy.asarray(_Interface(root, address, value[10::2][:et.ndim], [x * root.itemsize for x in value[11::2][:et.ndim]]))
        return read

    def _list_result(self, entry, io, args):
        """A returned list: the argument itself, or a new list of what the kernel built, whose memory is then released."""
        q = struct.unpack_from('<6q', io, 0)
        if q[5]:
            return args[q[5] - 1]
        address, n = q[0], q[4]
        et = entry.ret.elem
        try:
            if not n:
                return []
            import ctypes
            code = _LIST_CODES[et.name]
            values = memoryview((ctypes.c_char * (n * struct.calcsize(code))).from_address(address)).cast('B').cast(code).tolist()
            return [bool(v) for v in values] if et is T.BOOL else values
        finally:
            if address:
                entry.free(address)

    def _box_tuple(self, io, r, at=None):
        """A returned tuple of numbers, nested ones too: each number has its pair of slots, in order."""
        at = at if at is not None else [0]
        out = []
        for et in r.elem:
            if et.kind == 'tuple':
                out.append(self._box_tuple(io, et, at))
            else:
                out.append(self._box(io, 32 + 16 * at[0], 40 + 16 * at[0], et))
                at[0] += 1
        return tuple(out)

    def _array_result(self, entry, io, args):
        arrays = self._arrays(entry, io, args)
        r = entry.ret
        if r.kind == 'array':
            return arrays[0]
        arrays = iter(arrays)
        return tuple([next(arrays) if et.kind == 'array' else self._box(io, 32 + 16 * i, 40 + 16 * i, et) for i, et in enumerate(r.elem)])

    def _arrays(self, entry, io, args):
        """Returned arrays. One made in the kernel is copied into a numpy array of its own, once however many results
        are in it, and its block freed; so it's what numpy would have made. The argument itself, or the array made, if
        it's that; else a real numpy view of its memory."""
        out, made = [], None
        numpy = self._np
        try:
            for at, layout, typestr in entry.records:
                value = layout.unpack_from(io, at)
                address, which, same, block, data, ndim = value[:6]
                if isinstance(typestr, T.Type):          # a list of lists: the one passed in, or new lists of what was made
                    if block:
                        out.append(_lists_from(address, value[10::2][:typestr.ndim], typestr.elem))
                    else:
                        i, key = entry.array_args[which]
                        out.append(args[i] if key is None else _member(args[i], key))
                    continue
                if block:
                    if made is None:
                        made = {}
                    root = made.get(block)
                    if root is None:
                        root = made[block] = numpy.array(_Interface(None, data, value[6:6 + ndim], None, typestr))
                    if not same:
                        address += root.__array_interface__['data'][0] - data
                else:
                    i, key = entry.array_args[which]
                    root = args[i] if key is None else _member(args[i], key)
                if same:
                    out.append(root)
                else:
                    size = root.itemsize
                    out.append(numpy.asarray(_Interface(root, address, value[10::2], [s * size for s in value[11::2]])))
        finally:
            if entry.free is not None:
                blocks = {layout.unpack_from(io, at)[3] for at, layout, _ in entry.records}
                for block in blocks:
                    if block:
                        entry.free(block)
        return out

    def _box(self, io, at, flag_at, r):
        if r.kind == 'enum':                      # its value, back as the member
            return r.elem(int.from_bytes(io[at:at + 8], 'little', signed=True))
        if r.np == 'dyn' and r.kind == 'float' and io[flag_at] == RETURNS_INT:
            return int(struct.unpack_from('<d', io, at)[0])
        np = r.np if r.np != 'dyn' else io[flag_at] != 0
        if np and self._np is None:
            import numpy
            self._np = numpy
        if r.kind == 'float':
            v = struct.unpack_from('<d', io, at)[0]
            if not np and r.py == 'int':           # still the int it started as
                return int(v)
        else:
            v = int.from_bytes(io[at:at + 8], 'little', signed=r is not T.U64 and r is not T.NP_UINT64)
        if np:
            name = T.as_narrow(r).name[3:] if T.is_narrow_dyn(r) or r.np is True else {'int': 'int64', 'float': 'float64', 'bool': 'bool_'}[r.kind]
            return getattr(self._np, 'bool_' if name == 'bool' else name)(v)
        return bool(v) if r.kind == 'bool' else v


_GL_CTYPES = None
SJ_BLOCK = 72          # sizeof(sj_block) (_mathlib's np_make), native and on wasm32
_GL_CODES = {'int64_t': 'q', 'double': 'd', 'int': 'i', 'float': 'f', 'int32_t': 'i', 'int16_t': 'h', 'int8_t': 'b',
             'uint64_t': 'Q', 'uint32_t': 'I', 'uint16_t': 'H', 'uint8_t': 'B'}


def _gl_leaf(t):
    """(ctypes type, Python value of a C value) for a number a list holds."""
    global _GL_CTYPES
    import ctypes
    if _GL_CTYPES is None:
        _GL_CTYPES = {'int64_t': ctypes.c_int64, 'double': ctypes.c_double, 'int': ctypes.c_int, 'float': ctypes.c_float,
                      'int32_t': ctypes.c_int32, 'int16_t': ctypes.c_int16, 'int8_t': ctypes.c_int8, 'uint64_t': ctypes.c_uint64,
                      'uint32_t': ctypes.c_uint32, 'uint16_t': ctypes.c_uint16, 'uint8_t': ctypes.c_uint8}
    ct = _GL_CTYPES[t.ctype]
    if t.np is True:
        import numpy
        make = getattr(numpy, t.name[3:])
    else:
        make = {'int': int, 'float': float, 'bool': bool}[t.kind]
    return ct, make


def _gl_struct(t):
    if ('t', t) not in _GL_STRUCTS:
        _GL_STRUCTS['t', t] = _gl_struct_new(t)
    return _GL_STRUCTS['t', t]


def _gl_struct_new(t):
    import ctypes
    fields = []

    def walk(tt):
        for x in tt.elem:
            walk(x) if x.kind == 'tuple' else fields.append(x)
    walk(t)
    cls = type('sjT', (ctypes.Structure,), {'_fields_': [(f'f{k}', _gl_leaf(x)[0]) for k, x in enumerate(fields)]})
    return cls, fields


def _numpy():
    import numpy
    return numpy


def _read_held(addr, t, memo, blocks, arrays=None):
    """A list or a dict the kernel holds by reference, as a Python object (None for a null pointer)."""
    if not addr:
        return None
    if t.kind == 'gset':
        return _read_set(addr, t, memo, blocks)
    return _read_glist(addr, t, memo, blocks, arrays) if t.kind == 'glist' else _read_record(addr, t, memo, blocks, arrays)


def _gs_entry(et):
    """ctypes for a set's table entry {hash, state, key} (_gset)."""
    if ('s', et) not in _GL_STRUCTS:
        import ctypes
        k = _gl_struct(et)[0] if et.kind == 'tuple' else _gl_leaf(et)[0]
        _GL_STRUCTS['s', et] = type('sjE', (ctypes.Structure,), {'_fields_': [('h', ctypes.c_int64), ('st', ctypes.c_int64), ('k', k)]})
    return _GL_STRUCTS['s', et]


def _gs_key_value(et):
    """A table entry's key (as ctypes gives it) -> the Python value."""
    if et.kind == 'tuple':
        return lambda v: _gl_value(v, et, None, None, None)
    return _gl_leaf(et)[1]


def _read_set_table(addr, t):
    """(fill, used, mask, finger, slots, table block) of the set whose header is at addr; slots as _pysets takes them,
    with Python values for keys."""
    import ctypes
    from . import _pysets
    raw = ctypes.string_at(addr, 48)                 # sj_gset: pointers are 8 bytes, natively and in cimo's wasm
    table = int.from_bytes(raw[:8], 'little')
    fill, used, mask, finger = struct.unpack_from('<qqqq', raw, 8)
    db = int.from_bytes(raw[40:48], 'little')
    slots = []
    if t.elem is T.PENDING:                          # never given an element: empty, as made
        slots = [(None, 0)] * (mask + 1)
    else:
        value = _gs_key_value(t.elem)
        for e in (_gs_entry(t.elem) * (mask + 1)).from_address(table):
            slots.append((None, 0) if e.st == 0 else (_pysets.DUMMY, -1) if e.st == 2 else (value(e.k), e.h))
    return fill, used, mask, finger, slots, db


def _read_set(addr, t, memo, blocks):
    """The Python set for the header at addr: its table as the kernel left it, slot for slot (_pysets)."""
    from . import _pysets
    if addr in memo:
        return memo[addr]
    fill, used, mask, finger, slots, db = _read_set_table(addr, t)
    blocks.append(addr - SJ_BLOCK)
    if db:
        blocks.append(db)
    memo[addr] = out = _pysets.make(fill, used, mask, finger, slots)
    return out


_GL_STRUCTS = {}


def _rc_struct(t):
    """ctypes for a dict's C struct (_glist's rc_struct): a field per key. One class per type."""
    if ('r', t) in _GL_STRUCTS:
        return _GL_STRUCTS['r', t]
    _GL_STRUCTS['r', t] = cls = _rc_struct_new(t)
    return cls


def _rc_struct_new(t):
    import ctypes
    fields = []
    for k, (_, x) in enumerate(t.elem):
        if x.kind in ('glist', 'record', 'gset'):
            ct = ctypes.c_uint64          # a C pointer: 8 bytes, natively and in cimo's wasm
        elif x.kind == 'array':
            ct = ctypes.c_int64 * (10 + 2 * x.ndim)
        elif x.kind == 'tuple':
            ct = _gl_struct(x)[0]
        elif x is T.PENDING:
            ct = ctypes.c_int64
        else:
            ct = _gl_leaf(x)[0]
        fields.append((f'f{k}', ct))
    return type('sjR', (ctypes.Structure,), {'_fields_': fields})


def _gl_value(v, x, memo, blocks, arrays):
    """A C field or element of type x (as ctypes gives it) as a Python value."""
    if x.kind in ('glist', 'record', 'gset'):
        return _read_held(v, x, memo, blocks, arrays)
    if x.kind == 'array':
        value = ('array',) + tuple(v)
        if value not in memo:
            memo[value] = arrays(value[1:], x, blocks)
        return memo[value]
    if x.kind == 'tuple':
        _, fields = _gl_struct(x)
        vals = iter([_gl_leaf(f)[1](getattr(v, f'f{k}')) for k, f in enumerate(fields)])

        def build(tt):
            return tuple(build(y) if y.kind == 'tuple' else next(vals) for y in tt.elem)
        return build(x)
    return _gl_leaf(x)[1](v)


def _read_record(addr, t, memo, blocks, arrays=None):
    """The Python dict for the record at addr: one dict per record, so a dict in two places is one dict."""
    if addr in memo:
        return memo[addr]
    out = {} if t.cls is None else t.cls.__new__(t.cls)      # an instance: made as its class makes them, then filled
    memo[addr] = out
    blocks.append(addr - SJ_BLOCK)
    rec = _rc_struct(t).from_address(addr)
    for k, (key, x) in enumerate(t.elem):
        v = _gl_value(getattr(rec, f'f{k}'), x, memo, blocks, arrays)
        if t.cls is None:
            out[key] = v
        else:
            object.__setattr__(out, key, v)
    return out


def _read_glist(addr, t, memo, blocks, arrays=None):
    """The Python list for the header at addr, of list type t; the blocks it used go in blocks. arrays: reads an array
    element's record (_array_reader). One list object per header: memo."""
    if addr in memo:
        return memo[addr]
    out = []
    memo[addr] = out
    out.extend(_read_glist_values(addr, t, memo, blocks, arrays))
    return out


def _read_glist_values(addr, t, memo, blocks, arrays=None):
    """The elements of the list whose header is at addr, as Python values."""
    import ctypes
    raw = ctypes.string_at(addr, 32)                 # sj_glist: pointers are 8 bytes, natively and in cimo's wasm
    p = int.from_bytes(raw[:8], 'little')
    n, _cap = struct.unpack_from('<qq', raw, 8)
    db = int.from_bytes(raw[24:32], 'little')
    blocks.append(addr - SJ_BLOCK)                    # its header's block: sizeof(sj_block) before it
    if db:
        blocks.append(db)
    out = []
    et = t.elem
    if not n:
        return out
    if et.kind in ('glist', 'record', 'gset'):
        ptrs = (ctypes.c_uint64 * n).from_address(p)
        out.extend(_read_held(x, et, memo, blocks, arrays) for x in ptrs)
    elif et.kind == 'array':
        width = 10 + 2 * et.ndim
        recs = (ctypes.c_int64 * (width * n)).from_address(p)
        for k in range(n):
            value = ('array',) + tuple(recs[k * width:(k + 1) * width])
            if value not in memo:                     # the same array put in twice: one object, as in Python
                memo[value] = arrays(value[1:], et, blocks)
            out.append(memo[value])
    elif et.kind == 'tuple':                         # struct.iter_unpack reads the C struct, padding and all, in C
        cls, fields = _gl_struct(et)
        size = ctypes.sizeof(cls)
        fmt = '@' + ''.join(_GL_CODES[x.ctype] for x in fields)
        fmt += 'x' * (size - struct.calcsize(fmt))
        rows = struct.iter_unpack(fmt, (ctypes.c_char * (size * n)).from_address(p))
        plain = all(x.np is False and x.kind != 'bool' for x in fields) and len(fields) == len(et.elem)
        if plain:
            out.extend(rows)
        else:
            makes = [_gl_leaf(x)[1] for x in fields]
            for row in rows:
                vals = iter([make(v) for make, v in zip(makes, row)])

                def build(tt):
                    return tuple(build(x) if x.kind == 'tuple' else next(vals) for x in tt.elem)
                out.append(build(et))
    elif et is T.PENDING:
        pass
    else:                                            # numbers: one cast and tolist, in C
        ct, make = _gl_leaf(et)
        vals = memoryview((ctypes.c_char * (ctypes.sizeof(ct) * n)).from_address(p)).cast('B').cast(_GL_CODES[et.ctype]).tolist()
        out.extend(map(make, vals) if (et.np is True or et.kind == 'bool') else vals)
    return out


def _gl_is_object(v):
    """An object of a class of the user's (what a record with a class holds)."""
    t = type(v)
    return t not in (int, float, bool, tuple, list, dict, str, bytes) and hasattr(v, '__dict__') and t.__module__ != 'builtins' \
        and not hasattr(v, '__array_interface__') and not callable(v) and not isinstance(v, enum.Enum)


def _has_instances(t):
    """Does a list or dict type hold instances (records with a class), at any depth?"""
    if t.kind == 'record':
        return t.cls is not None or any(_has_instances(x) for _, x in t.elem)
    if t.kind == 'glist':
        return _has_instances(t.elem)
    return False


def _count_lists(x, ids, depth=0):
    """How often each list object is reached, through lists, from the arguments."""
    ids[id(x)] += 1
    if ids[id(x)] == 1 and depth < 64:
        for v in x:
            if type(v) is list:
                _count_lists(v, ids, depth + 1)


def _reaches(x, targets, depth=0):
    if id(x) in targets:
        return True
    return depth < 64 and any(type(v) is list and _reaches(v, targets, depth + 1) for v in x)


def _glist_arg_type(x, depth=0):
    """The type of a list passed in, held by reference: list[T] with T what every element is, or None."""
    from ._glist import unify
    if depth > 32:
        return None
    kinds = set(map(type, x))                         # numbers of one kind: looked at in C, not one by one
    if len(kinds) == 1 and next(iter(kinds)) in (float, bool) or kinds == {int} and -(1 << 63) <= min(x) and max(x) < 1 << 63:
        return T.glist_of({int: T.INT, float: T.FLOAT, bool: T.BOOL}[type(x[0])])
    et = T.PENDING
    for v in x:
        vt = _glist_elem_type(v, depth)
        if vt is None:
            return None
        u = unify(et, vt)
        if u is None:
            return None
        et = u
    return T.glist_of(et)


def _set_elem_type(v):
    """The type of a set's element (a number or a tuple of numbers), or None. A NaN: None (its hash is its address)."""
    t = type(v)
    if t is int:
        return T.INT if -(1 << 63) <= v < (1 << 63) else None
    if t is float:
        return T.FLOAT if v == v else None
    if t is bool:
        return T.BOOL
    if t is tuple:
        es = [_set_elem_type(y) for y in v]
        return None if any(e is None for e in es) else T.tuple_of(es)
    return None


def _set_arg_type(x):
    """set[T], T what every element is, or None."""
    et = T.PENDING
    for v in x:
        vt = _set_elem_type(v)
        if vt is None or (et is not T.PENDING and vt is not et):
            return None
        et = vt
    return T.gset_of(et)


def _glist_elem_type(v, depth):
    t = type(v)
    if t is set:
        return _set_arg_type(v)
    if t is list:
        return _glist_arg_type(v, depth + 1)
    if t is not dict and t not in (int, float, bool, tuple, str, bytes) and not hasattr(v, '__array_interface__') \
            and not isinstance(v, enum.Enum) and hasattr(v, '__dict__') and type(v).__module__ != 'builtins' and not callable(v):
        attrs = vars(v)                               # an instance: its attributes, as a record with its class
        if depth > 32 or not attrs:
            return None
        fields = []
        for k, x in attrs.items():
            xt = _glist_elem_type(x, depth + 1)
            if xt is None:
                return None
            fields.append((k, xt))
        return T.record_of(fields, t)
    if t is dict:                                     # a record: string keys, each value one of these
        if depth > 32 or not v or not all(type(k) is str for k in v):
            return None
        fields = []
        for k, x in v.items():
            xt = _glist_elem_type(x, depth + 1)
            if xt is None:
                return None
            fields.append((k, xt))
        return T.record_of(fields)
    a = _arg_type(v)
    if a is None or a is T.U64 or a.np == 'dyn':
        return None
    if a.is_number or (a.kind == 'tuple' and all(x.is_number and x is not T.U64 for x in T.leaves(a))):
        return a
    if a.kind == 'array' and not T.is_pylist(a):
        return a
    return None


class _GLBuild:
    """Lists passed in, built as _glist's headers in memory Python owns (each with room for an sj_block before it, which
    the kernel's marking writes), and what they became, written back."""

    def __init__(self):
        import ctypes
        self.ctypes = ctypes
        self.keep = []            # the buffers, alive for the call
        self.headers = {}         # id(list) -> its header's address
        self.objs = {}            # header address -> the list
        self.types = {}           # header address -> its type
        self.own = set()          # block addresses Python owns: never freed through the kernel
        self.arrays = {}          # an array's record values -> the array
        self.done = set()         # blocks already released
        self.top = []             # (argument index, header) of each list passed in
        self.snap = {}            # header address -> the bytes of its numbers as passed in (lists of ints or floats)

    def buffer(self, size):
        b = (self.ctypes.c_char * (SJ_BLOCK + max(size, 8)))()
        self.keep.append(b)
        a = self.ctypes.addressof(b)
        self.own.add(a)
        return a + SJ_BLOCK

    def build_record(self, x, t):
        """A dict passed in, as its record: a struct of its values in key order."""
        if id(x) in self.headers:
            return self.headers[id(x)]
        cls = _rc_struct(t)
        h = self.buffer(self.ctypes.sizeof(cls))
        self.headers[id(x)] = h
        self.objs[h] = x
        self.types[h] = t
        rec = cls.from_address(h)
        for k, (key, ft) in enumerate(t.elem):
            setattr(rec, f'f{k}', self.field(x[key] if t.cls is None else getattr(x, key), ft))
        return h

    def field(self, v, ft):
        """What a value of type ft is in C, for a struct's field."""
        if ft.kind == 'glist':
            return self.build(v, ft)
        if ft.kind == 'record':
            return self.build_record(v, ft)
        if ft.kind == 'gset':
            return self.build_set(v, ft)
        if ft.kind == 'array':
            rec = [v.__array_interface__['data'][0], -1, 0, 0, 0, 0, 0, 0, 0, 0]
            for k in range(ft.ndim):
                rec += [v.shape[k], v.strides[k] // v.itemsize]
            self.arrays[tuple(rec)] = v
            return (self.ctypes.c_int64 * len(rec))(*rec)
        if ft.kind == 'tuple':
            cls, _ = _gl_struct(ft)
            flat = []

            def walk(y):
                for z in y:
                    walk(z) if type(z) is tuple else flat.append(z)
            walk(v)
            return cls(*flat)
        if ft is T.PENDING:
            return 0
        return v.item() if ft.np is True else v

    def build_set(self, x, t):
        """A set passed in: its header and table, slot for slot as CPython has them (_pysets), in memory Python owns."""
        from . import _pysets
        if id(x) in self.headers:
            return self.headers[id(x)]
        ctypes = self.ctypes
        table = _pysets.read(x) or _pysets.layout(x)
        fill, used, mask, finger, slots = table
        et = t.elem if t.elem is not T.PENDING else T.INT
        E = _gs_entry(et)
        h = self.buffer(48)
        d = self.buffer(ctypes.sizeof(E) * (mask + 1))
        ents = (E * (mask + 1)).from_address(d)
        flat_tuple = et.kind == 'tuple'
        for i, (o, hh) in enumerate(slots):
            if o is None:
                continue
            if o is _pysets.DUMMY:
                ents[i].st, ents[i].h = 2, -1
                continue
            ents[i].st, ents[i].h = 1, hh
            ents[i].k = self.field(o, et) if flat_tuple else o
        ctypes.memmove(h, struct.pack('<Qqqqq', d, fill, used, mask, finger) + struct.pack('<Q', d - SJ_BLOCK), 48)
        self.headers[id(x)] = h
        self.objs[h] = x
        self.types[h] = t
        self.snap[h] = ctypes.string_at(h, 48) + ctypes.string_at(d, ctypes.sizeof(E) * (mask + 1))
        return h

    def build(self, x, t):
        if id(x) in self.headers:
            return self.headers[id(x)]
        ctypes = self.ctypes
        h = self.buffer(32)
        self.headers[id(x)] = h
        self.objs[h] = x
        self.types[h] = t
        et = t.elem
        n = len(x)
        if et.kind in ('glist', 'record', 'gset'):
            esize = 8
        elif et.kind == 'array':
            esize = 8 * (10 + 2 * et.ndim)
        elif et.kind == 'tuple':
            esize = ctypes.sizeof(_gl_struct(et)[0])
        elif et is T.PENDING:
            esize = 8
        else:
            esize = ctypes.sizeof(_gl_leaf(et)[0])
        d = self.buffer(esize * n)
        if n:
            if et.kind == 'glist':
                (ctypes.c_uint64 * n).from_address(d)[:] = [self.build(v, et) for v in x]
            elif et.kind == 'record':
                (ctypes.c_uint64 * n).from_address(d)[:] = [self.build_record(v, et) for v in x]
            elif et.kind == 'gset':
                (ctypes.c_uint64 * n).from_address(d)[:] = [self.build_set(v, et) for v in x]
            elif et.kind == 'array':
                recs = (ctypes.c_int64 * ((10 + 2 * et.ndim) * n)).from_address(d)
                at = 0
                for v in x:
                    info = v.__array_interface__
                    address = info['data'][0]
                    strides = v.strides
                    rec = [address, -1, 0, 0, 0, 0, 0, 0, 0, 0]
                    for k in range(et.ndim):
                        rec += [v.shape[k], strides[k] // v.itemsize]
                    recs[at:at + len(rec)] = rec
                    at += len(rec)
                    self.arrays[tuple(rec)] = v
            elif et.kind == 'tuple':
                cls, fields = _gl_struct(et)
                arr = (cls * n).from_address(d)
                for k, v in enumerate(x):
                    flat = []

                    def walk(y):
                        for z in y:
                            walk(z) if type(z) is tuple else flat.append(z)
                    walk(v)
                    arr[k] = cls(*flat)
            elif et in (T.INT, T.FLOAT):          # one copy in C, and its bytes kept: what changed is found by comparing
                a = array.array('q' if et is T.INT else 'd', x)
                ctypes.memmove(d, a.buffer_info()[0], 8 * n)
                self.snap[h] = a.tobytes()
            elif et is not T.PENDING:
                ct, _ = _gl_leaf(et)
                vals = [v.item() for v in x] if et.np is True else x
                (ct * n).from_address(d)[:] = vals
        struct.pack_into('<qqqq', (ctypes.c_char * 32).from_address(h), 0, d, n, n, d - SJ_BLOCK)
        return h

    def write_back(self, entry, args):
        """Each list passed in, as the kernel left it: elements that didn't change stay the very objects they were."""
        reader = None
        memo = dict(self.objs)
        blocks = []
        news = {}
        refined = getattr(entry, 'gl_types', None) or {}

        def retype(h, t, seen):                  # what appends taught the kernel about lists that came empty
            if h in seen or h not in self.objs:
                return
            seen.add(h)
            self.types[h] = t
            if t.kind == 'record':
                rec = _rc_struct(t).from_address(h)
                for k, (_, ft) in enumerate(t.elem):
                    if ft.kind in ('glist', 'record'):
                        retype(getattr(rec, f'f{k}'), ft, seen)
                return
            if t.elem.kind in ('glist', 'record'):
                n = struct.unpack_from('<q', (self.ctypes.c_char * 16).from_address(h), 8)[0]
                p = struct.unpack_from('<q', (self.ctypes.c_char * 8).from_address(h), 0)[0]
                if n and p:
                    for x in (self.ctypes.c_uint64 * n).from_address(p):
                        retype(x, t.elem, seen)
        seen = set()
        for i, h in self.top:
            if i in refined:
                retype(h, refined[i], seen)
        now = {}
        for h in self.snap:                      # numbers: their bytes now, and nothing more read if they're the same
            if self.types[h].kind == 'gset':
                now[h] = self._set_bytes(h, self.types[h])
                continue
            p, n = struct.unpack_from('<qq', (self.ctypes.c_char * 16).from_address(h), 0)
            now[h] = self.ctypes.string_at(p, 8 * n) if n else b''
        for h, x in self.objs.items():
            if now.get(h) == self.snap.get(h, 0):
                continue
            if reader is None:
                reader = _array_reader_for(entry, args, self)
            t = self.types[h]
            if t.kind == 'gset':
                news[h] = _read_set_table(h, t)
                if news[h][5] and news[h][5] + SJ_BLOCK not in self.own:
                    blocks.append(news[h][5])
                continue
            if t.kind == 'record':
                rec = _rc_struct(t).from_address(h)
                news[h] = {key: _gl_value(getattr(rec, f'f{k}'), ft, memo, blocks, reader) for k, (key, ft) in enumerate(t.elem)}
            else:
                news[h] = _read_glist_values(h, t, memo, blocks, reader)
        for h, x in self.objs.items():
            if h not in news:
                continue
            new = news[h]
            if self.types[h].kind == 'gset':     # its table as the kernel left it; elements it had, the very objects
                from . import _pysets
                fill, used, mask, finger, slots, _ = new
                had = {_gs_identity(o): o for o in x}
                slots = [(had.get(_gs_identity(o), o), hh) if o is not None and o is not _pysets.DUMMY else (o, hh) for o, hh in slots]
                _pysets.replace(x, fill, used, mask, finger, slots)
                continue
            if type(x) is dict:                  # its keys stay; the values that changed are assigned, in place
                for key, v in new.items():
                    if not _same_value(x[key], v):
                        x[key] = v
                continue
            if self.types[h].cls is not None:    # an instance: the attributes that changed, set on it
                for key, v in new.items():
                    if not _same_value(getattr(x, key), v):
                        setattr(x, key, v)
                continue
            if self.types[h].elem in (T.INT, T.FLOAT, T.BOOL):
                _write_numbers(x, new, self.types[h].elem, self.snap.get(h), now.get(h))
                continue
            if len(new) == len(x) and all(_same_value(a, b) for a, b in zip(x, new)):
                continue
            x[:] = [x[k] if k < len(x) and _same_value(x[k], a) else a for k, a in enumerate(new)]
        self.release(entry, blocks)
        self.memo = memo

    def _set_bytes(self, h, t):
        table = int.from_bytes(self.ctypes.string_at(h, 8), 'little')
        mask = struct.unpack_from('<q', self.ctypes.string_at(h + 24, 8))[0]
        et = t.elem if t.elem is not T.PENDING else T.INT
        return self.ctypes.string_at(h, 48) + self.ctypes.string_at(table, self.ctypes.sizeof(_gs_entry(et)) * (mask + 1))

    def release(self, entry, blocks):
        for b in blocks:
            if b not in self.own and b not in self.done:
                self.done.add(b)
                entry.free(b)


def _gs_identity(v):
    """What tells a set's element from another of the same type: a float by its bits (-0.0 isn't 0.0 here)."""
    return (type(v), struct.pack('<d', v)) if type(v) is float else (type(v), v)


def _same_value(a, b):
    """Is b what a was: the same object, or the same number of the same type (bit for bit)?"""
    if a is b:
        return True
    if type(a) is not type(b) or type(a) in (list,) or hasattr(a, '__array_interface__'):
        return False
    if type(a) is float:
        return struct.pack('<d', a) == struct.pack('<d', b)
    try:
        return a == b and not (a != a)
    except Exception:                                          # noqa: BLE001
        return False


def _array_reader_for(entry, args, gl):
    """Records of arrays in lists, for lists passed in: the arrays they came as, views of those, or what the kernel made."""
    made = {}

    def read(value, et, blocks):
        numpy = _numpy()
        address, which, same, block, data, ndim = value[:6]
        typestr = numpy.dtype(et.elem.name[3:]).str
        if block:
            blocks.append(block)
            root = made.get(block)
            if root is None:
                root = made[block] = numpy.array(_Interface(None, data, value[6:6 + ndim], None, typestr))
            if not same:
                address += root.__array_interface__['data'][0] - data
            else:
                return root
        else:
            if gl is not None and tuple(value) in gl.arrays:
                return gl.arrays[tuple(value)]
            root = None
            if which >= 0 and entry.array_args:
                i, key = entry.array_args[which]
                root = args[i] if key is None else _member(args[i], key)
                if same:
                    return root
            if root is None:
                root = _owner_of(address, gl, args)
        return numpy.asarray(_Interface(root, address, value[10::2][:et.ndim], [x * root.itemsize for x in value[11::2][:et.ndim]]))
    return read


def _owner_of(address, gl, args):
    """The array whose memory holds address: one passed in, or in a list passed in."""
    cands = list(gl.arrays.values()) if gl is not None else []
    cands += [a for a in args if hasattr(a, '__array_interface__')]
    for a in cands:
        lo = a.__array_interface__['data'][0]
        span = sum((n - 1) * abs(st) for n, st in zip(a.shape, a.strides)) + a.itemsize
        if lo - span <= address < lo + span:
            return a.base if a.base is not None and hasattr(a.base, '__array_interface__') else a
    raise RuntimeError('an array in a list points at memory scramjit did not give it')
