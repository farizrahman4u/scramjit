# FAQ

## Why not use an existing JIT?

If one works for you, it has a head start. LLVM-based JITs are mature and vectorize via LLVM, but need LLVM wheels, don't run in Pyodide and take time to support new CPython releases. Pythran, Cython's pure-Python mode and Shed Skin need a C or C++ toolchain on the target. scramjit needs neither, because it and its compiler are pure Python.

## How do I know my function was compiled?

`scramjit preflight src/` shows every decorated function as compiled, partly compiled or Python, before you run anything, and which lines stay in Python. `scramjit preflight --run app.py` (or `--run -m pytest`) shows what happened with real argument types, and how much of the run was native. Each fallback also emits a `FallbackWarning`. See [Preflight](profiling.md).

## Is it only for numeric code?

Mostly: numbers, numpy arrays, tuples, lists (including lists that grow), sets, dicts with fixed keys and instances of your own classes. The target is any loop the interpreter is slow at, not only math: searches, flood fills, dynamic programming and scans over byte buffers fit as long as they work on those types. Strings don't compile yet.

## What if it can't compile my function?

It runs as Python, so your program still works and gives the same results; you just don't get the speedup for that part. (Code that does compile has a few [documented differences](gotchas.md).) You get a `FallbackWarning` naming the line. Add `@scram(fallback=False)` to make that an error instead.

## Does it help with the GIL?

Yes: compiled code releases the GIL while it runs, so threads calling it run in parallel, including parts of one loop ([splitting a loop across threads](caching.md#splitting-one-loop-across-threads)).

## Is it a new language?

No. Decorated functions are ordinary Python and run unchanged without scramjit. scramjit compiles a subset of Python, and runs the rest as Python. There is no dialect to learn.

## What does it not compile?

Anything that needs Python objects at runtime: objects whose classes run code of their own on attribute access (properties, `__getattr__`), strings, `None`, and dicts that gain keys. That code runs as Python, and the loops around it still compile. [Dicts with fixed keys](data-structures.md#dicts), [lists that grow](data-structures.md#lists-that-grow-nest-and-are-shared), [sets](data-structures.md#sets) and [the attributes of an instance](data-structures.md#instances) do compile. See [Fallback and errors](errors.md).

## Will it vectorize my loop?

scramjit emits scalar C. It is built for loops that don't vectorize; for simple array arithmetic in an environment where numpy runs, numpy is already fast.

## Is the generated code sandboxed?

No. Like cimo, compiled code runs with the full privileges of your Python process.
