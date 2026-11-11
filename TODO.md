# TODO

Biggest first. `requirements.md` is the spec; this is the work queue. Smaller items are GitHub issues
(https://github.com/farizrahman4u/scramjit/issues): an item moved there leaves a one-line pointer here.

## Release

0. **0.0.5 released (2026-11-09),** with cimo 0.0.5: CI on every platform, CPython 3.12+. The rule stays: the README, site and
   docs describe the released version, and a feature on `main` but not on PyPI is released the same day or marked
   "from 0.0.6" wherever it's mentioned (`AGENTS.md`). `tests/test_version.py` checks the version agrees everywhere it's written.

## Object model: what's left

Built 2026-11-05 and 2026-11-06 (see "Object model" in `requirements.md`); tests in `test_alias_pattern.py`, `test_new_instances.py`,
`test_element_objects.py`, `test_references.py`. Left, smaller than what's done:

1. **References:** a compiled callee making instances to keep in the caller's pool (it gets no count, so it stays Python), a pool
   when no object and no argument says what a made class's attributes hold. A pooled call costs about 10 µs plus 0.45 µs per
   object reached (`_pool_build`), so a single walk over a list is faster as Python: worth a pool kept between calls, checked by
   identity, if a workload calls repeatedly on the same structure.
2. **Static preflight** assumes a pool's class from its annotations and that references reach that same class.

## Demos

1. **Standalone demo pages, one per demo, with the LLM demo's shell.** `docs/demos/llm/` is the
   model: a full page at `scramjit.dev/demos/<name>/` with the canvas, the editable code, the
   scramjit switch, the stats strip (frames per second, iterations per second, speedup, compile
   time), the tabs for the syntax tree, C and WebAssembly, the share-an-edit link, and the
   "compiles stay between visits" cache. Make the shell one module (`docs/demos/shell/`: `page.js`,
   `worker.js`, the CSS) that each demo fills with a Python file, a draw function and presets, so a
   new demo is one `.py` and one small `.js`. Then:
   - **Existing lab demos as pages:** `demos/flock`, `demos/erosion`, `demos/sand`, `demos/fractal`
     from `docs/lab/*.py`, unchanged in substance, each with room to show the full code (the home
     page's lab truncates). The lab on the home page keeps linking to them.
   - **`demos/3d`: strange attractors.** Aizawa, Dadras, Halvorsen, Lorenz, Rössler, Thomas (cimo's
     docs have posters of each): a `@scram` function integrates a million points with RK4 per frame
     into a float32 array that three.js (already on the site) draws as a rotating point cloud;
     plain Python manages a few thousand. Presets switch the system; the parameters are editable
     and recompile live. Color by speed.
   - **`demos/lenia`: continuous cellular automaton.** Lenia (Chan 2019) with one kernel ring and
     growth function, 256x256 at first, the convolution as a plain loop (no FFT: that is the point),
     drawn to a 2D canvas with a colormap. Presets: Orbium and two other known creatures. Edit the
     growth function's `mu`/`sigma` and watch it change. SmoothLife as a second preset family if
     cheap.
   - **`demos/tracer`: progressive path tracer.** A Cornell-box scene of spheres and planes, one
     sample per pixel per frame accumulated into a float32 buffer, tone-mapped to the canvas: noise
     resolves to a clean image in seconds under scramjit, barely clears the first pass as Python.
     Editable material parameters and light. 256x256 to start; measure what Pyodide sustains.
   - **`demos/fluid`: the race.** Stable fluids (Stam 1999: diffuse, advect, project, about 80
     lines) with dye injected by the mouse, **two panels on one clock**: plain Python on the left,
     scramjit on the right, both given the same real time per frame as `tools/readme_gif.py` does
     for the Mandelbrot race, with the clock and iteration counts on screen and nothing staged.
     This is the social-media asset: it needs no explanation.
   - **A recording mode** on every page: `?record=1` hides everything but the canvas, the stats and
     the switch, sizes the canvas to 1920x1080 (2x device pixel ratio), and `?autorun=<seconds>` flips
     the switch after that many seconds, so a clean 1080p screen capture needs no clicks. Document it
     in `docs/demos/README.md` with the shot list (3 s plain Python, flip, 5 s smooth, one edit, hold).
   - Each page states its own numbers from a measurement in Pyodide on a named machine, as the lab
     table in the README does, and the README's demo table gains a row per new demo.
   - Order: the shell with the four existing demos first (they're known to work), then fluid (the
     asset), then 3d, lenia, tracer. Each lands with its page, a preset test in `tests/`, and a line
     in the README.

2. **A case study from a real application.** A private CAD-drawing cleanup app (ours, not open source: name no
   product, client or code) moved its geometry from vectorized numpy to scramjit loops (2026-11-03, scramjit 0.0.3).
   numpy tested every candidate pair and de-duplicated afterwards; the loops stop at the first hit, walk only the grid
   cells a segment passes through and stamp what they've counted. Per kernel: a conflict graph 3.4 -> 0.8 s,
   crossing counts 2.9 -> 0.2 s; with compiled code releasing the GIL, a thread pool replaced processes (peak memory
   1.15 -> 0.63 GB, 1.7x the solver's repair windows in the same budget); the module halved (401 -> 187 lines, one
   implementation per test instead of numpy plus Python); a whole run 17.4 -> 12.4 s, the rest being file I/O and a
   CP-SAT solver. Write it up for the site and README as "already-vectorized code, made faster by un-vectorizing it",
   with a small open stand-in for the kernels (segments against a grid, early exit, stamping) in `examples/` that
   shows the same effect against its own numpy version. What it worked around goes in the items it points at:
   outputs of unknown size (a preallocated array, counted past, the kernel rerun with more room: lists that grow
   with `append`), text built digit by digit into a `uint8` array, a first call of about 4.5 s (#19 in "Next").

## Directions (priority order; each is weeks, not days, and has no oracle to diff against)

1. **Autodiff: gradients of any `@scram` function.** Source-level differentiation on scramjit's own
   IR (after type inference, before C): forward mode first (dual numbers per scalar, arrays of
   tangents), then reverse mode on loop bodies over arrays with a tape only where control flow
   needs one (Tapenade/Enzyme-style; the generated C stays plain loops cimo optimizes as usual).
   API: `grad(f)`, `jvp`, `vjp`, matching JAX's names for the subset. Correctness oracle: finite
   differences and, where it runs, JAX on the same function. Targets, in order: ODE/PDE parameter
   fitting (a `@scram` RK4 integrator differentiated through time), a physics-informed loss,
   inverse problems. It runs in Pyodide, so the demo is "fit a model to data in a tab."
   Hard parts, measured before designed: reverse mode through `while`, in-place array updates
   (checkpoint vs. recompute), and `math` functions (cimo's libm needs derivatives).
2. **Equation in, compiled solver out.** Tools that already generate C and shell out to a compiler
   run on cimo instead (cimo's `TODO.md`, "Directions"). scramjit's part: the Python-side ones,
   where the model is written as Python and the solver is a `@scram` loop: an ODE/SDE system from
   a right-hand side function, a stencil PDE from an update rule, with the time loop, boundary
   conditions and output stepping generated. First target: a minimal Devito-shaped API (grid,
   stencil expression, time loop) over scramjit, running in JupyterLite.
3. **GPU from a `@scram` function.** When cimo has a WGSL target (its `TODO.md`), `@scram(target='gpu')`
   maps a kernel over an index space: the function body becomes the per-element kernel, arrays
   become buffers, reductions are the first thing that won't map. Through WebGPU in the browser and
   wgpu natively; no CUDA, ROCm or Metal SDK. scramjit's work is the mapping rules (what a loop
   over `range(n)` means on a GPU, which loops stay on the CPU) and the buffer transfer, measured
   against the CPU path on one matmul and one stencil.
4. **Autotuning per machine.** Expose cimo's tuner (its `TODO.md`, "Directions") as
   `@scram(tune=True)`: the first call compiles the variants cimo proposes, times them on the
   user's CPU, keeps the best in the cache keyed by CPU model. Report what won in `preflight`.
5. **Model-written code, fast and sandboxed.** The product surface for code an LLM wrote: RL
   environments and reward functions, verifiers, data-pipeline steps, run as `@scram` on the WASM
   target in-process with a time limit and a memory limit, and natively when the code is trusted.
   Build the harness: `scramjit.sandbox(source, inputs, limits)` returning results or a typed
   failure; a Gymnasium-shaped environment API; a benchmark against a subprocess/container sandbox
   on throughput and startup. The measurement decides whether this is a product: do it before the API.
6. **Inference kernels: quantized formats, written once in Python.** Assumes the pieces elsewhere
   (narrow types below, `# scram: parallel` (Adoption #6), cimo's reductions and #3). The draw: the undecorated
   function is the reference and the kernel at once, so a new format needs no second implementation
   in C intrinsics, CUDA or WGSL to check against.
   - **Types:** `float16` and `bfloat16` arrays (bfloat16 has no numpy dtype: accept a `uint16`
     array with `scramjit.bf16(x)` as the view, or `ml_dtypes` as a dev extra), computed in float32
     and stored narrow; packed `int4` read through a helper (`scramjit.nibble(x, i)`), with group
     scales as a separate array. Needs cimo's half types (its `TODO.md`, Adoption).
   - **Kernels:** fused dequantize-and-dot for llama.cpp's `Q8_0` and `Q4_0` layouts first (an
     oracle: their outputs on the same blocks), then one format they lack (ternary or a codebook).
     Measured per token in `bench/llm.py` against ggml's own kernels on arm64 and x86-64; losses
     reported as they are.
   - **Fusion with the branchy part:** logits processing, a grammar mask and sampling (top-p, min-p,
     speculative acceptance) in one kernel after the last projection, measured against the same
     steps as separate numpy calls.
   - **Also needed:** torch tensors in and out zero-copy through `__dlpack__` (Adoption), or
     researchers can't call it from their models.
   - **Not the target:** datacenter GPU kernels (tensor cores, shared-memory tiling, async copies);
     WebGPU barely exposes matrix units, and Triton and CUDA already own that work.
   - **Decide first:** this contradicts the spec's "Not for: vectorized code" and the rule that public
     copy doesn't advertise ML use. Settle the positioning in `requirements.md` before any public page.
7. **Proofs about `@scram` functions, in pure Python.** Agents write code faster than it can be
   reviewed; a contract that is proven can't be gamed the way tests can. A Lean-style system written
   in Python: the toolchain (kernel, elaborator, automation) stays pure Python, no Lean, z3 or other
   native prover, so proofs check in Pyodide too.
   - **Kernel:** a dependent type theory (inductives, universes, definitional equality), small and
     audited, nothing clever in it. Everything else, automation included, is untrusted and emits
     proof terms the kernel checks.
   - **Reflection:** a terminating `@scram` function becomes a definition; `requires`/`ensures`,
     `decreases` and `assert`s become theorem statements, still valid Python undecorated (a property
     test for free). The semantics are the compiled ones, 64-bit wrapping included, stated explicitly;
     the differential tests against CPython back the translation.
   - **Automation, in order:** linear integer arithmetic (bounds, off-by-one, overflow), then
     congruence closure, then induction. Success is measured by how many goals close with no
     explicit proof.
   - **Proof surface:** Python syntax for what automation can't close (intermediate lemmas,
     invariants, induction hints), written by agents. No separate tactic language.
   - **Preflight:** each assert and contract reports proven / runtime check / counterexample, as
     bounds checks do now; `--json` for agents.
   - **Library:** program-verification lemmas (integers, lists, arrays, inequalities), not Mathlib.
     Optional later: check a slice of Mathlib from `lean4export` output as data (Lean runs once,
     offline; the toolchain stays pure Python).
   - **Measure first:** the kernel's speed on a realistic proof term (a list-sorting proof exported
     from Lean) before building the elaborator; CPython is likely 50-100x slower than Lean's C.
     Long term, scramjit compiles the prover's hot paths, which needs recursion, strings and
     growing dicts to compile. Also measure: on ~50 agent-written kernels, what fraction of their
     bugs linear integer arithmetic alone catches or proves absent.
   - **First step (days):** assert discharge in preflight from the range facts bounds-check
     elimination already computes, before any kernel.
   - **Decide first:** this is a new positioning ("verified Python for agents"); settle it in
     `requirements.md` before any public page.

## Adoption (priority order, below Directions; each is days, and has an oracle: the existing behavior, numpy, or Numba)

Rules for every item here: no new dependency (numpy and the optional library under test stay dev
extras); every step lands with its tests in `tests/` and its page in `docs/docs/content/`, and the
README's feature table changes only when PyPI has the feature (Release #0); measure before and
after with `bench/`, and write the numbers into the item when done.

1. **Shipping a package built on scramjit: pure-Python wheel, native speed, no compile for the user.**
   cimo's `distribution.md` covers the dependency and the vendored one-file cimo; what's missing is
   the scramjit side and shipping compiled code so a user never waits for a compile.
   - **Precompiled cache in the wheel.** cimo's public cache export/import has landed (`module.cache_entry()`,
     `cimo.add_cache_entry()`, 2026-11-07): add `scramjit build <package>`: imports the
     package in a subprocess with `SCRAMJIT_RECORD=1`, calls nothing, but compiles every `@scram`
     function for the signatures listed in a `scramjit.toml` (`[[precompile]] function = "pkg.mod.f"
     types = ["float64[:]", "float64[:]", "float"]`), one entry per target platform the build runs on
     (`linux-arm64`, `linux-x86_64`, `macos-arm64`, `windows-x86_64`, `windows-arm64`, `wasm32`), and
     writes the cache entries under `pkg/_scramjit_cache/<platform>/`. At import, `scram` looks
     there before compiling.
   - **Signatures without a call.** `@scram` needs argument types to compile; add
     `f.compile("float64[:], float64[:], float")` (the type-string syntax `_types.py` already prints)
     so a package can compile at import time deliberately, and `scramjit.toml` uses the same strings.
   - **An example package** at `examples/package/`: `pyproject.toml` building a `py3-none-any` wheel,
     one `@scram` kernel, the `scramjit.toml`, a GitHub Actions workflow that builds the wheel once
     and runs its tests on the five native platforms and in Pyodide (`tools/run_pyodide.mjs`), and
     a README that is the guide. Copy its `pyproject.toml` into `docs/docs/content/packaging.md`.
   - **Tests:** `tests/test_packaging.py` builds the example wheel with `python -m build`, installs
     it in a temp venv, imports it with `SCRAMJIT_LOG=1`, and asserts no compile happened (log is
     empty) when the precompiled cache matches, and exactly one compile when it doesn't.
   - **Done when:** `pip install` of the example wheel followed by the first call takes under 50 ms
     on a machine the cache was built for, and the guide fits on one page.

2. **Notebook integrations: marimo, JupyterLite, Jupyter Book.** The Pyodide path works; the
   gap is that nobody finds it. Three small deliverables, in this order:
   - **marimo.** A `marimo` notebook in `examples/notebooks/` that runs in marimo's WASM mode
     (`marimo export html-wasm`) with one `@scram` kernel behind a slider; a PR or issue to marimo
     listing scramjit in their docs' "packages that work in WASM" page, with the notebook as the
     example. Check `micropip.install("scramjit")` works from a stock Pyodide index (both wheels are
     pure Python, so it should; if `cimo` isn't resolvable, that's a packaging bug to fix first).
   - **JupyterLite.** A `jupyterlite` site config in `docs/notebooks/` (`jupyter_lite_config.json`
     with `piplite` wheels for `cimo` and `scramjit`) deployed to `scramjit.dev/notebooks/`, with the
     Gray-Scott and the EMA notebooks. Pyodide's kernel runs in a worker, so the main-thread sync
     compile limit (cimo `requirements.md`, "Platform loading") doesn't apply; note it.
   - **Jupyter Book / Sphinx.** A `sphinx` directive `.. scramjit-demo::` is too much; instead a
     doc page `notebooks.md` showing the `thebe`/JupyterLite embedding so a book's figure runs a
     `@scram` kernel live, and one such page on the site as proof.
   - **Tests:** `tests/test_notebooks.py` executes each notebook with `nbclient` natively (dev extra)
     and `tools/run_pyodide.mjs` runs the kernels' code in Pyodide. **Done when:** both notebook
     environments show scramjit in their own docs or examples, or the PRs are open.

3. **Native UDFs over Arrow, DuckDB and Polars buffers.** Their columns expose the buffer protocol
   or `__array__` zero-copy, so a `@scram` function can be a vectorized UDF with no C++ extension.
   - **Arrow first** (`pyarrow` dev extra): accept a `pyarrow.Array`/`ChunkedArray` of a primitive
     type as an argument, mapped to numpy via `to_numpy(zero_copy_only=True)` in `_dispatch`
     (`_arg_type`, `_push`), with the validity bitmap passed as a second hidden `uint8[:]` argument
     when nulls exist and `scramjit.isnull(x, i)` to read it. Chunked arrays: one call per chunk,
     results concatenated, in order. Strings and nested types stay Python (fallback message says so).
   - **DuckDB:** `duckdb.create_function(name, f, type="arrow")` takes a function over Arrow
     batches; show `@scram` inside it on a 10-million-row aggregate, measured against DuckDB's own
     SQL and against a plain-Python UDF. **Polars:** `df.with_columns(pl.col("x").map_batches(f))`
     over `to_numpy()` the same way. One example each in `examples/udf_*.py`.
   - **Tests:** `tests/test_arrow.py` (skipped without `pyarrow`): zero-copy asserted by writing
     through the array and reading from Arrow; nulls; chunked; results against numpy. **Done when:**
     the DuckDB example runs the UDF within 2x of the equivalent SQL expression, numbers in a new
     `udfs.md`.

4. **`deterministic=True`: the same bits on every platform.** cimo bundles its own libm and controls
   instruction selection; `parity=False` already uses that libm. What's missing is the promise and
   the proof.
   - **Define it:** `@scram(deterministic=True)` implies `parity=False`, `fma=False` in the cimo
     compile (no contraction, which differs between arm64 and x86-64), no reassociated reductions,
     and `# scram: parallel` with a fixed thread count only. Document what it does *not* promise:
     numpy calls that stay Python (their results follow numpy's SIMD paths), and `uint64`/narrow
     dtypes until their rules are platform-independent (item 1 under "Next").
   - **Prove it:** `tests/test_deterministic.py` runs a fixed set of kernels (the lab demos,
     `bench/kernels.py`, the `math`-heavy viewshed) on 1,000 fixed inputs and writes SHA-256 of the
     outputs to `tests/data/deterministic.json`; CI on all five platforms and Pyodide compares
     against the committed hashes and fails on any difference. The first run finds the places cimo
     isn't deterministic (expect: libm `pow`/`exp` paths that differ between the scalar and vector
     versions, x86 `sqrt` vs arm64); each becomes a cimo issue, fixed or excluded with a reason.
   - **Done when:** the hash file is identical from every CI job and the doc page `determinism.md`
     lists exactly what is covered.
   - **Parallel float reductions:** with `# scram: parallel` the combine order is fixed per thread count;
     a fixed reduction tree independent of thread count (and, under #3, of the GPU's workgroup size)
     would give batch- and thread-invariant results, the property reproducible LLM inference needs.
     Measure what the fixed tree costs against a free one.

5. **Torch, JAX and CuPy tensors through `__dlpack__`:** issue #7.
6. **`# scram: parallel` on a loop, checked.** Measured first (2026-11-06): splitting a loop by hand already
   scales, since compiled code releases the GIL. A `ThreadPoolExecutor.map` over four ranges of one array ran
   4.0x on 4 threads and 5.0x on 8 (Linux, 8 logical cores), bit-identical, and it's now documented ("Splitting one
   loop across threads" in `caching.md`, `examples/threads.py`). Lists, lists of instances and lists of lists split
   this way lost writes until the write-back compared against a snapshot taken at the call instead of the list's
   current values (`_snapshot`, `tests/test_threads.py`). So what the hint would add is what the hand-written split lacks:
   - **A proof the iterations are independent,** reusing the dependence analysis bounds-check versioning does; a
     hand split with a dependence across parts gives wrong results silently. The comment is a request, not a promise:
     anything not proven runs serially with a `FallbackWarning` naming the line and the reason. No `prange`, no
     decorator flag: a comment keeps the code standard Python, and preflight can suggest it on loops that qualify.
   - **Exceptions:** writes before one must match CPython, so only a loop that can't raise goes parallel; the copy
     bounds-check versioning makes after its checks pass is that loop.
   - **Reductions:** ints, `min` and `max` exact; float sums only with `parity=False`, combined in block order.
   - **Arguments copied in once,** not once per part.
   - **The cost of starting the parts:** four tiny calls through a pool took 390 µs against 3.8 µs for one direct call
     (Linux, WSL). The hint is worth little more than the hand split while parts start from Python: threads started from
     the C would need cimo to offer them (not checked).

7. **Arguments that need no wrapping:** issue #6.

## From cimo

- **A module's WebAssembly is public (cimo 2026-11-07):** use it in the lab and LLM demo: issue #1.

- **Ctrl-C stops a running kernel (cimo 2026-11-07, Linux and macOS):** the generated `if (*sj_stop) SJ_FAIL(...)` check
  in every loop can go, once cimo leaves a kernel running under a SIGINT handler of the program's own (its TODO,
  "Big"; `tests/test_interrupt.py::test_a_custom_handler_doesnt_stop_the_kernel` fails until then). Keep it for Windows.
- **`module.cached` (cimo 2026-11-07):** `preflight --run` can show `disk`: issue #4.
- **Public cache export/import (cimo 2026-11-07):** the LLM demo can switch to it: issue #2.
- **The CLI framework is public (cimo 2026-11-07):** replace the copies in `scramjit/cli/`: issue #3.
- **Inlining spends its budget on the deepest loops first (cimo 2026-11-07),** and helpers no bigger than their call
  always inline: the four wrapping helpers made macros for this can go back to functions. llama2 tok/s not
  re-measured yet.
- **gcc -O3 parity (cimo 2026-11-07, bench/gcc.py on mains):** gray_scott level, viewshed 0.64x of gcc's time, hand
  flood_fill level; generated flood_fill 2.1-2.2 vs 1.84 ms. Two things on your side help it: emit the floor-division
  sign fix as an explicit branch (`q = p / w; r = p % w; if (r != 0 && ((r < 0) != (w < 0))) { q -= 1; r += w; }` when
  `w != 0` is known): 2.47 to 2.00 ms on this loop with cimo (gcc 1.82); and drop the `*sj_stop` check (see Ctrl-C).
  cimo's remaining gap is gcc's unswitching of that loop on `w`.

- **cimo's `hypot` is glibc's algorithm (cimo 2026-11-07):** one `sqrt` of an fma on arm64 (a correction step on x86-64 and
  wasm), under 1 ulp; a `hypot` loop 1.47x faster than before, about glibc's speed by your viewshed ratio. Still a
  call (no inlining of the runtime yet). Its last bits differ from musl's in places.

- **A kernel can call a host function through a pointer in Pyodide (cimo 2026-11-07):** cimo's pointers are i32 at a wasm
  function's boundary now, as Emscripten's, so Pyodide's libm `frexp(double, int *)` by address works; the bit-level
  `frexp` workaround can go if wanted. A host `long` parameter still differs (Emscripten's is 32-bit).

- **Large `malloc` in Pyodide returns NULL (cimo 2026-11-07),** where a 4 GiB request returned a small block; and
  `module.free` works there, so `sj_release` can go: issue #5.

- **Half types are accepted (cimo 2026-11-06):** `_Float16` and `__bf16` as storage, computed in `float`, converted
  by runtime calls (hardware conversions and vectorized loads still open, cimo's `TODO.md` Adoption #6). Enough
  to start Directions #6's types; not yet fast.

- **max, min, argmax vectorize (cimo 2026-11-09):** cimo now vectorizes float maximum/minimum and argmax exactly (the
  scalar loop's ties, zeros and NaNs) and int ones at numpy's speed. `sj_np_max_f`, `min_f`, `argmax_f`, `argmin_f` are
  now a strict loop beside a NaN detector (`z += e - e` under `#pragma omp simd reduction(+:z)`), the loop that stops
  at a NaN only when z is NaN or a zero needs rule 1 or 2: 1M float64, Snapdragon X: `max` 1016 to 171 us (numpy 113),
  `argmax` 1366 to 258 (numpy 314); `max_i` 122 (116) and `argmax_i` 177 (174) unchanged C. Same bits (4000 random
  arrays of zeros, NaNs and infinities, strides 1, 2, 3, -1, every rule, against the old code).

## Workarounds waiting on cimo

Each has a matching entry in cimo's `TODO.md`, which asks for a note here when it lands.

- **Float `math.sumprod` falls back:** Python returns the int 0 when it's empty, so its type depends on
  the data. Same for float `math.prod` without `start=1.0`, where the fix is in the user's hands.
- **Math functions call the host libm through a table of addresses** (`_dispatch._libm_table`,
  `SJ_LIBM1` in the prelude), passed to each kernel that uses them, because cimo resolved `sin`
  etc. to its bundled libm. cimo now has `compile(libm='host')` (2026-11-09): the host's libm, the functions CPython's
  `math` calls, bit for bit (not `hypot`, which CPython computes itself). Call them directly with it under
  `parity=True` and drop the table.

- **Code shaped around cimo's missing CSE:** wraparound indices hoisted to the top of the loop
  body (`hoist_wraps`), `//` and `%` computed as one division and one remainder into
  temporaries. Both help any compiler; revisit what else can go once cimo merges repeated
  expressions (cimo `TODO.md`: "Slower than gcc -O3 on scramjit's kernels").

- **The LLM demo moves compiled code by listing what changes in `CIMO_CACHE_DIR`:** issue #2.

## Line counts, what's left

Spills per line, once cimo can report them.

## Next

1. **SIMD through cimo, what's left:** what vectorizes was checked loop by loop (cimo's `vectorize.plan`, wrapped):
   `saxpy`, scaling, `a + b + c`, stencils, int `+`, `^`, `&`, 2-D rows, offset bases and float32 do, and int sums now that a
   variable's numpy flag is set after the loop, not in it (`sinkable_flags`; int32 sum about 2x, int64 1.2x, memory-bound).
   Not yet: anything with a condition (`relu`, `clip`, `max`), which needs cimo to vectorize them (its `TODO.md`); `math.sqrt`
   and other functions that check their domain on every element, which only `parity=False` lets vectorize (a check after the
   loop would need its stores undone); float sums and `dot`, kept scalar on purpose, since their order is Python's. Re-run the
   probe on `bench/` and the examples when cimo lands conditions.
2. **Narrow dtypes, what's left:** arithmetic, `**`, `sum()`, the exact numpy functions and most reductions compile (see "Narrow dtypes" in `numpy.md`).
   Not yet: `np.sin`, `np.exp` and the like on float32 (numpy's own SIMD versions: check them against numpy per CPU first), float functions of narrow ints (float16 results),
   `.std()` and `.var()` on float32 (numpy computes them in float32), `.sum()` of signed ints on wasm32 (numpy totals in int32 there), and `np.int8(x)` of a float.
   A literal becoming narrow must fit exactly: not `s = 0.1` on float32, nor a parameter, and such a variable can't meet a Python number afterwards, since the type depends on its flag.
   Speed: byte and short lanes **done in cimo** (2026-11-09): `uint8` brighten 0.24 to 0.018 ns per element (int32 0.071, numpy
   0.022), its per-iteration `amount` check unswitched out of the loop by cimo; widening (`uint8` to `int32`, byte sums) is still in cimo's `TODO.md`.
   `bench/llm.py` still runs on float64 weights, from before float32 compiled: measure float32 (half the memory the matrix products read), with `bench/gcc.py` on `float` alongside.
3. **64-bit ints, what's left:** hashes masked with `& 0xFFFFFFFFFFFFFFFF` compile (`examples/hash64.py`). `np.uint64`
   elements compile with numpy's rules (`tests/test_uint64.py`). Not yet: unsigned 64-bit Python ints in tuples, dicts, as `/`, passed to
   compiled functions or `math`, compared with float32, as slice bounds, or choosing a row (`m[int(h)]` of a 2-D array).
   `examples/xorshift.py` takes 3.9 ms per million numbers against 2.05 for the same loop written in C on `uint32_t` through cimo:
   the masks sit on the chain each number waits on, and storing into `uint32` checks the range.
4. **Bounds-check hoisting, what's left:** versioning covers `arr[i + c]` and, in innermost loops, `arr[base + i]`
   with `base` built from Python ints that don't change in the loop, also with a positive literal step and a start
   checked at run time. Not yet: negative steps, numpy int bases, `arr[i // 2]` and `arr[2 * i]`, and outer loops,
   where a second copy of everything inside costs more than it wins.
5. **Store-to-load forwarding, what's left:** `y[i - 1]` to `y[i - 4]` come from temporaries when `y[i] = ...` is y's
   only store in the loop and runs every iteration, with other arrays stored to behind a test that they don't share
   y's memory. Not yet: 2-D rows (`m[r, j - 1]`), strided views among the other stores, and stores inside an `if`.
6. **Loop lifting, what's left:** a loop stays in Python if a variable read afterwards may be unbound
   (`for i ...` then reading `i`, with nothing assigned first), inside `try` or `with`, or containing `return`.
   "Partly compiled" is per function, not per signature. Tracebacks show one extra frame for a loop that raises.
   The first two are hard for one reason: compiled code has static types. A name the loop may leave unbound has no
   value of any type to come back as, and a `return` inside means returning either its value or the loop's outputs.
7. **numpy calls in compiled functions, what's left.** No cimo work is required for any step; its SIMD only adds speed:
   - **numpy functions on single numbers: built.** See `numpy.md`. Left in Python: `np.tanh` and `np.power`, which aren't libm's, and
     combinations whose result depends on the platform or numpy version. Bit for bit checked on arm64, wasm and x86-64 with AVX2 (Windows, emulated). Check AVX-512, where numpy
     may use its own `exp` and `sin`. Not yet: `np.mod`, `np.floor_divide`.
   - **Views: built,** as arguments and results too, with `.T`, slice assignment and tuples of views. Not yet: `a, b = m` (unpacking rows), `x[1:] += y`,
     a two-dimensional-or-more array assigned into three dimensions or more.
   - **Reductions: built,** with `std`, `var`, integer `mean` and `argmax` on wasm32, for one number from `float64` or `int64`. `axis=` that returns an array, `keepdims=True`, `dtype=`, `out=`, `where=`, `initial=` stay Python.
     Not yet: views with gaps in more than one dimension (numpy's order depends on strides), other dtypes.
   - **Arrays made inside a function: built.** See "Making arrays" in `numpy.md`. Not yet: `x.reshape()`, `np.eye`, `.size` or `.shape` directly on a call (`np.ones(n).size`), a conditional expression of arrays (`y = a if c else b`),
     five dimensions or more, `uint64` and `bool` dtypes, an array made in a comprehension. Returning one costs a copy (3.1 µs for 10 elements): worth removing if a Gymnasium-style `reset` or `step` shows it.
   - **Whole-array arithmetic: built (2026-11-07),** scoped: `+ - * / **`, unary `-`, and the numpy functions on single
     numbers, applied to arrays of up to 2 dimensions (`float64`, `int64`, narrow), views without gaps in 2-D, and
     plain numbers (names, constants, `w[k]`), with broadcasting (`OPERANDS_ERROR`, numpy's message). One pass per
     expression: each operand's element goes in a name (`cnames`) and the scalar rules, which are numpy's, compute it;
     a flat loop when every operand has the result's shape. `tests/test_elementwise.py`, `examples/whole_array.py`.
     Faster than numpy up to tens of thousands of elements; a big new array each pass of a loop too, since cimo
     reuses freed large blocks (1,000,000 elements: 6.0 ms against 6.9). Not yet: `//`, `%`, comparisons (bool arrays), more than 2
     dimensions, `x[1:] += y`, writing straight into the target of `y[:] = ...` when they don't overlap (it makes the
     right side first), and reusing a dead array's block for `y = y * 0.5 + b`.
   - **Not `np.dot`, and no MKL or BLAS:** they would break "standard library plus cimo", and their results depend on the library, CPU and thread count.
     `A @ B` and `np.linalg` already run at full speed in Python. An opt-in through cimo's `link=` for BLAS inside a hot loop is possible
     later; untried.
8. **Python data structures, nested, when statically sized and statically typed, after the loop work above:** so
   that containers don't hide speedups we can still get from loops, each step must leave `bench/` unchanged.
   Dicts, lists, tuples, classes and enums that don't resize at runtime and whose element types don't change. See "Containers" in
   `requirements.md`; lists, dicts, instances and sets that grow and nest are #23. What's left, by kind:
   - **Module-level constants: built** (lists too): hidden arguments, read at each call, about 0.2 µs each.
   - **Enums: built.** See "Enums" in `data-structures.md`. Not yet: `.name`, `Cell(1)`, iterating over the class, storing a member into an array, `Flag` and string enums.
   - **Nested tuples: built,** as locals, arguments, results and globals, and loops over tuples. Not yet: a tuple holding arrays as a local or an argument, a tuple holding lists, and a tuple indexed with a variable.
   - **Lists as arguments, returned, and at module level: built** (see `data-structures.md`). Lists of lists passed in
     too: one array, rows as views, so a row is a reference as in Python. Lists of lists made in the
     function too, from `[[c] * w for _ in range(h)]` and its comprehension form (`[[c] * w] * h`, one row repeated, stays Python).
     Returned too. Not yet: returning a row of a rectangular list, numpy scalars mixed with Python numbers in one list, and a call
     plan for list arguments (each call takes the general path).
   - **Instances: built** (see "Instances" in `data-structures.md`, and "Object model: what's left" above). A list's copy is
     about 0.15 µs per instance and attribute, checked and built in Python twice (`_key`, `_run`): once would halve it. A call plan for an
     array among the attributes: about 12 µs per call now, against 2 for numbers and 2.8 for the array passed separately.
   - **Dicts, what's left:** passed in, nested, made with constant keys and returned all compile (#23). Not yet: `len(d)`,
     which is known from the keys, a key in a variable, `.get` and iterating. A dict that gains keys still runs as Python.
9. **Call boundary cost:** about 1.4 µs with numbers (was 3.2) and 2.4 µs with an array (was 6.6), through
   a function generated per signature (`_make_plan`). For numbers: ctypes about 450 ns (the floor), cimo's
   `Kernel.__call__` about 450 then, now close to ctypes (cimo 2026-11-05: a void call 365 ns against 364, one int 501,
   a numpy array 1,250 from 2,431, laptop), scramjit about 500. cimo also now takes a numpy integer for a 64-bit integer
   parameter as its value (pointers are told apart), so calls needn't wrap such arguments in `int(...)`. In Pyodide about 13 µs, not yet
   broken down. Break-even is 30 to 50 loop iterations per call. A Gymnasium-style `step` is below it, with no loop
   at all: CartPole from a Python loop does 396k steps/s compiled against 698k plain, and 53M with the loop compiled too, which calls between compiled functions now allow without pasting. Rejected: caching an array's address (saves
   200 ns, but `resize(refcheck=False)` would corrupt memory silently). **`__call__` by argument count, done
   (2026-11-07)** after all: measured, it saves 350 ns, not 100 to 150 (`Function.__call__` 517 to 166 ns). Each
   Function of up to 8 positional parameters is an instance of a generated subclass (`_front`) whose `__call__`
   takes them by position and calls the last plan made when every type matches (`_fast`); keywords, defaults, extra
   arguments and other types take the general path. `tests/test_front.py`. `self` as a keyword argument works now.
10. **Variables that change type, what's left:** a name used for separate things compiles (each web is its own
   variable), and so does an int that becomes a float, where the int is a literal up to 2^53 or a parameter, and it's only
   returned or combined with a float. Not yet: other uses of such a variable (`-s`, `s + 1`, `t = s`, `a[s]`, `if s:`), and
   `s = n` from an int variable, each of which keeps the function in Python.
11. **Rest of M3:** run the M0 kernels in Pyodide as a test, not only through the site; preflight's
   per-loop report once cimo exposes one (cimo's `TODO.md`); the `disk` label is issue #4.
12. **`n ** k` with a negative `k` and `x ** 0.5` with a negative `x`, what's left:** a function that writes nothing
   reruns the call as Python there. One that writes still raises: resuming in Python alongside loop lifting would remove
   that difference too.
13. **A NaN's sign differs from CPython's on Windows** (arm64, and x86-64 under emulation). `inf % 1.0`, `inf // 1.0`
   and `math.atan2(0.0, nan)` give `7ff8...` where CPython gives `fff8...`, and `nan * -1` gives `fff8...`
   where it gives `7ff8...`; `tests/test_m1.py` and `test_m2.py` fail there on these, 7 in all. The values are
   all NaN, and only `math.copysign` or the raw bytes can tell. Likely fix: call the CRT's `fmod` through the
   libm table, as for `sin`, and make the `* -1` negation depend on the platform. Nothing runs scramjit's tests
   on Windows automatically yet.
14. **Test Ctrl-C on macOS.** `tests/test_interrupt.py` has run on Linux and Windows arm64, not macOS. It
   should work, since it uses `signal.set_wakeup_fd` and sockets, but nothing has checked it.
15. **`parity=False` asks cimo for FMA, and it never happens.** cimo contracts `a * b + c` within one C
    expression, and scramjit gives each operation its own temporary. Nothing is lost on reductions:
    FMA makes them slower on arm64 (a `dot` through cimo: 2.15 ns per element with, 1.81 without;
    `bench/llm.py` through `gcc -O3`: 63 tok/s with, 82 without), probably the latency of the
    instruction the sum's next add waits on (not measured). Measure non-reductions (stencils) before making it fire or removing it.
16. **Calls between compiled functions, what's left:** recursion, defaults other than numbers left out by a compiled caller, keyword-only parameters, dicts and lists as arguments, a function passed as an argument, `module.f()`, and a line profile of what's called.
    **Calling C, what's left:** modules with state (calling across modules works in cimo since 2026-11-07, `malloc` included), `include_dirs=` and `link=`, returning a pointer or `uint64_t`, a C function's module compiled with `fma=` on alongside exact math in one file (cimo has `#pragma STDC FP_CONTRACT` since 2026-11-07).
17. **Compile time on large functions, what's left:** versioning's copies are capped (`VERSION_TOTAL`, `VERSION_MIN` in
    `_compiler.py`), most deeply nested loops first: a generated 60-loop function 14.7 to 5.1 s, `bench/llm.py` unchanged.
    The cost was mostly cimo's, now much less (cimo 2026-11-05): this shape of function, 60 loops all versioned (4,150 lines
    of C), 32.6 to 7.9 s, 120 small loops 21.4 to 2.7 s; what's left grows about 3x per doubling. Raise the cap to match.
    Depth is a guess at which loops run most: a profile (`counts=True`) would know.
18. **scramjit against Numba and Cython: measured on arm64 (2026-11-07), now close the gaps.** `bench/compare.py`
    (kernels in `bench/compare_kernels.py`, typed Cython in `bench/compare_cy.pyx`; numba and cython are dev extras,
    never dependencies), table in the README ("Against Numba and Cython"). Every column matches CPython's results but
    fastmath's. On par: Gray-Scott, flood fill, dot, axpy and (since its fix below) the histogram without fastmath. Numba doesn't compile the dict and dataclass
    kernels, and Cython as written gains nothing on them (scramjit 19x and 49x). Still to do: **run on x86-64** (a
    reviewer's machine is x86; needs another machine) and put the table on the site. Gaps, biggest first, each to fix with
    a measurement before and after:
    - **Viewshed, 2.3x behind Numba** (3.2 against 1.4 ms; Cython 3.3, gcc 2.5): was 8.7 ms. The exact `math.hypot`
      was 6.5 of the 8.3 ms (`round` costs nothing measurable). `sj_hypot2` does CPython's vector-norm arithmetic for
      two values with `frexp`/`ldexp` as bit fields (23 to 6.5 ns a call, bit-identical on 240,000 pairs), its rare cases
      and `sj_truediv`'s long division `noinline` (8.3 to 3.4 ms). cimo's inliner now counts only what its cleanup
      keeps and allows `inline` functions twice the limit, so `sj_hypot2` inlines (3.4 to 3.1 ms; `parity=False` 3.2
      to 2.5). Where the rest goes (2026-11-07): Numba calls the C library's `hypot` (glibc's differs from CPython's
      `math.hypot` in the last bit for 12% of 2M random pairs; viewshed's 0/1 output hides it) and checks no bounds. The
      same loop in C: `sqrt(a*a + b*b)` cimo 0.98, gcc 1.12 ms; glibc `hypot` (gcc) 1.53; cimo's bundled `hypot` 2.19
      (since then glibc's algorithm in cimo, 1.47x faster: re-measure `parity=False`); `sj_hypot2` cimo 2.73, gcc 2.93. Bounds checks 0.25 ms, a branchless max nothing. So the gap is
      the exact `hypot`. **Tried, rejected (2026-11-07):** a certified fast path (`h = sqrt(fl(a*a + b*b))`, the
      residual `s - h*h` from fma splits, `h` or a neighbour picked against the halfway points with 2**-90 * s to
      spare, the exact code otherwise; CPython's result is t correctly rounded unless t is within about 2**-98 * s of a
      halfway point). It was bit-identical on 240,000 pairs and never fell back on 10M random pairs, with a latency of
      16 ns against 42, but throughput didn't improve (about 10 ns a call both), and viewshed got slower: 4.5 against
      3.8 ms (branches or selects), 4.2 as a function of its own. The loop overlaps iterations, so latency doesn't count
      there, and the fast path has as many instructions plus moves between FP and integer registers. Speeding up the
      exact code itself is what's left (fewer instructions, not a shorter chain).
    - ~~**Histogram, 2x behind everyone**~~ **done** (2026-11-07): 1.35 to 0.73 ms (Numba 0.69, gcc 0.62); `guard_facts`.
    - **Float sums with fastmath, 3x** (dot: 0.34 against 1.02 ms): Numba reorders the sum to vectorize it. Plan
      (2026-11-07): `reassoc=True`, separate from `parity` (reordering changes plain `+` everywhere, by more than
      a last bit under cancellation, and with the vector width; `parity=False` never reorders), emitting
      `#pragma omp simd reduction(+:s)` before a loop whose float sums (products, min, max) it lets go out of order,
      and nothing else. **Unblocked (cimo 2026-11-07):** cimo honors the pragma now (four vectors of accumulators;
      `+`, `-`, `*`, and fmin/fmax without it on arm64); re-measure dot with it before adding the flag.
    - **First call, 2 to 8x Numba's** (0.1 to 1 s against 0.04 to 0.12 s, cold, after one warm-up compile each): #19.
    - **Call overhead, 7 to 13x Numba's** (`bench/compare.py`'s per-call table, 2026-11-07): `a + b` on two floats
      829 ns a call, from 1,221 before `__call__` by argument count (#9) (CPython 36, Numba 112, Cython 24); `x[0]`
      of a numpy array 1,775 ns, from 2,199 (CPython 66, Numba 140, Cython 146). Since then cimo's calls skip ctypes (cimo 2026-11-07: a `double, double, long` call
      661 to 92 ns) and the per-signature fast path is in (#9): re-measure the table, then profile what's left.
    - On this CPU (Snapdragon X), LLVM crashes tuning for `oryon-1`, so the script points Numba at a generic arm64 CPU
      with the host's features and says so.
19. **Compile time.** A first call costs seconds, not the 0.1 s the README's `ema` example suggests: 0.53 s
    on Gray-Scott, 0.76 s native and 1.75 s in Pyodide for the LLM demo's forward pass, 3 s for a 200-line
    generated loop and up to 15 s with bounds-check versioning (#16). Every candidate in a generated-code
    workflow compiles once, so this decides what's practical. An optimization pass on the compiler
    itself, measured with `cProfile` on `bench/` and `bench/llm.py` in both runtimes, before and after:
    - Split the time between scramjit (parse, type inference, C generation) and cimo (preprocessor,
      front end, middle end, backend, encoder, link). Record the split here; cimo's share goes to its
      `TODO.md` ("Compile time").
    - In scramjit: generate less C (smaller versioned copies, no dead helpers, one prelude shared across
      compiles), and cache type inference per signature.
    - Across the boundary: cimo's compile of a kernel that reuses the prelude and libm table from a
      previous compile shouldn't redo their work; ask for a per-process cache of parsed headers and the
      prelude, or pass them pre-parsed.
    - Target: under 0.3 s warm for a 100-line function natively, under 1 s in Pyodide.
    - **An edit above a function recompiles it.** Line numbers and the file's path are in the C, so cimo's key changes: Gray-Scott first call 86 ms warm, 562 ms after
      adding one line above it, or renaming the file; a comment that keeps the lines stays warm. **Done here (2026-11-05):** the C's lines count from the function's
      `def` and `#line` names `sj` (or `sj<k>`, the k-th function called) instead of the file, put back for tracebacks (`_failed`), preflight's notes (`py_line`) and
      cimo's messages (`cimo_diagnostic`); moved 3 lines down and renamed, Gray-Scott's first call stayed warm. `tests/test_line_independence.py`.
      cimo's half is done too (2026-11-05: its key leaves out line markers, blank lines and indentation), so `#line` could name the real file again; `__LINE__`
      and `__FILE__` still go into the key wherever they're used.
    - **A warm first call (2026-11-05):** Gray-Scott about 47 ms: 13 generating C, 30 preprocessing it (14 of that the prelude), about 1 loading; `small` about 45, nearly all preprocessing.
      A cold one is 96% cimo (496 ms, 13 generating). A cache keyed on the Python would save only warm time, and can use cimo's public cache export and import (`module.cache_entry()`, `cimo.add_cache_entry()`) to skip the preprocessing.
20. **Split `_compiler.py` into a package.** 4.8k lines holding every lowering in one file. Make
    `scramjit/_compiler/` with one module per concern, in the order the compile runs: `infer` (types),
    `index` (indexing, views, slices), `checks` (bounds checks and versioning), `calls` (`@scram` and C
    callees, inlining), `reduce` (builtins and numpy reductions), `make` (arrays and containers made
    inside a function), `emit` (C text). No behavior change: the suite and `bench/` are the gate, and
    the `docs/py/scramjit` copy is regenerated, not edited.
21. **Adversarial code reading before a public launch.** `_compiler.py` (4.8k lines) and `_dispatch.py`
    were written in days and no one has read them end to end. The parity tests check results; a read
    checks what they can't: a lowering that is right for every tested type combination and wrong for
    one the generator never produces, a bounds check hoisted when it shouldn't be, a fallback that
    silently changes semantics. Read `_compiler.py` by lowering (indexing and views first, then
    bounds-check versioning, then calls), and record what was checked and found here.
22. **Value-range narrowing:** the rest of M2's loop work, using the range facts the compiler already has. Little to win: arrays already have the user's dtype, cimo has no byte or short lanes, and scalars gain nothing, which leaves temporary arrays.
23. ~~**Containers that compose: lists that grow, and any nesting of the types that compile.**~~ **Done (2026-11-07 to
    2026-11-09),** with slices, `in`, sorting, comparisons and sets; the design is under "Containers" in `requirements.md`.
    **Left:** `frozenset`; sets of records or strings; numpy scalars in sets and lists mixing them with Python numbers; CI
    for Pyodide (`tools/run_pyodide_tests.mjs`, run by hand so far).
24. **Preflight's advice: boxed data, and the flags that would help.** Two kinds of note:
    - **Converting costs more than computing.** `preflight --run` splits each function's time into native and
      Python; split off the seconds converting arguments and results too (copying lists in, writing back, boxing the
      return), and say so when they dominate: "`score` spends 85% of its time converting its list of lists: pass a
      numpy array, or move the loop that calls it inside it". Without `--run`, the static shape: a compiled function
      taking a nested list, called from a Python loop.
    - **Flags, measured.** For each hot function under `--run`, compile it again with `parity=False` (and
      `reassoc=True` once it exists: #18), time it on the same calls, and note the gain only past a
      threshold, with what changes ("1.4x with `parity=False`: `math` results may differ in the last bit, float
      division by zero gives inf"). `boundscheck=False` is reported, not recommended: the checks left in each hot
      loop and what dropping them measured, next to a rewrite that lets scramjit prove the index (the histogram's
      `0 <= k < nb`), since without checks a bad index corrupts memory instead of raising. Also from the case study:
      arguments wrapped in `float()`, `int()` or `np.ascontiguousarray()` that make no difference (issue #6).
