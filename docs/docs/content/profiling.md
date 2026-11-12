# Preflight

Code scramjit can't compile runs as Python, so nothing breaks, but speed can go missing quietly. Preflight shows the state of every decorated function, which lines run as Python and why, what compiling cost, and what to change.

Every function is in one of three states:

| | State | Means |
| --- | --- | --- |
| ✓ | **compiled** | the whole function runs as native code |
| ~ | **partly compiled** | some loops run native, the rest as Python |
| ✗ | **runs as Python** | nothing compiled |

`scramjit` is a command installed with the package; `python -m scramjit` works the same. Every command has `--help`, and `scramjit completion bash` (or `zsh`, `fish`) prints a shell completion script. Colors follow `--color`, `SCRAMJIT_COLOR`, `NO_COLOR` and `FORCE_COLOR`.

## Preflight, before you run

```console
$ scramjit preflight src/
```

reads every `@scram` function under the path (the current directory if you give none), without importing or running anything, and reports its state and, for each line that stops it, the construct, the reason and what to do about it:

```text
src/sim.py
  ✓ ema        compiled
  ✓ grid       compiled
  ✗ summarize  runs as Python
      22 │     names = {}
         │             ^^ dicts aren't compiled unless their keys are constant strings, written where the dict is made (an empty one would gain keys)
         │ hint: if the keys are small integers, use an array indexed by key; otherwise
         │       do this outside the compiled function
      24 │         names[i] = str(a[i])
         │                    ^^^^^^^^^ calls str()
         │ hint: keep text and printing out of the function: return numbers and format
         │       them after the call

2 compiled, 1 runs as Python
```

A line that only fails because an earlier one did isn't listed. `-q` shows only what doesn't compile, and `--json` prints the report for tools (`-o FILE` writes it to a file).

Undecorated functions with a loop that would compile are listed too, as suggestions that don't affect the exit status:

```text
src/stats.py
  → smooth      would compile with @scram (line 10)
  → count_hits  would compile with @scram (line 21)
```

Static analysis assumes their arguments are numbers or arrays, so check with `--run` once decorated. Methods, nested functions, tests and functions with other decorators aren't suggested. `--no-suggest` turns this off.

Annotations are used. Unannotated arguments are assumed to be numbers or arrays, so a fallback caused by the types of an actual call (a `str`, say) needs `--run`.

## Preflight while it runs

```console
$ scramjit preflight --run simulate.py --steps 2000
$ scramjit preflight --run -m pytest tests/
```

runs the script, or a module such as your test suite, and reports what actually happened, with the real argument types. The program keeps stdout; the report goes to stderr:

```text
scramjit 4.21 s run  ████████████░░░░░░░░ 61% native
    function      calls  compile   native   python
  ✓ flood           120   0.14 s   0.22 s        –
  ✓ ema ×2            2   0.07 s   0.02 s        –
  ✗ summarize     2,000        –        –   1.52 s

ema: a arrives as float and int, so it compiles once per type
  hint: annotate it, such as a: float, or convert once before calling
summarize, line 22: dicts aren't compiled unless their keys are constant strings, written where the dict is made (an empty one would gain keys)
  hint: if the keys are small integers, use an array indexed by key; otherwise do this
        outside the compiled function

summarize runs as Python
```

- **native** and **python** are the time each function spent in compiled code and in code that fell back. The native share at the top is native time over the whole run; the rest is your own Python and everything else.
- **compile** is the time spent compiling; it's much shorter when the code came from the [disk cache](caching.md). **×2** means two signatures.
- Functions that weren't called don't appear; run the static mode for those.

Arguments after the script or module are passed to it.

## Which lines ran most

`--counts` compiles copies of your functions that count how often each line runs, and adds the hottest lines to the report:

```console
$ scramjit preflight --counts --run examples/hot_lines.py
```

```text
hottest lines, exact counts; counting slowed this run, so the times above read high
  100,196,159  hot_lines.py:24  a[j + 1] = a[j]
  100,196,159  hot_lines.py:25  moved += 1
  100,196,159  hot_lines.py:26  j -= 1
       21,998  hot_lines.py:21  v = a[i]
       21,998  hot_lines.py:22  j = i - 1
```

Give it a function to count only that one, and to see every line of it, with how long each loop ran and how often each branch was taken:

```console
$ scramjit preflight examples/hot_lines.py:insertion_sort --counts --run examples/hot_lines.py
```

```text
  19 │ ✓           2     moved = 0
  20 │ ✓           2     for i in range(1, len(a)):              10,999.0 per entry
  21 │ ✓      21,998         v = a[i]                            bounds proven
  22 │ ✓      21,998         j = i - 1
  23 │ ✓      21,998         while j >= 0 and a[j] > v:          1 bounds check kept  4,554.8 per entry
  24 │ ✓ 100,196,159             a[j + 1] = a[j]                 2 bounds checks kept
  25 │ ✓ 100,196,159             moved += 1
  26 │ ✓ 100,196,159             j -= 1
  27 │ ✓      21,998         a[j + 1] = v                        1 bounds check kept
```

- A loop's line shows how many times the loop was entered, and **per entry** how many iterations each entry ran on average. An `if` shows how many times it was tested and how often it was **taken**.
- Counts cover native code, including loops compiled on their own in a partly compiled function. Lines that run as Python have none.
- Results don't change, and the compiled copies are cached apart from the normal ones.

In a notebook or the browser, where there's no command line, use `@scram(counts=True)` on the function. `f.preflight()` then shows the column, and `f.line_counts()` returns the numbers: a dict of line to times run, and one of each `if` and loop's line to its kind and how often its body ran.

Use the counts, not the times, from a run with `--counts`. Counting took from 3% to 45% more time on our benchmarks, the most in tight loops full of branches, so it changes which lines look expensive. Plain `--run` doesn't count anything.

Two cases aren't exact. When a line raises, the lines after it in the same block read one too many. And threads running the same function at once can lose some counts, since counters are plain additions; their results are still right.

## Watching a run

`--live` shows the report while the program runs, updating calls and time even in the middle of a long call. With `--counts` it adds the hottest lines:

```console
$ scramjit preflight --live --counts --run simulate.py
```

```text
scramjit 2.3 s  90% native
* settle          1 call     2.04 s native     0.00 s python
     853,290,288  simulate.py:9  x[i] = 0.5 * x[i] + 0.5 * x[i - 1]
     853,290,288  simulate.py:10  if x[i] > 0.25:
         854,145  simulate.py:8  for i in range(1, len(x)):
         280,875  simulate.py:11  t += 1.0
               1  simulate.py:6  t = 0.0
```

`*` marks what's running now. The normal report follows at the end, and the exit status is the same.

- On a terminal the block redraws in place, and clears when your program prints through `sys.stdout` or `sys.stderr`. Output from native libraries that write directly can leave a block behind.
- When stderr is piped, it prints one line every 5 seconds instead.
- With `--json` it writes one JSON object per line, about once a second: `type` is `snapshot` until the last line, which is the `report`. Use `-o run.jsonl` to keep them apart from your program's output.

`--live` on its own costs under 1% and doesn't recompile. Counters can be read during a call because compiled code releases the GIL. In Pyodide nothing can run during a call, so counts move only between calls.

## In CI

Both modes exit non-zero when any function is less than compiled; `--allow-partial` accepts partly compiled ones. The strongest check runs your tests, so every function is checked with the types it really gets:

```console
$ scramjit preflight --run -m pytest
```

## One function

`f.preflight()` prints the function with each line marked native or Python, and what the compiler decided. It works in any REPL or notebook:

```text
>>> ema.preflight()
ema(float64[:], float64[:], float)
✓ compiled · sim.py · compiled in 38 ms
────────────────────────────────────────────────────────────
   5 │   @scram
   6 │   def ema(x, y, a: float):
   7 │ ✓     y[0] = x[0]                               2 bounds checks kept
   8 │ ✓     for i in range(1, len(x)):
   9 │ ✓         y[i] = a * x[i] + (1 - a) * y[i - 1]  bounds hoisted out of the loop
────────────────────────────────────────────────────────────
types: x: float64[:], y: float64[:], a: float, i: int
```

It reports:

- **state per line:** native or Python, and for Python lines, the construct that kept them there;
- **types:** what each variable was inferred as, and where a type was narrowed;
- **bounds checks:** proven unnecessary, hoisted out of the loop, or kept, and what would let a kept one go;
- **loops and the call boundary:** planned; what cimo's optimizer did to each loop isn't available yet, and arrays are always passed in place (a strided one runs as Python).

Pass a signature to report on only that one: `f.preflight(f.signatures[0])`.

### From the command line

Give preflight a function instead of a directory:

```console
$ scramjit preflight src/sim.py:ema                    # static: one assumed signature
$ scramjit preflight src/sim.py:ema --run app.py       # every signature compiled during the run
$ scramjit preflight pkg.sim:ema                       # a dotted module path works too
```

It prints the same report as `ema.preflight()`, with a column of [how often each line ran](#which-lines-ran-most) if you add `--counts`. Without `--run`, argument types come from annotations, and unannotated arguments are assumed to be numbers or arrays. With `--run`, the report covers each signature the function was actually compiled for. Put the function before `--run`: everything after `--run` is the program and its arguments.

## Specializations and the generated C

```python
>>> ema.signatures
[(float64[:], float64[:], float), (float64[:], float64[:], int)]
>>> print(ema.c(ema.signatures[0]))
```

`f.c()` returns the C scramjit generated, with `#line` directives pointing back at your Python. Without an argument it uses the most recent signature. From the command line, without running anything: `scramjit c src/sim.py:ema` (add `--runtime` for a file you can compile on its own).

## Log compiles and fallbacks

```console
$ SCRAMJIT_LOG=1 python simulate.py
scramjit: compiled flood(uint8[:, :], int64[:, :], int, int) in 0.14 s
scramjit: compiled ema(float64[:], float64[:], float) in 0.04 s
scramjit: summarize(float64[:]) runs as Python: line 18: the Dict expression
```

One line per compile or fallback. A function that keeps compiling new signatures is being called with arguments of changing types; annotate them, or convert once before the loop.

## Acting on hints

| Hint | Why it costs | Fix |
| --- | --- | --- |
| a loop stays in Python | it runs at interpreter speed | move the dict, string or object work out of the loop, and keep the loop on numbers and arrays |
| an argument arrives as several types | one compile and one specialization per type | annotate it, such as `a: float` |
| a list is copied in and out | O(n) work on every call | pass a numpy array |
| an array is strided | every index multiplies by a stride | `np.ascontiguousarray` once, outside the loop |
| a bounds check stays in a hot loop | a compare and branch per access | index with the loop variable over `range(len(a))`, or plus a constant or a value fixed in an innermost loop (`range(lo, n, 2)` works too), or `@scram(boundscheck=False)` once verified |
