// The live demo's Python side, in a worker so compiling never blocks the page: Pyodide, numpy,
// cimo, scramjit and lab.py run here. The page sends code and controls; this sends back what each
// stage shows and, one frame at a time, the arrays the page draws.
'use strict';
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const SJ_FILES = ['__init__.py', '_compare.py', '_compiler.py', '_dispatch.py', '_errors.py', '_glist.py', '_gset.py', '_interrupt.py', '_lift.py', '_mathlib.py', '_nploops.py', '_pysets.py', '_runtime.py', '_types.py', '_webs.py'];
let py = null, lab = null, compiled = false, preset = null, ready = false;
let active = true, looping = false, inFlight = false, paced = false, nextCode = null;

async function text(base, path) {
  const r = await fetch(new URL(path, base), { cache: 'no-cache' });
  if (!r.ok) throw new Error(path + ': ' + r.status);
  return r.text();
}
const status = msg => postMessage({ type: 'status', msg });

async function boot(base) {
  status('Loading Pyodide (CPython in WebAssembly, 7 MB)…');
  importScripts(PYODIDE + 'pyodide.js');
  py = await loadPyodide({ indexURL: PYODIDE });
  status('Loading numpy…');
  await py.loadPackage('numpy');
  status('Loading cimo and scramjit (pure Python)…');
  py.FS.mkdirTree('/home/pyodide/scramjit');
  py.FS.writeFile('/home/pyodide/cimo.py', await text(base, 'py/cimo.py'));
  for (const f of SJ_FILES) py.FS.writeFile('/home/pyodide/scramjit/' + f, await text(base, 'py/scramjit/' + f));
  py.FS.writeFile('/home/pyodide/lab.py', await text(base, 'lab/lab.py'));
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  status('Importing…');
  await py.runPythonAsync('import lab');
  lab = py.pyimport('lab');
  ready = true;
  postMessage({ type: 'ready' });
}

// Work in time-boxed steps. Free-running demos (the terrains) keep working and send a frame
// whenever the page has drawn the last; paced ones (a fractal frame, a flock pass, a few sand
// steps per frame) wait for the page. Nothing runs while the demo is out of sight.
function loop() {
  looping = false;
  if (!ready || !active || !preset || (paced && inFlight)) return;
  try { work(); } catch (err) { postMessage({ type: 'loop-error', msg: err.message }); }
}
function work() {
  const r = JSON.parse(lab.step(compiled ? 22 : 14));
  paced = !!r.paced;
  if (!inFlight || r.error) {
    const arrays = {}, transfer = [];
    for (const name of JSON.parse(lab.views())) {
      const a = lab.view(name), b = a.getBuffer();
      const data = b.data.slice();                    // a copy: the page gets its own buffer
      arrays[name] = { data, shape: b.shape.slice() };
      transfer.push(data.buffer);
      b.release(); a.destroy();
    }
    inFlight = true;
    postMessage({ type: 'frame', preset, rate: r.rate, error: r.error, compiled, arrays }, transfer);
  }
  if (!(paced && inFlight)) kick();
}
function kick() { if (!looping) { looping = true; setTimeout(loop, 0); } }

// Code: only the latest matters. A compile can't be interrupted (that needs cross-origin
// isolation), but edits queued behind one collapse into a single compile of the newest code.
function takeCode() {
  const m = nextCode;
  if (!m) return;
  nextCode = null;
  compiled = m.compiled;
  preset = m.preset;
  const info = lab.set_code(m.preset, m.src, m.compiled);
  postMessage({ type: 'info', id: m.id, info: JSON.parse(info) });
  paced = false; inFlight = false;
  kick();
}

onmessage = async e => {
  const m = e.data;
  try {
    if (m.type === 'boot') return await boot(m.base);
    if (!ready) return;
    if (m.type === 'ack') { inFlight = false; kick(); }
    else if (m.type === 'active') { active = m.on; kick(); }
    else if (m.type === 'code') { nextCode = m; setTimeout(takeCode, 0); }
    else if (m.type === 'reset') { lab.reset(m.seed ?? 7, m.preset ?? null); if (m.preset) preset = m.preset; kick(); }
    else if (m.type === 'paint') lab.paint(m.x, m.y, m.r, m.kind);
    else if (m.type === 'pointer') lab.pointer(m.x, m.y);
    else if (m.type === 'target') lab.fractal_target(m.x, m.y);
    else if (m.type === 'wasm') {
      let t;
      try { t = lab.wasm_text(); } catch (err) { t = ';; could not disassemble: ' + err.message; }
      postMessage({ type: 'wat', text: t });
    }
  } catch (err) {
    postMessage({ type: 'fatal', msg: err.message });
  }
};
