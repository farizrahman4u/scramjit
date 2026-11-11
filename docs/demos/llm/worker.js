// Python, numpy, the compiler and the model all run in workers, so the page never blocks. One worker holds the
// weights and writes the story. A second, with no weights, compiles edits and hands over the compiled files.
'use strict';
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const SJ = ['__init__.py', '_compare.py', '_compiler.py', '_dispatch.py', '_errors.py', '_glist.py', '_gset.py', '_interrupt.py', '_lift.py', '_mathlib.py', '_nploops.py', '_preflight.py', '_pysets.py', '_runtime.py', '_types.py', '_webs.py'];
const WEIGHTS = 'https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin';
const TOKENIZER = 'https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin';
const CACHE = 'scramjit-llm-1';
let py, demo;
const text = async p => { const r = await fetch(p); if (!r.ok) throw new Error(p + ': ' + r.status); return r.text(); };
const say = msg => postMessage({ type: 'status', msg });
const breathe = () => new Promise(r => setTimeout(r));
const jsonOf = s => JSON.parse(s);

// Compiled code stays between visits, as a cache of its source: cimo uses a file only for the source it was built
// from. Files live in the browser's cache, with a list of which source made which. Nothing here may stop the demo.
const COMPILED = 'scramjit-llm-compiled-1', LIST = 'https://compiled.invalid/list', KEEP = 8;
const fileUrl = path => 'https://compiled.invalid' + path;
let keptList = null;                                   // { stamp, codes: [{ code, paths }] }, oldest first

async function restoreCompiled() {
  try {
    const stamp = demo.cache_stamp();
    keptList = { stamp, codes: [] };
    const store = await caches.open(COMPILED);
    const r = await store.match(LIST);
    const list = r && await r.json();
    if (list && list.stamp !== stamp) await caches.delete(COMPILED);   // another compiler or Python: start again
    else if (list) {
      for (const k of list.codes) {
        let whole = true;
        for (const path of k.paths) {
          const f = await store.match(fileUrl(path));
          if (!f) { whole = false; continue; }
          py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
          py.FS.writeFile(path, new Uint8Array(await f.arrayBuffer()));
        }
        if (whole) keptList.codes.push(k);
      }
    }
  } catch { keptList = null; }
  demo.cache_changes();
  return keptList ? keptList.codes.map(k => k.code) : [];
}

async function keepCompiled(code) {
  const paths = jsonOf(demo.cache_changes());
  if (!keptList) return;
  try {
    const store = await caches.open(COMPILED);
    for (const path of paths) await store.put(fileUrl(path), new Response(py.FS.readFile(path)));
    const old = keptList.codes.find(k => k.code === code);
    if (old && !paths.length) { keptList.codes = keptList.codes.filter(k => k !== old).concat(old); }   // used again: newest
    else keptList.codes = keptList.codes.filter(k => k !== old).concat({ code, paths: [...new Set([...(old?.paths || []), ...paths])] });
    const gone = keptList.codes.slice(0, -KEEP);
    keptList.codes = keptList.codes.slice(-KEEP);
    const used = new Set(keptList.codes.flatMap(k => k.paths));
    for (const k of gone) for (const path of k.paths) if (!used.has(path)) await store.delete(fileUrl(path));
    await store.put(LIST, new Response(JSON.stringify(keptList)));      // last: it says the files are all there
  } catch { /* out of space, or private browsing: the next visit compiles */ }
}

async function opened() { try { return await caches.open(CACHE); } catch { return null; } }

// A file from the browser's cache, or downloaded with progress and kept for the next visit.
async function download(url, label) {
  const cache = await opened();
  const kept = cache && await cache.match(url).catch(() => null);
  if (kept) return new Uint8Array(await kept.arrayBuffer());
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${label}: download failed (${r.status})`);
  const total = +r.headers.get('content-length') || 0, chunks = [], reader = r.body.getReader();
  let got = 0, shown = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (got - shown > 2e6) { shown = got; say(`Downloading ${label}: ${Math.round(got / 1e6)}${total ? ' of ' + Math.round(total / 1e6) : ''} MB`); postMessage({ type: 'alive' }); }
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  if (cache) await cache.put(url, new Response(out.slice())).catch(() => {});
  return out;
}

async function python() {
  say('Loading Python');
  importScripts(PYODIDE + 'pyodide.js');
  py = await loadPyodide({ indexURL: PYODIDE });
  say('Loading numpy');
  await py.loadPackage('numpy');
  postMessage({ type: 'alive' });
  say('Loading the compiler');
  py.FS.mkdirTree('/home/pyodide/scramjit');
  py.FS.writeFile('/home/pyodide/cimo.py', await text('../../py/cimo.py'));
  for (const f of SJ) py.FS.writeFile('/home/pyodide/scramjit/' + f, await text('../../py/scramjit/' + f));
  py.FS.writeFile('/home/pyodide/lab.py', await text('../../lab/lab.py'));
  py.FS.writeFile('/home/pyodide/demo_py.py', await text('demo_py.py'));
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  await py.runPythonAsync('import demo_py');
  demo = py.pyimport('demo_py');
}

async function boot() {
  const t = performance.now();
  await python();
  const ready = (performance.now() - t) / 1000;
  const kept = await restoreCompiled();
  py.FS.writeFile('/home/pyodide/tokenizer.bin', await download(TOKENIZER, 'the tokenizer'));
  py.FS.writeFile('/home/pyodide/weights.bin', await download(WEIGHTS, 'the model'));
  say('Loading the weights');
  await breathe();
  const info = jsonOf(demo.setup('/home/pyodide/weights.bin', '/home/pyodide/tokenizer.bin'));
  postMessage({ type: 'ready', python: ready, seconds: (performance.now() - t) / 1000, robot: demo.token_of('robot'), kept, ...info });
}

onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'boot') await boot();
    if (m.type === 'bootCompiler') { const t = performance.now(); await python(); postMessage({ type: 'compilerReady', seconds: (performance.now() - t) / 1000 }); }
    if (m.type === 'precompile') {
      await breathe();
      const r = jsonOf(demo.precompile(m.code));
      const files = (r.files || []).map(path => ({ path, data: py.FS.readFile(path) }));
      postMessage({ type: 'precompiled', id: m.id, seconds: r.seconds, error: r.error, files }, files.map(f => f.data.buffer));
    }
    if (m.type === 'precompileHeat') {
      await breathe();
      const r = jsonOf(demo.precompile(m.code, true));
      const files = (r.files || []).map(path => ({ path, data: py.FS.readFile(path) }));
      postMessage({ type: 'heatFiles', code: m.code, error: r.error, files }, files.map(f => f.data.buffer));
    }
    if (m.type === 'heat') {
      await breathe();
      const r = jsonOf(demo.count_lines(m.code || '', m.on));
      postMessage({ type: 'heat', code: m.code, on: m.on, ...r });
      if (m.on && !r.error && !r.stale) await keepCompiled('counted\0' + m.code);
    }
    if (m.type === 'adopt') for (const f of m.files) { py.FS.mkdirTree(f.path.slice(0, f.path.lastIndexOf('/'))); py.FS.writeFile(f.path, f.data); }
    if (m.type === 'compile') {
      await breathe();
      const r = jsonOf(demo.compile_forward(m.code, m.compiled));
      postMessage({ type: 'compiled', id: m.id, ...r });
      if (m.compiled && !r.error) await keepCompiled(m.code);
    }
    if (m.type === 'start') demo.start(m.prompt);
    if (m.type === 'wat') {
      let t;
      try { t = demo.wasm_text(m.code); } catch (err) { t = ';; could not disassemble: ' + err.message; }
      postMessage({ type: 'wat', code: m.code, text: t });
    }
    if (m.type === 'gen') postMessage({ type: 'text', ...jsonOf(demo.step(m.ms)) });
  } catch (err) { postMessage({ type: 'failed', compiler: ['bootCompiler', 'precompile', 'precompileHeat'].includes(m.type), error: err.message }); }
};
