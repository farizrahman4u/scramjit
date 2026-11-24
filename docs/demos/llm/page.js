// The page: the story, the scramjit switch, the tabs and the raster (as in the home page's demos), the preset edits.
'use strict';
const ICON = {"pause": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M8 5v14M16 5v14\"/></svg>", "play": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M7 4.5v15l12-7.5z\"/></svg>"};
const CHIP = {"short": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><circle cx=\"12\" cy=\"13\" r=\"8\"/><path d=\"M12 9v4l2 2M9 2h6\"/></svg>", "none": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"m3 3 18 18M10.6 5.1A10 10 0 0 1 22 12a14 14 0 0 1-3 3.6M6.6 6.6A14 14 0 0 0 2 12a10 10 0 0 0 13.4 5.4\"/></svg>", "flat": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M4 9h16M4 15h16\"/></svg>", "rope": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5\"/></svg>", "layers": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"m12 2 10 5-10 5L2 7z\"/><path d=\"m2 12 10 5 10-5M2 17l10 5 10-5\"/></svg>", "hot": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M12 22c4 0 7-3 7-7 0-5-4-6-5-11-3 2-5 5-4 8-2-1-3-3-3-3s-2 3-2 6c0 4 3 7 7 7z\"/></svg>", "cold": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M12 2v20M2 12h20M5 5l14 14M19 5 5 19\"/></svg>", "robot": "<svg class=\"ic\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><rect x=\"4\" y=\"8\" width=\"16\" height=\"12\" rx=\"2\"/><path d=\"M12 8V4M9 13v2M15 13v2M2 14h2M20 14h2\"/></svg>"};
const $ = id => document.getElementById(id);
const TYPING_MS = 700, BOOT_MS = 300000, COMPILE_MS = 60000, GEN_MS = 30000, PLAIN_MS = 90000, CHUNK_MS = 60, MAX_CHARS = 30000;
const ATT = 'range(pos + 1)', TEMP = 'logits[i] /= 0.8', MARK = '    # temperature\n';
const RM = matchMedia('(prefers-reduced-motion:reduce)').matches;
let robot = -1;
const EDITS = [
  { key: 'short', group: 'memory', label: 'Remember only 8 tokens', from: ATT, to: 'range(max(0, pos - 8), pos + 1)' },
  { key: 'none', group: 'memory', label: 'No memory', from: ATT, to: 'range(pos, pos + 1)' },
  { key: 'flat', label: 'Flat attention', from: 'att[t] = math.exp(att[t] - top)', to: 'att[t] = 1.0' },
  { key: 'rope', label: 'No positions', from: 'for i in range(0, dim, 2):', to: 'for i in range(0, 0, 2):' },
  { key: 'layers', label: 'Drop 2 layers', from: 'for l in range(n_layers):', to: 'for l in range(n_layers - 2):' },
  { key: 'hot', group: 'temp', label: 'Hotter', from: TEMP, to: 'logits[i] /= 1.6' },
  { key: 'cold', group: 'temp', label: 'Colder', from: TEMP, to: 'logits[i] /= 0.3' },
  { key: 'robot', label: 'Obsessed with robots', from: MARK, to: () => `    logits[${robot}] += 8.0  # "robot"\n${MARK}` },
];
const toText = e => typeof e.to === 'function' ? e.to() : e.to;

const root = $('demo');
let worker, editor, viewer, watchdog, original = '', lastGood = '', previousGood = '', id = 0, pendingLabel = '';
let ready = false, running = false, asking = false, compiling = false, booted = false, restarting = false;
let compiled = false, runningPlain = true, nudged = false, startedAt = 0, stage = 'python', info = {}, lastCompile = 5000, barTimer = null;
let tokens = 0, recent = [], version = 0, span = null, pyRate = 0;
let compiler = null, compilerUp = false, compilerDog, background = null, want = null, editTimer = null, typing = false, carryIntro = false, barOn = false;
const known = new Set(), infos = new Map();
let heatPrompted = false, heatPrompt;
let heatWanted = false, heatFor = null, heatAsked = null, heatDog, heatPainted = 0, heatLines = [];   // the heatmap
let sharedWaiting = false;                             // the editor holds code from a link: nothing runs it until asked
// A second Python uses memory; a machine that can't spare it compiles in the writing worker, pausing the story.
const roomy = () => (navigator.deviceMemory || 8) >= 4 && (navigator.hardwareConcurrency || 4) >= 4 && !/Mobi/.test(navigator.userAgent);
let firstToken = false;
const say = m => { $('status').textContent = m; if (!firstToken) $('waittext').textContent = m; };
const waitStops = () => { $('wait').classList.add('stopped'); };
// plain Python can take seconds per token, so something on screen has to say it's still working
const showWorking = () => { $('story').classList.toggle('slow', firstToken && running && runningPlain); };
const hidden = () => document.hidden;
const state = o => { window.demoState = { ...window.demoState, ...o }; };

function arm(ms) { clearTimeout(watchdog); watchdog = setTimeout(hung, ms); }
function calm() { clearTimeout(watchdog); }
function showError(text) { $('err').textContent = text; $('err').style.display = text ? 'block' : 'none'; }
function controls(on) {
  for (const b of ['reset', 'again', 'pause', 'sj', 'runshared']) $(b).disabled = !on;
  document.querySelectorAll('.chip').forEach(c => { c.disabled = !on; });
}
function note(t, busy) { $('note').textContent = t; $('note').classList.toggle('busy', !!busy); }
// a progress bar while scramjit compiles, paced by the last compile's time
function progress(on, ms) {
  const bar = $('progress'), fill = bar.firstElementChild;
  clearTimeout(barTimer);
  barOn = on;
  if (on) {
    bar.classList.add('on');
    fill.style.transition = 'none'; fill.style.width = '0%'; void fill.offsetWidth;
    fill.style.transition = `width ${(ms ?? lastCompile) * 1.1}ms cubic-bezier(.15,.6,.3,1)`;
    fill.style.width = '92%';
  } else {
    fill.style.transition = 'width .18s ease-out'; fill.style.width = '100%';
    barTimer = setTimeout(() => bar.classList.remove('on'), 260);
  }
}
// stops at once, without filling up as if the compile had finished
function cancelBar() {
  const bar = $('progress'), fill = bar.firstElementChild;
  clearTimeout(barTimer);
  barOn = false;
  bar.classList.remove('on');
  fill.style.transition = 'none'; fill.style.width = '0%';
  state({ barCancels: (window.demoState?.barCancels || 0) + 1 });
}

// ---- the workers
function hung() {
  worker.terminate();
  showError(booted ? 'That code stopped responding, so the tab stopped it and went back to the last code that worked. An infinite loop?' : 'Loading took too long. Reload the page to try again.');
  state({ hung: (window.demoState?.hung || 0) + 1 });
  asking = compiling = running = ready = false;
  want = background = null; dropCompiler(); clearTimeout(editTimer);
  controls(false); progress(false); note('');
  if (!booted) { say('Stopped'); return waitStops(); }
  say('Restarting');
  restarting = lastGood || original; known.clear();
  heatFor = heatAsked = null; clearHeat();
  start();
}
function start() { worker = new Worker('worker.js'); worker.onmessage = onMessage; arm(BOOT_MS); worker.postMessage({ type: 'boot' }); }
function startCompiler() {
  if (compiler || !roomy()) return;
  compiler = new Worker('worker.js'); compilerUp = false;
  compiler.onmessage = onCompiler;
  compiler.onerror = dropCompiler;
  compiler.postMessage({ type: 'bootCompiler' });
}
function dropCompiler() { clearTimeout(compilerDog); if (compiler) compiler.terminate(); compiler = null; compilerUp = false; }
function onCompiler({ data: m }) {
  if (m.type === 'compilerReady') { compilerUp = true; state({ compilerUp: m.seconds }); pump(); }
  if (m.type === 'heatEntries') {
    if (!heatWanted || m.code !== heatAsked) return;
    if (m.error) return heatFailed(m.error);
    worker.postMessage({ type: 'adopt', entries: m.entries }, m.entries.map(e => e.data.buffer));
    worker.postMessage({ type: 'heat', code: m.code, on: true });
  }
  if (m.type === 'failed') {                           // no second worker: compile in the writing one
    const b = background;
    dropCompiler(); background = null;
    if (b && !want && !b.stale) want = { code: b.code, name: b.name, intro: b.intro, compiled: b.compiled };
    pump();
  }
  if (m.type === 'precompiled' && background && m.id === background.id) {
    clearTimeout(compilerDog);
    const b = background;
    background = null;
    if (want || b.stale) { state({ dropped: (window.demoState?.dropped || 0) + 1 }); return pump(); }   // overtaken by newer code
    if (m.error) return rejected(m.error, b.intro);
    worker.postMessage({ type: 'adopt', entries: m.entries }, m.entries.map(e => e.data.buffer));
    swapIn(b.code, b.name, b.intro, m.seconds, b.compiled);
  }
}
function rejected(error, intro) {
  compiling = false;
  state({ compiles: (window.demoState?.compiles || 0) + 1, compile: null, error });
  if (want || typing) return pump();                   // already out of date: say nothing about it
  progress(false); note('');
  showError(error);
  if (intro) setSwitch(false);
  say(`That code did not ${compiled ? 'compile' : 'load'}. The story is still running the last code that worked.`);
  ask();
}
function compilerHung() {
  const intro = background?.intro;
  dropCompiler(); background = null;
  state({ hung: (window.demoState?.hung || 0) + 1, compiles: (window.demoState?.compiles || 0) + 1, compile: null });
  startCompiler();
  if (want || typing) return pump();
  progress(false); note('');
  if (intro) setSwitch(false);
  showError(`That code didn't finish compiling in ${COMPILE_MS / 1000} s, so it was stopped. The story is still running the last code that worked.`);
  say('Stopped that compile');
}
function onMessage({ data: m }) {
  if (m.type === 'status') say(m.msg);
  if (m.type === 'alive') arm(BOOT_MS);
  if (m.type === 'failed') { calm(); showError(m.error); progress(false); note(''); say(booted ? 'Stopped: see the error beside the editor.' : 'Could not start: ' + m.error); if (!firstToken) waitStops(); asking = compiling = false; state({ failed: m.error }); }
  if (m.type === 'ready') {
    calm(); booted = true; robot = m.robot;
    state({ python: m.python, boot: m.seconds, keptCodes: (m.kept || []).length });
    for (const code of m.kept || []) known.add(code);    // compiled on an earlier visit
    buildChips();
    if (compiled) { note('compiling…', true); progress(true); }
    swapIn(restarting || (sharedWaiting ? original : editor.getValue()), '', false, undefined, compiled);
  }
  if (m.type === 'compiled') swapped(m);
  if (m.type === 'heat' && m.on && m.code === heatAsked) {
    clearTimeout(heatDog);
    if (m.error) heatFailed(m.error);
    else if (m.stale) heatAsked = null;                // swapped meanwhile: swapped() asks again
    else { heatFor = m.code; note(''); say('Counting every line. The story runs a little slower while this is on.'); state({ heatOn: true }); }
  }
  if (m.type === 'text') gotText(m);
  if (m.type === 'wat') {
    const v = infos.get(m.code);
    if (v) v.wat = m.text;
    if (stage === 'wasm' && info.code === m.code) view(m.text, 'wast');
    state({ wat: m.text.length });
  }
}

// ---- code: as plain Python, or compiled in the background and swapped in
// Only the newest code matters: a request replaces any that's waiting, and a compile it overtakes is thrown
// away. A compile that has outlived what one should take is stopped, since it may never finish: the second
// worker can be killed and restarted while the story keeps writing.
function request(code, name, intro) {
  clearTimeout(editTimer);
  typing = false;
  if (!ready) return;
  if (sharedWaiting) code = original;                  // whatever asked, code from a link isn't run without saying so
  showError('');
  if (compiled && (want?.intro || background?.intro || carryIntro)) intro = true;
  carryIntro = false;
  want = { code, name, intro, compiled };
  if (compiled && !known.has(code) && !(code === lastGood && !runningPlain)) {
    note('compiling…', true);
    progress(true, lastCompile + (background ? Math.max(0, background.at + lastCompile - performance.now()) : 0));   // the old compile has to end first
  }
  if (background) {
    clearTimeout(request.limit);
    const running = background;
    request.limit = setTimeout(() => {
      if (background !== running || !want) return;
      state({ cancelled: (window.demoState?.cancelled || 0) + 1 });
      dropCompiler(); background = null;
      startCompiler();
      pump();
    }, Math.max(0, running.at + Math.max(3000, lastCompile * 1.5) - performance.now()));
  }
  pump();
}
function pump() {
  if (!ready || !want || background || compiling) return;
  const w = want, first = !known.has(w.code);
  if (w.code === lastGood && w.compiled === !runningPlain) {   // back where it already is
    want = null; if (barOn) cancelBar();
    note(w.compiled && $('compile').textContent !== 'cached' ? $('compile').textContent : '');
    say('Same story, same code.');
    return;
  }
  if (w.compiled && first) { note('compiling…', true); if (!barOn) progress(true); }
  if (w.compiled && first && compiler && !compilerUp) return;  // the second worker is on its way: compilerReady calls here
  want = null;
  if (!w.compiled) return swapIn(w.code, w.name, false, undefined, false);
  if (first && compilerUp) {
    say('Compiling; the story keeps writing');
    background = { id: ++id, code: w.code, name: w.name, intro: w.intro, compiled: true, at: performance.now() };
    clearTimeout(compilerDog); compilerDog = setTimeout(compilerHung, COMPILE_MS);
    compiler.postMessage({ type: 'precompile', id, code: w.code });
    state({ tokensAtCompile: tokens });
    return;
  }
  say(first ? 'Compiling; the story waits here' : 'Swapping');
  swapIn(w.code, w.name, w.intro, undefined, true);
}
function swapIn(code, name, intro, seconds, isCompiled) {
  compiling = true; pendingLabel = name;
  swapIn.pending = { code, intro, seconds, compiled: isCompiled };
  arm(COMPILE_MS);
  worker.postMessage({ type: 'compile', id: ++id, code, compiled: isCompiled });
}
function swapped(m) {
  calm();
  compiling = false;
  const p = swapIn.pending, took = p.seconds ?? m.seconds;
  controls(true);
  if (!want && !typing) progress(false);
  state({ compile: p.compiled ? took : null, pause: m.seconds, compiles: (window.demoState?.compiles || 0) + 1, error: m.error || null, writtenDuring: tokens - (window.demoState?.tokensAtCompile ?? tokens), mode: p.compiled ? 'compiled' : 'python' });
  if (m.error) return rejected(m.error, p.intro);
  previousGood = lastGood; lastGood = p.code;
  if (runningPlain === p.compiled) recent = [];
  runningPlain = !p.compiled;
  heatFor = heatAsked = null; clearHeat();
  $('heat').disabled = runningPlain;
  if (heatWanted && p.compiled) startHeat();
  const fresh = !m.cached;
  if (p.compiled) {
    known.add(p.code);
    if (fresh) lastCompile = took * 1000;
    if (m.c) infos.set(p.code, { code: p.code, c: m.c, tree: m.tree });
    info = infos.get(p.code) || info;
    $('compile').textContent = fresh ? took.toFixed(1) + ' s' : 'cached';
    if (!typing) note(fresh ? took.toFixed(1) + ' s' : '');
  } else { if (!typing) note(''); $('rate').textContent = '–'; }
  root.classList.toggle('fast', p.compiled);
  if (!ready) {
    ready = true; running = true; startedAt = performance.now();
    $('mode').hidden = false;
    if (restarting) { restarting = false; say('Restarted on the last code that worked'); } else say(p.compiled ? 'Writing' : 'Writing, as plain Python');
    if (!firstToken) $('waittext').textContent = p.compiled ? 'Writing the first token' : 'Writing the first token as plain Python. It stays this slow until you turn on scramjit.';
    worker.postMessage({ type: 'start', prompt: $('prompt').value });
    state({ firstAt: performance.now() });
    setTimeout(startCompiler, 1500);
  } else {
    version = (version % 4) + 1;
    const when = fresh ? took.toFixed(1) + ' s' : 'cached';
    say(!p.compiled ? 'Swapped in as plain Python. Same story, new code.' : !fresh ? 'Swapped in. Same story, new code.' : p.seconds !== undefined ? `Compiled in ${when} while the story kept writing; the swap paused it ${m.seconds.toFixed(1)} s.` : `Compiled and swapped in, ${when}. Same story, new code.`);
    newSpan(p.compiled ? `${pendingLabel || 'your edit'} · ${when}` : pendingLabel || 'your edit');
  }
  showWorking();
  if (typing && p.intro) carryIntro = true;
  else if (want && p.intro && want.compiled) want.intro = true;
  else if (p.compiled && p.intro) sweep();
  else if (p.compiled) { if (stage !== 'python') showStage(stage); } else if (stage !== 'python') showStage('python');
  ask();
  pump();
}
function setSwitch(on) {
  compiled = on;
  $('sj').setAttribute('aria-checked', on);
  $('sj').querySelector('.lab-state').textContent = on ? 'on' : 'off';
  root.classList.toggle('sj', on);
}
function setMode(on) {
  if (on === compiled || !ready) return;
  setSwitch(on);
  root.classList.remove('nudge'); nudged = true;
  if (!on) { showStage('python'); endPrompt(); }
  request(editor.getValue(), on ? 'scramjit on' : 'scramjit off', on);
}

// ---- the tabs, as in the home page's demos
function showStage(s) {
  stage = s;
  document.querySelectorAll('.lab-tab').forEach(c => c.classList.toggle('open', c.dataset.s === s));
  $('codepane').hidden = s !== 'python';
  $('panel').hidden = s !== 'c' && s !== 'wasm';
  $('tree').hidden = s !== 'tree';
  if (s === 'python') { editor.refresh(); return; }
  if (s === 'tree') { renderTree(); return; }
  if (s === 'c') return view(info.c || '', 'text/x-csrc');
  if (info.wat) return view(info.wat, 'wast');
  view(';; disassembling…', 'wast');
  worker.postMessage({ type: 'wat', code: info.code });
}
function view(text, mode) {
  if (viewer.getValue() !== text || viewer.getOption('mode') !== mode) {
    viewer.setOption('mode', mode);
    viewer.setValue(text);
    viewer.scrollTo(0, 0);
  }
  viewer.refresh();
}
const esc = t => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const KIND = { FunctionDef: 'def', For: 'loop', While: 'loop', If: 'cond', Else: 'cond' };
function tyClass(t) { return t.includes('[') ? 'arr' : /float/.test(t) ? 'flt' : /int|bool/.test(t) ? 'int' : ''; }
function badge(name, t) { return `<span class="lab-ty ${tyClass(t)}">${name ? `<b>${esc(name)}</b>: ` : ''}${esc(t)}</span>`; }
function exprHTML(e) {
  const kids = e.c ? `<ul>${e.c.map(exprHTML).join('')}</ul>` : '';
  return `<li><div class="lab-row"><span class="lab-k ex">${esc(e.k)}</span><span class="lab-t">${esc(e.t)}</span>${e.ty ? badge('', e.ty) : ''}</div>${kids}</li>`;
}
function stmtHTML(n) {
  const x = n.e ? ' x' : '';
  const kids = n.c ? `<ul>${n.c.map(stmtHTML).join('')}</ul>` : '';
  const ex = n.e ? `<div class="lab-ex" hidden><ul>${n.e.map(exprHTML).join('')}</ul></div>` : '';
  return `<li><div class="lab-row${x}"><span class="lab-caret">${n.e ? '▸' : ''}</span><span class="lab-k ${KIND[n.k] || ''}">${esc(n.k)}</span>` +
    `<span class="lab-t">${esc(n.t)}</span>${(n.b || []).map(b => badge(b[0], b[1])).join('')}</div>${ex}${kids}</li>`;
}
function renderTree() {
  const el = $('tree');
  if (!info.tree || el.dataset.for === info.code) return;
  el.dataset.for = info.code;
  el.innerHTML = `<p class="lab-tree-head">The function's syntax tree, statement by statement, with the type scramjit inferred for each name it binds. Click a statement for its expressions.</p><ul>${stmtHTML(info.tree)}</ul>`;
}
// the raster line sweeps down the editor; the C replaces the Python line by line, then the C tab opens
function sweep() {
  const pane = $('pane');
  editor.scrollTo(0, 0);
  $('codepane').hidden = false; $('panel').hidden = false; $('tree').hidden = true;
  view(info.c, 'text/x-csrc');
  viewer.scrollTo(0, 0);
  pane.style.setProperty('--p', 0);
  pane.classList.add('sweep');
  state({ swept: (window.demoState?.swept || 0) + 1 });
  const t0 = performance.now(), d = RM ? 0 : 1500;
  (function step(now) {
    const u = d ? Math.min(1, (now - t0) / d) : 1, e = u < .5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u);
    pane.style.setProperty('--p', e.toFixed(4));
    if (u < 1 && compiled) requestAnimationFrame(step);
    else { pane.classList.remove('sweep'); showStage('python'); if (compiled) promptHeat(); }
  })(t0);
}

// ---- the story
function ask() {
  if (!ready || !running || asking || compiling || hidden()) return;
  asking = true; arm(runningPlain ? PLAIN_MS : GEN_MS);
  worker.postMessage({ type: 'gen', ms: CHUNK_MS });
}
function newSpan(markText) {
  const box = $('story');
  if (markText) { const b = document.createElement('mark'); b.textContent = markText; box.append(b); }
  span = document.createElement('span');
  if (version) span.className = 'v' + version;
  box.append(span);
}
function gotText(m) {
  if (!compiling) calm();
  asking = false;
  if (m.lines) paintHeat(m.lines);
  if (m.text.length) {
    if (!firstToken) { firstToken = true; $('wait').hidden = true; state({ firstToken: true }); }
    write(m.text);
  }
  showWorking();
  tokens += m.tokens;
  recent.push([m.tokens, m.seconds]);
  if (recent.length > 24) recent.shift();
  const n = recent.reduce((a, r) => a + r[0], 0), s = recent.reduce((a, r) => a + r[1], 0), rate = s ? n / s : 0;
  $('rate').textContent = rate >= 10 ? Math.round(rate) : rate.toFixed(2);
  if (runningPlain) pyRate = rate;
  $('times').textContent = !runningPlain && pyRate ? Math.round(rate / pyRate) + '×' : '–';
  $('tokens').textContent = tokens.toLocaleString();
  state({ tokens, rate, pyRate });
  if (!compiled && !nudged && performance.now() - startedAt > 4000) { root.classList.add('nudge'); nudged = true; state({ nudged: true }); }
  if (m.error) {
    showError(m.error + '\n\nThat edit failed partway through the story, so it went back to the code before it and started a new story.');
    say('Went back to the previous code');
    lastGood = previousGood || original; state({ runtimeError: m.error });
    version = 0; newSpan('back to previous code');
  }
  ask();
}
function write(pieces) {
  const box = $('story'), follow = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  if (!span) newSpan();
  let t = pieces.join('');
  const before = box.textContent;
  if (!before || before.endsWith('\n')) t = t.replace(/^ +/, '');
  t = t.replace(/\n\n +/g, '\n\n');
  span.append(t);
  while (box.textContent.length > MAX_CHARS && box.firstChild !== span) box.firstChild.remove();
  if (follow) box.scrollTop = box.scrollHeight;
  state({ tail: box.textContent.slice(-280) });
}

// ---- preset edits: only the characters that differ change, and the editor shows where
let flashRun = 0;
function differ(a, b) {
  let p = 0, q = 0;
  while (p < a.length && p < b.length && a[p] === b[p]) p++;
  while (q < a.length - p && q < b.length - p && a[a.length - 1 - q] === b[b.length - 1 - q]) q++;
  return { p, q };
}
function replaceEvery(a, b, out) {
  const text = editor.getValue(), { p, q } = differ(a, b), put = b.slice(p, b.length - q);
  let at = text.lastIndexOf(a);
  while (at >= 0) {
    const from = editor.posFromIndex(at + p), to = editor.posFromIndex(at + a.length - q);
    editor.replaceRange(put, from, to, '+edit');
    out.lines.push(editor.getLineHandle(from.line));
    if (put) out.marks.push(editor.markText(from, editor.posFromIndex(editor.indexFromPos(from) + put.length), { className: 'edit-mark' }));
    at = at ? text.lastIndexOf(a, at - 1) : -1;
  }
}
function glideTo(line, done) {
  const box = editor.getScrollerElement(), start = editor.getScrollInfo().top;
  const end = Math.max(0, editor.heightAtLine(line, 'local') - box.clientHeight * 0.38), t0 = performance.now(), d = RM ? 0 : 340;
  (function step(now) {
    const u = d ? Math.min(1, (now - t0) / d) : 1, e = 1 - (1 - u) * (1 - u);
    editor.scrollTo(null, start + (end - start) * e);
    if (u < 1) requestAnimationFrame(step); else if (done) done();
  })(t0);
}
function flash(found) {
  const mine = ++flashRun, lines = found.lines.map(h => editor.getLineNumber(h)).filter(l => l !== null).sort((a, b) => a - b);
  if (!lines.length) return;
  state({ flashed: { lines, marks: found.marks.length } });
  // side by side, the code may sit partly below the window; stacked on a phone, stay with the story
  if (matchMedia('(min-width:1001px)').matches) $('pane').scrollIntoView({ block: 'nearest', behavior: RM ? 'auto' : 'smooth' });
  const pulse = line => {
    const h = editor.getLineHandle(line);
    editor.removeLineClass(h, 'background', 'pulse'); editor.removeLineClass(h, 'gutter', 'pulse-g');
    void editor.getWrapperElement().offsetWidth;
    editor.addLineClass(h, 'background', 'pulse'); editor.addLineClass(h, 'gutter', 'pulse-g');
    setTimeout(() => { editor.removeLineClass(h, 'background', 'pulse'); editor.removeLineClass(h, 'gutter', 'pulse-g'); }, 1800);
  };
  let i = 0;
  (function next() {
    if (mine !== flashRun || i >= lines.length) return;
    const line = lines[i++];
    glideTo(line, () => { if (mine === flashRun) { pulse(line); setTimeout(next, RM ? 0 : 420); } });
  })();
  setTimeout(() => found.marks.forEach(m => m.clear()), 2700 + lines.length * 760);
}
const isOn = e => editor.getValue().includes(toText(e));
function toggle(e) {
  showStage('python');
  if (sharedWaiting) return showError('Run the code from the link, or go back to the original, before trying an edit.');
  const found = { lines: [], marks: [] }, on = isOn(e), to = toText(e);
  showError('');
  if (on) {
    replaceEvery(to, e.from, found);
  } else {
    const mates = EDITS.filter(o => o.group && o.group === e.group && isOn(o));
    const check = mates.reduce((code, o) => code.split(toText(o)).join(o.from), editor.getValue());
    if (!check.includes(e.from)) return showError(`"${e.label}" changes a line that's no longer there. Press Reset, then try it again.`);
    for (const o of mates) replaceEvery(toText(o), o.from, { lines: [], marks: [] });
    replaceEvery(e.from, to, found);
  }
  showChips();
  changedLines();
  flash(found);
  request(editor.getValue(), (on ? 'undo: ' : '') + e.label.toLowerCase());
}
function showChips() {
  document.querySelectorAll('.chip').forEach(c => { const e = EDITS.find(x => x.key === c.dataset.key); c.setAttribute('aria-pressed', robot >= 0 || e.key !== 'robot' ? isOn(e) : false); });
}
function buildChips() {
  if ($('chips').children.length) return;
  for (const e of EDITS) {
    if (e.key === 'robot' && robot < 0) continue;
    const b = document.createElement('button');
    b.className = 'chip'; b.type = 'button'; b.innerHTML = CHIP[e.key]; b.append(e.label); b.dataset.key = e.key; b.disabled = true; b.onclick = () => toggle(e);
    $('chips').append(b);
  }
  showChips();
}
function changedLines() {
  const orig = new Set(original.split('\n'));
  editor.operation(() => { for (let i = 0; i < editor.lineCount(); i++) { const c = !orig.has(editor.getLine(i)); editor[c ? 'addLineClass' : 'removeLineClass'](i, 'background', 'changed'); } });
}
// ---- controls
$('reset').onclick = () => { showStage('python'); endShared(); editor.setValue(original); showChips(); request(original, 'reset'); };
// ---- a link to the code: it travels in the part of the URL no server sees, compressed
const toLink = b => btoa(String.fromCharCode(...b)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const fromLink = s => Uint8Array.from(atob(s.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0));
async function packed(code) {
  return toLink(new Uint8Array(await new Response(new Blob([code]).stream().pipeThrough(new CompressionStream('deflate-raw'))).arrayBuffer()));
}
async function unpacked(text) {
  const reader = new Blob([fromLink(text)]).stream().pipeThrough(new DecompressionStream('deflate-raw')).getReader(), parts = [];
  for (let got = 0; ;) {
    const { done, value } = await reader.read();
    if (done) return new TextDecoder('utf-8', { fatal: true }).decode(await new Blob(parts).arrayBuffer());
    if ((got += value.length) > 1e6) throw new Error('too long');
    parts.push(value);
  }
}
function endShared() { sharedWaiting = false; $('shared').hidden = true; }
function clearLink() { history.replaceState(null, '', location.pathname + location.search); }
// ---- the heatmap: a second compiled copy that counts its lines takes over writing while this is on
function startHeat() {
  const code = lastGood;
  if (!heatWanted || runningPlain || !code || heatAsked === code) return;
  heatAsked = code;
  note('counting…', true);
  clearTimeout(heatDog);
  heatDog = setTimeout(() => heatFailed(`The copy that counts lines didn't compile in ${COMPILE_MS / 1000} s.`), COMPILE_MS);
  if (compiler && compilerUp) compiler.postMessage({ type: 'precompileHeat', code });
  else worker.postMessage({ type: 'heat', code, on: true });
}
function heatFailed(error) {
  clearTimeout(heatDog);
  setHeat(false);
  showError(error);
}
function clearHeat() {
  editor.operation(() => {
    for (const [h, cls] of heatLines) { editor.removeLineClass(h, 'background', cls); editor.removeLineClass(h, 'gutter', 'heat-lit'); }
    heatLines = [];
    editor.clearGutter('heat');
  });
  editor.getGutterElement().querySelector('.heat-strip')?.remove();
}
// once, after the first compile: the button is easy to miss, and this is what to look at next
function promptHeat() {
  if (heatPrompted || heatWanted) return;
  heatPrompted = true;
  root.classList.add('heatnudge');
  state({ heatPrompted: true });
  heatPrompt = setTimeout(endPrompt, 15000);
}
function endPrompt() { clearTimeout(heatPrompt); root.classList.remove('heatnudge'); }
// cold to hot, as a line's share of the hottest line
const HEAT = [[0, [37, 99, 235]], [.35, [124, 58, 237]], [.7, [234, 88, 12]], [1, [220, 38, 38]]];
function heatColor(u) {
  let i = 1;
  while (i < HEAT.length - 1 && u > HEAT[i][0]) i++;
  const [a, ca] = HEAT[i - 1], [b, cb] = HEAT[i], k = (u - a) / (b - a);
  return `rgb(${ca.map((c, n) => Math.round(c + (cb[n] - c) * k)).join()})`;
}
// one gradient down the whole column, with a stop at the middle of each line, so that lines blend into each other
function paintStrip(lines, top) {
  const column = editor.getGutterElement().querySelector('.CodeMirror-linenumbers');
  if (!column) return;
  let strip = column.querySelector('.heat-strip');
  if (!strip) { strip = document.createElement('div'); strip.className = 'heat-strip'; column.prepend(strip); }
  const half = editor.defaultTextHeight() / 2, stops = [], span = Math.log(top + 1);   // counts span nine orders of magnitude
  for (let i = 0; i < editor.lineCount(); i++) {
    const v = lines[i + 1];
    stops.push(`${v ? heatColor(Math.log(v + 1) / span) : 'transparent'} ${(editor.heightAtLine(i, 'local') + half).toFixed(1)}px`);
  }
  strip.style.background = `linear-gradient(to bottom,${stops.join()})`;
}
function setHeat(on) {
  heatWanted = on;
  endPrompt();
  $('heat').setAttribute('aria-pressed', on);
  editor.setOption('gutters', on ? ['heat', 'CodeMirror-linenumbers'] : ['CodeMirror-linenumbers']);
  if (on) { showStage('python'); return startHeat(); }
  clearTimeout(heatDog);
  heatFor = heatAsked = null;
  clearHeat();
  note('');
  worker.postMessage({ type: 'heat', on: false });
  say('Writing');
}
const shortCount = v => v >= 1e9 ? (v / 1e9).toFixed(1) + 'B' : v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e4 ? Math.round(v / 1e3) + 'K' : String(v);
function paintHeat(lines) {
  const now = performance.now();
  if (!heatWanted || now - heatPainted < 250 || heatFor !== editor.getValue()) return;
  heatPainted = now;
  const top = Math.max(1, ...Object.values(lines));
  clearHeat();
  editor.operation(() => {
    for (const [line, v] of Object.entries(lines)) {
      const h = editor.getLineHandle(+line - 1);
      if (!h || !v) continue;
      const cls = 'h' + Math.max(1, Math.ceil(5 * Math.sqrt(v / top)));
      editor.addLineClass(h, 'background', cls);
      editor.addLineClass(h, 'gutter', 'heat-lit');
      heatLines.push([h, cls]);
      const n = document.createElement('div');
      n.className = 'heat-n'; n.textContent = shortCount(v);
      editor.setGutterMarker(h, 'heat', n);
    }
  });
  paintStrip(lines, top);
  state({ heatTop: top, heatPaints: (window.demoState?.heatPaints || 0) + 1, heatHot: Object.entries(lines).sort((a, b) => b[1] - a[1]).slice(0, 3).map(x => +x[0]) });
}
$('heat').onclick = () => setHeat(!heatWanted);
$('share').onclick = async () => {
  const code = editor.getValue();
  if (code === original) clearLink(); else history.replaceState(null, '', '#code=' + await packed(code));
  let copied = false;
  try { await navigator.clipboard.writeText(location.href); copied = true; } catch { /* the link is in the address bar */ }
  say(code === original ? (copied ? 'Copied a link to the page: the code is unedited' : 'The code is unedited, so this page\'s address is the link')
    : copied ? 'Copied a link to this code' : 'The link to this code is in the address bar');
  state({ shareLength: location.href.length });
};
$('runshared').onclick = () => { endShared(); showChips(); request(editor.getValue(), 'the shared code'); };
$('dropshared').onclick = () => { endShared(); clearLink(); editor.setValue(original); showChips(); changedLines(); };
addEventListener('hashchange', () => location.reload());
$('sj').onclick = () => setMode(!compiled);
$('ask').onsubmit = e => {
  e.preventDefault();
  if (!ready) return;
  $('story').append('\n\n'); newSpan();
  worker.postMessage({ type: 'start', prompt: $('prompt').value }); setRunning(true); ask();
};
function setRunning(on) {
  running = on;
  const b = $('pause'), name = on ? 'Pause' : 'Resume';
  b.innerHTML = on ? ICON.pause : ICON.play; b.title = name; b.setAttribute('aria-label', name);
  showWorking();
}
$('pause').onclick = () => { setRunning(!running); say(running ? 'Writing' : 'Paused'); ask(); };
$('clear').onclick = () => { $('story').textContent = ''; span = null; state({ tail: '' }); };
$('load').onclick = () => { $('load').hidden = true; $('wait').hidden = false; start(); };
document.querySelectorAll('.lab-tab').forEach(c => c.addEventListener('click', () => showStage(c.dataset.s)));
$('tree').addEventListener('click', e => {
  const row = e.target.closest('.lab-row.x');
  if (!row) return;
  row.classList.toggle('on');
  row.nextElementSibling.hidden = !row.classList.contains('on');
});
document.addEventListener('visibilitychange', () => { if (!hidden()) ask(); });

editor = CodeMirror.fromTextArea($('code'), { mode: 'python', theme: 'sj', lineNumbers: true, indentUnit: 4, scrollbarStyle: null, viewportMargin: Infinity });
viewer = CodeMirror($('panel'), { theme: 'sj', lineNumbers: true, readOnly: 'nocursor', mode: 'text/x-csrc', scrollbarStyle: null });
// typing recompiles once it pauses, as in the home page's demos; presets and Reset ask for themselves
editor.on('change', (cm, c) => {
  if (heatLines.length) clearHeat();                   // the counts are for the code before this change
  clearTimeout(changedLines.t);
  changedLines.t = setTimeout(() => { changedLines(); showChips(); }, 120);
  if (c.origin === 'setValue' || c.origin === '+edit') return;
  clearTimeout(editTimer);
  if (!ready) return;
  // typing cancels what's under way: the code it was for is already out of date
  typing = true;
  carryIntro = carryIntro || !!(want?.intro || background?.intro);
  want = null;
  if (background) background.stale = true;
  if (barOn) cancelBar();
  note('');
  say(compiled ? 'Compiles when you pause typing' : 'Runs when you pause typing');
  editTimer = setTimeout(() => request(editor.getValue(), ''), TYPING_MS);
});
new ResizeObserver(() => { editor.refresh(); viewer.refresh(); }).observe($('pane'));   // the code panel is as tall as the story's
window.demoApi = { get heatFor() { return heatFor; }, editor, viewer, EDITS, setMode, showStage, get lastGood() { return lastGood; }, get stage() { return stage; } };

async function kept() { try { const c = await caches.open('scramjit-llm-1'); return !!(await c.match('https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin')); } catch { return false; } }
(async () => {
  original = await (await fetch('forward.py')).text();
  editor.setValue(original);
  if (location.hash.startsWith('#code=')) {
    try {
      const code = await unpacked(location.hash.slice(6));
      if (code !== original) { editor.setValue(code); sharedWaiting = true; $('shared').hidden = false; changedLines(); }
    } catch { showError('This link\'s code couldn\'t be read, so this is the original.'); clearLink(); }
  }
  const net = navigator.connection || {};
  const careful = net.saveData || net.type === 'cellular' || /(^|-)2g$/.test(net.effectiveType || '');
  if (careful && !(await kept())) { $('load').hidden = false; $('wait').hidden = true; say('Ready when you are: 60 MB to download.'); } else start();
})();
