import { $, bars, esc, fail, int, kpis, legend, load, num, pct, seg, select, table, xy } from './kit.js';

// promptab/experiment.py in the browser: pairwise always-valid test, Bonferroni over ordered pairs, Beta-posterior Thompson sampling.
const TAU = 0.05, MIN_N = 20, ALPHA = 0.05, BATCH = 50, HORIZON = 20000, TOL = 0.01, WINDOW = 500;

function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const logNdtr = (z) => (z < -20 ? -0.5 * z * z - Math.log(-z) - 0.5 * Math.log(2 * Math.PI) + Math.log(1 - 1 / (z * z)) : Math.log(0.5 * erfc(-z / Math.SQRT2)));
function evidence([n0, s0, q0], [n1, s1, q1], tol) {   // that the second is worse than the first by more than tol
  if (n0 < MIN_N || n1 < MIN_N) return { llr: -Infinity, d: 0 };
  const mean = (s0 + s1) / (n0 + n1), v = ((q0 + q1) / (n0 + n1) - mean * mean) * (1 / n0 + 1 / n1);
  if (!(v > 1e-12)) return { llr: -Infinity, d: 0 };
  const d = s1 / n1 - s0 / n0, t2 = TAU * TAU, x = d + tol;
  return { llr: 0.5 * Math.log(v / (v + t2)) + (t2 * x * x) / (2 * v * (v + t2)) + Math.LN2 + logNdtr((-x * TAU) / Math.sqrt(v * (v + t2))), d };
}
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function normal(rnd) { let u = 0; while (!u) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); }
function gamma(k, rnd) {   // Marsaglia and Tsang; k >= 1 here
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) { let x, v; do { x = normal(rnd); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = rnd(); if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v; }
}
const beta = (a, b, rnd) => { const x = gamma(a, rnd); return x / (x + gamma(b, rnd)); };

function experiment(e, allocation, seed) {
  const rnd = mulberry32(seed), k = e.variants.length, thr = Math.log((k * (k - 1)) / ALPHA);
  const samplers = e.variants.map((v) => { let acc = 0; const cum = v.hist.map(([, n]) => (acc += n)); return () => { const r = rnd() * acc; let i = 0; while (cum[i] <= r) i++; return v.hist[i][0]; }; });
  const best = Math.max(...e.variants.map((v) => v.mean));
  const s = { counts: e.variants.map(() => [0, 0, 0]), alive: e.variants.map(() => true), dropped: {}, seen: 0, state: 'running', winner: null, lost: 0, recent: [], history: e.variants.map(() => []) };
  s.batch = () => {
    const alive = s.alive.flatMap((a, i) => (a ? [i] : []));
    for (let j = 0; j < BATCH; j++) {
      let arm;
      if (allocation === 'thompson') {
        let top = -1;
        for (const i of alive) { const [n, sum] = s.counts[i]; const x = beta(1 + sum, 1 + n - sum, rnd); if (x > top) { top = x; arm = i; } }
      } else arm = alive[Math.floor(rnd() * alive.length)];
      const score = samplers[arm](), c = s.counts[arm];
      c[0] += 1; c[1] += score; c[2] += score * score;
      s.lost += best - e.variants[arm].mean;
      s.recent.push(arm);
    }
    s.seen += BATCH;
    if (s.recent.length > WINDOW) s.recent.splice(0, s.recent.length - WINDOW);
    if (s.seen % WINDOW === 0) e.variants.forEach((_, i) => s.history[i].push({ x: s.seen, y: s.recent.filter((a) => a === i).length / s.recent.length }));
    for (let b = 0; b < k; b++) {
      if (!s.alive[b]) continue;
      let hit = null;
      for (let a = 0; a < k; a++) if (a !== b) { const ev = evidence(s.counts[a], s.counts[b], TOL); if (ev.llr >= thr && (!hit || ev.llr > hit.llr)) hit = { ...ev, a }; }
      if (hit) { s.alive[b] = false; s.dropped[b] = { at: s.seen, d: hit.d, vs: hit.a }; }
    }
    const left = s.alive.flatMap((a, i) => (a ? [i] : []));
    const mean = (i) => s.counts[i][1] / Math.max(1, s.counts[i][0]);
    if (left.length === 1) { s.state = 'decided'; s.winner = left[0]; }
    else if (s.seen >= HORIZON) { s.state = 'ended'; s.winner = left.reduce((x, y) => (mean(y) > mean(x) ? y : x)); }
    if (s.state !== 'running') s.lost += (HORIZON - s.seen) * (best - e.variants[s.winner].mean);
  };
  return s;
}

const show = (s) => esc(s).replace(/\n/g, '↵');
const template = (p) => `<code>${show(p.instructions || '')}${show(p.input_prefix || '')}<b>&lt;input&gt;</b>${show(p.input_suffix || '')}${show(p.output_prefix || '')}<b>&lt;answer&gt;</b>${show(p.output_suffix || '')}</code>`;

try {
  const { summary, experiments } = await load();
  const strat = Object.fromEntries(summary.strategies.map((x) => [x.strategy, x]));
  const aa = Object.fromEntries(summary.identical.map((x) => [x.allocation, x]));
  const gain = (e) => Math.max(...e.variants.map((v) => v.mean)) - e.variants[0].mean;
  kpis($('#kpis'), [
    { label: 'Best format beats default by 2+ pts', value: `${experiments.filter((e) => gain(e) >= 0.02).length} of ${experiments.length}`, note: 'experiments: 10 models × 3 tasks' },
    { label: 'Score lost per 1,000, Thompson', value: num(strat.thompson.shortfall_per_1000, 1), note: `uniform split ${num(strat.uniform.shortfall_per_1000, 1)}, keeping the default ${num(strat['keep the default'].shortfall_per_1000, 1)}` },
    { label: 'Picked a prompt within a point of the best', value: pct(strat.thompson.within_1_point), note: `Thompson; keeping the default: ${pct(strat['keep the default'].within_1_point)}` },
    { label: 'False drops, identical prompts', value: pct(aa.thompson.false / aa.thompson.runs), note: `Thompson; uniform ${pct(aa.uniform.false / aa.uniform.runs)}; target under 5%` },
  ]);

  seg($('#task'), ['civil_comments', 'imdb', 'natural_qa'], 'civil_comments', (task) => {
    const rows = experiments.filter((e) => e.task === task).map((e) => {
      const sorted = [...e.variants].sort((a, b) => b.mean - a.mean), d = e.variants[0];
      return { model: e.model, default: d.mean, best: sorted[0].mean, bestName: sorted[0].name, worst: sorted.at(-1).mean, worstName: sorted.at(-1).name, gain: sorted[0].mean - d.mean };
    });
    const f = (v) => (task === 'natural_qa' ? `${num(100 * v, 1)} F1` : pct(v));
    table($('#spread'), [
      { key: 'model', label: 'Model' },
      { key: 'default', label: 'Default', num: true, fmt: f },
      { key: 'best', label: 'Best', num: true, fmt: f },
      { key: 'bestName', label: 'Best format' },
      { key: 'gain', label: 'Gain', num: true, fmt: (v) => `+${num(100 * v, 1)}` },
      { key: 'worst', label: 'Worst', num: true, fmt: f },
      { key: 'worstName', label: 'Worst format' },
    ], rows, { hl: (r) => r.gain >= 0.02 });
  });

  let e = experiments[0], allocation = 'thompson', seed = 1, sim = null, raf = 0;
  const colors = ['var(--text)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)', 'var(--accent)'];
  function paint() {
    const s = sim;
    const series = e.variants.map((v, i) => ({ name: v.name + (i || v.name === 'default' ? '' : ' (default)'), color: colors[i % 6], width: i === s.winner ? 3 : 1.8, points: s.history[i].length ? s.history[i] : [{ x: 0, y: 1 / e.variants.length }] }));
    xy($('#share'), { label: 'Share of traffic per prompt', height: 220, series, x: { min: 0, max: HORIZON, fmt: (v) => (v ? `${v / 1000}k` : '0'), label: 'requests' }, y: { min: 0, max: 1, fmt: (v) => `${Math.round(v * 100)}%` } });
    legend($('#shareKey'), series);
    const verdict = s.state === 'running' ? `running, ${s.alive.filter(Boolean).length} prompts left` : `${s.state}: ${esc(e.variants[s.winner].name)}`;
    $('#out').innerHTML = `<div>Requests<b>${int(s.seen)}</b><span class="muted small">${verdict}</span></div>` +
      `<div>Score lost per 1,000 requests<b>${num((1000 * s.lost) / Math.max(s.seen, s.state === 'running' ? s.seen : HORIZON), 1)}</b><span class="muted small">expected, against always serving the best prompt</span></div>` +
      `<div>Best prompt in the recorded data<b>${esc([...e.variants].sort((a, b) => b.mean - a.mean)[0].name)}</b><span class="muted small">default: ${esc(e.variants[0].name)}</span></div>`;
    table($('#variants'), [
      { key: 'name', label: 'Prompt' },
      { key: 'tpl', label: 'Template', html: true },
      { key: 'mean', label: 'Recorded', num: true, fmt: (v) => num(100 * v, 1) },
      { key: 'n', label: 'Served', num: true, fmt: int },
      { key: 'obs', label: 'Observed', num: true, fmt: (v) => (v == null ? '–' : num(100 * v, 1)) },
      { key: 'status', label: 'Status', html: true },
    ], e.variants.map((v, i) => ({
      name: v.name, tpl: template(v.prompt), mean: v.mean, n: s.counts[i][0], obs: s.counts[i][0] ? s.counts[i][1] / s.counts[i][0] : null,
      status: i === s.winner ? '<span class="pill ok">serves</span>' : s.dropped[i] ? `<span class="pill no">dropped at ${int(s.dropped[i].at)}</span>` : '<span class="pill mid">in the running</span>',
    })));
  }
  function start() {
    cancelAnimationFrame(raf);
    sim = experiment(e, allocation, seed);
    const tick = () => { for (let j = 0; j < 8 && sim.state === 'running'; j++) sim.batch(); paint(); if (sim.state === 'running') raf = requestAnimationFrame(tick); };
    tick();
  }
  const order = [...experiments].sort((a, b) => gain(b) - gain(a));
  select($('#exp'), experiments.map((x, i) => [i, `${x.model} · ${x.task} (${x.variants.length} prompts)`]), experiments.indexOf(order[0]), (i) => { e = experiments[+i]; if (sim) start(); });
  seg($('#alloc'), [['thompson', 'Thompson sampling'], ['uniform', 'even split']], 'thompson', (v) => { allocation = v; if (sim) start(); });
  $('#again').onclick = () => { seed += 1; start(); };
  start();

  table($('#strategies'), [
    { key: 'strategy', label: 'Strategy' },
    { key: 'within_1_point', label: 'Within 1 pt of best', num: true, fmt: (v) => pct(v) },
    { key: 'decided', label: 'Decided early', num: true, fmt: (v) => (v == null ? '–' : pct(v)) },
    { key: 'shortfall_per_1000', label: 'Shortfall / 1k', num: true, fmt: (v) => num(v, 1) },
  ], summary.strategies, { hl: (r) => r.strategy === 'thompson' });
  bars($('#aa'), summary.identical.map((r) => ({ label: r.allocation, value: r.false / r.runs, text: `${pct(r.false / r.runs)} (${r.false}/${r.runs})` })), { max: 0.05 });
} catch (err) {
  fail(err);
}
