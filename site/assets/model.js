/* Wafer Defects: baselines, the method, evaluation design and validation (S5 to S8). */
(() => {
  'use strict';
  const K = window.KIT;
  if (!K) return;
  const { R, W, M, CLASSES, SIZE, $, $$, el, sv, segmented, press, F, wafer, unpack32, codesFor, paint, sum,
    removeIsolated, reduceMotion, intervalChart, probBars, cssColor, mix, whileVisible, onReveal, addCheck } = K;
  const V = R.validation;
  const story = W.story;
  const fmt = F.int;
  const pct = (v, d = 0) => K.roundHalf(v * 100, d).toFixed(d) + '%';
  const demoNote = M.source === 'saved'
    ? 'The numbers come from the saved final model.'
    : `The model here is the frozen recipe refit on the ${fmt(R.split.train_rows)} training wafers with the same seed, because the saved model file isn't in the repository. It gives the same answer as the saved final-test predictions on ${pct(R.demo_model.agreement, 1)} of test wafers.`;

  // ================================================================ S5 baselines
  (() => {
    const info = [
      ['B0-majority', 'B0 majority', 'always says the biggest class', 'the floor: no learning at all'],
      ['B0-uniform', 'B0 uniform', 'picks a random class', 'the other floor'],
      ['B1-failing-fraction', 'B1 failing share', 'logistic regression on one number: how much of the wafer failed', 'does “how much” alone tell you the pattern? (no)'],
      ['B2-flat-logistic', 'B2 logistic on pixels', 'the 32×32 fail map flattened into 1,024 numbers', 'the plain “no structure” model, same classifier as ours'],
      ['B3-rotation-pixel-logistic', 'B3 rotated pixels', 'B2 logistic, training maps copied at 0/90/180/270°', 'can you buy rotation-robustness just by showing turned copies? (a bit)'],
      ['B2-flat-random-forest', 'B2 random forest on pixels', 'same 1,024 numbers, 100 trees', 'the strongest simple model: the one to beat'],
    ];
    const ours = V['P3-kmeans-logreg'].macro_f1;
    intervalChart($('#baseline-chart'), {
      min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], ref: { v: ours, label: `our model, validation: ${F.f3(ours)}` },
      rows: info.map(([key, name, what]) => {
        const r = V[key], s = r.fold_std || 0;
        return { name, strong: key === 'B2-flat-random-forest',
          marks: [{ cls: key === 'B2-flat-random-forest' ? 'forest' : 'other', v: r.macro_f1, lo: r.macro_f1 - s, hi: r.macro_f1 + s, label: 'macro-F1 ± fold spread' }],
          val: `<b>${F.f3(r.macro_f1)}</b>${s > 0.0005 ? ` ± ${F.f3(s)}` : ''}`,
          tip: `<b>${name}</b><br>${what}<br>macro-F1 ${F.f3(r.macro_f1)} · accuracy ${F.f3(r.accuracy)}` };
      }),
    });
    const table = $('#baseline-table');
    table.append(el('thead', {}, el('tr', {}, ['Baseline', 'What it does', 'Macro-F1', 'Why it’s here'].map((h) => el('th', {}, h)))),
      el('tbody', {}, info.map(([key, name, what, why]) => {
        const r = V[key];
        return el('tr', { class: key === 'B2-flat-random-forest' ? 'row-strong' : '' }, el('td', {}, name), el('td', {}, what),
          el('td', { class: 'num' }, F.f3(r.macro_f1) + (r.fold_std > 0.0005 ? ` ± ${F.f3(r.fold_std)}` : '')), el('td', {}, why));
      })));
  })();

  // ================================================================ S6 pipeline
  (() => {
    const icon = (body) => `<svg viewBox="0 0 64 64" aria-hidden="true">${body}</svg>`;
    const disc = '<circle class="i-pass" cx="32" cy="32" r="27"/>';
    const steps = [
      { name: 'Clean up', learns: false, icon: icon(`${disc}<path class="i-fail" d="M40 14h5v5h-5zM44 19h5v5h-5zM47 24h5v5h-5zM49 29h5v5h-5z"/><rect class="i-speck" x="18" y="22" width="5" height="5"/><rect class="i-speck s2" x="26" y="42" width="5" height="5"/>`),
        text: 'Switch off failing dies with no failing neighbour (S4). Save the two summary numbers first: failing share and removed share.' },
      { name: 'Resize', learns: false, icon: icon(`<g class="i-shrink"><rect class="i-pass" x="8" y="8" width="48" height="48" rx="4"/><path class="i-gridline" d="M20 8v48M32 8v48M44 8v48M8 20h48M8 32h48M8 44h48"/></g>`),
        text: 'Shrink every map to 32×32 by picking the nearest die, and keep a wafer mask so empty corners are ignored.' },
      { name: 'Patches', learns: true, icon: icon(`${disc}<path class="i-fail" d="M40 14h5v5h-5zM44 19h5v5h-5zM47 24h5v5h-5zM49 29h5v5h-5z"/><rect class="i-window" x="8" y="8" width="20" height="20" rx="2"/>`),
        text: `We cut ${fmt(200000)} small 5×5 squares from training wafers, picked fairly across classes so the ${fmt(R.audit.subset_class_counts['Near-full'])} Near-full wafers give as many patches as the clean ones. Each patch is normalised (subtract its average, divide by its spread), so a filter cares about the <em>shape</em> in a patch, not how many dies failed. Then whitening (ZCA) removes the obvious “neighbours look alike” part, so k-means finds edges and lines instead of blurry blobs.`,
        fit: 'Learns: the patch average and the whitening matrix (training wafers only).' },
      { name: 'Filters', learns: true, icon: icon(`<g class="i-tile"><rect class="i-pass" x="3" y="22" width="18" height="18" rx="2"/><path class="i-fail" d="M9 22h6v18h-6z"/></g><g class="i-tile"><rect class="i-pass" x="23" y="22" width="18" height="18" rx="2"/><path class="i-fail" d="M23 22h6v6h-6zM29 28h6v6h-6zM35 34h6v6h-6z"/></g><g class="i-tile"><rect class="i-pass" x="43" y="22" width="18" height="18" rx="2"/><path class="i-fail" d="M49 28h6v6h-6z"/></g>`),
        text: `Group all those patches into ${R.recipe.k} clusters with k-means. Each cluster's centre is one filter: the ${R.recipe.k} most common little shapes on wafers, like spots, lines and edges. No labels and no backprop.`,
        fit: 'Learns: the 32 filters (training wafers only).' },
      { name: 'Convolve', learns: false, icon: icon(`<g class="i-turn"><rect class="i-pass" x="17" y="17" width="30" height="30" rx="3"/><path class="i-fail" d="M27 17h10v30h-10z"/></g>`),
        text: 'Slide every filter over every spot of the wafer. At each spot, each filter gets a score for how well it matches (the triangle activation: only patches closer than average score above zero). We try each filter turned 4 ways and keep the best one.' },
      { name: 'Ring pooling', learns: false, icon: icon('<circle class="i-ring r4" cx="32" cy="32" r="27"/><circle class="i-ring r3" cx="32" cy="32" r="20"/><circle class="i-ring r2" cx="32" cy="32" r="13"/><circle class="i-ring r1" cx="32" cy="32" r="6"/>'),
        text: 'Average each filter’s scores inside 4 rings (0-25-50-75-100% of the radius) and over the whole wafer, counting only dies on the wafer. 32 filters × 5 regions = 160 numbers, plus the 2 summary numbers = <strong>162 features</strong>.' },
      { name: 'Classifier', learns: true, icon: icon('<path class="i-axis" d="M6 54h52"/><g class="i-bars"><rect x="8" y="42" width="4" height="12"/><rect x="14" y="46" width="4" height="8"/><rect x="20" y="38" width="4" height="16"/><rect class="i-win" x="26" y="12" width="4" height="42"/><rect x="32" y="44" width="4" height="10"/><rect x="38" y="40" width="4" height="14"/><rect x="44" y="48" width="4" height="6"/><rect x="50" y="45" width="4" height="9"/></g>'),
        text: 'Logistic regression with balanced class weights (so rare classes count as much as common ones), then temperature scaling: one extra number that makes the probabilities honest. If it says 80% sure, it’s right about 80% of the time. The temperature is picked on training wafers the classifier hadn’t seen while fitting (3 inner folds), and it never changes which class wins.',
        fit: 'Learns: the feature scaler, 9 × 162 weights + 9 offsets, and the temperature (training wafers only).' },
    ];
    const list = $('#pipe'), panel = $('#pipe-panel');
    const items = steps.map((s, i) => {
      const button = el('button', { type: 'button', 'aria-pressed': 'false' },
        el('span', { class: 'pnum' }, String(i + 1)), el('span', { class: 'picon', html: s.icon }), el('b', {}, s.name),
        s.learns ? el('span', { class: 'learn-tag' }, 'learns') : null);
      button.addEventListener('click', () => show(i));
      list.append(el('li', {}, button));
      return button;
    });
    function show(i) {
      items.forEach((b, k) => b.setAttribute('aria-pressed', String(k === i)));
      const s = steps[i];
      panel.classList.remove('swap'); void panel.offsetWidth; panel.classList.add('swap');
      panel.innerHTML = `<h4>${i + 1}. ${s.name}</h4><p>${s.text}</p>${s.fit ? `<p class="fit">${s.fit}</p>` : '<p class="fit nofit">Nothing to learn here: each wafer is handled on its own.</p>'}`;
    }
    show(2);
  })();

  // ================================================ conv features (mirrors src/features.py)
  const conv = (() => {
    const P = 5, Kf = M.filters.length, N = SIZE * SIZE;
    const rot = (f) => { const out = new Array(25); for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) out[i * P + j] = f[j * P + (P - 1 - i)]; return out; };
    const bankSets = {};
    function banks(filters) {
      const key = filters === M.filters ? 'k' : 'r';
      if (bankSets[key]) return bankSets[key];
      const b = [filters]; for (let k = 1; k < 4; k++) b.push(b[k - 1].map(rot));
      return (bankSets[key] = { b, norms: b.map((bank) => bank.map((f) => f.reduce((t, v) => t + v * v, 0))) });
    }
    const region = new Uint8Array(N);
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const r = Math.hypot(x - (SIZE - 1) / 2, y - (SIZE - 1) / 2) / (SIZE / 2);
      region[y * SIZE + x] = r < 0.25 ? 0 : r < 0.5 ? 1 : r < 0.75 ? 2 : 3;
    }
    function features(fail, mask, filters = M.filters) {
      const { b, norms } = banks(filters);
      const act = new Float32Array(N * Kf), patch = new Float64Array(25), white = new Float64Array(25), dist = new Float64Array(Kf);
      for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
        let mean = 0;
        for (let dy = 0; dy < P; dy++) for (let dx = 0; dx < P; dx++) {
          const yy = y + dy - 2, xx = x + dx - 2;
          const v = yy >= 0 && yy < SIZE && xx >= 0 && xx < SIZE ? fail[yy * SIZE + xx] : 0;
          patch[dy * P + dx] = v; mean += v;
        }
        mean /= 25;
        let variance = 0;
        for (let i = 0; i < 25; i++) { patch[i] -= mean; variance += patch[i] * patch[i]; }
        const scale = 1 / Math.sqrt(variance / 25 + 0.01);
        for (let i = 0; i < 25; i++) patch[i] = patch[i] * scale - M.mean[i];
        let x2 = 0;
        for (let j = 0; j < 25; j++) { let t = 0; for (let i = 0; i < 25; i++) t += patch[i] * M.zca[i][j]; white[j] = t; x2 += t * t; }
        const base = (y * SIZE + x) * Kf;
        for (let r = 0; r < 4; r++) {
          let avg = 0;
          for (let k = 0; k < Kf; k++) {
            const f = b[r][k]; let dot = 0;
            for (let i = 0; i < 25; i++) dot += white[i] * f[i];
            dist[k] = Math.sqrt(Math.max(x2 - 2 * dot + norms[r][k], 0)); avg += dist[k];
          }
          avg /= Kf;
          for (let k = 0; k < Kf; k++) { const a = avg - dist[k]; if (a > act[base + k]) act[base + k] = a; }
        }
      }
      const pooled = new Float64Array(5 * Kf), count = new Float64Array(5);
      for (let p = 0; p < N; p++) {
        if (!mask[p]) continue;
        count[region[p]]++; count[4]++;
        for (let k = 0; k < Kf; k++) { const a = act[p * Kf + k]; pooled[region[p] * Kf + k] += a; pooled[4 * Kf + k] += a; }
      }
      for (let r = 0; r < 5; r++) for (let k = 0; k < Kf; k++) pooled[r * Kf + k] /= Math.max(count[r], 1);
      return { act, pooled };
    }
    /** Logistic regression + temperature, mirrors CalibratedLogReg.predict_proba. */
    function predict(pooled, extras) {
      const x = [...pooled, ...extras].map((v, i) => (v - M.scaler_mean[i]) / M.scaler_scale[i]);
      const scores = M.coef.map((row, c) => row.reduce((t, w, i) => t + w * x[i], M.intercept[c]) / M.temperature);
      const top = Math.max(...scores), exp = scores.map((s) => Math.exp(s - top)), total = exp.reduce((a, b) => a + b, 0);
      return exp.map((e) => e / total);
    }
    /** Full pipeline from a full-size wafer (cleanup, saved numbers, resize, features, classifier). */
    function fromWafer(wf) {
      const s = wf.small('isolated');
      const extras = [wf.nFail / Math.max(1, wf.dies), (wf.nFail - sum(wf.iso)) / Math.max(1, wf.nFail)];
      return { s, extras, ...features(s.fail, s.mask) };
    }
    return { features, predict, fromWafer, region, K: Kf };
  })();
  K.conv = conv;

  // ============================================================ S6 filter explorer
  (() => {
    const Kf = conv.K, N = SIZE * SIZE;
    const rot90 = (image) => { const out = new image.constructor(N); for (let i = 0; i < SIZE; i++) for (let j = 0; j < SIZE; j++) out[i * SIZE + j] = image[j * SIZE + (SIZE - 1 - i)]; return out; };
    const names = ['Centre', 'Inner ring', 'Outer ring', 'Edge', 'Whole wafer'];
    const waferCanvas = el('canvas', { role: 'img', 'aria-label': 'The 32 by 32 fail map the model sees' });
    const heatCanvas = el('canvas', { role: 'img', 'aria-label': 'Where the selected filter fires on this wafer' });
    $('#lab-wafer').prepend(waferCanvas);
    $('#lab-heat').prepend(heatCanvas);
    const scan = $('#lab-scan');
    let choice = CLASSES.indexOf('Edge-Ring'), filter = 0, turns = 0, base = null, now = null, playing = null, upTo = N;

    const chips = $('#lab-wafers');
    chips.append(...CLASSES.map((c, i) => el('button', { type: 'button', 'data-value': i, 'aria-pressed': String(i === choice) }, c)));
    const tiles = M.filters.map((f, i) => {
      const canvas = el('canvas', { width: 5, height: 5 });
      canvas._filter = f;
      return el('button', { type: 'button', 'data-value': i, 'aria-pressed': 'false', 'aria-label': `Filter ${i + 1}`, title: `Filter ${i + 1}` }, canvas);
    });
    $('#lab-filters').append(...tiles);
    const paintFilter = (canvas, f) => {
      const zero = cssColor('--line'), plus = cssColor('--m-ours'), minus = cssColor('--die-fail');
      const top = Math.max(...f.map(Math.abs)), context = canvas.getContext('2d'), image = context.createImageData(5, 5);
      f.forEach((v, i) => image.data.set([...mix(zero, v > 0 ? plus : minus, Math.abs(v) / top), 255], i * 4));
      context.putImageData(image, 0, 0);
    };
    K.paintFilter = paintFilter;
    const paintTiles = () => tiles.forEach((b) => paintFilter(b.firstChild, b.firstChild._filter));

    function load() {
      const wf = wafer(story.examples[CLASSES[choice]][0]);
      const s = wf.small('isolated');
      base = { fail: s.fail, mask: s.mask, ...conv.features(s.fail, s.mask) };
      now = base;
      if (turns) {
        let f = s.fail, m = s.mask;
        for (let t = 0; t < turns; t++) { f = rot90(f); m = rot90(m); }
        now = { fail: f, mask: m, ...conv.features(f, m) };
      }
    }
    function pickFilter() {   // start on the filter whose ring profile best follows where this wafer fails
      const share = [0, 0, 0, 0], count = [0, 0, 0, 0];
      for (let p = 0; p < N; p++) if (base.mask[p]) { count[conv.region[p]]++; share[conv.region[p]] += base.fail[p]; }
      const target = share.map((s, r) => s / Math.max(1, count[r]));
      const centred = (v) => { const m = v.reduce((a, b) => a + b, 0) / v.length; return v.map((x) => x - m); };
      const t = centred(target), tn = Math.hypot(...t) || 1;
      let best = 0, bestScore = -Infinity;
      for (let k = 0; k < Kf; k++) {
        const rings = centred([0, 1, 2, 3].map((r) => base.pooled[r * Kf + k]));
        const norm = Math.hypot(...rings) || 1;
        const score = (rings.reduce((s, v, i) => s + v * t[i], 0) / (norm * tn)) * norm;   // correlation × contrast
        if (score > bestScore) { bestScore = score; best = k; }
      }
      filter = best;
    }
    function drawHeat() {
      const low = cssColor('--die-pass'), high = cssColor('--m-ours');
      let top = 1e-9;
      for (let p = 0; p < N; p++) top = Math.max(top, now.act[p * Kf + filter], base.act[p * Kf + filter]);
      heatCanvas.width = SIZE; heatCanvas.height = SIZE;
      const context = heatCanvas.getContext('2d'), image = context.createImageData(SIZE, SIZE);
      for (let p = 0; p < N; p++) {
        const v = p < upTo ? now.act[p * Kf + filter] / top : 0;
        image.data.set(now.mask[p] ? [...mix(low, high, Math.min(1, v)), 255] : [0, 0, 0, 0], p * 4);
      }
      context.putImageData(image, 0, 0);
    }
    function draw(animateTurn) {
      paint(waferCanvas, codesFor.result(now.mask, now.fail), SIZE, SIZE);
      if (animateTurn && !reduceMotion) {
        [waferCanvas, heatCanvas].forEach((c) => {
          c.classList.remove('turning'); c.style.transform = 'rotate(90deg)'; void c.offsetWidth;
          c.classList.add('turning'); c.style.transform = 'none';
        });
      }
      drawHeat();
      const values = names.map((_, r) => now.pooled[r * Kf + filter]), before = names.map((_, r) => base.pooled[r * Kf + filter]);
      const scale = Math.max(...values, ...before, 1e-9) * 1.12;
      $('#lab-bars').replaceChildren(...names.flatMap((name, r) => [
        el('span', { class: 'name' }, name),
        el('span', { class: 'rtrack' }, el('i', { class: 'rbar', style: `width:${(values[r] / scale) * 100}%` }),
          turns ? el('i', { class: 'rtick', style: `left:calc(${(before[r] / scale) * 100}% - 1px)` }) : null),
        el('span', { class: 'val' }, F.f2(values[r]))]));
      tiles.forEach((b, i) => b.setAttribute('aria-pressed', String(i === filter)));
      const rings = values.slice(0, 4), hi = rings.indexOf(Math.max(...rings)), lo = rings.indexOf(Math.min(...rings));
      const where = ['in the centre', 'in the inner ring', 'in the outer ring', 'along the rim'][hi];
      let moved = 0, largest = 0;
      for (let i = 0; i < 5 * Kf; i++) { moved = Math.max(moved, Math.abs(now.pooled[i] - base.pooled[i])); largest = Math.max(largest, base.pooled[i]); }
      $('#lab-readout').innerHTML = `Filter ${filter + 1} fires most ${where}. On this ${CLASSES[choice]} wafer its ${names[hi].toLowerCase()} average is ` +
        `<strong>${F.f2(rings[hi])}</strong>, its ${names[lo].toLowerCase()} average is <strong>${F.f2(rings[lo])}</strong>. ` +
        (turns ? `Turned ${turns * 90}°: the 160 ring averages moved by at most <strong>${moved.toFixed(3)}</strong> (the largest is ${largest.toFixed(2)}). That’s why quarter turns don’t change our score.`
          : 'All 32 filters × 5 regions give the 160 picture numbers the classifier reads. Try “Turn 90°”.');
    }
    function stopPlay() { if (playing) cancelAnimationFrame(playing); playing = null; upTo = N; scan.style.opacity = 0; $('#lab-play').textContent = '▶ Slide the filter'; }
    function play() {
      if (playing) { stopPlay(); drawHeat(); return; }
      $('#lab-play').textContent = '■ Stop';
      const start = performance.now(), duration = 3600;
      scan.style.opacity = 1;
      const frame = (t) => {
        const p = Math.min(1, (t - start) / duration);
        upTo = Math.floor(p * N);
        const y = Math.floor(upTo / SIZE), x = upTo % SIZE;
        scan.setAttribute('x', x - 2); scan.setAttribute('y', y - 2);
        drawHeat();
        if (p < 1) playing = requestAnimationFrame(frame); else stopPlay();
      };
      playing = requestAnimationFrame(frame);
    }
    segmented(chips, (v) => { stopPlay(); choice = Number(v); turns = 0; load(); pickFilter(); draw(false); });
    $('#lab-filters').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { stopPlay(); filter = Number(b.dataset.value); draw(false); } });
    $('#lab-turn').addEventListener('click', () => { stopPlay(); turns = (turns + 1) % 4; load(); draw(true); });
    $('#lab-play').addEventListener('click', play);
    load(); pickFilter(); paintTiles(); draw(false);
    $('#lab-caption').textContent = `Computed live in your browser. Dashed circles are the ring borders. ${demoNote}`;
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { paintTiles(); draw(false); });
  })();

  // ===================================================== S6 rings, animated
  (() => {
    const svg = $('#rings-svg'), wf = wafer(story.examples.Donut[0]), s = wf.small('isolated');
    const cell = 5, ox = 14, oy = 20;
    const g = sv('g', { class: 'rw' });
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const k = y * SIZE + x;
      if (s.mask[k]) g.append(sv('rect', { class: s.fail[k] ? 'f' : 'p', x: ox + x * cell, y: oy + y * cell, width: cell - 0.4, height: cell - 0.4 }));
    }
    svg.append(g);
    const cx = ox + 16 * cell, cy = oy + 16 * cell, radii = [4, 8, 12, 16].map((r) => r * cell);
    const rings = radii.map((r, i) => sv('circle', { class: `rr rr${i}`, cx, cy, r }));
    svg.append(...rings);
    const share = [0, 0, 0, 0], count = [0, 0, 0, 0];
    for (let p = 0; p < SIZE * SIZE; p++) if (s.mask[p]) { count[conv.region[p]]++; share[conv.region[p]] += s.fail[p]; }
    const names = ['centre', 'inner', 'outer', 'edge'];
    const bx = 196, bw = 108;
    names.forEach((n, i) => {
      const v = share[i] / Math.max(1, count[i]), y = 40 + i * 34;
      svg.append(sv('text', { class: 'rb-label', x: bx, y: y - 4 }, `${n} ring: ${pct(v)} failing`),
        sv('rect', { class: 'rb-track', x: bx, y, width: bw, height: 9, rx: 3 }),
        sv('rect', { class: `rb-bar rb${i}`, x: bx, y, width: Math.max(2, v * bw), height: 9, rx: 3 }));
    });
    svg.append(sv('text', { class: 'rb-note', x: bx, y: 180 }, 'Turned 90°:'), sv('text', { class: 'rb-note', x: bx, y: 194 }, 'same rings, same numbers.'));
    if (!reduceMotion) whileVisible(svg, () => svg.classList.add('play'), () => svg.classList.remove('play'));
  })();

  // ============================================================ S6 draw a wafer
  (() => {
    const canvas = $('#draw-canvas'), N = SIZE * SIZE;
    const mask = unpack32(M.draw_mask).mask;
    const fail = new Uint8Array(N);
    let tool = 'fail', pending = false;
    const update = probBars($('#draw-probs'));
    const presets = {
      'Edge ring': (x, y, r) => r > 13.3,
      'Centre blob': (x, y, r) => r < 5.5,
      Donut: (x, y, r) => r > 6 && r < 10,
      Scratch: (x, y) => Math.abs(x - y) < 0.6 && x > 6 && x < 24,
      'Random specks': () => Math.random() < 0.35,
    };
    $('#draw-presets').append(...Object.keys(presets).map((name) => el('button', { type: 'button', 'data-value': name }, name)));
    $('#draw-presets').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
        const r = Math.hypot(x - 15.5, y - 15.5), k = y * SIZE + x;
        fail[k] = mask[k] && presets[b.dataset.value](x, y, r) ? 1 : 0;
      }
      run();
    });
    function run() {
      pending = false;
      paint(canvas, codesFor.result(mask, fail), SIZE, SIZE);
      const nFail = sum(fail), dies = sum(mask);
      const clean = removeIsolated(fail, SIZE, SIZE);
      const extras = [nFail / Math.max(1, dies), (nFail - sum(clean)) / Math.max(1, nFail)];
      const { pooled } = conv.features(clean, mask);
      const probs = conv.predict(pooled, extras);
      const said = update(probs);
      const p = Math.max(...probs);
      $('#draw-notice').innerHTML = nFail === 0
        ? 'A clean wafer. Draw some failing dies, or start from a shape.'
        : `The model says <strong>${said}</strong> (${pct(p)} sure). ${nFail - sum(clean) ? `Our cleanup removed ${nFail - sum(clean)} lonely ${nFail - sum(clean) === 1 ? 'die' : 'dies'} first. ` : ''}Try turning your shape into a thin line, or moving it to the edge.`;
    }
    const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(run); } };
    let down = false;
    const brush = (event) => {
      const box = canvas.getBoundingClientRect();
      const x = Math.floor(((event.clientX - box.left) / box.width) * SIZE), y = Math.floor(((event.clientY - box.top) / box.height) * SIZE);
      if (x < 0 || y < 0 || x >= SIZE || y >= SIZE) return;
      const k = y * SIZE + x;
      if (!mask[k]) return;
      fail[k] = tool === 'fail' ? 1 : 0;
      paint(canvas, codesFor.result(mask, fail), SIZE, SIZE);
      schedule();
    };
    canvas.addEventListener('pointerdown', (e) => { down = true; canvas.setPointerCapture(e.pointerId); brush(e); });
    canvas.addEventListener('pointermove', (e) => { if (down) brush(e); });
    canvas.addEventListener('pointerup', () => { down = false; });
    canvas.addEventListener('pointercancel', () => { down = false; });
    segmented($('#draw-tool'), (v) => { tool = v; });
    $('#draw-clear').addEventListener('click', () => { fail.fill(0); run(); });
    // Start with a short scratch.
    for (let i = 8; i < 22; i++) fail[i * SIZE + i + 1] = mask[i * SIZE + i + 1];
    run();
    $('#draw-caption').textContent = `Drag on the wafer to mark failing dies. The whole pipeline (cleanup, ${conv.K} filters × 4 turns, rings, logistic regression, temperature) runs in your browser. Your drawing is treated as the full-size map. ${demoNote}`;
  })();

  // ============================================================ S7 bootstrap
  (() => {
    const B = W.bootstrap, nLots = B.lots, C = CLASSES.length;
    // Per lot: true-positive, predicted and true counts per class.
    const per = Array.from({ length: nLots }, () => new Int32Array(3 * C));
    B.lot.forEach((lot, i) => {
      const t = B.truth[i], p = B.ours[i], v = per[lot];
      if (t === p) v[t]++;
      v[C + p]++; v[2 * C + t]++;
    });
    const macro = (weights) => {
      const total = new Float64Array(3 * C);
      for (let l = 0; l < nLots; l++) if (weights[l]) { const v = per[l], w = weights[l]; for (let j = 0; j < 3 * C; j++) total[j] += v[j] * w; }
      let f = 0;
      for (let c = 0; c < C; c++) { const d = total[C + c] + total[2 * C + c]; f += d ? (2 * total[c]) / d : 0; }
      return f / C;
    };
    const full = macro(new Int32Array(nLots).fill(1));
    addCheck('bootstrap', () => ({ ok: Math.abs(full - R.test.models.ours.macro_f1[0]) < 1e-6,
      detail: `Macro-F1 recomputed from the saved test predictions matches the logged ${F.f3(R.test.models.ours.macro_f1[0])}.` }));
    const draw = () => { const w = new Int32Array(nLots); for (let i = 0; i < nLots; i++) w[(Math.random() * nLots) | 0]++; return macro(w); };

    const svg = $('#boot-svg'), Wd = 720, H = 240, m = { l: 14, r: 14, t: 26, b: 34 };
    svg.setAttribute('viewBox', `0 0 ${Wd} ${H}`);
    const lo = 0.71, hi = 0.82, bins = 55, width = (Wd - m.l - m.r) / bins;
    const px = (v) => m.l + ((v - lo) / (hi - lo)) * (Wd - m.l - m.r);
    const counts = new Array(bins).fill(0);
    const barsG = sv('g'), marksG = sv('g');
    svg.append(sv('line', { class: 'axis', x1: m.l, x2: Wd - m.r, y1: H - m.b, y2: H - m.b }), barsG, marksG);
    for (let t = 0.72; t <= 0.8201; t += 0.02) svg.append(sv('text', { x: px(t), y: H - m.b + 18, 'text-anchor': 'middle' }, t.toFixed(2)));
    svg.append(sv('line', { class: 'boot-full', x1: px(full), x2: px(full), y1: m.t - 6, y2: H - m.b }),
      sv('text', { class: 'boot-full-label', x: px(full), y: m.t - 10, 'text-anchor': 'middle' }, `all test lots: ${F.f3(full)}`));
    const rects = counts.map((_, i) => { const r = sv('rect', { class: 'boot-bar', x: m.l + i * width + 0.5, width: width - 1, y: H - m.b, height: 0 }); barsG.append(r); return r; });
    let scores = [], running = null;
    function render() {
      const top = Math.max(4, ...counts);
      rects.forEach((r, i) => { const h = (counts[i] / top) * (H - m.t - m.b - 16); r.setAttribute('y', H - m.b - h); r.setAttribute('height', h); });
      marksG.replaceChildren();
      if (scores.length >= 40) {
        const sorted = [...scores].sort((a, b) => a - b);
        const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
        const a = q(0.025), b = q(0.975);
        marksG.append(sv('rect', { class: 'boot-range', x: px(a), y: m.t, width: px(b) - px(a), height: H - m.t - m.b }),
          sv('text', { class: 'boot-range-label', x: px(a) - 4, y: m.t + 12, 'text-anchor': 'end' }, F.f3(a)),
          sv('text', { class: 'boot-range-label', x: px(b) + 4, y: m.t + 12 }, F.f3(b)));
        $('#boot-readout').innerHTML = `<strong>${fmt(scores.length)}</strong> redraws so far. The middle 95% runs from <strong>${F.f3(a)}</strong> to <strong>${F.f3(b)}</strong>. Logged: ${F.f3(R.test.models.ours.macro_f1[1])} to ${F.f3(R.test.models.ours.macro_f1[2])}.`;
      } else {
        $('#boot-readout').innerHTML = scores.length ? `${scores.length} redraw${scores.length === 1 ? '' : 's'}: latest macro-F1 <strong>${F.f3(scores.at(-1))}</strong>. Each redraw picks ${fmt(nLots)} lots with replacement.` : 'Each redraw picks lots at random, with replacement, and scores the saved predictions again.';
      }
    }
    const add = (v) => { scores.push(v); counts[Math.max(0, Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins)))]++; };
    $('#boot-one').addEventListener('click', () => { add(draw()); render(); });
    $('#boot-many').addEventListener('click', () => {
      if (running) return;
      let left = 1000;
      const batch = () => {
        const n = reduceMotion ? left : Math.min(left, 25);
        for (let i = 0; i < n; i++) add(draw());
        left -= n; render();
        running = left > 0 ? requestAnimationFrame(batch) : null;
      };
      running = requestAnimationFrame(batch);
    });
    $('#boot-reset').addEventListener('click', () => { if (running) cancelAnimationFrame(running); running = null; scores = []; counts.fill(0); render(); });
    render();
  })();

  // ===================================================== S8 the 3x4 grid
  (() => {
    const kinds = [['k-means (ours)', 'kmeans'], ['PCA', 'pca'], ['Random filters', 'random']];
    const models = [['Logistic', 'logreg'], ['Ridge', 'ridge'], ['Linear SVM', 'svm'], ['Random forest', 'rf']];
    const values = kinds.flatMap(([, k]) => models.map(([, m]) => V[`P3-${k}-${m}`].macro_f1));
    const lo = Math.min(...values), hi = Math.max(...values), baseline = V['B2-flat-random-forest'].macro_f1;
    $('#grid-table').append(
      el('thead', {}, el('tr', {}, el('th', {}, 'Filters ↓ / Classifier →'), models.map(([n]) => el('th', {}, n)))),
      el('tbody', {}, kinds.map(([name, k]) => el('tr', {}, el('td', {}, name), models.map(([, mname]) => {
        const r = V[`P3-${k}-${mname}`], t = (r.macro_f1 - lo) / (hi - lo);
        return el('td', { class: ['heat', k === 'kmeans' && mname === 'logreg' ? 'pick' : '', r.macro_f1 < baseline ? 'below' : ''].join(' '), style: `--t:${t.toFixed(3)}` },
          el('b', {}, F.f3(r.macro_f1)), el('small', {}, ` ± ${F.f3(r.fold_std)}`));
      })))));
  })();

  // ===================================================== S8 ablations + cards
  (() => {
    const ref = V['P3-kmeans-logreg'];
    const cards = {
      'quadrant-pooling': { name: 'Quadrant pooling instead of rings', what: 'Average each filter in 4 quadrants (top-left, top-right, …) instead of 4 rings.', why: 'Quadrants care which way the wafer faces; rings don’t. Patterns defined by distance from the centre lose their best clue.', adopted: 'No. Rings stayed.', key: 'Center' },
      'no-rotation': { name: 'No rotation max', what: 'Each filter is tried only one way round instead of at 0/90/180/270°.', why: 'Lines at new angles go unseen.', adopted: 'No. Rotation stayed.', key: 'Scratch' },
      k16: { name: 'K = 16 filters', what: '16 k-means filters instead of 32.', why: 'Too few shapes to describe the patterns.', adopted: 'No.' },
      'denoise-none': { name: 'No cleanup', what: 'Skip the lonely-die cleanup entirely.', why: 'Speckle noise confuses the filters.', adopted: 'No.', key: 'none' },
      size64: { name: '64×64 instead of 32×32', what: 'Shrink maps to 64×64 instead of 32×32.', why: 'A small gain for 4× the pixels.', adopted: 'Not before the test: we stuck with the planned 32×32 (final_config.json records no reason for or against 64×64).' },
      k64: { name: 'K = 64 filters', what: '64 k-means filters instead of 32.', why: 'More shapes help, but double the features.', adopted: 'No: final_config.json records K = 32 to keep the feature vector small enough for the browser.' },
      'denoise-median': { name: 'Median cleanup', what: 'The 3×3 median filter instead of isolated-die removal.', why: 'Best overall score, but it erases thin lines. See S13 for the same trade-off on the test set.', adopted: 'No: final_config.json records that it costs scratches.', key: 'Scratch' },
    };
    const order = Object.keys(cards).sort((a, b) => V[`P3-ablation-${a}`].macro_f1 - V[`P3-ablation-${b}`].macro_f1);
    const card = $('#ablation-card');
    const rows = order.map((key) => {
      const r = V[`P3-ablation-${key}`], c = cards[key], d = r.macro_f1 - ref.macro_f1;
      return { name: c.name, link: [ref.macro_f1, r.macro_f1],
        marks: [{ cls: d < 0 ? 'down' : 'up', v: r.macro_f1, lo: r.macro_f1 - r.fold_std, hi: r.macro_f1 + r.fold_std, label: 'macro-F1 ± fold spread' }],
        val: `${F.f3(r.macro_f1)} <b class="${d < 0 ? 'neg' : 'posv'}">(${F.sf3(d)})</b>`,
        onClick: () => show(key) };
    });
    intervalChart($('#ablation-chart'), { min: 0.65, max: 0.8, ticks: [0.65, 0.7, 0.75, 0.8], rows, ref: { v: ref.macro_f1, label: `our recipe ${F.f3(ref.macro_f1)}` } });
    function show(key) {
      const r = V[`P3-ablation-${key}`], c = cards[key], d = r.macro_f1 - ref.macro_f1;
      // the class that moved most (or the one named in the card)
      const deltas = CLASSES.map((cl) => [cl, r.per_class[cl] - ref.per_class[cl]]);
      const [cls] = c.key ? [c.key] : deltas.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];
      const moved = r.per_class[cls] - ref.per_class[cls];
      rows.forEach((row, i) => row._nodes?.forEach((n) => n.classList.toggle('sel', order[i] === key)));
      card.classList.remove('swap'); void card.offsetWidth; card.classList.add('swap');
      card.innerHTML = `<h4>${c.name} <span class="${d < 0 ? 'neg' : 'posv'}">${F.sf3(d)}</span></h4>
        <dl><dt>What we changed</dt><dd>${c.what}</dd>
        <dt>What happened</dt><dd>Macro-F1 ${F.f3(ref.macro_f1)} → ${F.f3(r.macro_f1)}. ${cls} F1 ${moved < 0 ? 'fell' : 'rose'} from ${F.f3(ref.per_class[cls])} to ${F.f3(r.per_class[cls])}.</dd>
        <dt>Why</dt><dd>${c.why}</dd><dt>Did we adopt it?</dt><dd>${c.adopted}</dd></dl>`;
    }
    show(order[0]);
  })();

  // ===================================================== S8 random filters
  (() => {
    const tiles = $('#filter-tiles');
    const draw = (kind) => {
      const filters = kind === 'random' ? M.random_filters : M.filters;
      tiles.replaceChildren(...filters.map((f) => { const c = el('canvas', { width: 5, height: 5 }); K.paintFilter(c, f); return el('span', { class: 'ftile' }, c); }));
      tiles.classList.remove('swap'); void tiles.offsetWidth; tiles.classList.add('swap');
    };
    segmented($('#filter-kind'), draw);
    draw('kmeans');
    const pc = R.filter_check.per_class;
    const order = [...CLASSES].sort((a, b) => (pc[b].random - pc[b].kmeans) - (pc[a].random - pc[a].kmeans));
    intervalChart($('#filter-classes'), {
      min: 0.3, max: 1, ticks: [0.3, 0.5, 0.7, 0.9],
      rows: order.map((c) => ({ name: c, link: [pc[c].kmeans, pc[c].random],
        marks: [{ cls: 'other', v: pc[c].kmeans, label: 'k-means filters' }, { cls: 'ours', v: pc[c].random, label: 'random filters' }],
        val: F.sf3(pc[c].random - pc[c].kmeans) })),
    });
    $('#filter-classes').before(el('p', { class: 'legend' }, el('span', { class: 'key' }, el('i', { class: 'dot m-logistic' }), 'k-means'), el('span', { class: 'key' }, el('i', { class: 'dot m-ours' }), 'random'), el('span', { class: 'key muted' }, 'validation F1 per class')));
  })();

  // ---------------------------------------------------------------- self-check
  addCheck('model', () => {
    let worstF = 0, worstP = 0;
    for (const [id, expected] of Object.entries(M.check)) {
      const out = conv.fromWafer(wafer(id));
      const feats = [...out.pooled, ...out.extras];
      expected.features.forEach((v, i) => { worstF = Math.max(worstF, Math.abs(v - feats[i])); });
      conv.predict(out.pooled, out.extras).forEach((p, i) => { worstP = Math.max(worstP, Math.abs(p - expected.probs[i])); });
    }
    const n = Object.keys(M.check).length;
    return { ok: worstF < 2e-3 && worstP < 1e-3,
      detail: `The convolution, ring pooling and classifier in this page match the Python code on ${n} wafers (probabilities to 3 decimals; largest gap ${worstP.toFixed(4)}).` };
  });
})();
