/* Wafer Defects: the data story, S0 to S4 (hero, problem, data, split, preprocessing). */
(() => {
  'use strict';
  const K = window.KIT;
  if (!K) return;
  const { R, W, CLASSES, SIZE, $, $$, el, sv, segmented, press, F, wafer, unpack32, codesFor, paint, tile, sum,
    resize, removeIsolated, failingNeighbours, reduceMotion, intervalChart, showTip, hideTip, whileVisible, addCheck } = K;
  const story = W.story;
  const pct = (v, d = 0) => K.roundHalf(v * 100, d).toFixed(d) + '%';
  const fmt = F.int;

  // ============================================================== S0 hero wafer
  (() => {
    const svg = $('#hero-wafer'), caption = $('#hero-caption');
    const cell = 9.6, off = 26;                            // 32 cells x 9.6 = 307px, centred in 360
    svg.append(sv('circle', { class: 'hw-disc', cx: 180, cy: 180, r: 168 }),
      sv('path', { class: 'hw-notch', d: 'M166 348 a14 14 0 0 1 28 0' }));
    const cells = [];
    const grid = sv('g', { class: 'hw-grid' });
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const r = sv('rect', { x: off + x * cell + 0.6, y: off + y * cell + 0.6, width: cell - 1.2, height: cell - 1.2, rx: 1.4 });
      r.style.transitionDelay = `${y * 34 + x * 4}ms`;
      grid.append(r); cells.push(r);
    }
    const probe = sv('g', { class: 'hw-probe' },
      sv('rect', { class: 'hw-beam', x: 10, y: -3, width: 340, height: 6, rx: 3 }),
      sv('path', { class: 'hw-needle', d: 'M352 0 l-14 -8 v16 z' }));
    svg.append(grid, probe);
    const maps = CLASSES.map((c) => ({ label: c, small: wafer(story.examples[c][0]).small('isolated') }));
    let index = 0, timer = null;
    function show(i, animate) {
      const { label, small } = maps[i];
      svg.classList.toggle('scan', animate);
      cells.forEach((r, k) => {
        r.setAttribute('class', small.mask[k] ? (small.fail[k] ? 'f' : 'p') : 'o');
      });
      caption.innerHTML = `<b>${label}</b> · a real training wafer, cleaned and shrunk to 32×32`;
    }
    const onDisc = cells.map((_, k) => Math.hypot((k % SIZE) - 15.5, Math.floor(k / SIZE) - 15.5) <= 16);
    function cycle() {
      svg.classList.remove('scan'); void svg.getBoundingClientRect();
      cells.forEach((r, k) => r.setAttribute('class', onDisc[k] ? 'u' : 'o'));
      requestAnimationFrame(() => requestAnimationFrame(() => show(index, true)));
      index = (index + 1) % maps.length;
      timer = setTimeout(cycle, 4200);
    }
    if (reduceMotion) { show(3, false); return; }
    whileVisible(svg, () => { if (!timer) cycle(); }, () => { clearTimeout(timer); timer = null; });
  })();

  // ============================================== S1 the four story cards
  (() => {
    const disc = (cx, cy, r, n, cls, parent, pick = () => '') => {
      const g = sv('g'); const cell = (2 * r) / n;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const px = cx - r + x * cell, py = cy - r + y * cell;
        if (Math.hypot(px + cell / 2 - cx, py + cell / 2 - cy) > r - cell * 0.35) continue;
        g.append(sv('rect', { class: `${cls} ${pick(x, y, n)}`.trim(), x: px + 0.5, y: py + 0.5, width: cell - 1, height: cell - 1, rx: 0.8, 'data-x': x, 'data-y': y }));
      }
      parent.append(g); return g;
    };

    // 1 · the wafer: a silicon disc with a die grid and light sweeping across
    (() => {
      const svg = $('#sc-wafer');
      const defs = sv('defs');
      defs.append(sv('clipPath', { id: 'sc-clip' }));
      defs.firstChild.append(sv('circle', { cx: 100, cy: 70, r: 58 }));
      const grad = sv('linearGradient', { id: 'sc-sheen', x1: 0, x2: 1, y1: 0, y2: 0 });
      grad.append(sv('stop', { offset: 0, 'stop-color': '#fff', 'stop-opacity': 0 }), sv('stop', { offset: 0.5, 'stop-color': '#fff', 'stop-opacity': 0.55 }), sv('stop', { offset: 1, 'stop-color': '#fff', 'stop-opacity': 0 }));
      defs.append(grad);
      svg.append(defs, sv('circle', { class: 'sc-si', cx: 100, cy: 70, r: 58 }));
      disc(100, 70, 58, 12, 'sc-die', svg);
      svg.append(sv('path', { class: 'sc-notch', d: 'M93 128 a7 7 0 0 1 14 0' }));
      const sheen = sv('g', { 'clip-path': 'url(#sc-clip)' });
      sheen.append(sv('rect', { class: 'sc-sheen', x: -40, y: 0, width: 40, height: 140, fill: 'url(#sc-sheen)' }));
      svg.append(sheen);
    })();

    // 2 · the fab: layers deposited, patterned and etched on a cross-section
    (() => {
      const svg = $('#sc-fab');
      svg.append(sv('rect', { class: 'sc-sub', x: 20, y: 96, width: 160, height: 22, rx: 2 }),
        sv('text', { class: 'sc-small', x: 24, y: 112 }, 'silicon'));
      const layers = [['l1', 86], ['l2', 76], ['l3', 66]];
      layers.forEach(([cls, y], i) => {
        const g = sv('g', { class: `sc-layer ${cls}`, style: `--i:${i}` });
        g.append(sv('rect', { x: 20, y, width: 160, height: 10 }));
        // etched gaps appear after each layer lands
        [40, 80, 120, 150].forEach((x, k) => g.append(sv('rect', { class: 'sc-gap', x: x + (i * 7) % 15, y: y - 0.5, width: 10 + ((k + i) % 2) * 6, height: 11, style: `--k:${k}` })));
        svg.append(g);
      });
      const steps = sv('text', { class: 'sc-step', x: 100, y: 34, 'text-anchor': 'middle' });
      ['deposit', 'pattern', 'etch', 'repeat…'].forEach((w, i) => steps.append(sv('tspan', { class: `sc-w w${i}`, x: 100 }, w)));
      svg.append(steps, sv('path', { class: 'sc-beam', d: 'M100 42 L60 64 L140 64 Z' }));
    })();

    // 3 · the test: a probe card steps over a row of dies; each one lights up pass or fail
    (() => {
      const svg = $('#sc-probe');
      const row = sv('g', { class: 'sc-row' });
      const verdicts = 'ppfpppfpppppfpp';
      const dies = [...verdicts].map((v, i) => {
        const r = sv('rect', { class: 'sc-pdie', x: 22 + i * 26, y: 92, width: 22, height: 14, rx: 2 });
        row.append(r); return { r, v };
      });
      svg.append(sv('rect', { class: 'sc-chuck', x: 8, y: 106, width: 184, height: 10, rx: 3 }), row);
      const head = sv('g', { class: 'sc-head' });
      head.append(sv('path', { class: 'sc-card', d: 'M70 30 h60 l-14 26 h-32 Z' }),
        sv('path', { class: 'sc-needle', d: 'M92 56 L97 86 M100 56 L100 86 M108 56 L103 86' }),
        sv('rect', { class: 'sc-cable', x: 96, y: 8, width: 8, height: 22 }));
      const label = sv('text', { class: 'sc-verdict', x: 150, y: 50 });
      svg.append(head, label);
      if (reduceMotion) { dies.forEach((d) => d.r.classList.add(d.v === 'f' ? 'fail' : 'pass')); return; }
      let k = 0, timer = null;
      const step = () => {
        if (k === 0) dies.forEach((d) => d.r.classList.remove('pass', 'fail'));
        row.style.transform = `translateX(${-k * 26 + 78 - 22}px)`;
        head.classList.remove('down'); void head.getBoundingClientRect(); head.classList.add('down');
        setTimeout(() => {
          const d = dies[k - 1]; if (!d) return;
          d.r.classList.add(d.v === 'f' ? 'fail' : 'pass');
          label.textContent = d.v === 'f' ? 'FAIL' : 'pass';
          label.setAttribute('class', `sc-verdict ${d.v === 'f' ? 'fail' : 'pass'}`);
        }, 420);
        k = (k + 1) % (dies.length + 1);
        timer = setTimeout(step, k === 0 ? 1600 : 900);
      };
      whileVisible(svg, () => { if (!timer) step(); }, () => { clearTimeout(timer); timer = null; });
    })();

    // 4 · the map: results land in place and a ring appears
    (() => {
      const svg = $('#sc-map');
      const n = 12, ring = (x, y) => Math.hypot(x - 5.5, y - 5.5) > 4.6 || (x === 3 && y === 5) || (x === 7 && y === 3);
      const g = disc(100, 70, 58, n, 'sc-cell', svg, (x, y) => (ring(x, y) ? 'is-fail' : 'is-pass'));
      svg.append(sv('circle', { class: 'sc-outline', cx: 100, cy: 70, r: 58 }));
      const cells = [...g.children];
      if (reduceMotion) { cells.forEach((c) => c.classList.add('on')); return; }
      let timer = null;
      const run = () => {
        cells.forEach((c) => c.classList.remove('on'));
        cells.forEach((c, i) => { c.style.transitionDelay = `${i * 22}ms`; });
        requestAnimationFrame(() => requestAnimationFrame(() => cells.forEach((c) => c.classList.add('on'))));
        timer = setTimeout(run, cells.length * 22 + 2600);
      };
      whileVisible(svg, () => { if (!timer) run(); }, () => { clearTimeout(timer); timer = null; });
    })();
  })();

  // ======================================== S1 from a wafer to a wafer map
  (() => {
    const svg = $('#w2m'), N = 22, R = 118, L = { x: 140, y: 172 }, M = { x: 580, y: 172 }, cell = (2 * R) / N;
    const choices = ['Edge-Ring', 'Center', 'Donut', 'Edge-Loc', 'Scratch'];
    svg.append(
      sv('text', { class: 'w2m-title', x: L.x, y: 30, 'text-anchor': 'middle' }, 'The wafer on the prober'),
      sv('text', { class: 'w2m-title', x: M.x, y: 30, 'text-anchor': 'middle' }, 'The wafer map we get'),
      sv('circle', { class: 'w2m-wafer', cx: L.x, cy: L.y, r: R + 4 }),
      sv('circle', { class: 'w2m-mapdisc', cx: M.x, cy: M.y, r: R + 4 }),
      sv('path', { class: 'w2m-arrow', d: `M${L.x + R + 18} ${L.y} H${M.x - R - 22}` }),
      sv('path', { class: 'w2m-head', d: `M${M.x - R - 30} ${L.y - 7} l9 7 l-9 7` }),
      sv('text', { class: 'w2m-label', x: (L.x + M.x) / 2, y: L.y - 26, 'text-anchor': 'middle' }, 'pass or fail,'),
      sv('text', { class: 'w2m-label', x: (L.x + M.x) / 2, y: L.y - 11, 'text-anchor': 'middle' }, 'one die at a time'));
    const left = sv('g'), right = sv('g'), flash = sv('g');
    svg.append(left, right, flash);
    const probe = sv('g', { class: 'w2m-probe' });
    probe.append(sv('circle', { class: 'w2m-spot', r: cell * 0.9 }), sv('rect', { class: 'w2m-arm', x: -3, y: -64, width: 6, height: 38, rx: 2 }),
      sv('path', { class: 'w2m-needle', d: 'M0 0 L-6 -28 L6 -28 Z' }));
    svg.append(probe);
    let order = [], tiles = [], timer = null, at = 0, fails = 0, current = choices[0];

    function load(label) {
      current = label;
      const wf = wafer(story.examples[label][0]);
      const s = { fail: resize(wf.iso, wf.h, wf.w, N), mask: resize(wf.mask, wf.h, wf.w, N) };
      left.replaceChildren(); right.replaceChildren();
      order = []; tiles = [];
      for (let y = 0; y < N; y++) {
        const xs = [...Array(N).keys()];
        if (y % 2) xs.reverse();                               // the prober snakes back and forth
        for (const x of xs) {
          const k = y * N + x;
          if (!s.mask[k]) continue;
          const lx = L.x - R + x * cell, ly = L.y - R + y * cell, mx = M.x - R + x * cell;
          const a = sv('rect', { class: 'w2m-die', x: lx + 0.6, y: ly + 0.6, width: cell - 1.2, height: cell - 1.2, rx: 1 });
          const b = sv('rect', { class: 'w2m-cell', x: mx + 0.6, y: ly + 0.6, width: cell - 1.2, height: cell - 1.2, rx: 1 });
          left.append(a); right.append(b);
          order.push({ a, b, fail: !!s.fail[k], cx: lx + cell / 2, cy: ly + cell / 2 });
        }
      }
      tiles = order;
      press($('#w2m-pick'), label);
      restart();
    }
    function show(k) {
      const t = order[k];
      t.a.classList.add('done');
      t.b.classList.add(t.fail ? 'fail' : 'pass');
      if (t.fail) fails++;
      probe.style.transform = `translate(${t.cx}px, ${t.cy}px)`;
      $('#w2m-readout').innerHTML = `Tested <strong>${k + 1}</strong> of ${order.length} dies · <strong>${fails}</strong> failed` +
        (k === order.length - 1 ? ` · that's a <strong>${current}</strong> wafer: the failures make a shape.` : '');
    }
    function finish() { for (; at < order.length; at++) show(at); }
    function tick() {
      if (at >= order.length) { timer = setTimeout(() => { restart(); tick(); }, 3600); return; }
      show(at++);
      timer = setTimeout(tick, Math.max(8, 5200 / order.length));
    }
    function restart() {
      clearTimeout(timer); timer = null; at = 0; fails = 0;
      tiles.forEach((t) => { t.a.classList.remove('done'); t.b.classList.remove('pass', 'fail'); });
      if (reduceMotion) finish();
    }
    $('#w2m-pick').append(...choices.map((c) => el('button', { type: 'button', 'data-value': c, 'aria-pressed': 'false' }, c)));
    segmented($('#w2m-pick'), (v) => { load(v); if (!reduceMotion) tick(); });
    $('#w2m-replay').addEventListener('click', () => { restart(); if (!reduceMotion) tick(); });
    load(choices[0]);
    if (!reduceMotion) whileVisible(svg, () => { if (!timer) tick(); }, () => { clearTimeout(timer); timer = null; });
  })();

  // ===================================================== S1 meet the 9 patterns
  (() => {
    const shapes = {
      Center: 'a blob of failures in the middle', Donut: 'a ring of failures around a clean middle',
      'Edge-Loc': 'a patch of failures touching one part of the edge', 'Edge-Ring': 'failures all the way around the rim',
      Loc: 'a patch of failures somewhere inside, not at the edge', 'Near-full': 'almost the whole wafer failed',
      Random: 'failures sprinkled everywhere, no shape', Scratch: 'a thin line of failures',
      none: 'no pattern (may still have a few lone failures)',
    };
    const host = $('#patterns');
    CLASSES.forEach((label) => {
      const ids = story.examples[label];
      let at = 0;
      const canvas = el('canvas', { role: 'img' });
      const counter = el('span', { class: 'pcount' });
      const draw = () => {
        const wf = wafer(ids[at]);
        paint(canvas, codesFor.raw(wf), wf.h, wf.w);
        canvas.setAttribute('aria-label', `${label} wafer ${at + 1} of ${ids.length}, ${wf.h} by ${wf.w} dies`);
        counter.textContent = `${at + 1} / ${ids.length}`;
      };
      const extra = label === 'none' ? ` (picked from ${fmt(R.audit.none_rows)})` : '';
      const card = el('button', { type: 'button', class: 'pattern', 'aria-label': `${label}: ${shapes[label]}. Show another example.` },
        el('span', { class: 'pmap' }, canvas),
        el('span', { class: 'ptitle' }, el('i', { class: `cdot c-${label}` }), label, counter),
        el('span', { class: 'pshape' }, shapes[label]),
        el('span', { class: 'pn' }, `${fmt(R.audit.subset_class_counts[label])} wafers in our data${extra}`));
      card.addEventListener('click', () => {
        at = (at + 1) % ids.length;
        card.classList.remove('flip'); void card.offsetWidth; card.classList.add('flip');
        draw();
      });
      draw();
      host.append(card);
    });
  })();

  // ============================================================ S2 audit funnel
  // Centred bars, width to scale with the count, one row per step of the audit.
  (() => {
    const a = R.audit, host = $('#funnel');
    const rows = [
      { name: 'All wafer maps in the file', n: a.source_rows, cls: 'all', why: 'Every map in LSWMD.pkl.' },
      { name: 'Have a label', n: a.labeled_rows, cls: 'label', why: `${pct(a.labeled_share)} of the file.` },
      { name: 'No label: set aside', n: a.unlabeled_rows, cls: 'unlabeled', why: `${pct(a.unlabeled_share)} of the file. We can't learn from or check a wafer without a label.` },
      { name: "Labeled 'none' (no pattern)", n: a.none_rows, cls: 'none', why: `${pct(a.none_share_of_labeled)} of the labeled wafers.` },
      { name: 'Labeled with a defect pattern: keep all', n: a.defect_rows, cls: 'defect', why: `${pct(a.defect_share_of_labeled)} of the labeled wafers. Every one stays.` },
      { name: "'none' kept (random sample)", n: a.none_kept, cls: 'kept', why: `A random ${fmt(a.none_kept)} of the ${fmt(a.none_rows)} clean wafers (seed 42).` },
      { name: 'Our project sample', n: a.subset_rows, cls: 'sample', why: `${fmt(a.none_kept)} clean + ${fmt(a.defect_rows)} with a defect.` },
    ];
    host.append(...rows.map((r, i) => {
      const width = (r.n / a.source_rows) * 100;
      const bar = el('span', { class: `cf-bar cf-${r.cls}`, style: `width:${width}%;--i:${i}` });
      const tipHtml = `<b>${r.name}</b><br>${fmt(r.n)} wafers · ${pct(r.n / a.source_rows, r.n / a.source_rows < 0.1 ? 1 : 0)} of the file<br>${r.why}`;
      const row = el('div', { class: 'cf-row', tabindex: '0', 'aria-label': `${r.name}: ${fmt(r.n)}` },
        el('span', { class: 'cf-name' }, r.name), el('span', { class: 'cf-track' }, bar), el('b', { class: 'cf-val' }, fmt(r.n)));
      row.addEventListener('pointermove', (e) => showTip(e, tipHtml));
      row.addEventListener('pointerleave', hideTip);
      row.addEventListener('focus', () => { const b = row.getBoundingClientRect(); showTip({ clientX: b.left + b.width / 2, clientY: b.top }, tipHtml); });
      row.addEventListener('blur', hideTip);
      return row;
    }));
  })();

  // ========================================================= S2 class counts
  (() => {
    const host = $('#class-counts'), logBox = $('#counts-log');
    let set = 'subset';
    function draw() {
      const counts = set === 'subset' ? R.audit.subset_class_counts : R.audit.labeled_class_counts;
      const order = [...CLASSES].sort((a, b) => counts[b] - counts[a]);
      const top = Math.max(...order.map((c) => counts[c]));
      const scale = (n) => (logBox.checked ? Math.log10(n) / Math.log10(top) : n / top);
      const existing = new Map($$('.hrow', host).map((r) => [r.dataset.c, r]));
      const rows = order.map((c) => {
        let row = existing.get(c);
        if (!row) {
          row = el('div', { class: 'hrow', 'data-c': c },
            el('span', { class: 'name' }, el('i', { class: `cdot c-${c}` }), c),
            el('span', { class: 'track' }, el('span', { class: `bar c-bar-${c === 'none' ? 'none' : 'defect'}` }), el('span', { class: 'val' })));
        }
        $('.bar', row).style.width = `calc((100% - 70px) * ${scale(counts[c])})`;
        $('.val', row).textContent = fmt(counts[c]);
        return row;
      });
      host.replaceChildren(...rows);
    }
    segmented($('#counts-set'), (value) => { set = value; draw(); });
    logBox.addEventListener('change', draw);
    draw();
  })();

  // ========================================================== S2 size extremes
  (() => {
    const small = wafer(story.small), large = wafer(story.large);
    const stage = $('#size-stage');
    const make = (wf, name) => {
      const canvas = el('canvas', { role: 'img', 'aria-label': `${name} wafer` });
      const label = el('figcaption');
      const fig = el('figure', { class: 'size-wafer' }, canvas, label);
      return { wf, name, canvas, label, fig };
    };
    const items = [make(small, 'Smallest'), make(large, 'Biggest')];
    stage.append(...items.map((i) => i.fig));
    function show(mode) {
      stage.dataset.mode = mode;
      items.forEach((item) => {
        const { wf } = item;
        if (mode === 'real') {
          paint(item.canvas, codesFor.raw(wf), wf.h, wf.w);
          item.fig.style.width = `${(wf.w / large.w) * 72}%`;
          item.label.innerHTML = `<b>${wf.h}×${wf.w}</b> · ${fmt(wf.h * wf.w)} cells · ${wf.label}`;
        } else {
          const s = wf.small('isolated');
          paint(item.canvas, codesFor.result(s.mask, s.fail), SIZE, SIZE);
          item.fig.style.width = '34%';
          item.label.innerHTML = `<b>32×32</b> · was ${wf.h}×${wf.w}`;
        }
      });
      $('#size-notice').innerHTML = mode === 'real'
        ? `Left: ${small.h}×${small.w} (${fmt(small.h * small.w)} cells). Right: ${large.h}×${large.w} (${fmt(large.h * large.w)} cells). Notice the gap: one die is the same size in both pictures, and the big grid has ${F.x(R.sizes.ratio)} more cells.`
        : 'Notice: after resizing, both are the same 32×32 grid, so the same filters can slide over both. The big wafer’s rim is thinner because many dies were skipped.';
    }
    segmented($('#size-mode'), show);
    show('real');
  })();

  // =========================================================== S3 lot rows
  (() => {
    const blocks = story.same_lot;
    const seg = $('#lot-tabs');
    seg.append(...blocks.map((b, i) => el('button', { type: 'button', 'data-value': i, 'aria-pressed': String(i === 0) }, `${b.lot} · ${b.label}`)));
    function show(i) {
      const b = blocks[i];
      $('#lot-row-a').textContent = `Same lot: six wafers from ${b.lot}, all ${b.label}`;
      $('#lot-row-b').textContent = `Six different lots: six ${b.label} wafers, each from another lot`;
      $('#lot-mates').replaceChildren(...b.wafers.map((id, k) => {
        const wf = wafer(id);
        return tile(codesFor.raw(wf), wf.h, wf.w, `${b.lot}, wafer ${k + 1}`, `${b.label} wafer from ${b.lot}`);
      }));
      $('#lot-strangers').replaceChildren(...b.strangers.map((id) => {
        const wf = wafer(id);
        return tile(codesFor.raw(wf), wf.h, wf.w, wf.lot, `${b.label} wafer from ${wf.lot}`);
      }));
    }
    segmented(seg, (v) => show(Number(v)));
    show(0);

    const table = $('#overlap-table');
    const order = Object.keys(R.overlap).sort((a, b) => R.overlap[b].same_lot - R.overlap[a].same_lot);
    table.append(el('thead', {}, el('tr', {}, el('th', {}, 'Class'), el('th', {}, 'Two wafers, same lot'), el('th', {}, 'Two wafers, different lots'), el('th', {}, 'How much more alike'))),
      el('tbody', {}, order.map((c) => {
        const o = R.overlap[c];
        const bar = (v, cls) => el('span', { class: 'mini' }, el('i', { class: cls, style: `width:${v * 100}%` }), el('span', {}, F.f2(v)));
        return el('tr', {}, el('td', {}, el('i', { class: `cdot c-${c}` }), c), el('td', {}, bar(o.same_lot, 'same')), el('td', {}, bar(o.other_lots, 'other')),
          el('td', {}, `${(o.same_lot / o.other_lots).toFixed(1)}×`));
      })));
  })();

  // ===================================================== S3 split it yourself
  (() => {
    const svg = $('#splitter'), lots = W.split_demo;
    const Wd = 720, cols = 8, tileW = 84, tileH = 40, gapX = 4, gapY = 8, top = 10;
    const rows = Math.ceil(lots.length / cols);
    const boxY = top + rows * (tileH + gapY) + 34, boxH = 150;
    const trainBox = { x: 4, y: boxY, w: 470, h: boxH }, testBox = { x: 490, y: boxY, w: 226, h: boxH };
    const H = boxY + boxH + 6;
    svg.setAttribute('viewBox', `0 0 ${Wd} ${H}`);
    const tilesG = sv('g'), dotsG = sv('g');
    svg.append(
      sv('rect', { class: 'sbox train', x: trainBox.x, y: trainBox.y, width: trainBox.w, height: trainBox.h, rx: 10 }),
      sv('rect', { class: 'sbox test', x: testBox.x, y: testBox.y, width: testBox.w, height: testBox.h, rx: 10 }),
      sv('text', { class: 'sbox-label', x: trainBox.x + 12, y: trainBox.y + 20 }, 'Train'),
      sv('text', { class: 'sbox-label', x: testBox.x + 12, y: testBox.y + 20 }, 'Test'),
      sv('path', { class: 'sarrow', d: `M${Wd / 2} ${boxY - 26} v14 m-6 -6 l6 6 l6 -6` }),
      tilesG, dotsG);
    const dots = [];
    lots.forEach((lot, i) => {
      const x = (i % cols) * (tileW + gapX) + 6, y = top + Math.floor(i / cols) * (tileH + gapY);
      lot.tile = sv('rect', { class: 'stile', x, y, width: tileW, height: tileH, rx: 6 });
      lot.home = { x, y };
      tilesG.append(lot.tile, sv('text', { class: 'stile-label', x: x + 5, y: y + tileH - 5 }, lot.lot.replace('lot', '')));
      lot.dots = lot.maps.map((_, k) => {
        const hx = x + 10 + k * 13, hy = y + 13;
        const dot = sv('circle', { class: 'sdot', r: 5, cx: 0, cy: 0 });
        dot.style.transform = `translate(${hx}px, ${hy}px)`;
        dotsG.append(dot);
        const d = { dot, lot: i, home: [hx, hy] };
        dots.push(d);
        return d;
      });
    });
    function place(box, items) {
      const per = Math.floor((box.w - 24) / 13);
      items.forEach((d, k) => {
        const x = box.x + 16 + (k % per) * 13, y = box.y + 38 + Math.floor(k / per) * 13;
        d.dot.style.transitionDelay = reduceMotion ? '0ms' : `${Math.min(k * 6, 700)}ms`;
        d.dot.style.transform = `translate(${x}px, ${y}px)`;
      });
    }
    function run(mode) {
      const isTest = new Map();
      if (mode === 'naive') {
        const shuffled = [...dots].sort(() => Math.random() - 0.5);
        const nTest = Math.round(dots.length * 0.2);
        shuffled.forEach((d, k) => isTest.set(d, k < nTest));
      } else {
        const order = lots.map((_, i) => i).sort(() => Math.random() - 0.5);
        let n = 0; const testLots = new Set();
        for (const i of order) { if (n >= dots.length * 0.2) break; testLots.add(i); n += lots[i].dots.length; }
        dots.forEach((d) => isTest.set(d, testLots.has(d.lot)));
      }
      let leaked = 0, splitLots = 0;
      lots.forEach((lot) => {
        const t = lot.dots.filter((d) => isTest.get(d)).length;
        const leak = t > 0 && t < lot.dots.length;
        lot.tile.classList.toggle('leak', leak);
        if (leak) splitLots++;
        lot.dots.forEach((d) => d.dot.classList.toggle('leak', leak));
      });
      dots.forEach((d) => { d.dot.classList.toggle('test', isTest.get(d)); d.dot.classList.toggle('train', !isTest.get(d)); });
      const train = dots.filter((d) => !isTest.get(d)), test = dots.filter((d) => isTest.get(d));
      leaked = test.filter((d) => d.dot.classList.contains('leak')).length;
      // keep lot-mates next to each other inside each box
      const byLot = (a, b) => a.lot - b.lot;
      place(trainBox, train.sort(byLot)); place(testBox, test.sort(byLot));
      $('#split-readout').innerHTML = mode === 'naive'
        ? `<strong>${leaked} of ${test.length}</strong> test wafers have a sibling in training, across ${splitLots} lots cut in two. Notice the red rings: the model would be graded on lots it has already seen.`
        : `<strong>0 of ${test.length}</strong> test wafers have a sibling in training. Whole lots moved together, so every test lot is new to the model.`;
    }
    $('#split-naive').addEventListener('click', () => run('naive'));
    $('#split-lot').addEventListener('click', () => run('lot'));
    $('#split-readout').textContent = 'Press a button to split these 40 lots into train and test.';
  })();

  // ================================================================= S3 folds
  (() => {
    const s = R.split, pick = $('#fold-pick'), bar = $('#foldbar'), table = $('#fold-table');
    pick.append(...[0, 1, 2, 3, 4].map((k) => el('button', { type: 'button', 'data-value': k, 'aria-pressed': String(k === 0) }, `Round ${k + 1}`)));
    const blocks = [0, 1, 2, 3, 4].map((k) => el('button', { type: 'button', class: 'fold', 'data-value': k }));
    bar.append(...blocks, el('div', { class: 'fold locked', html: `<b>Locked test set</b>${fmt(s.test_rows)} wafers, not touched` }));
    table.append(
      el('thead', {}, el('tr', {}, el('th', {}, 'Class'), [0, 1, 2, 3, 4].map((k) => el('th', { 'data-col': k }, `Fold ${k + 1}`)), el('th', { class: 'locked' }, 'Test'))),
      el('tbody', {}, CLASSES.map((c) => el('tr', {}, el('td', {}, c),
        [0, 1, 2, 3, 4].map((k) => el('td', { 'data-col': k }, fmt(s.fold_class_counts[k][c]))),
        el('td', { class: 'locked' }, fmt(s.test_class_counts[c]))))),
      el('tfoot', {}, el('tr', {}, el('td', {}, 'Wafers'), [0, 1, 2, 3, 4].map((k) => el('td', { 'data-col': k }, fmt(s.fold_rows[k]))),
        el('td', { class: 'locked' }, fmt(s.test_rows)))));
    function show(round) {
      press(pick, round);
      blocks.forEach((b, k) => {
        b.classList.toggle('val', k === round);
        b.innerHTML = `<b>Fold ${k + 1}</b>${k === round ? 'validate' : 'train'}`;
      });
      $$('[data-col]', table).forEach((c) => c.classList.toggle('on', Number(c.dataset.col) === round));
      $('#fold-readout').innerHTML = `Round ${round + 1}: train on four folds (${fmt(s.train_rows - s.fold_rows[round])} wafers), then check on fold ${round + 1} ` +
        `(<strong>${fmt(s.fold_rows[round])} wafers from ${fmt(s.fold_lots[round])} lots</strong> the model hasn't seen).`;
    }
    segmented(pick, (v) => show(Number(v)));
    blocks.forEach((b, k) => b.addEventListener('click', () => show(k)));
    show(0);
  })();

  // ===================================================== S3 union-find picture
  (() => {
    const svg = $('#uf-svg');
    // Three lots of wafers; two identical maps tie lot B to lot C.
    const lotsDef = [{ name: 'lot A', n: 5, x: 30 }, { name: 'lot B', n: 4, x: 200 }, { name: 'lot C', n: 4, x: 345 }, { name: 'lot D', n: 3, x: 445 }];
    const nodes = [];
    lotsDef.forEach((l, li) => {
      for (let k = 0; k < l.n; k++) nodes.push({ lot: li, x: l.x + (k % 3) * 26 + (k >= 3 ? 13 : 0), y: 50 + Math.floor(k / 3) * 30 });
      svg.append(sv('text', { class: 'uf-lot', x: l.x, y: 24 }, l.name));
    });
    const edges = [];
    lotsDef.forEach((_, li) => {
      const members = nodes.filter((n) => n.lot === li);
      members.slice(1).forEach((n, k) => edges.push({ a: members[k], b: n, kind: 'lot' }));
    });
    const bLast = nodes.filter((n) => n.lot === 1).at(-1), cFirst = nodes.find((n) => n.lot === 2);
    edges.push({ a: bLast, b: cFirst, kind: 'dup' });
    const hulls = [
      { lots: [0], cls: 'g1', label: 'group 1 → train' }, { lots: [1, 2], cls: 'g2', label: 'group 2 (two lots) → test' },
      { lots: [3], cls: 'g3', label: 'group 3 → train' }];
    hulls.forEach((h) => {
      const members = nodes.filter((n) => h.lots.includes(n.lot));
      const x0 = Math.min(...members.map((n) => n.x)) - 16, x1 = Math.max(...members.map((n) => n.x)) + 16;
      const y0 = Math.min(...members.map((n) => n.y)) - 16, y1 = Math.max(...members.map((n) => n.y)) + 16;
      svg.append(sv('rect', { class: `uf-hull ${h.cls}`, x: x0, y: y0, width: x1 - x0, height: y1 - y0, rx: 16 }),
        sv('text', { class: `uf-glabel ${h.cls}`, x: x0, y: y1 + 20 }, h.label));
    });
    edges.forEach((e) => svg.append(sv('line', { class: `uf-edge ${e.kind}`, x1: e.a.x, y1: e.a.y, x2: e.b.x, y2: e.b.y, pathLength: 1 })));
    nodes.forEach((n) => svg.append(sv('circle', { class: 'uf-node', cx: n.x, cy: n.y, r: 7 })));
    svg.append(sv('text', { class: 'uf-dup', x: (bLast.x + cFirst.x) / 2, y: bLast.y - 14, 'text-anchor': 'middle' }, 'identical map'));
  })();

  // ======================================================== S4 healthy wafer
  (() => {
    const wf = wafer(story.noise);
    const canvas = el('canvas', { role: 'img', 'aria-label': 'A wafer labeled none with scattered failing dies' });
    $('#noise-wafer').append(canvas);
    const lonely = wf.nFail - sum(wf.iso);
    const views = {
      raw: () => [codesFor.raw(wf), `<strong>${wf.nFail}</strong> of ${fmt(wf.dies)} dies fail (${pct(wf.nFail / wf.dies)}). An engineer looked at this wafer and saw no pattern: its label is <em>none</em>.`],
      mark: () => [codesFor.marked(wf.mask, wf.fail, wf.iso), `<strong>${lonely}</strong> of the ${wf.nFail} failing dies have no failing neighbour among the 8 dies around them. These are the purest form of noise.`],
      clean: () => [codesFor.result(wf.mask, wf.iso), `With the lonely dies removed, <strong>${sum(wf.iso)}</strong> failing dies remain: small chance clusters. Our rule is cautious on purpose, as the stepper below shows.`],
    };
    const show = (mode) => { const [codes, text] = views[mode](); paint(canvas, codes, wf.h, wf.w); $('#noise-readout').innerHTML = text; };
    segmented($('#noise-mode'), show);
    show('raw');
  })();

  // ================================================== S4 preprocessing stepper
  const stepper = (() => {
    const select = $('#step-class'), slider = $('#step-slider'), labels = $$('#slider-labels li');
    select.append(...CLASSES.map((c) => el('option', { value: c }, c)));
    const state = { cls: 'Scratch', idx: 0, clean: 'isolated', step: 1 };
    select.value = state.cls;
    const names = { isolated: 'isolated-die removal', median: 'the 3×3 median filter', none: 'no cleanup' };

    function render(animate = true) {
      const wf = wafer(story.examples[state.cls][state.idx]);
      const cleaned = wf.cleaned(state.clean);
      const kept = sum(Uint8Array.from(cleaned, (v, i) => v & wf.fail[i]));
      const removed = wf.nFail - kept;
      const added = sum(Uint8Array.from(cleaned, (v, i) => v & !wf.fail[i]));
      const small = wf.small(state.clean);
      const views = $('#stage-views');
      let content, title, desc;
      switch (state.step) {
        case 1:
          content = [tile(codesFor.raw(wf), wf.h, wf.w, `raw map, ${wf.h}×${wf.w}`, 'Raw wafer map')];
          title = 'Raw map';
          desc = `This is what comes out of the tester: ${fmt(wf.nFail)} failing dies on a ${wf.h}×${wf.w} grid. Grey-blue passed, red failed. Every wafer has its own size.`;
          break;
        case 2:
          content = [tile(codesFor.marked(wf.mask, wf.fail, cleaned), wf.h, wf.w, `after ${names[state.clean]}`, 'Wafer after cleanup, removed dies in orange')];
          title = 'Clean up lonely dies';
          desc = state.clean === 'isolated'
            ? `${fmt(removed)} lonely failing dies removed (orange). We switch off any failing die that has no failing neighbour among the 8 around it. A lone dot is usually random noise. Dies in a line always have a failing neighbour, so lines survive.`
            : state.clean === 'median'
              ? `The median filter takes a vote in every 3×3 box. Here it removes ${fmt(removed)} of the ${fmt(wf.nFail)} failing dies (orange)${added ? ` and fills in ${fmt(added)} new ones` : ''}. Strong on noise, but it wipes out 1-die-thin lines.`
              : 'No cleanup: every failing die stays, specks and all. On validation this scored lowest of the three options.';
          break;
        case 3:
          content = [tile(codesFor.result(small.mask, small.fail), SIZE, SIZE, `32×32, was ${wf.h}×${wf.w}`, '32 by 32 map')];
          title = 'Shrink to 32×32';
          desc = `Maps range from ${R.sizes.subset_min_side} to ${R.sizes.subset_max_side} dies on a side. We shrink every one to 32×32 by picking the nearest die. We never blend neighbouring dies, because blending would invent “half-failed” dies that don’t exist.`;
          break;
        default:
          content = [tile(codesFor.failOnly(small.fail), SIZE, SIZE, 'fail map: where the failures are', 'Fail map'),
            tile(codesFor.maskOnly(small.mask), SIZE, SIZE, 'wafer mask: where the wafer is', 'Wafer mask')];
          title = 'Fail map + wafer mask';
          desc = 'We keep two 32×32 pictures: where the failures are, and where the wafer is. The mask lets later steps ignore the empty corners of the square, so an edge failure isn’t confused with “off the wafer”.';
      }
      views.classList.toggle('two', content.length === 2);
      if (animate && !reduceMotion) { views.classList.remove('swap'); void views.offsetWidth; views.classList.add('swap'); }
      views.replaceChildren(...content);
      $('#step-count').textContent = `Step ${state.step} of 4 · ${state.cls} example ${state.idx + 1} of ${story.examples[state.cls].length}`;
      $('#stage-title').textContent = title;
      $('#stage-desc').textContent = desc;
      $('#stage-facts').replaceChildren(...[
        ['Failing share (saved)', `${pct(wf.nFail / wf.dies, 1)} of ${fmt(wf.dies)} dies`],
        ['Removed share (saved)', state.clean === 'none' ? '0% (no cleanup)' : `${pct(removed / Math.max(1, wf.nFail), 1)} of failing dies`],
        ['Cleanup', names[state.clean]],
      ].flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, v)]));
      labels.forEach((l, i) => { l.classList.toggle('on', i + 1 === state.step); l.classList.toggle('done', i + 1 < state.step); });
      slider.value = state.step;
      slider.style.setProperty('--p', `${((state.step - 1) / 3) * 100}%`);
      $('#step-notice').innerHTML = state.cls === 'Scratch' && state.step === 2 && state.clean === 'median'
        ? `Notice: the scratch is gone. The median filter removed ${fmt(removed)} of ${fmt(wf.nFail)} failing dies, line included.`
        : state.cls === 'Random' && state.step === 2
          ? `Notice: on this Random wafer ${state.clean === 'isolated' ? 'our rule barely touches anything' : state.clean === 'median' ? 'the median filter strips out much of the scatter' : 'nothing is removed'}. The two saved numbers keep the scatter on record either way.`
          : state.step === 2 && state.clean === 'isolated'
            ? 'Notice: lines and blobs survive; only specks with no failing neighbour go.'
            : '';
    }
    slider.addEventListener('input', () => { state.step = Number(slider.value); render(); });
    labels.forEach((l, i) => { l.addEventListener('click', () => { state.step = i + 1; render(); }); });
    select.addEventListener('change', () => { state.cls = select.value; state.idx = 0; render(); });
    $('#step-another').addEventListener('click', () => { state.idx = (state.idx + 1) % story.examples[state.cls].length; render(); });
    segmented($('#step-clean'), (v) => { state.clean = v; render(); });
    render(false);
    return {
      show(next) {
        Object.assign(state, next); select.value = state.cls; press($('#step-clean'), state.clean); render();
        $('#stepper').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
      },
    };
  })();
  K.stepper = stepper;

  // =============================================== S4 the two rules, animated
  (() => {
    const N = 9, cell = 20, off = 10;
    const line = [[1, 7], [2, 6], [3, 5], [4, 4], [5, 3], [6, 2], [7, 1]];
    const specks = [[1, 1], [7, 7], [4, 1], [1, 4]];
    const failing = new Set([...line, ...specks].map(([x, y]) => y * N + x));
    const fail = new Uint8Array(N * N); failing.forEach((i) => { fail[i] = 1; });
    const mask = new Uint8Array(N * N).fill(1);
    const order = [...line, ...specks].sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    function setup(svg, rule) {
      const rects = [];
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const r = sv('rect', { x: off + x * cell + 1, y: off + y * cell + 1, width: cell - 2, height: cell - 2, rx: 3 });
        svg.append(r); rects.push(r);
      }
      const win = sv('rect', { class: 'rule-win', x: 0, y: 0, width: cell * 3, height: cell * 3, rx: 4 });
      const label = sv('text', { class: 'rule-label', x: 100, y: 197, 'text-anchor': 'middle' });
      svg.append(win, label);
      const reset = () => { rects.forEach((r, i) => r.setAttribute('class', fail[i] ? 'f' : 'p')); win.style.opacity = 0; label.textContent = ''; };
      let k = 0, timer = null;
      const step = () => {
        if (k === 0) reset();
        if (k < order.length) {
          const [x, y] = order[k], i = y * N + x;
          const n = failingNeighbours(fail, N, N, y, x);
          const keep = rule === 'iso' ? n > 0 : n + 1 >= 5;
          win.style.opacity = 1;
          win.style.transform = `translate(${off + (x - 1) * cell}px, ${off + (y - 1) * cell}px)`;
          label.textContent = rule === 'iso' ? `${n} failing neighbour${n === 1 ? '' : 's'} → ${keep ? 'keep' : 'remove'}` : `${n + 1} of 9 fail → ${keep ? 'keep' : 'erase'}`;
          rects[i].setAttribute('class', keep ? 'f keep' : 'f gone');
          k++;
          timer = setTimeout(step, 720);
        } else {
          win.style.opacity = 0;
          const left = order.filter(([x, y]) => (rule === 'iso' ? failingNeighbours(fail, N, N, y, x) > 0 : failingNeighbours(fail, N, N, y, x) + 1 >= 5)).length;
          label.textContent = rule === 'iso' ? `line kept, ${specks.length} specks removed` : `everything erased (${left} left)`;
          rects.forEach((r) => { if (r.classList.contains('gone')) r.setAttribute('class', 'p was'); });
          k = 0;
          timer = setTimeout(step, 2600);
        }
      };
      reset();
      if (reduceMotion) {
        const out = rule === 'iso' ? removeIsolated(fail, N, N) : K.medianFilter(fail, mask, N, N);
        rects.forEach((r, i) => r.setAttribute('class', out[i] ? 'f' : fail[i] ? 'p was' : 'p'));
        label.textContent = rule === 'iso' ? 'line kept, specks removed' : 'everything erased';
        return;
      }
      whileVisible(svg, () => { if (!timer) step(); }, () => { clearTimeout(timer); timer = null; k = 0; });
    }
    setup($('#rule-iso'), 'iso');
    setup($('#rule-med'), 'med');
  })();

  // ========================================== S4 median vs scratch + inspector
  (() => {
    const ids = story.scratches, pick = $('#scratch-pick'), triple = $('#scratch-triple'), mark = $('#scratch-mark');
    pick.append(...ids.map((id, i) => el('button', { type: 'button', 'data-value': i, 'aria-pressed': String(i === 0) }, `Scratch wafer ${i + 1}`)));
    let wf = null, cellAt = null, marker = null;
    function inspect(y, x) {
      cellAt = [y, x];
      const { h, w, fail, mask } = wf;
      marker.style.cssText = `left:${(x / w) * 100}%;top:${(y / h) * 100}%;width:${100 / w}%;height:${100 / h}%`;
      const view = [];
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        const inside = yy >= 0 && yy < h && xx >= 0 && xx < w && mask[yy * w + xx];
        view.push(el('i', { class: [inside ? (fail[yy * w + xx] ? 'f' : '') : 'o', !dy && !dx ? 'c' : ''].join(' ').trim() }));
      }
      $('#inspector-window').replaceChildren(...view);
      const n = failingNeighbours(fail, h, w, y, x), here = fail[y * w + x];
      let html;
      if (!mask[y * w + x]) html = 'This position is off the wafer. Neither filter does anything here.';
      else if (!here) html = `This die passed. ${n} of its 8 neighbours fail. ` + (n >= 5 ? 'The median filter would turn it into a failing die, because most of the box fails. ' : 'The median filter leaves it alone. ') + 'Our rule only ever switches failing dies off, so it never touches this one.';
      else {
        const votes = n + 1;
        html = `<span class="verdict"><strong>Median filter:</strong> ${votes} of the 9 dies in this box fail, and it needs 5. <b class="${votes >= 5 ? 'kept' : 'gone'}">${votes >= 5 ? 'Kept' : 'Erased'}.</b></span>` +
          `<span class="verdict"><strong>Our rule:</strong> ${n} failing ${n === 1 ? 'neighbour' : 'neighbours'}, and it needs 1. <b class="${n ? 'kept' : 'gone'}">${n ? 'Kept' : 'Removed'}.</b></span>`;
      }
      $('#inspector-verdict').innerHTML = html;
    }
    function draw() {
      const show = (after) => (mark.checked ? codesFor.marked(wf.mask, wf.fail, after) : codesFor.result(wf.mask, after));
      const nMed = sum(wf.med), nIso = sum(wf.iso);
      const raw = tile(codesFor.raw(wf), wf.h, wf.w, `<b>Raw map</b>${wf.nFail} failing dies · ${wf.h}×${wf.w}`, 'Raw Scratch wafer');
      const canvas = $('canvas', raw);
      marker = el('div', { class: 'marker' });
      const wrap = el('div', { class: 'canvas-wrap' }, canvas, marker);
      raw.prepend(wrap);
      triple.replaceChildren(raw,
        tile(show(wf.med), wf.h, wf.w, `<b>After the median filter</b>${nMed} failing ${nMed === 1 ? 'die remains' : 'dies remain'}`, 'After the median filter'),
        tile(show(wf.iso), wf.h, wf.w, `<b>After our rule</b>${nIso} failing dies remain`, 'After isolated-die removal'));
      const pointer = (event) => {
        const box = canvas.getBoundingClientRect();
        inspect(Math.min(wf.h - 1, Math.max(0, Math.floor(((event.clientY - box.top) / box.height) * wf.h))),
          Math.min(wf.w - 1, Math.max(0, Math.floor(((event.clientX - box.left) / box.width) * wf.w))));
      };
      wrap.addEventListener('pointermove', pointer);
      wrap.addEventListener('pointerdown', pointer);
      inspect(cellAt[0], cellAt[1]);
    }
    function choose(i) {
      wf = wafer(ids[i]);
      const onLine = [];
      for (let y = 0; y < wf.h; y++) for (let x = 0; x < wf.w; x++) {
        const k = y * wf.w + x;
        if (wf.iso[k] && !wf.med[k] && failingNeighbours(wf.fail, wf.h, wf.w, y, x) === 2) onLine.push([y, x]);
      }
      const cy = onLine.reduce((t, p) => t + p[0], 0) / onLine.length, cx = onLine.reduce((t, p) => t + p[1], 0) / onLine.length;
      cellAt = onLine.sort((a, b) => Math.hypot(a[0] - cy, a[1] - cx) - Math.hypot(b[0] - cy, b[1] - cx))[0] || [wf.h >> 1, wf.w >> 1];
      draw();
    }
    segmented(pick, (v) => choose(Number(v)));
    mark.addEventListener('change', draw);
    choose(0);
  })();

  // ===================================================== S4 gallery, 9 classes
  (() => {
    const gallery = $('#gallery');
    const show = (mode) => gallery.replaceChildren(...CLASSES.map((label) => {
      const wf = wafer(story.examples[label][0]);
      const codes = mode === 'raw' ? codesFor.raw(wf) : mode === 'mark' ? codesFor.marked(wf.mask, wf.fail, wf.iso) : codesFor.result(wf.mask, wf.iso);
      return tile(codes, wf.h, wf.w, `<b>${label}</b>removed ${fmt(wf.nFail - sum(wf.iso))} of ${fmt(wf.nFail)} failing dies`, `${label} wafer`);
    }));
    segmented($('#gallery-mode'), show);
    show('mark');
  })();

  // ===================================================== S4 summary numbers
  (() => {
    const S = R.summary_numbers, table = $('#summary-table');
    const bar = (v, cls) => el('span', { class: 'mini' }, el('i', { class: cls, style: `width:${v * 100}%` }), el('span', {}, F.f3(v)));
    table.append(
      el('thead', {}, el('tr', {}, el('th', {}, 'Class'), el('th', {}, 'Failing share'), el('th', {}, 'Removed by our cleanup'), el('th', {}, 'Removed by a median filter'))),
      el('tbody', {}, CLASSES.map((c) => el('tr', { class: c === 'Random' || c === 'Scratch' ? 'hl' : '' },
        el('td', {}, el('i', { class: `cdot c-${c}` }), c), el('td', {}, bar(S[c].failing_share, 'share')),
        el('td', {}, bar(S[c].removed_isolated, 'ours')), el('td', {}, bar(S[c].removed_median, 'median'))))));
  })();

  // ============================================== S4 clean first, then resize
  (() => {
    const wf = wafer(story.scratches[0]);
    const ours = wf.small('isolated');
    const other = removeIsolated(resize(wf.fail, wf.h, wf.w), SIZE, SIZE);
    $('#order-triple').append(
      tile(codesFor.raw(wf), wf.h, wf.w, `<b>Raw map, ${wf.h}×${wf.w}</b>a thin scratch plus scattered specks`, 'Raw Scratch wafer'),
      tile(codesFor.result(ours.mask, ours.fail), SIZE, SIZE, `<b>Clean, then shrink (ours)</b>${sum(ours.fail)} failing pixels: the scratch leaves a trace`, 'Clean then shrink'),
      tile(codesFor.result(ours.mask, other), SIZE, SIZE, `<b>Shrink, then clean</b>${sum(other)} failing pixels: much of the trace is erased`, 'Shrink then clean'));
  })();

  // ---------------------------------------------------------------- self-check
  addCheck('preprocessing', () => {
    const weighted = (a) => { let t = 0; for (let i = 0; i < a.length; i++) t += a[i] * ((i % 97) + 1); return t; };
    const ids = Object.keys(W.raw), bad = [];
    ids.forEach((id) => {
      const wf = wafer(id), c = W.raw[id].check;
      const ok = wf.nFail === c.fail && sum(wf.iso) === c.isolated && sum(wf.med) === c.median &&
        weighted(wf.small('isolated').fail) === c.isolated32 && weighted(wf.small('isolated').mask) === c.mask32 &&
        weighted(wf.small('median').fail) === c.median32 && weighted(wf.small('none').fail) === c.none32;
      if (!ok) bad.push(id);
    });
    return { ok: !bad.length, detail: bad.length ? `${bad.length} wafers differ` : `Cleanup and resizing in this page match the Python code on all ${ids.length} training wafers shown.` };
  });
})();
