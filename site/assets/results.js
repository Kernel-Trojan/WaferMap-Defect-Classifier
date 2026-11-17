/* Wafer Defects: final results, robustness, errors, CNN, denoise trade-off, real fab (S9 to S14). */
(() => {
  'use strict';
  const K = window.KIT;
  if (!K) return;
  const { R, W, M, CLASSES, SIZE, $, $$, el, segmented, press, F, unpack32, codesFor, paint, tile,
    reduceMotion, intervalChart, lineChart, probBars, addCheck, whileVisible } = K;
  const T = R.test;
  const fmt = F.int;
  const pct = (v, d = 0) => K.roundHalf(v * 100, d).toFixed(d) + '%';
  const MODELS = { ours: 'Ours', svm: 'Ours + SVM', forest: 'Random forest on pixels', logistic: 'Logistic on pixels', cnn: 'Textbook CNN' };
  const demoNote = M.source === 'saved' ? 'Probabilities from the saved final model.'
    : `Probabilities come from the frozen recipe refit on the training lots (the saved model file isn't in the repository); at 0° it gives the same answer as the saved final-test prediction for every wafer listed here.`;

  // ================================================================= S9 gaps
  (() => {
    const g = T.gaps;
    const rows = [
      ['logistic', 'B2 logistic on pixels', 'clear win: our whole pipeline over raw pixels'],
      ['forest', 'B2 random forest', 'clear win over the best baseline'],
      ['svm', 'Linear SVM on our features', 'a tie: the range crosses 0'],
      ['cnn', 'Textbook CNN (post-hoc)', 'clear loss'],
    ];
    intervalChart($('#gap-plot'), {
      min: -0.12, max: 0.26, ticks: [-0.1, 0, 0.1, 0.2], ref: { v: 0, label: 'no difference' },
      tickFormat: (t) => (t > 0 ? '+' : t < 0 ? '−' : '') + Math.abs(t).toFixed(1),
      rows: rows.map(([key, name, read]) => {
        const [v, lo, hi] = g[key];
        return { name: `Ours minus ${name}`, sub: read, marks: [{ cls: v > 0 && lo > 0 ? 'up' : hi < 0 ? 'down' : 'other', v, lo, hi, label: 'gap in macro-F1' }],
          val: `<b>${F.sf3(v)}</b> (${F.sf3(lo)} to ${F.sf3(hi)})` };
      }),
    });
  })();

  // ============================================================ S9 per class
  (() => {
    const shown = new Set(['ours', 'forest', 'cnn']);
    const box = $('#perclass-models');
    Object.entries(MODELS).forEach(([key, name]) => {
      const input = el('input', { type: 'checkbox', checked: shown.has(key) ? 'checked' : null });
      input.addEventListener('change', () => { input.checked ? shown.add(key) : shown.delete(key); draw(); });
      box.append(el('label', { class: 'check' }, input, el('i', { class: `dot m-${key}` }), name));
    });
    const order = [...CLASSES].sort((a, b) => T.per_class.ours[b].f1 - T.per_class.ours[a].f1);
    function draw() {
      intervalChart($('#perclass'), {
        min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1],
        rows: order.map((c) => {
          const marks = Object.keys(MODELS).filter((k) => shown.has(k)).map((k) => {
            const p = T.per_class[k][c];
            return { cls: k, v: p.f1, lo: k === 'ours' ? p.range[0] : null, hi: k === 'ours' ? p.range[1] : null, label: MODELS[k] };
          });
          const tipRows = Object.keys(MODELS).filter((k) => shown.has(k)).map((k) => {
            const p = T.per_class[k][c];
            return `<div class="trow"><span><i class="t-${k}"></i>${MODELS[k]}</span><span>P ${F.f2(p.precision)} · R ${F.f2(p.recall)} · F1 ${F.f3(p.f1)}</span></div>`;
          }).join('');
          const ours = T.per_class.ours[c];
          return { name: c, sub: `${fmt(R.split.test_class_counts[c])} test wafers`, marks,
            tip: `<b>${c}</b>${tipRows}<div class="trow"><span>Ours, 95% range</span><span>${F.f3(ours.range[0])} to ${F.f3(ours.range[1])}</span></div>`,
            val: `<b>${F.f3(ours.f1)}</b>` };
        }),
      });
    }
    draw();
  })();

  // ================================================================ S9 ECE
  (() => {
    const rows = [['ours', 'Ours (temperature-scaled)'], ['logistic', 'Logistic on pixels'], ['forest', 'Random forest on pixels']];
    intervalChart($('#ece-bars'), {
      min: 0, max: 0.3, ticks: [0, 0.1, 0.2, 0.3],
      rows: rows.map(([k, name]) => {
        const [v, lo, hi] = T.models[k].ece;
        return { name, marks: [{ cls: k, v, lo, hi, label: 'ECE' }], val: `<b>${F.f3(v)}</b> (${F.f3(lo)} to ${F.f3(hi)})` };
      }),
    });
  })();

  // ================================================== S10 rotation playground
  (() => {
    const demos = W.rotation, angles = R.rotation.angles;
    const select = $('#rot-wafer'), slider = $('#rot-angle'), canvas = $('#rot-canvas');
    const counters = {};
    demos.forEach((d, i) => {
      counters[d.label] = (counters[d.label] || 0) + 1;
      const right = d.saved_prediction === d.label;
      select.append(el('option', { value: i }, `${d.label} #${counters[d.label]} (${right ? 'called right' : `called ${d.saved_prediction}`})`));
    });
    const update = probBars($('#rot-probs'));
    const chart = lineChart($('#rotation-chart'), {
      label: 'Test macro-F1 for four models as the test wafers are rotated from 0 to 345 degrees',
      xs: angles, bands: [90, 180, 270],
      xTicks: [0, 45, 90, 135, 180, 225, 270, 315].map((v) => ({ v, label: `${v}°`, wideOnly: v % 90 !== 0 })),
      fmtX: (v) => `Rotated ${v}°`, xTitle: 'Rotation of the test wafers (quarter turns shaded)', yMin: 0.3, yMax: 0.9, yStep: 0.1,
      series: [{ name: 'ours', cls: 'ours', ys: R.rotation.ours }, { name: 'textbook CNN', cls: 'cnn', ys: R.rotation.cnn },
        { name: 'random forest', cls: 'forest', ys: R.rotation.forest }, { name: 'logistic', cls: 'logistic', ys: R.rotation.logistic }],
    });
    let timer = null;
    function show() {
      const d = demos[Number(select.value)], k = angles.indexOf(Number(slider.value));
      const frame = unpack32(d.frames[k]);
      paint(canvas, frame.codes, SIZE, SIZE);
      const said = update(d.probs[k], d.label);
      $('#rot-out').textContent = `${angles[k]}°`;
      $('#rot-truth').innerHTML = `True label: <strong>${d.label}</strong> · ${d.lot}`;
      chart.setMarker(k);
      const p0 = Math.max(...d.probs[0]), said0 = CLASSES[d.probs[0].indexOf(p0)];
      const p = Math.max(...d.probs[k]);
      $('#rot-notice').innerHTML = k === 0
        ? `At 0° the model says <strong>${said0}</strong> (${pct(p0)}). Drag the angle: at 90°, 180° and 270° the answer should hardly move.`
        : `At ${angles[k]}° it says <strong>${said}</strong> (${pct(p)}); at 0° it said ${said0} (${pct(p0)}). ` +
          (angles[k] % 90 === 0 ? 'A quarter turn: the same dies, just turned, and the rings and 4-way filters see the same thing.' : 'An in-between angle: dies get moved to the nearest grid spot and the corners fall off, so the picture really changes.');
    }
    select.addEventListener('change', show);
    slider.addEventListener('input', show);
    $('#rot-play').addEventListener('click', () => {
      if (timer) { clearInterval(timer); timer = null; $('#rot-play').textContent = '▶ Spin'; return; }
      $('#rot-play').textContent = '■ Stop';
      timer = setInterval(() => { slider.value = (Number(slider.value) + 15) % 360; show(); }, reduceMotion ? 900 : 450);
    });
    select.value = String(Math.max(0, demos.findIndex((d) => d.label === 'Scratch' && d.saved_prediction === 'Scratch')));
    show();
    $('#rot-caption').innerHTML = `Test wafers, nothing refit. We rotate the 32×32 maps, the same way the robustness test did. The curve is test macro-F1 for every model at every angle; quarter turns are shaded. ${demoNote} <a class="src" href="${R.meta.repo}/blob/${R.meta.commit}/experiments/phase4_analysis.json" target="_blank" rel="noopener">phase4_analysis.json · rotation_test</a>`;
    addCheck('rotation demo', () => {
      const bad = demos.filter((d) => CLASSES[d.probs[0].indexOf(Math.max(...d.probs[0]))] !== d.saved_prediction);
      return { ok: !bad.length, detail: `The rotation playground's 0° answer matches the saved final-test prediction for all ${demos.length} demo wafers.` };
    });
  })();

  // ===================================================== S10 shift + learning
  lineChart($('#shift-chart'), {
    label: 'Test macro-F1 for four models as the test wafers are shifted by 0 to 3 pixels',
    xs: R.shift.pixels, xTicks: R.shift.pixels.map((v) => ({ v, label: `${v} px` })), fmtX: (v) => `Shifted ${v} px`,
    xTitle: 'Shift of the test wafers on the 32×32 grid', yMin: 0.1, yMax: 0.9, yStep: 0.2,
    series: [{ name: 'ours', cls: 'ours', ys: R.shift.ours }, { name: 'textbook CNN', cls: 'cnn', ys: R.shift.cnn },
      { name: 'random forest', cls: 'forest', ys: R.shift.forest }, { name: 'logistic', cls: 'logistic', ys: R.shift.logistic }],
  });
  lineChart($('#learning-chart'), {
    label: 'Validation macro-F1 against the share of training lots used', log: true,
    xs: R.learning.fractions, xTicks: R.learning.fractions.map((v) => ({ v, label: pct(v) })), fmtX: (v) => `${pct(v)} of training lots`,
    xTitle: 'Share of training lots used (log scale)', yMin: 0.4, yMax: 0.8, yStep: 0.1,
    series: [{ name: 'ours', cls: 'ours', ys: R.learning.ours }, { name: 'logistic on pixels', cls: 'logistic', ys: R.learning.logistic }],
    pins: [{ x: R.learning.fractions[1], y: R.learning.ours[1], label: `5%: ${F.f3(R.learning.ours[1])}` }],
  });

  // ================================================== S11 confusion explorer
  (() => {
    const C = T.confusion_counts;
    const short = { Center: 'Cen', Donut: 'Don', 'Edge-Loc': 'E-Loc', 'Edge-Ring': 'E-Ring', Loc: 'Loc', 'Near-full': 'N-full', Random: 'Rand', Scratch: 'Scr', none: 'none' };
    const rowTotal = C.map((row) => row.reduce((a, b) => a + b, 0));
    const colTotal = CLASSES.map((_, j) => C.reduce((t, row) => t + row[j], 0));
    const grid = $('#cm'), buttons = [];
    grid.append(el('span'), ...CLASSES.map((c) => el('span', { class: 'h', title: `Predicted ${c}` }, short[c])));
    CLASSES.forEach((t, i) => {
      grid.append(el('span', { class: 'r', html: `<span class="lbl-full">${t}</span><span class="lbl-short">${short[t]}</span>` }));
      CLASSES.forEach((p, j) => {
        const share = (C[i][j] / rowTotal[i]) * 100;
        const v = i === j ? share : Math.min(100, share * 3.4);
        const button = el('button', {
          type: 'button', style: `--v:${v.toFixed(1)};--i:${i * 9 + j}`, 'aria-pressed': 'false',
          class: [i === j ? 'diag' : '', i !== j && v > 55 ? 'dark' : ''].join(' ').trim(),
          'aria-label': `True ${t}, predicted ${p}: ${C[i][j]} wafers, ${share.toFixed(1)} percent of ${t}`,
        }, share >= 0.5 ? String(Math.round(share)) : '');
        button.addEventListener('click', () => select(i, j));
        buttons.push(button); grid.append(button);
      });
    });
    function select(i, j) {
      buttons.forEach((b, k) => b.setAttribute('aria-pressed', String(k === i * 9 + j)));
      const t = CLASSES[i], p = CLASSES[j], n = C[i][j], examples = W.confusion[`${i},${j}`] || [];
      const share = pct(n / rowTotal[i], n / rowTotal[i] < 0.1 ? 1 : 0);
      $('#cm-title').textContent = i === j ? `${t}: called right` : `True ${t}, called ${p}`;
      $('#cm-text').innerHTML = i === j
        ? `<strong>${fmt(n)} of ${fmt(rowTotal[i])}</strong> ${t} test wafers (${share}) were called ${t}. Of everything the model called ${t}, ${pct(n / colTotal[j])} really was.`
        : n === 0 ? 'This never happened on the test set.'
          : `<strong>${fmt(n)} of ${fmt(rowTotal[i])}</strong> ${t} test wafers (${share}) were called ${p}. They make up ${pct(n / colTotal[j])} of everything the model called ${p}.`;
      $('#cm-examples').replaceChildren(...examples.map((e) => {
        const m = unpack32(e.map);
        return tile(m.codes, SIZE, SIZE, `<b>${e.lot}</b>${pct(e.confidence)} sure it's ${p}`, `${t} test wafer the model called ${p}`);
      }));
    }
    select(CLASSES.indexOf('Loc'), CLASSES.indexOf('Scratch'));
    $('#cm-caption').innerHTML = `Locked test set, main model. Example wafers are the 32×32 maps the model saw, drawn at random from each cell. ${M.source === 'saved' ? '' : 'The “% sure” comes from the refit demo model (see S6); the cell each wafer sits in comes from the saved final-test predictions.'} <a class="src" href="${R.meta.repo}/blob/${R.meta.commit}/experiments/final_test_predictions.csv" target="_blank" rel="noopener">final_test_predictions.csv</a>`;
  })();

  // ===================================================== S11 mistake gallery
  (() => {
    const groups = [
      { title: 'Loc ↔ Scratch: a small patch that looks like a short line, or the other way round.', match: (m) => (m.true === 'Loc' && m.predicted === 'Scratch') || (m.true === 'Scratch' && m.predicted === 'Loc') },
      { title: 'Called none: faint patterns with only a few failing dies.', match: (m) => m.predicted === 'none' },
      { title: 'Clean wafers with a few specks in a row, called Scratch.', match: (m) => m.true === 'none' && m.predicted === 'Scratch' },
      { title: 'Loc called Edge-Loc: a patch near, but not touching, the edge.', match: (m) => m.true === 'Loc' && m.predicted === 'Edge-Loc' },
    ];
    const host = $('#mistakes');
    groups.forEach((g) => {
      const items = W.mistakes.filter(g.match);
      if (!items.length) return;
      host.append(el('div', { class: 'mgroup' }, el('p', { class: 'mtitle' }, g.title),
        el('div', { class: 'mrow' }, items.map((m) => {
          const map = unpack32(m.map);
          return tile(map.codes, SIZE, SIZE, `<b>${m.true} → ${m.predicted}</b>${pct(m.confidence)} sure · ${m.lot}`, `${m.true} wafer called ${m.predicted}`);
        }))));
    });
  })();

  // ================================================================= S11 PR-AUC
  intervalChart($('#prauc'), {
    min: 0.4, max: 1, ticks: [0.4, 0.6, 0.8, 1],
    rows: [...CLASSES].sort((a, b) => T.pr_auc[b][0] - T.pr_auc[a][0]).map((c) => {
      const [v, lo, hi] = T.pr_auc[c];
      return { name: c, marks: [{ cls: 'ours', v, lo, hi, label: 'PR-AUC' }], val: `<b>${F.f3(v)}</b>` };
    }),
  });

  // ================================================================ S12 CNN
  (() => {
    const ys = R.cnn.validation_by_epoch, xs = ys.map((_, i) => i + 1), best = R.cnn.epoch_picked;
    const lo = Math.floor(Math.min(...ys) * 10) / 10;
    lineChart($('#epoch-chart'), {
      label: 'Validation macro-F1 of the textbook CNN after each training round', xs,
      xTicks: [1, 5, 10, 15, 20, 25, 30].map((v) => ({ v, label: String(v) })), fmtX: (v) => `Round ${v}`,
      xTitle: 'Training round', yMin: lo, yMax: 0.9, yStep: 0.1, height: 240,
      series: [{ name: 'textbook CNN', cls: 'cnn', ys }],
      pins: [{ x: best, y: ys[best - 1], label: `picked: round ${best}, ${F.f3(ys[best - 1])}` }],
    });
  })();

  // ============================================================ S13 denoise
  (() => {
    const D = R.denoise, table = $('#denoise-table'), pick = $('#denoise-pick');
    const rows = [...CLASSES, 'macro avg'];
    const metrics = ['precision', 'recall', 'f1'];
    table.append(
      el('thead', {}, el('tr', {}, el('th', { rowspan: 2 }, 'Class'), el('th', { colspan: 3, class: 'grp' }, 'Isolated (our pick)'), el('th', { colspan: 3, class: 'grp' }, 'Median (post-hoc)')),
        el('tr', {}, [...metrics, ...metrics].map((m) => el('th', {}, m === 'f1' ? 'F1' : m[0].toUpperCase() + m.slice(1))))),
      el('tbody', {}, rows.map((c) => el('tr', { 'data-c': c, class: c === 'macro avg' ? 'row-strong' : '' }, el('td', {}, c === 'macro avg' ? 'Macro average' : c),
        ['isolated', 'median'].flatMap((side) => metrics.map((m) => {
          const v = D[side][c][m], other = D[side === 'isolated' ? 'median' : 'isolated'][c][m];
          return el('td', { class: v > other + 1e-9 ? 'win' : '' }, F.f3(v));
        }))))));
    pick.append(...CLASSES.map((c) => el('button', { type: 'button', 'data-value': c, 'aria-pressed': String(c === 'Scratch') }, c)));
    const highlight = (c) => $$('tbody tr', table).forEach((tr) => tr.classList.toggle('hl', tr.dataset.c === c));
    segmented(pick, highlight);
    highlight('Scratch');
    $('#denoise-show').addEventListener('click', () => K.stepper?.show({ cls: 'Scratch', idx: 0, clean: 'median', step: 2 }));
  })();

  // ===================================================== S14 real-world slider
  (() => {
    const C = T.confusion_counts, n = CLASSES.length;
    const rates = C.map((row) => { const s = row.reduce((a, b) => a + b, 0); return row.map((v) => v / s); });
    const testShare = R.real_world.test_share, realShare = R.real_world.real_share;
    function at(t) {
      const share = testShare.map((s, i) => (1 - t) * s + t * realShare[i]);
      const precision = CLASSES.map((_, j) => {
        let num = share[j] * rates[j][j], den = 0;
        for (let i = 0; i < n; i++) den += share[i] * rates[i][j];
        return den ? num / den : 0;
      });
      const f1 = precision.map((p, j) => { const r = rates[j][j]; return p + r ? (2 * p * r) / (p + r) : 0; });
      return { share, precision, macro: f1.reduce((a, b) => a + b, 0) / n };
    }
    const base = at(0);
    const slider = $('#real-t');
    function show() {
      const t = Number(slider.value) / 100, s = at(t);
      slider.style.setProperty('--p', `${slider.value}%`);
      $('#real-clean').textContent = pct(s.share[n - 1]);
      $('#real-macro').textContent = F.f3(s.macro);
      const sp = s.precision[CLASSES.indexOf('Scratch')];
      $('#real-scratch').textContent = `1 in ${Math.max(1, Math.round(1 / sp))}`;
      const order = [...CLASSES.keys()].sort((a, b) => base.precision[b] - base.precision[a]);
      intervalChart($('#real-bars'), {
        min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1],
        rows: order.map((j) => ({ name: CLASSES[j], sub: `${pct(s.share[j], 1)} of wafers`, link: [base.precision[j], s.precision[j]],
          marks: [{ cls: 'logistic', v: base.precision[j], label: 'precision, our test mix' }, { cls: 'ours', v: s.precision[j], label: 'precision at this mix' }],
          val: `<b>${F.f3(s.precision[j])}</b>` })),
      });
      $('#real-notice').innerHTML = t === 0
        ? 'Notice: this is our test set as scored. Drag right and watch the rare classes’ precision fall while recall stays put.'
        : `Notice: recall hasn’t moved for any class, but at ${pct(s.share[n - 1])} clean wafers only about 1 in ${Math.max(1, Math.round(1 / sp))} Scratch alarms is a real scratch. The model is a first filter, and a person checks what it flags.`;
    }
    slider.addEventListener('input', show);
    show();
    addCheck('real-world slider', () => {
      const a = at(0).macro, b = at(1).macro;
      const ea = R.real_world.table['macro avg']['F1 on test'], eb = R.real_world.table['macro avg']['F1 at real shares'];
      return { ok: Math.abs(a - ea) < 1e-3 && Math.abs(b - eb) < 1e-3,
        detail: `The real-fab slider gives ${F.f3(a)} at our test mix and ${F.f3(b)} at the real mix, matching real_world_precision.csv.` };
    });
  })();
})();
