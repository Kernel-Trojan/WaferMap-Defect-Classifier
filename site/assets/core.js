/* Wafer Defects: shared helpers, number binding, navigation, charts and motion.
   Plain JavaScript, no dependencies. Data comes from data/*.js, written by
   scripts/11_site_data.py (python main.py site-data). */
(() => {
  'use strict';
  const SITE = window.SITE || {};
  const R = SITE.results, W = SITE.wafers, M = SITE.model;
  if (!R || !W) return;
  const CLASSES = R.meta.classes;
  const SIZE = 32;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ DOM helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  function el(tag, attrs = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === false || value == null) continue;
      if (key === 'class') node.className = value;
      else if (key === 'html') node.innerHTML = value;
      else if (key === 'style') node.style.cssText = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) node.append(kid);
    return node;
  }
  const NS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) if (value != null) node.setAttribute(key, value);
    if (text != null) node.textContent = text;
    return node;
  }

  /** A row of buttons where exactly one is pressed. */
  function segmented(root, onPick) {
    root.addEventListener('click', (event) => {
      const button = event.target.closest('button');
      if (!button || !root.contains(button)) return;
      $$('button', root).forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
      onPick(button.dataset.mode ?? button.dataset.value, button);
    });
  }
  function press(root, value) {
    $$('button', root).forEach((b) => b.setAttribute('aria-pressed', String((b.dataset.mode ?? b.dataset.value) === String(value))));
  }

  // --------------------------------------------------------------------- numbers
  /** Round half away from zero; scripts/11_site_data.py uses the same rule. */
  function roundHalf(value, digits) {
    const scale = 10 ** digits;
    const out = Math.sign(value) * Math.floor(Math.abs(value) * scale + 0.5 + 1e-6) / scale;
    return out === 0 ? 0 : out;
  }
  const signed = (v, d = 3) => { const r = roundHalf(v, d); return (r > 0 ? '+' : r < 0 ? '−' : '') + Math.abs(r).toFixed(d); };
  const F = {
    int: (v) => Math.floor(v + 0.5).toLocaleString('en-US'),
    f3: (v) => roundHalf(v, 3).toFixed(3),
    f2: (v) => roundHalf(v, 2).toFixed(2),
    pct: (v) => roundHalf(v * 100, 0).toFixed(0) + '%',
    pct1: (v) => roundHalf(v * 100, 1).toFixed(1) + '%',
    sf3: (v) => signed(v, 3),
    af2: (v) => roundHalf(Math.abs(v), 2).toFixed(2),
    af3: (v) => roundHalf(Math.abs(v), 3).toFixed(3),
    x: (v) => roundHalf(v, 0).toFixed(0) + '×',
    per100: (v) => roundHalf(v * 100, 0).toFixed(0),
    inv: (v) => roundHalf(1 / v, 0).toFixed(0),
    thousand: (v) => Math.floor(v / 1000).toLocaleString('en-US') + ' thousand',
    raw: (v) => String(v),
  };
  const get = (path) => path.split('.').reduce((node, key) => (node == null ? undefined : node[key]), R);

  // Bind every number (the export script already wrote them; this keeps them in sync).
  $$('[data-key]').forEach((node) => {
    const value = get(node.dataset.key);
    if (value == null) { console.warn('No data for', node.dataset.key); return; }
    node.textContent = (F[node.dataset.f] || F.raw)(value);
  });
  // Source chips, commit links, repository links.
  const repo = R.meta.repo;
  const blob = (path) => `${repo}/blob/${R.meta.new_files.includes(path) ? 'main' : R.meta.commit}/${path}`;
  $$('a.src[data-src]').forEach((a) => { a.href = blob(a.dataset.src); a.target = '_blank'; a.rel = 'noopener'; });
  const commits = { head: R.meta.commit, freeze: R.meta.freeze.commit, results: R.meta.results.commit };
  $$('a[data-commit]').forEach((a) => {
    const hash = commits[a.dataset.commit];
    a.textContent = hash; a.href = `${repo}/commit/${hash}`; a.target = '_blank'; a.rel = 'noopener';
  });
  $$('#repo-link, #footer-repo').forEach((a) => { a.href = repo; a.target = '_blank'; a.rel = 'noopener'; });

  // ------------------------------------------------------------------ wafer maps
  function failingNeighbours(fail, h, w, y, x) {
    let count = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (!dy && !dx) continue;
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < h && xx >= 0 && xx < w) count += fail[yy * w + xx];
      }
    }
    return count;
  }
  /** Isolated-die removal: a failing die goes only if none of its 8 neighbours fails. */
  function removeIsolated(fail, h, w) {
    const out = fail.slice();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (fail[y * w + x] && failingNeighbours(fail, h, w, y, x) === 0) out[y * w + x] = 0;
    }
    return out;
  }
  /** 3x3 median filter on a 0/1 map: failing only if at least 5 of the 9 dies fail. */
  function medianFilter(fail, mask, h, w) {
    const out = new Uint8Array(h * w);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const votes = failingNeighbours(fail, h, w, y, x) + fail[y * w + x];
      out[y * w + x] = votes >= 5 && mask[y * w + x] ? 1 : 0;
    }
    return out;
  }
  /** Nearest-neighbour resize: every output pixel copies the closest die, never a blend. */
  function resize(image, h, w, size = SIZE) {
    const index = (length) => Array.from({ length: size }, (_, i) => {
      const position = i === size - 1 ? length - 1 : i * ((length - 1) / (size - 1));
      return Math.floor(position + 0.5);
    });
    const rows = index(h), cols = index(w), out = new Uint8Array(size * size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) out[y * size + x] = image[rows[y] * w + cols[x]];
    return out;
  }
  const sum = (array) => { let total = 0; for (let i = 0; i < array.length; i++) total += array[i]; return total; };

  const rawCache = new Map();
  /** A full-size training wafer from data/wafers.js, with its cleanups worked out lazily. */
  function wafer(id) {
    if (rawCache.has(id)) return rawCache.get(id);
    const meta = W.raw[id];
    const { h, w } = meta;
    const mask = new Uint8Array(h * w), fail = new Uint8Array(h * w);
    for (let i = 0; i < h * w; i++) {
      const v = meta.map.charCodeAt(i) - 48;
      mask[i] = v > 0 ? 1 : 0; fail[i] = v === 2 ? 1 : 0;
    }
    const item = {
      id, label: meta.label, lot: meta.lot, h, w, mask, fail, dies: sum(mask), nFail: sum(fail),
      get iso() { return this._iso ??= removeIsolated(fail, h, w); },
      get med() { return this._med ??= medianFilter(fail, mask, h, w); },
      cleaned(method) { return method === 'median' ? this.med : method === 'none' ? fail : this.iso; },
      /** The 32x32 fail map + mask the model sees, after the chosen cleanup. */
      small(method = 'isolated') {
        const key = '_s_' + method;
        return this[key] ??= { fail: resize(this.cleaned(method), h, w), mask: resize(mask, h, w) };
      },
    };
    rawCache.set(id, item);
    return item;
  }
  /** Decode a packed 32x32 map (2 bits per die: 0 off, 1 pass, 2 fail). */
  function unpack32(b64) {
    const bytes = atob(b64);
    const codes = new Uint8Array(SIZE * SIZE), fail = new Uint8Array(SIZE * SIZE), mask = new Uint8Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const v = (bytes.charCodeAt(i >> 2) >> ((i & 3) * 2)) & 3;
      codes[i] = v; mask[i] = v > 0 ? 1 : 0; fail[i] = v === 2 ? 1 : 0;
    }
    return { codes, fail, mask };
  }

  // Codes for painting: 0 off the wafer, 1 passed, 2 failed, 3 removed by cleanup, 4 plain background.
  const codesFor = {
    raw: (wf) => wf.mask.map((m, i) => m + (wf.fail[i] & m)),
    result: (mask, fail) => mask.map((m, i) => (fail[i] && m ? 2 : m)),
    marked: (mask, before, after) => mask.map((m, i) => (after[i] && m ? 2 : before[i] && m ? 3 : m)),
    failOnly: (fail) => fail.map((f) => (f ? 2 : 4)),
    maskOnly: (mask) => mask.slice(),
  };

  let palette = null;
  function readPalette() {
    const style = getComputedStyle(document.documentElement);
    const rgba = (name) => {
      const hex = style.getPropertyValue(name).trim().replace('#', '');
      return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)).concat(255);
    };
    palette = [[0, 0, 0, 0], rgba('--die-pass'), rgba('--die-fail'), rgba('--die-removed'), rgba('--die-bg')];
  }
  function paint(canvas, codes, h, w) {
    if (!palette) readPalette();
    canvas.width = w; canvas.height = h;
    canvas._wafer = { codes, h, w };
    const context = canvas.getContext('2d');
    const image = context.createImageData(w, h);
    for (let i = 0; i < codes.length; i++) image.data.set(palette[codes[i]], i * 4);
    context.putImageData(image, 0, 0);
  }
  function tile(codes, h, w, caption, label) {
    const canvas = el('canvas', { role: 'img', 'aria-label': label || 'Wafer map' });
    paint(canvas, codes, h, w);
    return el('figure', { class: 'wafer' }, canvas, caption ? el('figcaption', { html: caption }) : null);
  }
  const colorScheme = window.matchMedia('(prefers-color-scheme: dark)');
  colorScheme.addEventListener('change', () => {
    readPalette();
    $$('canvas').forEach((c) => { if (c._wafer) paint(c, c._wafer.codes, c._wafer.h, c._wafer.w); });
  });
  const cssColor = (name) => {
    const hex = getComputedStyle(document.documentElement).getPropertyValue(name).trim().replace('#', '');
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  };
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

  // --------------------------------------------------------------------- tooltip
  const tip = $('#tooltip');
  function showTip(event, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, Math.max(8, event.clientX + 14)) + 'px';
    tip.style.top = Math.max(8, event.clientY - tip.offsetHeight - 12) + 'px';
  }
  const hideTip = () => { tip.hidden = true; };
  window.addEventListener('scroll', hideTip, { passive: true });

  // ------------------------------------------------------- dot-and-interval chart
  // rows: [{ name, sub, strong, marks: [{ cls, v, lo, hi, label }], link: [a, b], val, onClick }]
  function intervalChart(host, { min, max, ticks, rows, ref, tickFormat = (t) => String(t) }) {
    const at = (v) => ((Math.max(min, Math.min(max, v)) - min) / (max - min)) * 100;
    const grid = `--grid:${100 / (ticks.length - 1)}%`;
    const out = rows.flatMap((row, index) => {
      const track = el('span', { class: 'itrack', style: grid });
      if (ref) track.append(el('i', { class: 'refline', style: `left:${at(ref.v)}%` }));
      if (row.link) {
        const [a, b] = row.link.map(at);
        track.append(el('i', { class: 'link', style: `left:${Math.min(a, b)}%;width:${Math.abs(b - a)}%` }));
      }
      row.marks.forEach((m) => {
        if (m.lo != null) track.append(el('i', { class: `ci ${m.cls}`, style: `left:${at(m.lo)}%;width:${at(m.hi) - at(m.lo)}%` }));
      });
      row.marks.forEach((m) => track.append(el('i', { class: `pt ${m.cls}`, style: `left:${at(m.v)}%;transition-delay:${index * 60}ms` })));
      const html = row.tip || (`<b>${row.name}</b>` + row.marks.map((m) =>
        `<div class="trow"><span><i class="t-${m.cls}"></i>${m.label}</span><span>${F.f3(m.v)}` +
        (m.lo != null ? ` (${F.f3(m.lo)} to ${F.f3(m.hi)})` : '') + '</span></div>').join(''));
      track.addEventListener('pointermove', (event) => showTip(event, html));
      track.addEventListener('pointerleave', hideTip);
      const name = el('span', { class: 'name' + (row.strong ? ' strong' : ''), html: row.name + (row.sub ? `<small>${row.sub}</small>` : '') });
      const val = el('span', { class: 'val', html: row.val || '' });
      if (row.onClick) {
        [name, track, val].forEach((n) => { n.classList.add('click'); n.addEventListener('click', row.onClick); });
        name.setAttribute('tabindex', '0'); name.setAttribute('role', 'button');
        name.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); row.onClick(); } });
        row._nodes = [name, track, val];
      }
      return [name, track, val];
    });
    out.push(el('span', { class: 'spacer' }),
      el('span', { class: 'iaxis' }, ticks.map((t) => el('span', { style: `left:${at(t)}%` }, tickFormat(t))),
        ref && ref.label ? el('em', { class: 'reflabel', style: `left:${at(ref.v)}%` }, ref.label) : null),
      el('span', { class: 'spacer' }));
    host.replaceChildren(...out);
  }

  // ---------------------------------------------------- line chart (1 unit = 1px)
  function lineChart(host, cfg) {
    let drawn = 0, markerAt = null, api = {};
    function build() {
      const W = Math.max(280, Math.min(940, Math.round(host.clientWidth)));
      if (W === drawn) return;
      drawn = W;
      const wide = W >= 600;
      const H = cfg.height || (wide ? 320 : 260), m = { l: 44, r: wide && cfg.series.length > 1 ? 128 : 16, t: 14, b: 44 };
      const tx = cfg.log ? Math.log10 : (v) => v;
      const x0 = tx(cfg.xs[0]), x1 = tx(cfg.xs[cfg.xs.length - 1]);
      const px = (v) => m.l + ((tx(v) - x0) / (x1 - x0)) * (W - m.l - m.r);
      const py = (v) => H - m.b - ((v - cfg.yMin) / (cfg.yMax - cfg.yMin)) * (H - m.t - m.b);
      const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.label });
      (cfg.bands || []).forEach((v) => svg.append(sv('rect', { class: 'band', x: px(v) - 8, y: m.t, width: 16, height: H - m.t - m.b })));
      for (let v = cfg.yMin; v <= cfg.yMax + 1e-9; v += cfg.yStep) {
        svg.append(sv('line', { class: Math.abs(v - cfg.yMin) < 1e-9 ? 'axis' : 'grid', x1: m.l, x2: W - m.r, y1: py(v), y2: py(v) }),
          sv('text', { x: m.l - 8, y: py(v) + 4, 'text-anchor': 'end' }, v.toFixed(cfg.yDigits ?? 1)));
      }
      cfg.xTicks.filter((t) => wide || !t.wideOnly).forEach((t) =>
        svg.append(sv('text', { x: px(t.v), y: H - m.b + 18, 'text-anchor': 'middle' }, t.label)));
      svg.append(sv('text', { class: 'axis-title', x: m.l, y: H - 6 }, cfg.xTitle));
      const marker = sv('line', { class: 'marker-line', y1: m.t, y2: H - m.b, x1: 0, x2: 0, visibility: 'hidden' });
      const cross = sv('line', { class: 'cross', y1: m.t, y2: H - m.b, x1: 0, x2: 0, visibility: 'hidden' });
      svg.append(marker, cross);
      cfg.series.forEach((s) => {
        const d = s.ys.map((y, i) => `${i ? 'L' : 'M'}${px(cfg.xs[i]).toFixed(1)} ${py(y).toFixed(1)}`).join('');
        svg.append(sv('path', { class: `ln ${s.cls}`, d, pathLength: 1 }));
      });
      const dots = cfg.series.map((s) => s.ys.map((y, i) => {
        const dot = sv('circle', { class: `mk ${s.cls}`, cx: px(cfg.xs[i]).toFixed(1), cy: py(y).toFixed(1), r: cfg.xs.length > 10 ? 3 : 4.5 });
        svg.append(dot); return dot;
      }));
      (cfg.pins || []).forEach((p) => {
        svg.append(sv('circle', { class: 'pin', cx: px(p.x), cy: py(p.y), r: 7 }),
          sv('text', { class: 'pin-label', x: px(p.x) - 10, y: py(p.y) - 12, 'text-anchor': 'end' }, p.label));
      });
      if (wide && cfg.series.length > 1) {
        const ends = cfg.series.map((s) => ({ s, y: py(s.ys[s.ys.length - 1]) })).sort((a, b) => a.y - b.y);
        for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
        ends.forEach(({ s, y }) => svg.append(sv('text', { class: 'end', x: W - m.r + 10, y: y + 4 }, s.name)));
      }
      const nearest = (x) => { let best = 0; cfg.xs.forEach((v, i) => { if (Math.abs(px(v) - x) < Math.abs(px(cfg.xs[best]) - x)) best = i; }); return best; };
      const move = (event) => {
        const box = svg.getBoundingClientRect();
        const best = nearest((event.clientX - box.left) * (W / box.width));
        cross.setAttribute('x1', px(cfg.xs[best])); cross.setAttribute('x2', px(cfg.xs[best]));
        cross.setAttribute('visibility', 'visible');
        const rows = [...cfg.series].sort((a, b) => b.ys[best] - a.ys[best]).map((s) =>
          `<div class="trow"><span><i class="t-${s.cls}"></i>${s.name}</span><span>${F.f3(s.ys[best])}</span></div>`).join('');
        showTip(event, `<b>${cfg.fmtX(cfg.xs[best])}</b>${rows}`);
      };
      svg.addEventListener('pointermove', move);
      svg.addEventListener('pointerdown', move);
      svg.addEventListener('pointerleave', () => { hideTip(); cross.setAttribute('visibility', 'hidden'); });
      host.replaceChildren(svg);
      api.setMarker = (index) => {
        markerAt = index;
        if (index == null) { marker.setAttribute('visibility', 'hidden'); return; }
        marker.setAttribute('x1', px(cfg.xs[index])); marker.setAttribute('x2', px(cfg.xs[index]));
        marker.setAttribute('visibility', 'visible');
        dots.forEach((row) => row.forEach((d, i) => d.classList.toggle('on', i === index)));
      };
      if (markerAt != null) api.setMarker(markerAt);
    }
    build();
    if ('ResizeObserver' in window) new ResizeObserver(build).observe(host);
    return api;
  }

  // ------------------------------------------------------------- probability bars
  function probBars(host) {
    const rows = CLASSES.map((c) => {
      const bar = el('i', { class: 'pbar' }), val = el('span', { class: 'pval' });
      const row = el('div', { class: 'prow' }, el('span', { class: 'pname' }, el('i', { class: `cdot c-${c}` }), c), el('span', { class: 'ptrack' }, bar), val);
      host.append(row);
      return { row, bar, val };
    });
    return (probs, truth) => {
      const top = probs.indexOf(Math.max(...probs));
      rows.forEach((r, i) => {
        r.bar.style.width = (probs[i] * 100).toFixed(1) + '%';
        r.val.textContent = roundHalf(probs[i] * 100, 0).toFixed(0) + '%';
        r.row.classList.toggle('top', i === top);
        r.row.classList.toggle('truth', CLASSES[i] === truth);
      });
      return CLASSES[top];
    };
  }

  // -------------------------------------------------------------------- navigation
  (() => {
    const bar = $('.topbar'), links = $$('#toplinks a'), menu = $('#menu-btn'), strip = $('#toplinks');
    menu.addEventListener('click', () => {
      const open = bar.classList.toggle('open');
      menu.setAttribute('aria-expanded', String(open));
    });
    links.forEach((a) => a.addEventListener('click', () => { bar.classList.remove('open'); menu.setAttribute('aria-expanded', 'false'); }));
    const byId = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      links.forEach((a) => a.classList.remove('active'));
      const a = byId.get(entry.target.id);
      if (!a) return;
      a.classList.add('active');
      if (strip.scrollWidth > strip.clientWidth && !bar.classList.contains('open')) {
        strip.scrollTo({ left: a.offsetLeft - (strip.clientWidth - a.offsetWidth) / 2, behavior: reduceMotion ? 'auto' : 'smooth' });
      }
    }), { rootMargin: '-30% 0px -60% 0px' });
    $$('section.sec').forEach((s) => observer.observe(s));

    const progress = $('#progress-bar');
    let queued = false;
    const update = () => {
      queued = false;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      progress.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + '%';
    };
    window.addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
    update();

    // Skip link: jump to the results and move keyboard focus to their heading.
    $$('a[href="#main-content"]').forEach((a) => a.addEventListener('click', (event) => {
      event.preventDefault();
      const target = $('#main-content');
      target.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' });
      $('#results-title').focus({ preventScroll: true });
      history.replaceState(null, '', '#main-content');
    }));

    // A "#" link on every section heading, which also copies the link.
    $$('section.sec').forEach((section) => {
      const heading = $('h2', section);
      if (!heading) return;
      const anchor = el('a', { class: 'anchor', href: `#${section.id}`, 'aria-label': `Link to this section: ${heading.textContent}` }, '#');
      anchor.addEventListener('click', () => {
        const url = location.href.split('#')[0] + '#' + section.id;
        navigator.clipboard?.writeText(url).then(() => {
          anchor.dataset.copied = 'Link copied';
          setTimeout(() => delete anchor.dataset.copied, 1400);
        }).catch(() => {});
      });
      heading.append(' ', anchor);
    });

    // Copy buttons on commands.
    $$('.cmd .copy').forEach((button) => button.addEventListener('click', () => {
      navigator.clipboard?.writeText($('code', button.parentElement).textContent).then(() => {
        button.textContent = 'Copied'; setTimeout(() => { button.textContent = 'Copy'; }, 1400);
      }).catch(() => {});
    }));
    $$('#cmd-table code').forEach((code) => {
      code.setAttribute('tabindex', '0'); code.setAttribute('title', 'Click to copy');
      const copy = () => navigator.clipboard?.writeText(code.textContent).then(() => {
        code.dataset.copied = 'Copied'; setTimeout(() => delete code.dataset.copied, 1200);
      }).catch(() => {});
      code.addEventListener('click', copy);
      code.addEventListener('keydown', (e) => { if (e.key === 'Enter') copy(); });
    });
  })();

  // ------------------------------------------------------- motion: reveal + count-up
  function countUp(target) {
    const final = target.textContent, match = final.match(/^([^\d]*)([\d,]*\.?\d+)(.*)$/s);
    if (!match || reduceMotion) return;
    const goal = parseFloat(match[2].replace(/,/g, '')), decimals = (match[2].split('.')[1] || '').length;
    const start = performance.now();
    const frame = (time) => {
      const t = Math.min(1, (time - start) / 1100), value = goal * (1 - Math.pow(1 - t, 3));
      target.textContent = t === 1 ? final
        : match[1] + (decimals ? value.toFixed(decimals) : Math.round(value).toLocaleString('en-US')) + match[3];
      if (t < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
  const revealHooks = new Map();
  function onReveal(node, fn) { revealHooks.set(node, fn); }
  function startReveal() {
    document.documentElement.classList.add('js');
    const targets = $$('.figure, .pull, .callout, .otherwise, .takeaway, .big-compare, .limits li, .story-cards, .photo-strip, .stat-cards');
    const reveal = (node) => {
      node.classList.add('in');
      $$('.big b, .tile-n', node).forEach(countUp);
      revealHooks.get(node)?.();
    };
    if (!('IntersectionObserver' in window)) { targets.forEach(reveal); return; }
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
      if (entry.isIntersecting) { reveal(entry.target); observer.unobserve(entry.target); }
    }), { rootMargin: '0px 0px -6% 0px', threshold: 0.02 });
    targets.forEach((node) => observer.observe(node));
    $$('.hcard b').forEach(countUp);
  }
  /** Run fn only while node is on screen (for looping animations). */
  function whileVisible(node, start, stop) {
    if (!('IntersectionObserver' in window)) { start(); return; }
    new IntersectionObserver((entries) => entries.forEach((entry) => (entry.isIntersecting ? start() : stop())),
      { threshold: 0.05 }).observe(node);
  }

  // -------------------------------------------------------------------- self-check
  const checks = [];
  function addCheck(name, fn) { checks.push([name, fn]); }
  function runChecks() {
    const lines = checks.map(([name, fn]) => {
      try { const result = fn(); return { name, ok: result.ok, detail: result.detail }; }
      catch (error) { console.error(error); return { name, ok: false, detail: String(error) }; }
    });
    const failed = lines.filter((l) => !l.ok);
    window.waferSiteSelfCheck = () => lines;
    // The result used to be printed in the footer; it's still available from the console.
    if (failed.length) console.warn('Self-check failed:', failed);
    const box = $('#selfcheck');
    if (box) box.innerHTML = (failed.length
      ? `<strong>Self-check failed:</strong> ${failed.map((l) => `${l.name} (${l.detail})`).join('; ')}.`
      : '<strong>Self-check passed.</strong> ') + ' ' + lines.filter((l) => l.ok).map((l) => l.detail).join(' ');
  }

  window.KIT = {
    SITE, R, W, M, CLASSES, SIZE, reduceMotion, $, $$, el, sv, segmented, press, F, roundHalf, get,
    failingNeighbours, removeIsolated, medianFilter, resize, sum, wafer, unpack32, codesFor, paint, tile,
    cssColor, mix, showTip, hideTip, intervalChart, lineChart, probBars, onReveal, whileVisible, addCheck,
  };
  // Other scripts register their widgets first; motion and checks start once everything is built.
  window.addEventListener('DOMContentLoaded', () => {
    startReveal();
    // A link like …#main-content: the browser jumped before the widgets were built, so jump again.
    const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) {
      const jump = () => {
        target.scrollIntoView({ behavior: 'instant' });
        if (target.id === 'main-content') $('#results-title').focus({ preventScroll: true });
      };
      jump();
      window.addEventListener('load', () => setTimeout(jump, 0), { once: true });
    }
    (window.requestIdleCallback || setTimeout)(runChecks);
  });
})();
