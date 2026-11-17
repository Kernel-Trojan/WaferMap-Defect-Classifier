/* Faint page background: a grid of dies where defect patterns slowly appear and fade,
   while a soft line sweeps down like a tester checking each row. */
(() => {
  'use strict';
  const canvas = document.getElementById('bg');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const CELL = 22, LIFE = 10000, MAX = 5;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const rand = (a, b) => a + Math.random() * (b - a);
  const hash = (i) => ((i * 2654435761) >>> 0) % 100;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  let cols, rows, colors, events = [], last = 0, next = 0;

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cols = Math.ceil(innerWidth / CELL); rows = Math.ceil(innerHeight / CELL);
    colors = { line: css('--line-strong'), fail: css('--die-fail'), scan: css('--accent') };
  }

  // The same kinds of shapes the model sorts: a ring, a blob, a scratch, scattered specks.
  function spawn(born) {
    const R = rand(4, 9), a = rand(0, Math.PI), L = rand(9, 18), seed = Math.floor(rand(0, 1e6));
    const shapes = [
      (dx, dy, d) => Math.abs(d - R) < 0.7,
      (dx, dy, d) => d < R * 0.7,
      (dx, dy, d, i) => d < R && hash(i + seed) < 35,
      (dx, dy) => {
        const t = Math.max(0, Math.min(L, dx * Math.cos(a) + dy * Math.sin(a)));
        return Math.hypot(dx - t * Math.cos(a), dy - t * Math.sin(a)) < 0.6;
      },
    ];
    events.push({ x: rand(0, cols), y: rand(0, rows), born, shape: shapes[Math.floor(rand(0, shapes.length))] });
  }

  function draw(now) {
    events = events.filter((e) => now - e.born < LIFE);
    if (events.length < MAX && now >= next) { spawn(now); next = now + rand(1500, 3500); }   // by time, not frames
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    ctx.globalAlpha = 0.5; ctx.strokeStyle = colors.line; ctx.beginPath();
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) ctx.rect(c * CELL + 2.5, r * CELL + 2.5, CELL - 5, CELL - 5);
    ctx.stroke();
    const scan = ((now / 16000) % 1) * (rows + 8) - 4;
    for (let r = 0; r < rows; r++) {
      const s = Math.max(0, 1 - Math.abs(r - scan) / 3);
      for (let c = 0; c < cols; c++) {
        let v = 0;
        for (const e of events) {
          const dx = c - e.x, dy = r - e.y, d = Math.hypot(dx, dy);
          if (d < 20 &&e.shape(dx, dy, d, r * cols + c)) v = Math.max(v, Math.sin((Math.PI * (now - e.born)) / LIFE));
        }
        if (v) { ctx.globalAlpha = v * 0.8;ctx.fillStyle = colors.fail; ctx.fillRect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4); }
        if (s) { ctx.globalAlpha = s * 0.35; ctx.fillStyle = colors.scan; ctx.fillRect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4); }
      }
    }
    ctx.globalAlpha = 1;
  }

  function loop(now) {
    if (now - last > 40) { last = now; draw(now); }   // about 25 frames a second is plenty
    requestAnimationFrame(loop);
  }

  resize();
  addEventListener('resize', () => { resize(); if (still) draw(LIFE / 2); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', resize);
  if (still) {   // one still frame with a few patterns, no motion
    for (let i = 0; i < MAX; i++) spawn(0);
    draw(LIFE / 2);
  } else {
    requestAnimationFrame(loop);
  }
})();
