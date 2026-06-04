/* ── Feature-Space Shift Player ──────────────────────────────────────────── */

const DATA_URL = 'episode/feature_shift_data.json';

let shiftData = null;
let currentIter = 0;
let playTimer = null;
let isPlaying = false;

// Plotly colour map
const GROUP_PLOTLY = {
  A_query:   { color: 'rgb(217,38,38)',  symbol: 'circle',         name: 'Class A — query' },
  A_support: { color: 'rgb(250,166,166)', symbol: 'circle-open',   name: 'Class A — support' },
  B_query:   { color: 'rgb(38,90,217)',  symbol: 'circle',          name: 'Class B — query' },
  B_support: { color: 'rgb(166,186,250)', symbol: 'circle-open',   name: 'Class B — support' },
};

async function loadData() {
  try {
    const r = await fetch(DATA_URL);
    if (!r.ok) throw new Error(r.statusText);
    shiftData = await r.json();
    initPlayer();
  } catch (e) {
    document.getElementById('shift-loading').innerHTML =
      `<p style="color:#c0392b;text-align:center">Could not load feature_shift_data.json.<br>
       Run the feature_shift_analysis notebook first, then save the data.</p>`;
  }
}

function initPlayer() {
  const meta = shiftData.metadata;
  const n    = shiftData.iterations.length;

  document.getElementById('shift-loading').style.display = 'none';
  document.getElementById('shift-player').style.display  = 'block';

  // Build iteration dots
  const dotsEl = document.getElementById('iter-dots');
  for (let i = 0; i < n; i++) {
    const d = document.createElement('div');
    d.className = 'iter-dot' + (i === 0 ? ' active' : '');
    d.textContent = i === 0 ? '0' : i;
    d.title = i === 0 ? 'Before adaptation' : `After iteration ${i}`;
    d.addEventListener('click', () => goTo(i));
    dotsEl.appendChild(d);
  }

  // Slider range
  const slider = document.getElementById('iter-slider');
  slider.max   = n - 1;
  slider.value = 0;
  slider.addEventListener('input', () => goTo(+slider.value));

  document.getElementById('btn-play').addEventListener('click', togglePlay);

  // Set episode images (static)
  document.getElementById('img-query').src    = shiftData.episode.query_png;
  document.getElementById('img-query-gt').src = shiftData.episode.query_gt_png;

  goTo(0);

  // Kick off the substitution animation now that images are available
  startSubstitutionAnim();
}

function goTo(iter) {
  if (!shiftData) return;
  currentIter = iter;
  const data = shiftData.iterations[iter];
  const n    = shiftData.iterations.length;

  // Label
  const label = iter === 0 ? 'Before adaptation (vanilla encoder)' : `After iteration ${iter}`;
  document.getElementById('iter-label-text').textContent = label;
  document.getElementById('iter-label-pill').textContent =
    iter === 0 ? 'Baseline' : `Iter ${iter}/${n-1}`;

  // Slider
  document.getElementById('iter-slider').value = iter;

  // Dots
  document.querySelectorAll('.iter-dot').forEach((d, i) => {
    d.classList.toggle('active', i === iter);
  });

  // Segmentation image
  if (data.overlay_png) {
    document.getElementById('img-pred').src = data.overlay_png;
  }

  // Separation score bar
  const scoreEl  = document.getElementById('score-fill');
  const scoreTxt = document.getElementById('score-text');
  const maxScore = Math.max(...shiftData.iterations.map(d => d.separation_score || 0)) * 1.1;
  if (data.separation_score != null) {
    const pct = Math.min(100, (data.separation_score / maxScore) * 100);
    scoreEl.style.width = pct + '%';
    scoreTxt.textContent = `Class separation score: ${data.separation_score.toFixed(3)}`;
  } else {
    scoreEl.style.width = '0%';
    scoreTxt.textContent = 'Separation score: N/A';
  }

  // t-SNE
  renderTSNE(data.tsne);
}

function renderTSNE(points) {
  const plotEl = document.getElementById('tsne-plot');
  if (!points || points.length === 0) {
    Plotly.purge(plotEl);
    plotEl.innerHTML = '<p style="text-align:center;color:#aaa;padding:60px 0">No t-SNE data for this iteration</p>';
    return;
  }

  // Group points
  const byGroup = {};
  points.forEach(p => {
    if (!byGroup[p.group]) byGroup[p.group] = { x: [], y: [] };
    byGroup[p.group].x.push(p.x);
    byGroup[p.group].y.push(p.y);
  });

  const traces = Object.entries(byGroup)
    .filter(([g]) => GROUP_PLOTLY[g])
    .map(([g, pts]) => ({
      type: 'scatter', mode: 'markers',
      x: pts.x, y: pts.y,
      name: GROUP_PLOTLY[g].name,
      marker: {
        color:  GROUP_PLOTLY[g].color,
        symbol: GROUP_PLOTLY[g].symbol,
        size: 6,
        opacity: 0.80,
        line: { width: 1, color: GROUP_PLOTLY[g].color },
      },
    }));

  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    margin: { l: 10, r: 10, t: 10, b: 10 },
    legend: { orientation: 'h', y: -0.06, font: { size: 11 }, bgcolor: 'rgba(0,0,0,0)' },
    xaxis: { showticklabels: false, showgrid: false, zeroline: false },
    yaxis: { showticklabels: false, showgrid: false, zeroline: false },
    transition: { duration: 350, easing: 'cubic-in-out' },
  };

  const isMobile = window.matchMedia('(max-width: 600px)').matches;
  const config = { displayModeBar: false, responsive: true,
                   scrollZoom: false, staticPlot: isMobile };

  if (plotEl.data) {
    Plotly.react(plotEl, traces, layout, config);
  } else {
    Plotly.newPlot(plotEl, traces, layout, config);
  }
}

function togglePlay() {
  const btn = document.getElementById('btn-play');
  if (isPlaying) {
    clearInterval(playTimer);
    isPlaying = false;
    btn.innerHTML = '▶ Play';
  } else {
    isPlaying = true;
    btn.innerHTML = '⏸ Pause';
    const n = shiftData.iterations.length;
    if (currentIter >= n - 1) goTo(0);
    playTimer = setInterval(() => {
      const next = currentIter + 1;
      if (next >= n) {
        clearInterval(playTimer);
        isPlaying = false;
        btn.innerHTML = '▶ Play';
      } else {
        goTo(next);
      }
    }, 900);
  }
}

/* ── Copy BibTeX ─────────────────────────────────────────────────────────── */
function copyBib() {
  const text = document.getElementById('bibtex').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = '✓ Copied';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

/* ── Animated stat counters ──────────────────────────────────────────────── */
function animateCounter(el, target, duration = 1200, prefix = '+', decimals = 2) {
  const start = performance.now();
  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;  // easeInOut
    const val = ease * target;
    el.textContent = prefix + val.toFixed(decimals) + '%';
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ── Intersection Observer — trigger counters when in view ──────────────── */
const counterObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting && !e.target.dataset.counted) {
      e.target.dataset.counted = '1';
      animateCounter(e.target, +e.target.dataset.target, 1400, '+', 2);
    }
  });
}, { threshold: 0.5 });

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.counter').forEach(el => counterObs.observe(el));
  loadData();
  initSubstitutionAnim();
});

/* ═══════════════════════════════════════════════════════════════════════════
   Substitution animation
   ═══════════════════════════════════════════════════════════════════════════ */

/* initSubstitutionAnim  — sets up static DOM; startSubstitutionAnim() is called
   from initPlayer() once shiftData (with real images) is available.             */
function initSubstitutionAnim() {
  // Show a waiting message until images arrive
  document.getElementById('cards-a').innerHTML =
    '<span style="font-size:0.78rem;color:var(--muted)">Loading images…</span>';
  document.getElementById('cards-b').innerHTML = '';
}

function startSubstitutionAnim() {
  if (!shiftData) return;

  const supportA = shiftData.episode.support_a;   // base64 data-URL array
  const supportB = shiftData.episode.support_b;
  const K        = supportA.length;
  const T_ITERS  = 3;
  const PHASE_MS = { select: 700, forward: 900, loss: 700, back: 900, pause: 400 };

  // ── Build image cards ──────────────────────────────────────────────────────
  const aContainer = document.getElementById('cards-a');
  const bContainer = document.getElementById('cards-b');
  aContainer.innerHTML = '';
  bContainer.innerHTML = '';
  const cards = [];

  const makeCard = (src, cls, idx) => {
    const d   = document.createElement('div');
    d.className = `s-card cls-${cls}`;
    d.dataset.cls = cls; d.dataset.idx = idx; d.dataset.src = src;
    const img = document.createElement('img');
    img.src = src; img.alt = `${cls.toUpperCase()}${(idx % K) + 1}`;
    d.appendChild(img);
    const lbl = document.createElement('span');
    lbl.className = 's-card-lbl';
    lbl.textContent = `${cls.toUpperCase()}${(idx % K) + 1}`;
    d.appendChild(lbl);
    return d;
  };

  supportA.forEach((src, k) => {
    const d = makeCard(src, 'a', k);
    aContainer.appendChild(d);
    cards.push(d);
  });
  supportB.forEach((src, k) => {
    const d = makeCard(src, 'b', k + K);
    bContainer.appendChild(d);
    cards.push(d);
  });

  document.getElementById('sub-total').textContent = cards.length;

  // ── Cache DOM ──────────────────────────────────────────────────────────────
  const pqCardDiag = document.getElementById('pq-card-diag');
  const ctxMini    = document.getElementById('ctx-mini');
  const dnEnc      = document.getElementById('dn-enc');
  const dnDec      = document.getElementById('dn-dec');
  const dnLoss     = document.getElementById('dn-loss');
  const loraChip   = document.getElementById('lora-chip');
  const svg        = document.getElementById('sub-svg');
  const phases     = ['ph-select','ph-forward','ph-loss','ph-back']
                      .map(id => document.getElementById(id));
  const stepEl     = document.getElementById('sub-step');
  const iterEl     = document.getElementById('sub-iter');

  svg.setAttribute('viewBox','0 0 100 100');
  svg.setAttribute('preserveAspectRatio','none');

  let currentStep = 0, currentIter = 1, animTimer = null;
  let isVisible = false;

  // ── Helpers ────────────────────────────────────────────────────────────────
  const setPhase = active => phases.forEach((p, i) => {
    p.classList.remove('active','done');
    if (i < active)   p.classList.add('done');
    if (i === active) p.classList.add('active');
  });

  const clearArrows = () => svg.querySelectorAll('path').forEach(p => p.remove());

  const drawArrow = (x1, y1, x2, y2, cls) => {
    const p  = document.createElementNS('http://www.w3.org/2000/svg','path');
    const mx = (x1+x2)/2, my = (y1+y2)/2 - 20;
    p.setAttribute('d', `M${x1},${y1} Q${mx},${my} ${x2},${y2}`);
    p.setAttribute('class', cls);
    svg.appendChild(p);
    return p;
  };

  const pct = (el, xf, yf) => {
    const box = document.querySelector('.sub-diagram').getBoundingClientRect();
    const r   = el.getBoundingClientRect();
    return {
      x: (r.left + r.width  * xf - box.left) / box.width  * 100,
      y: (r.top  + r.height * yf - box.top)  / box.height * 100,
    };
  };

  // ── Main animation loop ────────────────────────────────────────────────────
  const animate = () => {
    if (!isVisible) return;
    const total = cards.length;

    setPhase(0);
    clearArrows();
    [dnEnc, dnDec, dnLoss].forEach(n => n.classList.remove('active'));
    stepEl.textContent = currentStep + 1;
    iterEl.textContent = currentIter;

    // Highlight pseudo-query card; dim context cards
    cards.forEach((c, i) => {
      c.classList.remove('is-pq','is-ctx');
      c.classList.add(i === currentStep ? 'is-pq' : 'is-ctx');
    });

    // Pseudo-query node: show the real image
    const pqSrc = cards[currentStep].dataset.src;
    pqCardDiag.style.backgroundImage   = `url('${pqSrc}')`;
    pqCardDiag.style.backgroundSize    = 'cover';
    pqCardDiag.style.backgroundPosition= 'center';

    // Context mini thumbnails
    ctxMini.innerHTML = '';
    cards.forEach((c, i) => {
      if (i === currentStep) return;
      const m = document.createElement('img');
      m.className = 'ctx-mini-card';
      m.src = c.dataset.src;
      m.alt = '';
      ctxMini.appendChild(m);
    });

    animTimer = setTimeout(() => {
      if (!isVisible) return;
      setPhase(1);
      dnEnc.classList.add('active');

      const pqP  = pct(document.getElementById('dn-pq'),  0.9, 0.5);
      const ctxP = pct(document.getElementById('dn-ctx'), 0.1, 0.5);
      const encT = pct(dnEnc, 0.5, 0);
      const encB = pct(dnEnc, 0.5, 1);
      const decT = pct(dnDec, 0.5, 0);

      const a1 = drawArrow(pqP.x,  pqP.y,  encT.x - 5, encT.y, 'fwd animating');
      const a2 = drawArrow(ctxP.x, ctxP.y, encT.x + 5, encT.y, 'fwd animating');

      animTimer = setTimeout(() => {
        if (!isVisible) return;
        a1.classList.remove('animating'); a2.classList.remove('animating');
        dnDec.classList.add('active');
        const a3 = drawArrow(encB.x, encB.y + 1, decT.x, decT.y - 1, 'fwd animating');

        animTimer = setTimeout(() => {
          if (!isVisible) return;
          a3.classList.remove('animating');
          setPhase(2);
          dnLoss.classList.add('active');
          const decB  = pct(dnDec,  0.5, 1);
          const lossT = pct(dnLoss, 0.5, 0);
          const a4 = drawArrow(decB.x, decB.y + 1, lossT.x, lossT.y - 1, 'fwd animating');

          animTimer = setTimeout(() => {
            if (!isVisible) return;
            a4.classList.remove('animating');
            setPhase(3);

            // Backward gradient arrow: loss → encoder (curves left)
            const lossL = pct(dnLoss, 0.1, 0.5);
            const encR  = pct(dnEnc,  0.5, 1);
            const cx    = lossL.x - 18;
            const bwd   = document.createElementNS('http://www.w3.org/2000/svg','path');
            bwd.setAttribute('d',
              `M${lossL.x},${lossL.y} C${cx},${lossL.y} ${cx},${encR.y} ${encR.x},${encR.y}`);
            bwd.setAttribute('class','bwd animating');
            svg.appendChild(bwd);

            loraChip.classList.remove('pulse');
            void loraChip.offsetWidth;
            loraChip.classList.add('pulse');

            animTimer = setTimeout(() => {
              if (!isVisible) return;
              bwd.classList.remove('animating');
              animTimer = setTimeout(() => {
                if (!isVisible) return;
                clearArrows();
                phases.forEach(p => p.classList.remove('active','done'));
                [dnEnc, dnDec, dnLoss].forEach(n => n.classList.remove('active'));
                cards.forEach(c => c.classList.remove('is-pq','is-ctx'));
                currentStep++;
                if (currentStep >= total) {
                  currentStep = 0;
                  currentIter = currentIter < T_ITERS ? currentIter + 1 : 1;
                }
                animate();
              }, PHASE_MS.pause);
            }, PHASE_MS.back);
          }, PHASE_MS.loss);
        }, PHASE_MS.forward / 2);
      }, PHASE_MS.forward / 2);
    }, PHASE_MS.select);
  };

  // ── Intersection observer: pause when off-screen ──────────────────────────
  const subObs = new IntersectionObserver(entries => {
    isVisible = entries[0].isIntersecting;
    if (isVisible && !animTimer) animate();
    if (!isVisible) { clearTimeout(animTimer); animTimer = null; }
  }, { threshold: 0.2 });
  subObs.observe(document.getElementById('substitution'));
}

/* ── Active nav link on scroll ───────────────────────────────────────────── */
const sections = ['hero','abstract','method','substitution','shift','results','citation'];
const navLinks  = {};
sections.forEach(id => {
  const a = document.querySelector(`nav a[href="#${id}"]`);
  if (a) navLinks[id] = a;
});
window.addEventListener('scroll', () => {
  let current = 'hero';
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && el.getBoundingClientRect().top < 120) current = id;
  });
  Object.entries(navLinks).forEach(([id, a]) =>
    a.classList.toggle('active', id === current));
}, { passive: true });
