/* sensorlab governance dashboard — vanilla JS, SVG charts, no build step. */
(() => {
  "use strict";
  const $ = (sel) => document.querySelector(sel);
  const fmt = {
    pct: (v, d = 1) => (v == null ? "—" : (100 * v).toFixed(d) + " %"),
    num: (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d)),
    int: (v) => (v == null ? "—" : Math.round(v).toLocaleString("en-CH")),
    chf: (v) => (v == null ? "—" : Math.round(v).toLocaleString("en-CH") + " CHF"),
    min: (v, d = 0) => (v == null ? "—" : Number(v).toFixed(d) + " min"),
    date: (s) => (s ? s.replace("T", " ").replace("+00:00", " UTC") : "—"),
  };
  const DET_COLOR = { "PCA-T2Q": "var(--series-1)", IForest: "var(--series-2)", "LSTM-AE": "var(--series-3)" };
  const ACTIONS = ["wait", "investigate", "schedule_maintenance", "intervene_now"];
  const api = async (path, opts) => {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${path} → ${r.status} ${await r.text()}`);
    return r.status === 204 ? null : r.json();
  };
  const el = (tag, attrs = {}, children = []) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "html") n.innerHTML = v; else n.setAttribute(k, v);
    }
    for (const c of [].concat(children)) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return n;
  };
  const statusClass = (s) => ({ stable: "good", ok: "good", watch: "warn", degraded: "warn", alert: "crit" }[s] || "");

  // ------------------------------------------------------------------ tooltip
  const tip = $("#tooltip");
  const showTip = (x, y, html) => { tip.innerHTML = html; tip.hidden = false; const w = tip.offsetWidth; tip.style.left = Math.min(x + 14, window.innerWidth - w - 8) + "px"; tip.style.top = (y + 14) + "px"; };
  const hideTip = () => { tip.hidden = true; };

  // ------------------------------------------------------------------ SVG helpers
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs = {}) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };
  const niceTicks = (lo, hi, n = 5) => {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => (hi - lo) / s <= n) || raw;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
  };

  /** Multi-series line chart with optional shaded band, horizontal threshold, vertical markers, hover crosshair. */
  function lineChart(container, opts) {
    const { x, series, title, yLabel, threshold, band, markers = [], height = 220, xLabel = "minutes", yFmt = (v) => fmt.num(v, 2), yMin } = opts;
    container.innerHTML = "";
    if (title) container.append(el("p", { class: "title" }, title));
    const W = 1000, H = height, m = { t: 12, r: 16, b: 34, l: 52 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xs = x, x0 = Math.min(...xs), x1 = Math.max(...xs);
    let vals = series.flatMap((s) => [...s.values, ...(s.area ? [...s.area.lo, ...s.area.hi] : [])]).filter((v) => v != null && isFinite(v));
    if (threshold != null) vals.push(threshold);
    let y0 = yMin != null ? yMin : Math.min(...vals), y1 = Math.max(...vals);
    if (y1 === y0) y1 = y0 + 1;
    const pad = (y1 - y0) * 0.06; y1 += pad; if (yMin == null) y0 -= pad;
    const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw, sy = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title || "chart" });
    if (band && band.from != null) svg.append(svgEl("rect", { class: "band", x: sx(band.from), y: m.t, width: Math.max(0, sx(band.to ?? x1) - sx(band.from)), height: ih }));
    const g = svgEl("g", { class: "axis" });
    for (const t of niceTicks(y0, y1, 4)) { g.append(svgEl("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) })); const tx = svgEl("text", { x: m.l - 8, y: sy(t) + 4, "text-anchor": "end" }); tx.textContent = yFmt(t); g.append(tx); }
    for (const t of niceTicks(x0, x1, 8)) { const tx = svgEl("text", { x: sx(t), y: H - m.b + 18, "text-anchor": "middle" }); tx.textContent = t; g.append(tx); }
    g.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }));
    const xl = svgEl("text", { x: W - m.r, y: H - 4, "text-anchor": "end" }); xl.textContent = xLabel; g.append(xl);
    if (yLabel) { const yl = svgEl("text", { x: m.l, y: m.t - 2, "text-anchor": "start" }); yl.textContent = yLabel; g.append(yl); }
    svg.append(g);
    for (const s of series) {
      if (s.area) {
        const d = s.area.lo.map((v, i) => `${i ? "L" : "M"}${sx(xs[i])},${sy(v)}`).join(" ") + " " + s.area.hi.map((v, i) => `L${sx(xs[s.area.hi.length - 1 - i])},${sy(s.area.hi[s.area.hi.length - 1 - i])}`).join(" ") + " Z";
        svg.append(svgEl("path", { d, fill: s.color, "fill-opacity": 0.15, stroke: "none" }));
      }
      const d = s.values.map((v, i) => (v == null ? "" : `${i && s.values[i - 1] != null ? "L" : "M"}${sx(xs[i])},${sy(v)}`)).join(" ");
      svg.append(svgEl("path", { class: "series", d, stroke: s.color, "stroke-dasharray": s.dash || "" }));
    }
    if (threshold != null) svg.append(svgEl("line", { class: "threshold", x1: m.l, x2: W - m.r, y1: sy(threshold), y2: sy(threshold) }));
    markers.forEach((mk, k) => { svg.append(svgEl("line", { class: "marker" + (mk.dotted ? " dotted" : ""), x1: sx(mk.x), x2: sx(mk.x), y1: m.t, y2: m.t + ih })); const t = svgEl("text", { x: sx(mk.x) + 4, y: m.t + 12 + 14 * k, class: "axis" }); t.setAttribute("style", "font-size:11px;fill:var(--ink-2)"); t.textContent = mk.label; svg.append(t); });
    // hover
    const cross = svgEl("line", { class: "crosshair", y1: m.t, y2: m.t + ih, visibility: "hidden" }); svg.append(cross);
    const dots = series.map((s) => { const c = svgEl("circle", { class: "dot", r: 5, fill: s.color, visibility: "hidden" }); svg.append(c); return c; });
    const hit = svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }); svg.append(hit);
    hit.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
      const xv = x0 + ((px - m.l) / iw) * (x1 - x0); let i = 0, best = Infinity; xs.forEach((v, k) => { const d = Math.abs(v - xv); if (d < best) { best = d; i = k; } });
      cross.setAttribute("x1", sx(xs[i])); cross.setAttribute("x2", sx(xs[i])); cross.setAttribute("visibility", "visible");
      let html = `<div class="t">${xLabel} ${fmt.num(xs[i], 0)}</div>`;
      series.forEach((s, k) => { const v = s.values[i]; if (v == null) { dots[k].setAttribute("visibility", "hidden"); return; } dots[k].setAttribute("cx", sx(xs[i])); dots[k].setAttribute("cy", sy(v)); dots[k].setAttribute("visibility", "visible"); html += `<div class="row"><span><span class="swatch" style="background:${s.color};display:inline-block;width:8px;height:8px;margin-right:6px"></span>${s.name}</span><span>${(s.fmt || yFmt)(v)}</span></div>`; });
      if (opts.extra) html += opts.extra(i);
      showTip(e.clientX, e.clientY, html);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); hideTip(); });
    container.append(svg);
    if (series.length > 1) container.append(el("div", { class: "legend" }, series.map((s) => el("span", {}, [el("span", { class: "swatch", style: `background:${s.color}` }), s.name]))));
  }

  /** Horizontal bar chart (one series) with per-bar threshold marks and a colour by status. */
  function barChart(container, opts) {
    const { items, title, xLabel, height } = opts; // items: {label, value, color, marks:[{x,label}], tip}
    container.innerHTML = "";
    if (title) container.append(el("p", { class: "title" }, title));
    const rowH = 18, W = 1000, m = { t: 8, r: 16, b: 28, l: 90 }, H = height || m.t + m.b + items.length * rowH;
    const iw = W - m.l - m.r; const x1 = Math.max(...items.flatMap((it) => [it.value, ...(it.marks || []).map((k) => k.x)]), 0.01) * 1.08;
    const sx = (v) => m.l + (v / x1) * iw;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title || "bar chart" });
    const g = svgEl("g", { class: "axis" });
    for (const t of niceTicks(0, x1, 6)) { g.append(svgEl("line", { class: "gridline", x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); const tx = svgEl("text", { x: sx(t), y: H - m.b + 16, "text-anchor": "middle" }); tx.textContent = t; g.append(tx); }
    const xl = svgEl("text", { x: W - m.r, y: H - 4, "text-anchor": "end" }); xl.textContent = xLabel || ""; g.append(xl);
    svg.append(g);
    items.forEach((it, i) => {
      const y = m.t + i * rowH;
      const lab = svgEl("text", { x: m.l - 8, y: y + rowH * 0.72, "text-anchor": "end", class: "axis" }); lab.setAttribute("style", "font-size:11px;fill:var(--ink-2)"); lab.textContent = it.label; svg.append(lab);
      const bar = svgEl("rect", { x: m.l, y: y + 3, width: Math.max(0, sx(it.value) - m.l), height: rowH - 6, fill: it.color, rx: 2 }); svg.append(bar);
      for (const mk of it.marks || []) svg.append(svgEl("line", { x1: sx(mk.x), x2: sx(mk.x), y1: y + 1, y2: y + rowH - 1, stroke: mk.color || "var(--ink)", "stroke-width": mk.width || 1.5, "stroke-dasharray": mk.dash || "" }));
      const hit = svgEl("rect", { x: m.l, y, width: iw, height: rowH, fill: "transparent" });
      hit.addEventListener("mousemove", (e) => showTip(e.clientX, e.clientY, it.tip || `${it.label}: ${fmt.num(it.value, 3)}`));
      hit.addEventListener("mouseleave", hideTip); svg.append(hit);
    });
    container.append(svg);
  }

  // ------------------------------------------------------------------ sections
  let OV = null;

  async function loadHeader() {
    const h = await api("/api/health");
    $("#m-status").textContent = ""; $("#m-drift").textContent = "";
    $("#m-status").append(el("span", { class: "status " + statusClass(h.status) }, `${h.status} · ${h.checks_passed}/${h.checks_total} gates`));
    $("#m-model").textContent = `sensorlab ${h.model_version} · ${h.model_path.split("/").pop()}`;
    $("#m-fitted").textContent = fmt.date(h.fitted_at);
    $("#m-primary").textContent = h.primary_detector;
    $("#m-drift").append(el("span", { class: "status " + statusClass(h.drift_status) }, h.drift_status));
    $("#foot-version").textContent = `sensorlab ${h.sensorlab_version} · process started ${fmt.date(h.started_at)}`;
  }

  function kpi(label, value, unit, sub) { return el("div", { class: "kpi" }, [el("span", { class: "label" }, label), el("div", { class: "value" }, [value, unit ? el("span", { class: "unit" }, unit) : ""]), el("div", { class: "sub" }, sub || "")]); }

  async function loadOverview() {
    OV = await api("/api/overview");
    const k = OV.kpis, p = OV.model.primary_detector, ds = OV.model.dataset || {};
    $("#lede").textContent = `Deployed pipeline (primary detector ${p}) evaluated on ${OV.split.test_runs} held-out runs (${OV.split.test_faulty_runs} faulty) · ${OV.split.train_runs} train / ${OV.split.val_runs} validation runs · source ${ds.data || "synthetic"}, seed ${ds.seed ?? "—"} · ${OV.model.n_sensors} sensors at ${OV.model.sample_minutes} min.`;
    const box = $("#kpis"); box.innerHTML = "";
    box.append(
      kpi("AUROC · " + p, fmt.num(k.auroc, 3), "", `FAR observed ${fmt.pct(k.far_observed)} vs target ${fmt.pct(k.far_target, 0)}`),
      kpi("Faulty runs caught", fmt.pct(k.fraction_detected, 0), "", `median delay ${fmt.min(k.median_delay_min)}`),
      kpi("Expected cost · test", fmt.int(k.expected_cost), "CHF", `${k.false_alarms} false alarms · ${k.missed_faults} missed`),
      kpi("Diagnosis accuracy", fmt.num(k.diagnosis_accuracy, 3), "", `macro-F1 ${fmt.num(k.diagnosis_macro_f1, 3)} · 22 classes`),
      kpi("RUL 80 % coverage", fmt.pct(k.rul_coverage, 0), "", `raw ${fmt.pct(k.rul_coverage_raw, 0)} · MAE ${fmt.min(k.rul_mae_min)}`),
      kpi("Alarm precision / recall", `${fmt.num(k.alarm_precision, 2)} / ${fmt.num(k.alarm_recall, 2)}`, "", `outcomes: ${Object.entries(OV.outcomes).map(([a, b]) => `${b} ${a}`).join(", ")}`),
    );
    // published multi-seed
    try {
      const r = await api("/api/results");
      if (r && r.aggregate) { const a = r.aggregate.detection[p]; $("#published-note").textContent = `Published multi-seed results (seeds ${Object.keys(r.seeds).join(", ")}, ${fmt.date(r.generated_at)}): ${p} AUROC ${fmt.num(a.auroc.mean, 3)} ± ${fmt.num(a.auroc.std, 3)}, runs caught ${fmt.pct(a.fraction_detected.mean, 0)}, median delay ${fmt.min(a.median_delay_min.mean)} ± ${fmt.num(a.median_delay_min.std, 0)}; diagnosis accuracy ${fmt.num(r.aggregate.diagnosis.accuracy.mean, 3)} ± ${fmt.num(r.aggregate.diagnosis.accuracy.std, 3)}; RUL coverage ${fmt.pct(r.aggregate.rul.coverage_80.mean, 0)}. This process shows a single seed.`; }
      else $("#published-note").textContent = "No published multi-seed results found (run `sensorlab evaluate --seeds 0 1 2`).";
    } catch { $("#published-note").textContent = ""; }

    // acceptance
    const t = $("#acceptance"); t.innerHTML = "";
    t.append(el("thead", {}, el("tr", {}, [el("th", {}, "Gate"), el("th", { class: "num" }, "Observed"), el("th", { class: "num" }, "Limit"), el("th", {}, "Result")])));
    const fmtBy = (f, v) => (f === "pct" ? fmt.pct(v) : f === "int" ? fmt.int(v) : String(v));
    t.append(el("tbody", {}, OV.acceptance.map((c) => el("tr", {}, [el("td", {}, c.name), el("td", { class: "num" }, fmtBy(c.format, c.value)), el("td", { class: "num" }, fmtBy(c.format, c.limit)), el("td", {}, el("span", { class: c.passed ? "pass" : "fail" }, c.passed ? "PASS" : "FAIL"))]))));

    // ownership + thresholds
    const o = $("#ownership"); o.innerHTML = "";
    o.append(el("thead", {}, el("tr", {}, [el("th", {}, "Knob"), el("th", {}, "Owner"), el("th", { class: "num" }, "Value in force")])));
    o.append(el("tbody", {}, OV.ownership.map((r) => el("tr", {}, [el("td", {}, r.knob), el("td", {}, r.owner), el("td", { class: "num" }, r.key === "far_target" ? fmt.pct(r.value, 1) : typeof r.value === "number" ? fmt.int(r.value) : String(r.value))]))));
    const th = $("#thresholds"); th.innerHTML = "";
    th.append(el("thead", {}, el("tr", {}, [el("th", {}, "Detector"), el("th", { class: "num" }, "FAR"), el("th", { class: "num" }, "Operating")])));
    th.append(el("tbody", {}, OV.model.detectors.map((d) => el("tr", {}, [el("td", {}, [el("span", { class: "swatch", style: `background:${DET_COLOR[d]}` }), d]), el("td", { class: "num" }, fmt.num(OV.thresholds.far[d], 3)), el("td", { class: "num" }, fmt.num(OV.thresholds.operating[d], 3))]))));

    // detection table
    const d = $("#detection"); d.innerHTML = "";
    d.append(el("thead", {}, el("tr", {}, ["Detector", "AUROC", "TPR @ FAR", "FAR observed", "Runs caught", "Median delay", "p90 delay", "Test cost", "False alarms", "Missed", "Fit"].map((h, i) => el("th", { class: i ? "num" : "" }, h)))));
    d.append(el("tbody", {}, OV.model.detectors.map((n) => { const r = OV.detection[n], c = OV.decision[n]; return el("tr", {}, [el("td", {}, [el("span", { class: "swatch", style: `background:${DET_COLOR[n]}` }), n + (n === p ? " · primary" : "")]), el("td", { class: "num" }, fmt.num(r.auroc, 3)), el("td", { class: "num" }, fmt.num(r.tpr_at_far, 2)), el("td", { class: "num" }, fmt.pct(r.far_observed)), el("td", { class: "num" }, fmt.pct(r.fraction_detected, 0)), el("td", { class: "num" }, fmt.min(r.median_delay_min)), el("td", { class: "num" }, fmt.min(r.p90_delay_min)), el("td", { class: "num" }, fmt.int(c.expected_cost)), el("td", { class: "num" }, fmt.int(c.false_alarms)), el("td", { class: "num" }, `${c.missed_faults}/${c.n_faulty_runs}`), el("td", { class: "num" }, fmt.num(r.fit_seconds, 1) + " s")]); })));

    // decision controls
    const sel = $("#dec-detector"); sel.innerHTML = ""; OV.model.detectors.forEach((n) => sel.append(el("option", { value: n }, n))); sel.value = p;
    $("#dec-fa").value = OV.cost_model.false_alarm_cost; $("#dec-mf").value = OV.cost_model.missed_fault_cost; $("#dec-ld").value = OV.cost_model.delay_cost_per_min;
  }

  async function loadDecision() {
    const q = new URLSearchParams({ detector: $("#dec-detector").value, fa: $("#dec-fa").value, mf: $("#dec-mf").value, ld: $("#dec-ld").value });
    const d = await api("/api/decision?" + q);
    lineChart($("#cost-chart"), { x: d.grid, series: [{ name: "expected cost", values: d.costs, color: DET_COLOR[d.detector], fmt: fmt.chf }], title: `Expected cost vs threshold · ${d.detector} · test runs`, xLabel: "threshold", yLabel: "CHF", yFmt: (v) => fmt.int(v), yMin: 0, markers: [{ x: d.oracle.threshold, label: "oracle" }, { x: d.deployed.threshold, label: "deployed", dotted: true }], extra: (i) => `<div class="row"><span>false alarms</span><span>${d.false_alarms[i]}</span></div><div class="row"><span>missed</span><span>${d.missed_faults[i]}</span></div>` });
    const s = $("#dec-stats"); s.innerHTML = "";
    const row = (label, a, b) => el("div", { class: "stat" }, [el("span", { class: "label" }, label), el("div", { class: "value" }, a), el("div", { class: "sub" }, `deployed: ${b}`)]);
    s.append(row("Oracle threshold (test)", fmt.num(d.oracle.threshold, 3), fmt.num(d.deployed.threshold, 3)), row("Expected cost", fmt.chf(d.oracle.expected_cost), fmt.chf(d.deployed.expected_cost)), row("False alarms", fmt.int(d.oracle.false_alarms), fmt.int(d.deployed.false_alarms)), row("Missed faults", `${d.oracle.missed_faults} / ${d.oracle.n_faulty_runs}`, `${d.deployed.missed_faults} / ${d.deployed.n_faulty_runs}`), row("Mean delay", fmt.min(d.oracle.mean_delay_min, 1), fmt.min(d.deployed.mean_delay_min, 1)));
  }

  let RUNS = [], currentRun = null;
  async function loadRuns() {
    RUNS = await api("/api/runs");
    const t = $("#runs"); t.innerHTML = "";
    t.append(el("thead", {}, el("tr", {}, ["Run", "True fault", "Onset", "First alarm after onset", "Delay", "Pre-onset alarms", "Diagnosed at alarm", "RUL p50 at alarm", "Final action", "Outcome"].map((h, i) => el("th", { class: [2, 3, 4, 5, 7].includes(i) ? "num" : "" }, h)))));
    const cls = { caught: "good", clean: "good", "false alarm": "warn", missed: "crit" };
    t.append(el("tbody", {}, RUNS.map((r) => { const tr = el("tr", { class: "clickable", "data-run": r.run_id }, [el("td", {}, `#${String(r.run_id).padStart(2, "0")}`), el("td", {}, r.true_fault), el("td", { class: "num" }, fmt.min(r.onset_min)), el("td", { class: "num" }, fmt.min(r.first_alarm_min)), el("td", { class: "num" }, fmt.min(r.delay_min)), el("td", { class: "num" }, r.pre_onset_alarms), el("td", {}, r.diagnosed_at_alarm || "—"), el("td", { class: "num" }, fmt.min(r.rul_p50_at_alarm)), el("td", {}, r.final_action), el("td", {}, el("span", { class: "status " + cls[r.outcome] }, r.outcome))]); tr.addEventListener("click", () => loadRun(r.run_id)); return tr; })));
    const first = RUNS.find((r) => r.true_fault_id === 13) || RUNS.find((r) => r.true_fault_id > 0) || RUNS[0];
    if (first) loadRun(first.run_id);
  }

  async function loadRun(rid) {
    currentRun = rid;
    document.querySelectorAll("#runs tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.run == rid));
    const d = await api(`/api/runs/${rid}?sensors=${encodeURIComponent($("#run-sensors").value)}`);
    $("#run-title").textContent = `Run #${String(rid).padStart(2, "0")} · ${d.true_fault}`;
    const band = d.onset_index >= 0 ? { from: d.onset_min } : null;
    const palette = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--ink-2)", "var(--ink-3)"];
    lineChart($("#run-traces"), { x: d.t_minutes, series: Object.entries(d.sensors).map(([n, v], i) => ({ name: n, values: v, color: palette[i % palette.length] })), title: "Sensor traces (raw units) · shaded = fault active", band, height: 200 });
    const prim = d.scores[d.primary_detector];
    lineChart($("#run-scores"), { x: d.t_minutes, series: [{ name: `${d.primary_detector} score`, values: prim, color: DET_COLOR[d.primary_detector] }], title: `Primary detector score vs operating threshold (dashed)`, threshold: d.operating_threshold, band, height: 200, yMin: 0, extra: (i) => `<div class="row"><span>alarm</span><span>${d.alarm[i] ? "confirmed" : "—"}</span></div><div class="row"><span>diagnosis</span><span>${d.fault_pred[i]} (${fmt.num(d.fault_confidence[i], 2)})</span></div><div class="row"><span>action</span><span>${d.action[i]}</span></div>` });
    const others = Object.entries(d.scores).filter(([n]) => n !== d.primary_detector).map(([n, v]) => ({ name: `${n} score ÷ its FAR threshold`, values: v.map((x) => x / d.far_thresholds[n]), color: DET_COLOR[n] }));
    if (others.length) lineChart($("#run-others"), { x: d.t_minutes, series: others, title: "Other detectors · score divided by each one's FAR threshold (1.0 = threshold)", threshold: 1.0, band, height: 160, yMin: 0 });
    lineChart($("#run-rul"), { x: d.t_minutes, series: [{ name: "RUL p50", values: d.rul_p50, color: "var(--series-1)", area: { lo: d.rul_p10, hi: d.rul_p90 } }], title: "Remaining useful life · median and conformal 80 % interval (minutes)", band, height: 180, yMin: 0, yFmt: (v) => fmt.int(v) });
    const tl = $("#run-actions"); tl.innerHTML = ""; const n = d.action.length;
    let i = 0; while (i < n) { let j = i; while (j < n && d.action[j] === d.action[i]) j++; const seg = el("span", { class: "act-" + d.action[i], style: `width:${(100 * (j - i)) / n}%`, title: `${d.action[i]} · ${fmt.min(d.t_minutes[i])} → ${fmt.min(d.t_minutes[j - 1])}` }); tl.append(seg); i = j; }
    $("#action-legend").innerHTML = ""; ACTIONS.forEach((a) => $("#action-legend").append(el("span", {}, [el("span", { class: "swatch act-" + a }), a])));
  }

  async function loadDiagnosis() {
    const d = await api("/api/diagnosis");
    const s = $("#diag-stats"); s.innerHTML = "";
    const stat = (l, v, sub) => el("div", { class: "stat" }, [el("span", { class: "label" }, l), el("div", { class: "value" }, v), el("div", { class: "sub" }, sub || "")]);
    s.append(stat("Accuracy (test)", fmt.num(d.metrics.accuracy, 3)), stat("Balanced accuracy", fmt.num(d.metrics.balanced_accuracy, 3)), stat("Macro-F1", fmt.num(d.metrics.macro_f1, 3), `${d.metrics.n_samples} test windows`));
    const t = $("#diag-table"); t.innerHTML = "";
    t.append(el("thead", {}, el("tr", {}, ["Fault", "Driver 1", "Driver 2", "Driver 3"].map((h) => el("th", {}, h)))));
    t.append(el("tbody", {}, d.top_sensors_per_fault.map((r) => el("tr", {}, [el("td", {}, r.fault_name), ...r.sensors.map((x) => el("td", {}, `${x.sensor} (${fmt.num(x.importance, 2)})`))]))));
  }

  async function loadDrift() {
    const sensor = $("#drift-sensor").value, shift = $("#drift-shift").value;
    $("#drift-shift-val").textContent = fmt.num(shift, 2);
    const d = await api(`/api/drift?sensor=${encodeURIComponent(sensor)}&shift=${shift}`);
    const s = $("#drift-status"); s.innerHTML = "";
    s.append(el("div", { class: "stat" }, [el("span", { class: "label" }, "Status on normal held-out runs"), el("div", { class: "value" }, el("span", { class: "status " + statusClass(d.status) }, d.status)), el("div", { class: "sub" }, `${d.n_current} samples · alert: ${d.alert.length ? d.alert.join(", ") : "none"} · watch: ${d.watch.length ? d.watch.join(", ") : "none"}`)]));
    const items = Object.entries(d.psi).map(([n, v]) => ({ label: n, value: v, color: d.alert.includes(n) ? "var(--crit)" : d.watch.includes(n) ? "var(--warn)" : "var(--series-1)", marks: [{ x: d.watch_thresholds[n], color: "var(--warn)", dash: "2 2" }, { x: d.alert_thresholds[n], color: "var(--crit)" }], tip: `<div class="t">${n}</div><div class="row"><span>PSI</span><span>${fmt.num(v, 3)}</span></div><div class="row"><span>noise floor</span><span>${fmt.num(d.noise_floor[n], 3)}</span></div><div class="row"><span>watch / alert</span><span>${fmt.num(d.watch_thresholds[n], 2)} / ${fmt.num(d.alert_thresholds[n], 2)}</span></div>` }));
    barChart($("#drift-chart"), { items, title: "PSI per sensor · dashed mark = watch threshold, solid = alert threshold", xLabel: "PSI" });
  }

  async function loadAudit() {
    const rows = await api("/api/audit");
    const t = $("#audit"); t.innerHTML = "";
    t.append(el("thead", {}, el("tr", {}, ["When", "File", "Rows", "Runs", "Alarms", "Actions", "Drift"].map((h, i) => el("th", { class: [2, 3, 4].includes(i) ? "num" : "" }, h)))));
    t.append(el("tbody", {}, rows.length ? rows.map((r) => el("tr", {}, [el("td", {}, fmt.date(r.at)), el("td", {}, r.file || "—"), el("td", { class: "num" }, fmt.int(r.rows)), el("td", { class: "num" }, r.runs), el("td", { class: "num" }, fmt.int(r.alarms)), el("td", {}, Object.entries(r.actions).map(([a, b]) => `${b} ${a}`).join(", ")), el("td", {}, r.drift_status ? el("span", { class: "status " + statusClass(r.drift_status) }, r.drift_status + (r.drift_alert.length ? " · " + r.drift_alert.slice(0, 3).join(", ") : "")) : "—")])) : [el("tr", {}, el("td", { colspan: 7 }, "No scoring calls yet."))]));
  }

  async function scoreBatch(asCsv) {
    const f = $("#score-file").files[0]; const msg = $("#score-msg");
    if (!f) { msg.textContent = "Choose a file first."; return; }
    msg.textContent = "Scoring…"; $("#score-btn").disabled = true; $("#score-csv").disabled = true;
    try {
      const fd = new FormData(); fd.append("file", f);
      if (asCsv) {
        const r = await fetch("/api/score?format=csv", { method: "POST", body: fd }); if (!r.ok) throw new Error(await r.text());
        const blob = await r.blob(); const a = el("a", { href: URL.createObjectURL(blob), download: `scored_${f.name.replace(/\.[^.]+$/, "")}.csv` }); document.body.append(a); a.click(); a.remove(); msg.textContent = "Downloaded.";
      } else {
        const d = await api("/api/score", { method: "POST", body: fd });
        const s = $("#score-stats"); s.innerHTML = "";
        const stat = (l, v, sub) => el("div", { class: "stat" }, [el("span", { class: "label" }, l), el("div", { class: "value" }, v), el("div", { class: "sub" }, sub || "")]);
        s.append(stat("Rows scored", fmt.int(d.summary.rows), `${d.summary.runs} run(s)`), stat("Confirmed alarms", fmt.int(d.summary.alarms)), stat("Actions", Object.entries(d.summary.actions).map(([a, b]) => `${b} ${a}`).join(" · ")), stat("Drift", d.drift ? el("span", { class: "status " + statusClass(d.drift.status) }, d.drift.status) : "—", d.drift && d.drift.alert.length ? "alert: " + d.drift.alert.join(", ") : ""));
        const t = $("#score-preview"); t.innerHTML = "";
        const cols = ["run_id", "t_minutes", "primary_score", "alarm", "fault_name_pred", "fault_confidence", "rul_p50_min", "action"];
        t.append(el("thead", {}, el("tr", {}, cols.map((c) => el("th", {}, c)))));
        t.append(el("tbody", {}, d.preview.map((r) => el("tr", {}, cols.map((c) => el("td", {}, typeof r[c] === "number" ? fmt.num(r[c], 2) : String(r[c])))))));
        msg.textContent = `Done — ${d.summary.rows} rows.`;
      }
      loadAudit();
    } catch (e) { msg.textContent = "Error: " + e.message.slice(0, 300); }
    finally { $("#score-btn").disabled = false; $("#score-csv").disabled = false; }
  }

  // ------------------------------------------------------------------ boot
  async function boot() {
    try {
      await loadHeader();
      await loadOverview();
      loadDecision();
      loadRuns();
      loadDiagnosis().catch((e) => { $("#diag-stats").textContent = "Diagnosis unavailable: " + e.message; });
      const sel = $("#drift-sensor"); OV.ownership && (await api("/api/manifest")).sensor_names.forEach((n) => sel.append(el("option", { value: n }, n)));
      loadDrift();
      loadAudit();
      ["#dec-detector", "#dec-fa", "#dec-mf", "#dec-ld"].forEach((s) => $(s).addEventListener("change", loadDecision));
      $("#drift-sensor").addEventListener("change", loadDrift); $("#drift-shift").addEventListener("input", loadDrift);
      $("#run-sensors").addEventListener("change", () => currentRun != null && loadRun(currentRun));
      $("#score-btn").addEventListener("click", () => scoreBatch(false)); $("#score-csv").addEventListener("click", () => scoreBatch(true));
    } catch (e) { document.body.prepend(el("p", { class: "fail", style: "padding:16px 24px" }, "Dashboard failed to load: " + e.message)); }
  }
  boot();
})();
