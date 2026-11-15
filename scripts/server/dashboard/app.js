// Panther dashboard: a small hash-routed app over /api/state (polled every 10 s), laid out the way W&B, MLflow and
// Airflow are: a sidebar of sections, list pages that lead to detail pages, and tabs on a detail page.
//   #/                          overview: what is running and when it ends, headline numbers, latest results
//   #/contests                  every contest, filterable and sortable
//   #/contests/<id>[/steps|/events]   one contest: results, step timeline, its events
//   #/experiments               experiments (EfficientQAT)
//   #/experiments/qat[/stage1|/stage2|/settings]
//   #/system                    GPU, bookings, disk, the Pi
//   #/logs                      every log, streamed live (server-sent events), searchable
// Plain DOM and SVG, no libraries.
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const GiB = 2 ** 30;
const sgn = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const pct = (a, b) => sgn(((a - b) / b) * 100, 0) + "%";
const BPW = { IQ1_S: 1.56, IQ1_M: 1.75, IQ2_XXS: 2.06, IQ2_XS: 2.31, IQ2_S: 2.5, Q2_K: 2.63, IQ3_XXS: 3.06, Q3_K: 3.44, IQ3_S: 3.44,
  IQ4_XS: 4.25, Q4_K: 4.5, Q5_K: 5.5, Q6_K: 6.56, Q8_0: 8.5 };
const TASKS = [["mmlu", "MMLU"], ["hellaswag", "HellaSwag"], ["arc_challenge", "ARC"], ["winogrande", "WinoGrande"], ["gsm8k", "GSM8K"]];
const LABEL = { store: "Store", error: "Error estimates", refs: "References", preset: "Preset", ga: "Genetic search", final: "Final", bundle: "Pi bundle" };
const QAT_NAME = "EfficientQAT on 14B @ 5 GiB";
const RUNS = { "stage 1 + 2": "EfficientQAT (Stage 1 + 2)", "stage 2 only": "Control (Stage 2 only)" };
const ui = { state: null, last: 0, path: null, block: null, picked: false, filter: "all", sort: null, asc: true, logq: "", logk: "all" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const dark = () => document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
const typeColor = (t) => { const b = Math.min(1, Math.max(0, ((BPW[t] ?? 4) - 1.5) / 7)); return `hsl(${200 - 175 * b} ${42 + 18 * b}% ${dark() ? 58 : 44}%)`; };
const mins = (s) => (s ? +s.slice(0, 2) * 60 + +s.slice(3, 5) : null);
const hhmm = (m) => `${String(Math.floor(((m % 1440) + 1440) % 1440 / 60)).padStart(2, "0")}:${String(Math.round(((m % 60) + 60) % 60)).padStart(2, "0")}`;
const dur = (m) => (m == null ? "–" : m < 60 ? `${Math.round(m)} min` : `${Math.floor(m / 60)} h ${String(Math.round(m % 60)).padStart(2, "0")}`);
const badge = (s) => `<span class="badge ${esc(s)}">${esc(s)}</span>`;
const verdict = (v) => (v ? `<span class="v ${v}">${v}</span>` : "");
const kpi = (k, v, s = "", extra = "") => `<div class="card kpi"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}${extra}</div>`;
const prog = (f) => (f == null ? "" : `<div class="prog"><i style="width:${(Math.max(0, Math.min(1, f)) * 100).toFixed(1)}%"></i></div>`);
const tabs = (base, list, on) => `<nav class="tabs">${list.map(([k, n]) => `<a href="${base}${k ? "/" + k : ""}" class="${k === on ? "on" : ""}">${n}</a>`).join("")}</nav>`;

// --- charts -----------------------------------------------------------------------------------------------------------
function nice(lo, hi, n = 4) {
  const span = hi - lo || Math.abs(hi) || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}
function chart(el, { series, refs = [], marks = [], vlines = [], xdomain = null, dots = false, xfmt = (x) => x, yfmt = (y) => y, empty = "no points yet" }) {
  if (!el) return;
  const w = el.clientWidth || 600, h = el.clientHeight || 200, L = 46, R = 10, T = 10, B = 22;
  const pts = series.flatMap((s) => s.pts);
  if (new Set(pts.map((p) => p[0])).size < (xdomain ? 1 : 2)) { el.innerHTML = `<div class="empty">${esc(empty)}</div>`; return; }
  const ys = pts.map((p) => p[1]).concat(refs.map((r) => r.y)), xs = pts.map((p) => p[0]);
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0) * 0.15 || Math.abs(y1) * 0.1 || 1; y0 -= pad; y1 += pad;
  const x0 = xdomain ? xdomain[0] : Math.min(...xs), x1 = xdomain ? xdomain[1] : Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (w - L - R), Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (h - T - B);
  let s = `<svg viewBox="0 0 ${w} ${h}"><g class="ticks">`;
  for (const t of nice(y0, y1, 4)) s += `<line x1="${L}" x2="${w - R}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${L - 6}" y="${Y(t) + 3.5}" text-anchor="end">${esc(yfmt(t))}</text>`;
  for (const t of (xdomain && xdomain[1] % 512 === 0 ? Array.from({ length: xdomain[1] / 512 + 1 }, (_, i) => i * 512).filter((t, i, a) => w > 520 || i % 2 === 0 || i === a.length - 1) : nice(x0, x1, Math.max(2, Math.floor(w / 110))))) s += `<text x="${X(t)}" y="${h - 5}" text-anchor="${X(t) > w - R - 14 ? "end" : X(t) < L + 14 ? "start" : "middle"}">${esc(xfmt(t))}</text>`;
  s += `</g>`;
  for (const r of refs) s += `<line x1="${L}" x2="${w - R}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="${r.color}" stroke-dasharray="4 4"/><text x="${w - R}" y="${Y(r.y) - 5}" text-anchor="end" font-size="10.5" fill="${r.color}">${esc(r.label)}</text>`;
  for (const se of series) {
    if (!se.pts.length) continue;
    const d = se.pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} stroke-linejoin="round"/>`;
  }
  for (const v of vlines) s += `<line x1="${X(v.x)}" x2="${X(v.x)}" y1="${T}" y2="${h - B}" stroke="${v.color}" stroke-dasharray="3 3"/><text x="${X(v.x) - 4}" y="${T + 10}" text-anchor="end" font-size="10.5" fill="${v.color}">${esc(v.label)}</text>`;
  if (dots) for (const se of series) for (const p of se.pts) s += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.5" fill="${se.color}"/>`;
  for (const m of marks) s += `<circle cx="${X(m.x)}" cy="${Y(m.y)}" r="4.5" fill="${css("--panel")}" stroke="${m.color}" stroke-width="2"/>`;
  el.innerHTML = s + `</svg>`;
}
const trend = (pts, k = 2) => pts.map((p, i) => { const w = pts.slice(Math.max(0, i - k), i + k + 1); return [p[0], w.reduce((a, q) => a + q[1], 0) / w.length]; });

// --- shared pieces ------------------------------------------------------------------------------------------------------
function scale(cs) {
  const ci = cs.filter((c) => c.verdict).flatMap((c) => c.verdict.ci);
  const lo = Math.min(-1, ...ci) - 0.5, hi = Math.max(1, ...ci) + 0.5;
  return (v) => (((v - lo) / (hi - lo)) * 100).toFixed(2) + "%";
}
function forest(v, X) {
  const col = v.verdict === "WIN" ? "var(--win)" : v.verdict === "LOSE" ? "var(--lose)" : "var(--tie)";
  return `<div class="fp" style="--c:${col}" title="ours − preset ${sgn(v.diff)} points, 95% interval [${sgn(v.ci[0])}, ${sgn(v.ci[1])}]"><i class="z" style="left:${X(0)}"></i><i class="ln" style="left:${X(v.ci[0])};right:calc(100% - ${X(v.ci[1])})"></i><i class="pt" style="left:${X(v.diff)}"></i></div>`;
}
function qatState(q) {
  if (!q) return null;
  const done = q.blocks.filter((b) => b.after != null), s2 = q.stage2 || [], res = q.results || {};
  if (res["stage 1 + 2"]) return { state: "done", step: "finished", done };
  if (done.length < q.total && q.blocks.length) return { state: "running", step: `Stage 1 · block ${done.length} of ${q.total}`, frac: done.length / q.total, eta: q.stage1_eta, etaWhat: "Stage 1 ends", tab: "stage1", done };
  if (s2.length) { const u = s2.at(-1); return { state: "running", step: `Stage 2 · update ${u.update} of ${u.of}`, frac: u.update / u.of, tab: "stage2", done }; }
  return { state: "running", step: "Stage 2 starting, then held-out tests", tab: "stage2", done };
}
function jobs(st) {
  const out = [], qs = qatState(st.qat);
  if (qs && qs.state === "running") out.push({ name: QAT_NAME, kind: "experiment", href: `#/experiments/qat/${qs.tab}`, ...qs });
  for (const c of (st.contests || []).filter((c) => c.state === "running")) {
    const r = c.steps.find((s) => s.state === "running");
    out.push({ name: `${c.size} @ ${c.budget} GiB`, kind: "contest", href: `#/contests/${c.id}/steps`, step: r ? LABEL[r.name] || r.name : "running",
      frac: c.steps.length ? c.steps.filter((s) => s.state === "done").length / c.steps.length : null, eta: c.eta, etaWhat: "contest ends" });
  }
  return out;
}
function kind(x) {
  if (/ note: /.test(x)) return "note";
  if (/FAILED|Traceback/.test(x)) return "fail";
  if (/held-out test vs|QAT .*held-out|: done$|validation MSE .* -> /.test(x)) return "res";
  return "quiet";
}
function feed(list) {
  return `<ul class="feed">${list.map((x) => `<li class="${kind(x)}"><time>${esc(x.slice(0, 5))}</time><span>${esc(x.slice(6).replace(/^note: /, ""))}</span></li>`).join("") || '<li><span class="muted">nothing yet</span></li>'}</ul>`;
}

// --- pages ----------------------------------------------------------------------------------------------------------------
const pages = {};

pages.overview = (st, el) => {
  const cs = st.contests || [], fin = cs.filter((c) => c.verdict), js = jobs(st), sys = st.system || {};
  const n = (v) => fin.filter((c) => c.verdict.verdict === v).length;
  const booked = (sys.bookings || []).reduce((a, b) => a + b.bytes, 0);
  const X = scale(cs);
  el.innerHTML = `
    <div class="ph"><h1>Overview</h1></div>
    <div class="grid cols-4">
      ${kpi("Contests won", `${n("WIN")}<small>/ ${fin.length}</small>`, `${n("TIE")} tied · ${n("LOSE")} lost · ${cs.filter((c) => c.state === "dropped").length} dropped`)}
      ${kpi("Running", `${js.length}`, js.map((j) => esc(j.name.split(" on ")[0])).join(" · ") || "nothing")}
      ${kpi("GPU memory", sys.gpu ? `${(sys.gpu.used / GiB).toFixed(0)}<small>/ ${(sys.gpu.total / GiB).toFixed(0)} GiB</small>` : `${(booked / GiB).toFixed(0)}<small>/ 92 GiB booked</small>`, sys.gpu ? `${sys.gpu.util.toFixed(0)}% busy · ${(booked / GiB).toFixed(0)} GiB booked` : "", sys.gpu ? prog(sys.gpu.used / sys.gpu.total) : prog(booked / (92 * GiB)))}
      ${kpi("Disk free", sys.disk ? `${(sys.disk.free / 1e9).toFixed(0)}<small>GB</small>` : "–", sys.disk ? `of ${(sys.disk.total / 1e12).toFixed(1)} TB` : "", sys.disk ? prog(1 - sys.disk.free / sys.disk.total) : "")}
    </div>
    <div class="card section"><h2>Running now <a class="more" href="#/experiments">all runs →</a></h2>
      ${js.map((j) => `<a class="job" href="${j.href}"><div><b>${esc(j.name)}</b><div class="muted">${esc(j.step)}</div></div>${prog(j.frac) || "<div></div>"}<div class="eta mono">${j.eta ? `~${esc(j.eta)}<small>${esc(j.etaWhat)}</small>` : "–"}</div></a>`).join("") || '<div class="empty">Nothing is running.</div>'}
    </div>
    <div class="grid cols-21 section">
      <div class="card"><h2>Results<a class="more" href="#/contests">all contests →</a></h2>
        <div class="tw"><table><tbody>${fin.map((c) => `<tr class="click" data-href="#/contests/${esc(c.id)}"><td><b>${esc(c.size)}</b> @ ${c.budget} GiB</td><td>${verdict(c.verdict.verdict)}</td><td class="n">${sgn(c.verdict.diff)}</td><td class="hide-s" style="width:42%">${forest(c.verdict, X)}</td></tr>`).join("")}</tbody></table></div>
      </div>
      <div class="card"><h2>Recent activity <a class="more" href="#/logs">logs →</a></h2>${feed((st.events || []).filter((x) => kind(x) !== "quiet").slice(0, 9))}</div>
    </div>`;
};

pages.contests = (st, el) => {
  const all = st.contests || [], X = scale(all);
  const groups = [["all", "All", () => true], ["WIN", "Won", (c) => c.verdict?.verdict === "WIN"], ["TIE", "Tied", (c) => c.verdict?.verdict === "TIE"],
    ["running", "Running", (c) => c.state === "running" || c.state === "queued"], ["dropped", "Dropped", (c) => c.state === "dropped"]];
  const g = groups.find((x) => x[0] === ui.filter) || groups[0];
  let cs = all.filter(g[2]);
  const keys = { contest: (c) => c.budget * 1000 + parseFloat(c.size), diff: (c) => c.verdict?.diff ?? -1e9, ours: (c) => c.held_out?.ours.accuracy ?? -1e9,
    ram: (c) => (st.pi || []).find((p) => p.contest === c.id)?.ram ?? -1e9 };
  if (ui.sort) cs = [...cs].sort((a, b) => (keys[ui.sort](a) - keys[ui.sort](b)) * (ui.asc ? 1 : -1));
  const th = (k, n, cls = "") => `<th class="sort ${cls}${ui.sort === k ? " on" + (ui.asc ? " asc" : "") : ""}" data-k="${k}">${n}</th>`;
  const pis = Object.fromEntries((st.pi || []).map((p) => [p.contest, p]));
  el.innerHTML = `
    <div class="ph"><h1>Contests</h1></div>
    <div class="toolbar"><div class="chips">${groups.map(([k, n, f]) => `<button data-f="${k}" class="${k === g[0] ? "on" : ""}">${n}<em>${all.filter(f).length}</em></button>`).join("")}</div></div>
    <div class="card tw"><table>
      <thead><tr>${th("contest", "Contest")}<th>Status</th><th>Verdict</th>${th("diff", "Ours − preset", "n")}<th class="hide-s">95% interval</th>${th("ours", "Held-out", "n")}<th class="n hide-s">Preset</th>${th("ram", "Pi RAM", "n hide-s")}</tr></thead>
      <tbody>${cs.map((c) => {
        const v = c.verdict, h = c.held_out, p = pis[c.id];
        return `<tr class="click${c.state === "dropped" ? " dim" : ""}" data-href="#/contests/${esc(c.id)}">
          <td><b>${esc(c.size)}</b> @ ${c.budget} GiB</td><td>${badge(c.state)}${c.state === "running" && c.eta ? ` <span class="muted mono small">~${esc(c.eta)}</span>` : ""}</td>
          <td>${v ? verdict(v.verdict) : ""}</td><td class="n">${v ? sgn(v.diff) : "–"}</td>
          <td class="hide-s" style="width:26%">${v ? forest(v, X) : ""}</td>
          <td class="n">${h ? h.ours.accuracy.toFixed(1) : "–"}</td><td class="n hide-s">${h ? `${h.preset_test.accuracy.toFixed(1)} <span class="muted small">${esc(v.preset)}</span>` : "–"}</td>
          <td class="n hide-s">${p ? `${p.ram.toFixed(2)}<span class="muted"> / ${c.budget}</span>` : "–"}</td></tr>`;
      }).join("") || `<tr><td colspan="8" class="empty">No contests here.</td></tr>`}</tbody></table></div>`;
  el.querySelector(".chips").onclick = (e) => { const b = e.target.closest("button"); if (b) { ui.filter = b.dataset.f; draw(); } };
  el.querySelector("thead").onclick = (e) => { const t = e.target.closest("th.sort"); if (!t) return; if (ui.sort === t.dataset.k) ui.asc = !ui.asc; else { ui.sort = t.dataset.k; ui.asc = t.dataset.k === "contest"; } draw(); };
};

function timeline(c, nowStr) {
  const steps = c.steps;
  if (!steps.length) return '<div class="empty">No step record for this contest.</div>';
  // place each step: its start (wrapped past midnight) or the end of the one before; done steps last as long as they took
  let first = steps.findIndex((s) => s.began);
  let cursor = first < 0 ? 0 : mins(steps[first].began) - steps.slice(0, first).reduce((a, s) => a + (s.took || 0), 0);
  const clock = first >= 0, rows = [];
  let now = mins(nowStr.slice(-8, -3));
  for (const s of steps) {
    let a = s.began ? mins(s.began) : null;
    if (a != null) while (a < cursor - 720) a += 1440;
    if (s.state === "waiting") a = Math.max(cursor, now);
    if (a == null) a = cursor;
    const d = s.state === "done" ? s.took ?? 0 : s.state === "waiting" ? s.left : s.spent ?? 0;
    rows.push({ s, a, d });
    cursor = a + d;
    if (s.state === "running" || s.state === "failed") { while (now < a) now += 1440; }
  }
  while (now < rows[0].a) now += 1440;
  const t0 = rows[0].a, t1 = Math.max(...rows.map((r) => r.a + r.d), t0 + 1), X = (m) => (((m - t0) / (t1 - t0)) * 100).toFixed(2) + "%";
  const step = [15, 30, 60, 120, 180, 360, 720, 1440].find((s) => (t1 - t0) / s <= 6) || 1440;
  const ticks = []; for (let m = Math.ceil(t0 / step) * step; m <= t1; m += step) ticks.push(m);
  const live = rows.some((r) => r.s.state !== "done") && clock;
  return `<div class="tl">${rows.map(({ s, a, d }) => `<div class="r"><span title="${esc(s.what)}">${esc(LABEL[s.name] || s.name)}</span>
      <div class="track"><i class="${s.state}" style="left:${X(a)};width:${(((d) / (t1 - t0)) * 100).toFixed(2)}%" title="${esc(LABEL[s.name] || s.name)}: ${s.state}${s.began ? `, began ${s.began}` : ""}, ${s.state === "waiting" ? "about " : ""}${dur(d)}"></i>${live && now <= t1 ? `<span class="now" style="left:${X(now)}"></span>` : ""}</div>
      <span class="mono small ${s.state === "done" ? "muted" : ""}" style="text-align:right">${s.state === "waiting" ? `~${dur(d)}` : s.state === "running" ? `${dur(s.spent)} · ${dur(s.left)} left` : dur(d)}</span></div>`).join("")}
    <div class="axis"><div></div><div>${clock ? ticks.map((m) => `<span style="left:${X(m)}">${hhmm(m)}</span>`).join("") : ""}</div><div></div></div></div>`;
}

pages.contest = (st, el, [id, tab]) => {
  const c = (st.contests || []).find((x) => x.id === id);
  if (!c) { el.innerHTML = `<div class="empty">No contest called ${esc(id)}.</div>`; return; }
  const v = c.verdict, h = c.held_out, p = (st.pi || []).find((x) => x.contest === c.id);
  tab = tab || (v ? "" : "steps");
  const base = `#/contests/${c.id}`;
  const lane = `${c.size.toLowerCase()}-b${c.budget}`;
  const evs = [...new Set([...(c.events || []).slice().reverse(), ...(st.events || []).filter((x) => x.includes(lane) || x.includes(c.id))])];
  let body = "";
  if (tab === "steps") {
    body = `<div class="card pad"><h2>Steps <span class="muted">${c.state === "running" ? `ends ~${esc(c.eta)}` : ""}</span></h2>${timeline(c, st.now || "")}</div>
      <div class="card tw section"><table><thead><tr><th>Step</th><th>Status</th><th class="n">Began</th><th class="n">Took</th></tr></thead><tbody>
      ${c.steps.map((s) => `<tr><td>${esc(LABEL[s.name] || s.name)}</td><td>${badge(s.state)}</td><td class="n">${s.began ?? "–"}</td><td class="n">${s.state === "done" ? dur(s.took) : s.state === "running" ? `${dur(s.spent)} so far` : `~${dur(s.left)}`}</td></tr>`).join("")}
      </tbody></table></div>`;
  } else if (tab === "events") {
    body = `<div class="card">${feed(evs.slice(0, 60))}</div>`;
  } else if (v) {
    const total = c.types.reduce((a, t) => a + t[1], 0);
    const rows = [["Original (16-bit)", h.original], ["Ours", h.ours], [`Preset ${v.preset}`, h.preset_test]];
    body = `<div class="grid cols-4">
        ${kpi("Ours − preset", `${sgn(v.diff)}<small>points</small>`, `95% interval [${sgn(v.ci[0])}, ${sgn(v.ci[1])}]`)}
        ${kpi("Held-out accuracy", `${h.ours.accuracy.toFixed(1)}<small>vs ${h.preset_test.accuracy.toFixed(1)}</small>`, `original ${h.original.accuracy.toFixed(1)}`)}
        ${kpi("Pi peak RAM", p ? `${p.ram.toFixed(2)}<small>/ ${c.budget} GiB</small>` : "–", p ? `predicted ${p.predicted.toFixed(2)} GiB` : "not measured", p ? prog(p.ram / c.budget) : "")}
        ${kpi("Pi speed", p ? `${p.decode.toFixed(2)}<small>tokens/s</small>` : "–", p ? `writing · ${p.prefill.toFixed(1)} reading` : "")}
      </div>
      <div class="card tw section"><h2>Held-out</h2><table>
        <thead><tr><th></th><th class="n">Accuracy</th>${TASKS.map(([, n]) => `<th class="n hide-s">${n}</th>`).join("")}<th class="n">KL</th><th class="n hide-s">Flips %</th></tr></thead>
        <tbody>${rows.map(([n, r]) => `<tr${n === "Ours" ? ' class="ours"' : ""}><td>${esc(n)}</td><td class="n">${r.accuracy.toFixed(2)}</td>${TASKS.map(([k]) => `<td class="n hide-s">${r.per_task?.[k]?.toFixed(1) ?? "–"}</td>`).join("")}<td class="n">${r.kl != null ? r.kl.toFixed(3) : "–"}</td><td class="n hide-s">${r.flips != null && !n.startsWith("Original") ? r.flips.toFixed(1) : "–"}</td></tr>`).join("")}</tbody></table></div>
      <div class="card pad section"><h2>Our mix of types <span class="muted">${total} matrices</span></h2>
        <div class="mix">${c.types.map(([t, n]) => `<i title="${t} ×${n}" style="width:${(n / total) * 100}%;background:${typeColor(t)}"></i>`).join("")}</div>
        <div class="legend">${c.types.map(([t, n]) => `<span><span class="sw" style="background:${typeColor(t)}"></span>${t} <span class="mono">×${n}</span></span>`).join("")}</div>
      </div>`;
  } else body = `<div class="empty">This contest has no result${c.state === "dropped" ? ": it was dropped" : " yet"}.</div>`;
  el.innerHTML = `
    <div class="ph"><h1>${esc(c.size)} @ ${c.budget} GiB</h1>${badge(c.state)}${v ? verdict(v.verdict) : ""}
      <div class="sub">${esc(c.model || "")}${v ? ` · against ${esc(v.preset)}` : ""}${c.state === "running" ? ` · ends ~${esc(c.eta)}` : ""}</div></div>
    ${tabs(base, [...(v ? [["", "Results"]] : []), ["steps", "Steps"], ["events", `Events <span class="muted">${evs.length}</span>`]], tab)}
    ${body}`;
};

pages.experiments = (st, el) => {
  const q = st.qat, qs = qatState(q);
  if (!q) { el.innerHTML = `<div class="ph"><h1>Experiments</h1></div><div class="card empty">No experiments.</div>`; return; }
  const runs = [{ name: RUNS["stage 1 + 2"], r: q.results?.["stage 1 + 2"], state: qs.state, step: qs.step, frac: qs.frac, eta: qs.eta },
    { name: RUNS["stage 2 only"], r: q.results?.["stage 2 only"], state: q.results?.["stage 2 only"] ? "done" : "running", step: q.results?.["stage 2 only"] ? "finished" : "running" }];
  el.innerHTML = `
    <div class="ph"><h1>Experiments</h1></div>
    <div class="card"><h2>${QAT_NAME} <a class="more" href="#/experiments/qat">open →</a></h2><div class="tw"><table>
      <thead><tr><th>Run</th><th>Status</th><th class="hide-s">Progress</th><th class="n">Held-out</th><th>vs winner</th><th class="hide-s">vs preset</th></tr></thead>
      <tbody>${runs.map((x) => `<tr class="click" data-href="#/experiments/qat"><td><b>${esc(x.name)}</b></td><td>${badge(x.state)}</td>
        <td class="hide-s" style="min-width:180px">${x.state === "running" ? `${prog(x.frac)}<div class="muted small">${esc(x.step)}${x.eta ? ` · ~${esc(x.eta)}` : ""}</div>` : '<span class="muted small">finished</span>'}</td>
        <td class="n">${x.r ? x.r.held_out.accuracy.toFixed(2) : "–"}</td><td>${x.r ? `${verdict(x.r.vs_winner.verdict)} <span class="num">${sgn(x.r.vs_winner.diff)}</span>` : '<span class="muted">pending</span>'}</td>
        <td class="hide-s">${x.r ? `${verdict(x.r.vs_preset.verdict)} <span class="num">${sgn(x.r.vs_preset.diff)}</span>` : ""}</td></tr>`).join("")}</tbody></table></div></div>`;
};

function strip(q) {
  return `<div class="strip" id="strip">${Array.from({ length: q.total }, (_, i) => {
    const b = q.blocks.find((x) => x.block === i);
    let cls = "cell", style = "", title = `block ${i}: waiting`;
    if (b && b.after != null) {
      const p = (b.after - b.before) / b.before;
      title = `block ${i}: ${pct(b.after, b.before)}${b.best === 0 ? " (kept as it was)" : ""}`;
      if (b.best === 0) cls += " kept";
      else style = `background:color-mix(in srgb, ${p < 0 ? "var(--win)" : "var(--lose)"} ${Math.round(20 + 75 * Math.min(1, Math.abs(p) / 0.4))}%, var(--sunk))`;
    } else if (b) { cls += " now"; title = `block ${i}: training`; }
    if (i === ui.block) cls += " sel";
    return `<i class="${cls}" style="${style}" title="${esc(title)}" data-b="${i}"></i>`;
  }).join("")}</div>`;
}
// Why Stage 1 trains what it trains: the block-0 tests side by side, each as the change in the block's validation
// error (a bar left of zero is better), the full-length runs last, the one in use marked.
function choice(q) {
  const ts = q.tests || [];
  if (!ts.length) return "";
  const ch = (t) => (t.after - t.before) / t.before, m = Math.max(0.1, ...ts.map((t) => Math.abs(ch(t))));
  const X = (v) => (50 + (50 * v) / m).toFixed(2) + "%";
  const using = (q.strategy || "").startsWith("weights only") ? "weights only" : "weights + scales";
  const full = (w) => ts.filter((t) => t.trains === w && t.steps === Math.max(...ts.map((x) => x.steps || 0))).at(-1);
  const a = full("weights + scales"), b = full("weights only");
  const order = [...ts.map((t, i) => ({ ...t, i }))].sort((x, y) => (x.steps || 0) - (y.steps || 0) || x.i - y.i);
  return `<div class="card"><h2>Block 0 tests</h2>
    <div class="body">
    <div class="cmp"><div class="cr ch"><span></span><div class="ax"><span style="left:${X(-m)}">${sgn(-m * 100, 0)}%</span><span style="left:50%">0</span><span style="left:${X(m)}">${sgn(m * 100, 0)}%</span></div><span></span></div>
    ${order.map((t) => {
      const v = ch(t), use = t.i === ts.indexOf(using === "weights only" ? b : a);
      return `<div class="cr${use ? " use" : ""}"><div class="lb"><b>${esc(t.trains)}</b><span class="muted">${t.steps ? t.steps.toLocaleString() + " steps" : ""}${t.first ? " · first run" : ""}</span>${use ? ' <span class="badge done">in use</span>' : ""}</div>
        <div class="tr"><i class="z"></i><i class="b ${v < 0 ? "good" : "bad"}" style="left:${X(Math.min(0, v))};right:calc(100% - ${X(Math.max(0, v))})"></i></div>
        <div class="nm"><span class="mono">${t.before.toFixed(4)} → ${t.after.toFixed(4)}</span> <b class="${v < 0 ? "good" : "bad"} mono">${pct(t.after, t.before)}</b></div></div>`;
    }).join("")}</div></div></div>`;
}
function resultsTable(res) {
  return `<table><thead><tr><th>Run</th><th class="n">Held-out</th><th>vs current winner</th><th class="hide-s">vs preset</th></tr></thead><tbody>${Object.entries(res).map(([n, r]) =>
    `<tr><td>${esc(RUNS[n] || n)}</td><td class="n">${r.held_out.accuracy.toFixed(2)}</td><td class="num">${verdict(r.vs_winner.verdict)} ${sgn(r.vs_winner.diff)} <span class="muted hide-s">[${sgn(r.vs_winner.ci[0])}, ${sgn(r.vs_winner.ci[1])}]</span></td><td class="num hide-s">${verdict(r.vs_preset.verdict)} ${sgn(r.vs_preset.diff)}</td></tr>`).join("")}</tbody></table>`;
}

pages.qat = (st, el, [tab = ""]) => {
  const q = st.qat;
  if (!q) { el.innerHTML = `<div class="empty">No experiment running.</div>`; return; }
  const qs = qatState(q), done = qs.done, cur = q.blocks.find((b) => b.after == null), s2 = q.stage2 || [], res = q.results || {};
  const main = res["stage 1 + 2"], ctl = res["stage 2 only"];
  const mean = done.length ? done.reduce((a, b) => a + (b.after - b.before) / b.before, 0) / done.length : 0;
  if (!ui.picked || !q.blocks.some((b) => b.block === ui.block)) ui.block = (cur && cur.checks.length > 1 ? cur : done.at(-1) || cur || {}).block ?? null;
  let body = "";
  if (tab === "stage1") {
    const b = q.blocks.find((x) => x.block === ui.block);
    body = `<div class="card pad"><h2>Blocks</h2>${strip(q)}</div>
      <div class="card pad section"><h2>Block ${ui.block ?? "–"} <span class="muted">${b ? (b.after != null ? `${b.before.toFixed(4)} → ${b.after.toFixed(4)} (${pct(b.after, b.before)})${b.stopped ? `, stopped early at ${b.stopped}` : ""}` : "training") : ""}</span></h2>
        <div class="chart" id="c-block"></div><div class="legend"><span><i style="background:var(--accent)"></i>validation error by step</span><span><i style="background:var(--muted)"></i>before training</span><span><i style="background:var(--win);height:8px;width:8px;border-radius:50%"></i>kept point</span></div></div>
      <div class="card tw section"><table><thead><tr><th>Block</th><th class="n">Before</th><th class="n">After</th><th class="n">Change</th><th class="n hide-s">Kept step</th><th class="hide-s">Stopped early</th><th class="n">Done</th></tr></thead><tbody>
        ${done.slice().reverse().map((b) => `<tr class="click" data-block="${b.block}"><td>${b.block}</td><td class="n">${b.before.toFixed(4)}</td><td class="n">${b.after.toFixed(4)}</td><td class="n">${b.best === 0 ? '<span class="muted">kept</span>' : pct(b.after, b.before)}</td><td class="n hide-s">${b.best}</td><td class="hide-s muted">${b.stopped ? `at ${b.stopped}` : ""}</td><td class="n muted">${esc(b.done || "")}</td></tr>`).join("")}
      </tbody></table></div>`;
  } else if (tab === "stage2") {
    const last = (a) => (a.length ? a.at(-1) : null), m = last(s2), c = last(q.control || []);
    body = `<div class="grid cols-2">
        ${kpi("EfficientQAT · Stage 2", m ? `${m.update}<small>/ ${m.of} updates</small>` : `<span class="muted">—</span>`, m ? `KL ${m.kl.toFixed(3)} at ${m.at}` : "starts after Stage 1", m ? prog(m.update / m.of) : "")}
        ${kpi("Control · Stage 2 only", c ? `${c.update}<small>/ ${c.of} updates</small>` : "–", c ? `KL ${c.kl.toFixed(3)} at ${c.at}` : "", c ? prog(c.update / c.of) : "")}
      </div>
      <div class="card pad section"><h2>KL</h2><div class="chart" id="c-s2"></div>
        <div class="legend"><span><i style="background:var(--s1)"></i>after Stage 1</span><span><i style="background:var(--s2)"></i>control</span></div></div>`;
  } else if (tab === "setup") {
    body = `${choice(q)}
      <div class="grid cols-2 section">${(q.config || []).map((c) => `<div class="card"><h2>${esc(c.stage)}</h2>
        <dl class="kv">${c.rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></div>`).join("")}</div>`;
  } else {
    body = `<div class="grid cols-4">
        ${kpi("Stage 1 · blocks", `${done.length}<small>/ ${q.total}</small>`, qs.state === "running" && q.stage1_eta && done.length < q.total ? `ends ~${esc(q.stage1_eta)} · ${q.minutes_per_block ? Math.round(q.minutes_per_block) + " min a block" : ""}` : "finished", prog(done.length / q.total))}
        ${kpi("Stage 1 · error change", done.length ? `${sgn(mean * 100, 0)}<small>%</small>` : "–", "mean over finished blocks")}
        ${kpi("Stage 2 · scales", s2.length ? `${s2.at(-1).update}<small>/ ${s2.at(-1).of}</small>` : `<span class="muted">—</span>`, s2.length ? "updates" : "after Stage 1")}
        ${kpi("Result vs winner", main ? `${verdict(main.vs_winner.verdict)} ${sgn(main.vs_winner.diff)}` : `<span class="muted">pending</span>`, ctl ? `control: ${ctl.vs_winner.verdict} ${sgn(ctl.vs_winner.diff)}` : "")}
      </div>
      <div class="card pad section"><h2>Stage 1 blocks <a class="more" href="#/experiments/qat/stage1">details →</a></h2>${strip(q)}</div>
      <div class="card tw section"><h2>Results</h2>${Object.keys(res).length ? resultsTable(res) : '<div class="empty">No results yet.</div>'}</div>
      ${q.now ? `<div class="card pad section prose"><h2>Latest note <span class="muted">${esc(q.now_at || "")}</span></h2><p>${esc(q.now)}</p></div>` : ""}`;
  }
  el.innerHTML = `
    <div class="ph"><h1>${QAT_NAME}</h1>${badge(qs.state)}<div class="sub">${esc((q.strategy || "").split(",")[0])}${qs.state === "running" ? ` · ${esc(qs.step)}` : ""}</div></div>
    ${tabs("#/experiments/qat", [["", "Overview"], ["stage1", "Stage 1 · blocks"], ["stage2", "Stage 2 · scales"], ["setup", "Setup"]], tab)}
    ${body}`;
  const s = el.querySelector("#strip");
  if (s) s.onclick = (e) => { const c = e.target.closest(".cell"); if (!c) return; ui.block = +c.dataset.b; ui.picked = true; if (tab !== "stage1") location.hash = "#/experiments/qat/stage1"; else draw(); };
  el.querySelectorAll("tr[data-block]").forEach((r) => (r.onclick = () => { ui.block = +r.dataset.block; ui.picked = true; draw(); scrollTo({ top: 0, behavior: "smooth" }); }));
  const b = q.blocks.find((x) => x.block === ui.block);
  chart(el.querySelector("#c-block"), {
    series: b ? [{ color: css("--accent"), pts: [...(b.before != null && !b.checks.some((c) => c.step === 0) ? [[0, b.before]] : []), ...b.checks.filter((c) => c.val != null).map((c) => [c.step, c.val])] }] : [],
    xdomain: [0, (b && b.of) || (b && b.checks[0]?.of) || 4096], dots: true,
    vlines: b && b.stopped ? [{ x: b.stopped, label: "stopped early", color: css("--muted") }] : [],
    refs: b && (b.before ?? b.start) != null ? [{ y: b.before ?? b.start, label: "before", color: css("--muted") }] : [],
    marks: b && b.best && b.checks.find((c) => c.step === b.best) ? [{ x: b.best, y: b.checks.find((c) => c.step === b.best).val, color: css("--win") }] : [],
    yfmt: (y) => (+y).toPrecision(3), empty: "first check after 512 steps",
  });
  chart(el.querySelector("#c-s2"), {
    series: [{ color: css("--s2"), pts: trend((q.control || []).map((u) => [u.update, u.kl])), width: 1.5, dash: "4 3" },
             { color: css("--s1"), pts: trend(s2.map((u) => [u.update, u.kl])) }], yfmt: (y) => (+y).toFixed(2), empty: "no updates yet",
  });
};

pages.system = (st, el) => {
  const sys = st.system || {}, bk = sys.bookings || [], booked = bk.reduce((a, b) => a + b.bytes, 0), pis = st.pi || [];
  el.innerHTML = `
    <div class="ph"><h1>System</h1></div>
    <div class="grid cols-4">
      ${kpi("GPU memory in use", sys.gpu ? `${(sys.gpu.used / GiB).toFixed(1)}<small>/ ${(sys.gpu.total / GiB).toFixed(0)} GiB</small>` : "–", sys.gpu ? "" : "", sys.gpu ? prog(sys.gpu.used / sys.gpu.total) : "")}
      ${kpi("GPU busy", sys.gpu ? `${sys.gpu.util.toFixed(0)}<small>%</small>` : "–")}
      ${kpi("Booked", `${(booked / GiB).toFixed(0)}<small>/ 92 GiB</small>`, `${bk.length} booking${bk.length === 1 ? "" : "s"}`, prog(booked / (92 * GiB)))}
      ${kpi("Disk free", sys.disk ? `${(sys.disk.free / 1e9).toFixed(0)}<small>GB</small>` : "–", sys.disk ? `${((sys.disk.total - sys.disk.free) / 1e12).toFixed(2)} of ${(sys.disk.total / 1e12).toFixed(2)} TB used` : "", sys.disk ? prog(1 - sys.disk.free / sys.disk.total) : "")}
    </div>
    ${(sys.paused || []).length ? `<div class="banner" style="margin:18px 0 0">Paused ${esc(sys.paused.join(", "))}</div>` : ""}
    <div class="grid cols-2 section">
      <div class="card tw"><h2>GPU bookings</h2><table><thead><tr><th>#</th><th class="n">Booked</th><th class="n">Lapses in</th></tr></thead>
        <tbody>${bk.map((b, i) => `<tr><td>${i + 1}</td><td class="n">${(b.bytes / GiB).toFixed(0)} GiB</td><td class="n">${dur(b.lapses_in / 60)}</td></tr>`).join("") || '<tr><td colspan="3" class="empty">No bookings.</td></tr>'}</tbody></table></div>
      <div class="card tw"><h2>Raspberry Pi 5</h2><table><thead><tr><th>Contest</th><th class="n">Peak RAM</th><th class="n hide-s">Predicted</th><th class="n">Writing</th><th class="n hide-s">Reading</th></tr></thead>
        <tbody>${pis.map((p) => `<tr class="click" data-href="#/contests/${esc(p.contest)}"><td>${esc(p.contest)}</td><td class="n">${p.ram.toFixed(2)}<span class="muted"> / ${p.budget}</span></td><td class="n hide-s">${p.predicted.toFixed(2)}</td><td class="n">${p.decode.toFixed(2)} <span class="muted small">t/s</span></td><td class="n hide-s">${p.prefill.toFixed(1)} <span class="muted small">t/s</span></td></tr>`).join("") || '<tr><td colspan="5" class="empty">No measurements.</td></tr>'}</tbody></table></div>
    </div>`;
};

// The log page is a live tail, like `kubectl logs -f` or a CI job's log: the last 3,000 lines of every log, then each
// new line pushed by the server as it is written (server-sent events). It scrolls with the tail while you are at the
// bottom; scroll up and it holds still and counts what arrived.
const LOG = { lines: [], es: null, follow: true, wrap: true, src: "all", unseen: 0, live: false };
const LEVELS = [["all", "All", () => true], ["key", "Key", (k) => k !== "quiet"], ["note", "Notes", (k) => k === "note"], ["fail", "Errors", (k) => k === "fail"]];
const srcHue = (s) => (s === "main" ? 35 : [...s].reduce((a, c) => a * 31 + c.charCodeAt(0), 7) % 360);
function logMatch(o) {
  const q = ui.logq.trim().toLowerCase(), lv = (LEVELS.find((l) => l[0] === ui.logk) || LEVELS[0])[2];
  return lv(o.k) && (LOG.src === "all" || o.src === LOG.src) && (!q || o.line.toLowerCase().includes(q));
}
function logRow(o) {
  const q = ui.logq.trim(), t = /^\d\d:\d\d /.test(o.line), msg = t ? o.line.slice(6) : o.line;
  let m = esc(msg);
  if (q) m = m.replace(new RegExp(esc(q).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), (x) => `<mark>${x}</mark>`);
  return `<div class="ln ${o.k}"><span class="no">${o.n}</span><span class="tm">${t ? o.line.slice(0, 5) : ""}</span><span class="src" style="--h:${srcHue(o.src)}">${esc(o.src)}</span><span class="msg">${m}</span></div>`;
}
function logAdd(list) {
  for (const o of list) { o.k = kind(o.line); o.n = (LOG.lines.at(-1)?.n || 0) + 1; LOG.lines.push(o); }
  if (LOG.lines.length > 6000) { LOG.lines.splice(0, LOG.lines.length - 5000); return logDraw(); }
  const box = $("#logbox"); if (!box) return;
  const hits = list.filter(logMatch);
  if (!hits.length) return logMeta();
  box.querySelector(".empty")?.remove();
  box.insertAdjacentHTML("beforeend", hits.map(logRow).join(""));
  if (LOG.follow) box.scrollTop = box.scrollHeight; else LOG.unseen += hits.length;
  logMeta();
}
function logDraw() {
  const box = $("#logbox"); if (!box) return;
  const hits = LOG.lines.filter(logMatch);
  box.innerHTML = hits.map(logRow).join("") || `<div class="empty">${LOG.lines.length ? "No line matches." : "Loading…"}</div>`;
  box.classList.toggle("nowrap", !LOG.wrap);
  if (LOG.follow) box.scrollTop = box.scrollHeight;
  const srcs = ["all", ...new Set(LOG.lines.map((o) => o.src))], sel = $("#lsrc");
  if (sel.options.length !== srcs.length) sel.innerHTML = srcs.map((s) => `<option value="${esc(s)}">${s === "all" ? "All logs" : s === "main" ? "main log" : `${s} progress`}</option>`).join("");
  sel.value = LOG.src;
  logMeta(hits.length);
}
function logMeta(shown) {
  if (shown == null) shown = $("#logbox")?.querySelectorAll(".ln").length ?? 0;
  $("#llive").className = `live ${LOG.live ? "on" : ""}`;
  $("#llive").textContent = LOG.live ? "streaming" : "reconnecting…";
  $("#lcount").textContent = `${shown.toLocaleString()} of ${LOG.lines.length.toLocaleString()} lines`;
  $("#lfollow").classList.toggle("on", LOG.follow);
  $("#lnew").hidden = LOG.follow || !LOG.unseen;
  $("#lnew").textContent = `↓ ${LOG.unseen} new line${LOG.unseen === 1 ? "" : "s"}`;
}
function logClose() { if (LOG.es) { LOG.es.close(); LOG.es = null; LOG.live = false; } }
async function logOpen() {
  logClose();
  try { LOG.lines = []; logAdd(await (await fetch("api/logs?n=3000", { cache: "no-store" })).json()); logDraw(); } catch (e) {}
  const es = (LOG.es = new EventSource("api/logs/stream"));
  let dropped = false;
  es.onopen = () => { LOG.live = true; logMeta(); if (dropped) { dropped = false; logOpen(); } };
  es.onerror = () => { LOG.live = false; dropped = true; if ($("#llive")) logMeta(); };
  es.onmessage = (e) => logAdd([JSON.parse(e.data)]);
}

pages.logs = (st, el, _, fresh) => {
  if (!fresh && el.querySelector("#logbox")) return;  // the stream keeps it current; polls leave it alone
  el.innerHTML = `
    <div class="ph"><h1>Logs</h1><span class="live" id="llive"></span></div>
    <div class="logv">
      <div class="lbar">
        <input class="search" id="q" placeholder="Search lines" value="${esc(ui.logq)}" spellcheck="false">
        <div class="seg" id="lk">${LEVELS.map(([k, n]) => `<button data-k="${k}" class="${k === ui.logk ? "on" : ""}">${n}</button>`).join("")}</div>
        <select id="lsrc" class="sel"></select>
        <span class="grow"></span>
        <span class="muted small mono" id="lcount"></span>
        <button class="tbtn" id="lwrap" title="Wrap long lines">Wrap</button>
        <button class="tbtn" id="lfollow" title="Keep the newest line in view">Follow</button>
      </div>
      <div class="logbox" id="logbox"></div>
      <button class="lnew" id="lnew" hidden></button>
    </div>`;
  const box = $("#logbox");
  $("#q").oninput = (e) => { ui.logq = e.target.value; logDraw(); };
  $("#lk").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; ui.logk = b.dataset.k; $("#lk").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); logDraw(); };
  $("#lsrc").onchange = (e) => { LOG.src = e.target.value; logDraw(); };
  $("#lwrap").classList.toggle("on", LOG.wrap);
  $("#lwrap").onclick = () => { LOG.wrap = !LOG.wrap; $("#lwrap").classList.toggle("on", LOG.wrap); box.classList.toggle("nowrap", !LOG.wrap); };
  const bottom = () => { LOG.follow = true; LOG.unseen = 0; box.scrollTop = box.scrollHeight; logMeta(); };
  $("#lfollow").onclick = () => { if (LOG.follow) { LOG.follow = false; logMeta(); } else bottom(); };
  $("#lnew").onclick = bottom;
  box.onscroll = () => {
    const end = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
    if (end !== LOG.follow) { LOG.follow = end; if (end) LOG.unseen = 0; logMeta(); }
  };
  if (LOG.lines.length && LOG.es) logDraw(); else { logDraw(); logOpen(); }
};

// --- router and shell ---------------------------------------------------------------------------------------------------
function resolve() {
  const p = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  if (p[0] === "contests" && p[1]) return { page: "contest", nav: "contests", args: p.slice(1), crumbs: [["Contests", "#/contests"], [p[1].toUpperCase().replace("@", " @ ") + " GiB"]] };
  if (p[0] === "experiments" && p[1]) return { page: "qat", nav: "experiments", args: p.slice(2), crumbs: [["Experiments", "#/experiments"], [QAT_NAME]] };
  const known = { contests: "Contests", experiments: "Experiments", system: "System", logs: "Logs" };
  if (known[p[0]]) return { page: p[0], nav: p[0], args: [], crumbs: [[known[p[0]]]] };
  return { page: "overview", nav: "overview", args: [], crumbs: [["Overview"]] };
}
function draw(fresh = false) {
  const st = ui.state, r = resolve();
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === r.nav));
  $("#crumbs").innerHTML = [["Panther", "#/"], ...r.crumbs].map(([n, h], i, a) => (i < a.length - 1 ? `<a href="${h}">${esc(n)}</a><span class="sep">/</span>` : `<b>${esc(n)}</b>`)).join("");
  document.title = `${r.crumbs.at(-1)[0]} · Panther`;
  if (!st) { if (r.page === "logs") pages.logs(null, $("#page"), [], fresh); else $("#page").innerHTML = '<div class="empty">Loading…</div>'; return; }
  const js = jobs(st), sys = st.system || {}, pz = sys.paused || [];
  $("#running").hidden = !js.length;
  $("#running").textContent = js.length === 1 ? `${js[0].step}${js[0].eta ? ` · ~${js[0].eta}` : ""}` : `${js.length} running`;
  $("#cnt-exp").textContent = js.some((j) => j.kind === "experiment") ? "●" : "";
  $("#cnt-con").textContent = js.filter((j) => j.kind === "contest").length || "";
  const ban = $("#banner"), low = sys.disk && sys.disk.free < 20e9;
  ban.hidden = !pz.length && !st.error && !low;
  ban.textContent = st.error ? `The server could not read the data: ${st.error}` : pz.length ? `Paused: ${pz.join(", ")}` : low ? `Disk almost full: ${(sys.disk.free / 1e9).toFixed(0)} GB free` : "";
  if (r.page !== "logs") logClose();
  if (st.error && !st.contests) return;
  pages[r.page](st, $("#page"), r.args, fresh);
}
document.addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-href]");
  if (tr && !e.target.closest("a")) location.hash = tr.dataset.href;
});
addEventListener("hashchange", () => { $("#side").classList.remove("open"); $("#scrim").classList.remove("open"); draw(true); scrollTo(0, 0); });
$("#menu").onclick = () => { $("#side").classList.add("open"); $("#scrim").classList.add("open"); };
$("#scrim").onclick = () => { $("#side").classList.remove("open"); $("#scrim").classList.remove("open"); };
$("#theme").onclick = () => {
  const next = dark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("panther-theme", next); } catch (e) {}
  draw();
};

async function poll() {
  try {
    const st = await (await fetch("api/state", { cache: "no-store" })).json();
    const first = !ui.state;
    ui.state = st; ui.last = Date.now();
    $("#dot").className = st.error ? "dot err" : "dot";
    draw(first);
  } catch (e) { $("#dot").className = "dot err"; }
}
setInterval(() => {
  const s = Math.round((Date.now() - ui.last) / 1000);
  $("#age").textContent = ui.last ? (s < 12 ? "live" : `${s}s ago`) : "…";
  if (ui.last && s > 35) $("#dot").className = "dot stale";
}, 1000);
let rs; addEventListener("resize", () => { clearTimeout(rs); rs = setTimeout(() => ui.state && draw(), 150); });
draw(); poll(); setInterval(poll, 10000);
