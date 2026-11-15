// Panther dashboard: polls /api/state every 10 s. Three questions, in order: what is running and when does it end;
// how is the experiment going; how did the contests do. Plain DOM and SVG, no libraries.
"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const GiB = 2 ** 30;
const sgn = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const pct = (a, b) => sgn(((a - b) / b) * 100, 0) + "%";
const BPW = { IQ1_S: 1.56, IQ1_M: 1.75, IQ2_XXS: 2.06, IQ2_XS: 2.31, IQ2_S: 2.5, Q2_K: 2.63, IQ3_XXS: 3.06, Q3_K: 3.44, IQ3_S: 3.44,
  IQ4_XS: 4.25, Q4_K: 4.5, Q5_K: 5.5, Q6_K: 6.56, Q8_0: 8.5 };
const TASKS = [["mmlu", "MMLU"], ["hellaswag", "HellaSwag"], ["arc_challenge", "ARC"], ["winogrande", "WinoGrande"], ["gsm8k", "GSM8K"]];
const ui = { open: new Set(), block: null, all: false, last: 0, state: null };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const dark = () => document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
const typeColor = (t) => { const b = Math.min(1, Math.max(0, ((BPW[t] ?? 4) - 1.5) / 7)); return `hsl(${200 - 175 * b} ${42 + 18 * b}% ${dark() ? 58 : 44}%)`; };

$("#theme").onclick = () => {
  const next = dark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("panther-theme", next); } catch (e) {}
  if (ui.state) render(ui.state);
};
$("#more").onclick = () => { ui.all = !ui.all; if (ui.state) feed(ui.state); };

// --- one line chart -------------------------------------------------------------------------------------------------
function nice(lo, hi, n = 4) {
  const span = hi - lo || Math.abs(hi) || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}
function chart(el, { series, refs = [], marks = [], xfmt = (x) => x, yfmt = (y) => y }) {
  const w = el.clientWidth || 600, h = el.clientHeight || 150, L = 44, R = 8, T = 8, B = 20;
  const pts = series.flatMap((s) => s.pts);
  if (new Set(pts.map((p) => p[0])).size < 2) { el.innerHTML = `<div class="empty">first checks in a few minutes</div>`; return; }
  const ys = pts.map((p) => p[1]).concat(refs.map((r) => r.y)), xs = pts.map((p) => p[0]);
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0) * 0.15 || Math.abs(y1) * 0.1 || 1; y0 -= pad; y1 += pad;
  const x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (w - L - R), Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (h - T - B);
  let s = `<svg viewBox="0 0 ${w} ${h}"><g class="ticks">`;
  for (const t of nice(y0, y1, 3)) s += `<line x1="${L}" x2="${w - R}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${L - 6}" y="${Y(t) + 3.5}" text-anchor="end">${esc(yfmt(t))}</text>`;
  for (const t of nice(x0, x1, 4)) s += `<text x="${X(t)}" y="${h - 4}" text-anchor="middle">${esc(xfmt(t))}</text>`;
  s += `</g>`;
  for (const r of refs) s += `<line x1="${L}" x2="${w - R}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="${r.color}" stroke-dasharray="4 4"/><text x="${w - R}" y="${Y(r.y) - 4}" text-anchor="end" font-size="10.5" fill="${r.color}">${esc(r.label)}</text>`;
  for (const se of series) {
    if (!se.pts.length) continue;
    const d = se.pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} stroke-linejoin="round"/>`;
  }
  for (const m of marks) s += `<circle cx="${X(m.x)}" cy="${Y(m.y)}" r="4.5" fill="${css("--panel")}" stroke="${m.color}" stroke-width="2"/>`;
  el.innerHTML = s + `</svg>`;
}
const trend = (pts, k = 2) => pts.map((p, i) => { const w = pts.slice(Math.max(0, i - k), i + k + 1); return [p[0], w.reduce((a, q) => a + q[1], 0) / w.length]; });

// --- the headline: what is running, when it ends --------------------------------------------------------------------
function headline(st) {
  const q = st.qat, sys = st.system || {};
  let h = "Idle: every contest is finished", sh = "Idle";
  const run = (st.contests || []).find((c) => c.state === "running");
  if (q) {
    const done = q.blocks.filter((b) => b.after != null).length, s2 = q.stage2 || [], res = q.results || {};
    if (done < q.total && q.blocks.length) { h = `QAT stage 1 · block ${done} of ${q.total}${q.stage1_eta ? ` · done ~${q.stage1_eta}` : ""}`; sh = `Stage 1 · ${done}/${q.total}${q.stage1_eta ? ` · ~${q.stage1_eta}` : ""}`; }
    else if (s2.length && !res["stage 1 + 2"]) { h = `QAT stage 2 · update ${s2.at(-1).update} of ${s2.at(-1).of}`; sh = `Stage 2 · ${s2.at(-1).update}/${s2.at(-1).of}`; }
    else if (!res["stage 1 + 2"]) h = sh = "QAT · held-out tests";
  } else if (run) h = sh = `${run.id} · ${run.steps.find((x) => x.state === "running")?.name ?? "running"} · ETA ${run.eta}`;
  $("#headline").innerHTML = `<span class="dot"></span><span class="long">${esc(h)}</span><span class="short">${esc(sh)}</span>`;
  const booked = (sys.bookings || []).reduce((a, b) => a + b.bytes, 0);
  $("#sys").textContent = [sys.gpu ? `GPU ${(sys.gpu.used / GiB).toFixed(0)}/${(sys.gpu.total / GiB).toFixed(0)} GiB · ${sys.gpu.util.toFixed(0)}%` : `GPU booked ${(booked / GiB).toFixed(0)}/92 GiB`,
    sys.disk ? `disk ${(sys.disk.free / 1e9).toFixed(0)} GB free` : ""].filter(Boolean).join(" · ");
  const ban = $("#banner"), pz = sys.paused || [];
  ban.hidden = !pz.length && !st.error && !(sys.disk && sys.disk.free < 20e9);
  ban.textContent = st.error ? `The server could not read the data: ${st.error}` : pz.length ? `Paused: ${pz.join(", ")}` : sys.disk && sys.disk.free < 20e9 ? `Disk almost full: ${(sys.disk.free / 1e9).toFixed(0)} GB free` : "";
}

// --- the experiment ----------------------------------------------------------------------------------------------------
function experiment(q) {
  const el = $("#exp");
  if (!q) { el.hidden = true; return; }
  el.hidden = false;
  const done = q.blocks.filter((b) => b.after != null), cur = q.blocks.find((b) => b.after == null);
  const s2 = q.stage2 || [], res = q.results || {}, main = res["stage 1 + 2"], ctl = res["stage 2 only"];
  if (!ui.picked || !q.blocks.some((b) => b.block === ui.block)) ui.block = (cur && cur.checks.length > 1 ? cur : done.at(-1) || cur || {}).block ?? null;
  const mean = done.length ? done.reduce((a, b) => a + (b.after - b.before) / b.before, 0) / done.length : 0;
  const cells = Array.from({ length: q.total }, (_, i) => {
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
  }).join("");
  const verdict = (r) => r ? `<span class="v ${r.vs_winner.verdict}">${r.vs_winner.verdict}</span> <span class="num">${sgn(r.vs_winner.diff)}</span> <span class="muted">vs winner</span>` : "";
  const stat = (k, v, sub) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`;
  const b = q.blocks.find((x) => x.block === ui.block);
  el.innerHTML = `
    <h2>EfficientQAT on 14B @ 5 GiB <span class="muted">${esc((q.strategy || "").split(",")[0])}</span></h2>
    <div class="stats">
      ${stat("Stage 1 · blocks", `${done.length}<small>/${q.total}</small>`, `mean ${sgn(mean * 100, 0)}% error${q.stage1_eta && done.length < q.total ? ` · done ~${q.stage1_eta}` : ""}`)}
      ${stat("Stage 2 · scales", s2.length ? `${s2.at(-1).update}<small>/${s2.at(-1).of}</small>` : `<span class="muted">—</span>`, s2.length ? "updates" : "after stage 1")}
      ${stat("Result", main ? verdict(main) : `<span class="muted">pending</span>`, ctl ? `control (stage 2 only): ${verdict(ctl)}` : "")}
    </div>
    <div class="strip" id="strip">${cells}</div>
    <div class="ch"><span>Block ${ui.block ?? "–"}</span><span class="muted">${b ? (b.after != null ? `${b.before.toFixed(4)} → ${b.after.toFixed(4)} (${pct(b.after, b.before)})${b.stopped ? ", stopped early" : ""}` : "training") : ""} · validation error by step</span></div>
    <div class="chart" id="c-block"></div>
    ${s2.length ? `<div class="ch"><span>Stage 2</span><span class="muted">KL to the original, trend</span></div><div class="chart" id="c-s2"></div>` : ""}
    ${Object.keys(res).length ? `<table class="res"><thead><tr><th>file</th><th>held-out</th><th>vs current winner</th><th>vs preset</th></tr></thead><tbody>${Object.entries(res).map(([n, r]) =>
      `<tr><td>${esc(n)}</td><td class="num">${r.held_out.accuracy.toFixed(2)}</td><td class="num">${sgn(r.vs_winner.diff)} <span class="muted">[${sgn(r.vs_winner.ci[0])}, ${sgn(r.vs_winner.ci[1])}]</span> <span class="v ${r.vs_winner.verdict}">${r.vs_winner.verdict}</span></td><td class="num">${sgn(r.vs_preset.diff)} <span class="v ${r.vs_preset.verdict}">${r.vs_preset.verdict}</span></td></tr>`).join("")}</tbody></table>` : ""}
    <details><summary>Details</summary>
      ${q.now ? `<p>${esc(q.now)}</p>` : ""}
      ${q.tests.length ? `<p class="muted">How Stage 1's settings were chosen (block 0):</p><ul class="plain">${q.tests.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : ""}
    </details>`;
  el.querySelector("#strip").onclick = (e) => { const c = e.target.closest(".cell"); if (c) { ui.block = +c.dataset.b; ui.picked = true; experiment(q); } };
  chart($("#c-block"), {
    series: b ? [{ name: "validation", color: css("--accent"), pts: b.checks.filter((c) => c.val != null).map((c) => [c.step, c.val]) }] : [],
    refs: b && b.before != null ? [{ y: b.before, label: "before", color: css("--muted") }] : [],
    marks: b && b.best && b.checks.find((c) => c.step === b.best) ? [{ x: b.best, y: b.checks.find((c) => c.step === b.best).val, color: css("--win") }] : [],
    xfmt: (x) => x, yfmt: (y) => (+y).toPrecision(2),
  });
  if (s2.length) chart($("#c-s2"), {
    series: [{ color: css("--s2"), pts: trend((q.control || []).map((u) => [u.update, u.kl])), width: 1.5, dash: "4 3" },
             { color: css("--s1"), pts: trend(s2.map((u) => [u.update, u.kl])) }], yfmt: (y) => (+y).toFixed(2) });
}

// --- contests ----------------------------------------------------------------------------------------------------------
function contests(st) {
  const cs = st.contests || [], pis = Object.fromEntries((st.pi || []).map((p) => [p.contest, p]));
  const ci = cs.filter((c) => c.verdict).flatMap((c) => c.verdict.ci);
  const lo = Math.min(-1, ...ci) - 0.5, hi = Math.max(1, ...ci) + 0.5, X = (v) => (((v - lo) / (hi - lo)) * 100).toFixed(2) + "%";
  let s = `<table class="ct"><thead><tr><th>contest</th><th></th><th class="fp-h">${nice(lo, hi, 5).map((t) => `<span style="left:${X(t)}">${t > 0 ? "+" : ""}${t}</span>`).join("")}</th><th>points</th><th>ours / preset / original</th><th>Pi RAM</th></tr></thead><tbody>`;
  for (const c of cs) {
    const v = c.verdict, p = pis[c.id];
    if (!v) {
      s += `<tr class="dim"><td>${esc(c.size)} @ ${c.budget}</td><td colspan="5"><span class="v ${c.state === "running" ? "run" : "drop"}">${esc(c.state)}</span> ${c.state === "running" && c.eta ? `<span class="muted">ETA ${esc(c.eta)}</span>` : ""}</td></tr>`;
      continue;
    }
    const h = c.held_out, col = v.verdict === "WIN" ? "var(--win)" : v.verdict === "LOSE" ? "var(--lose)" : "var(--tie)";
    s += `<tr class="row${ui.open.has(c.id) ? " open" : ""}" data-id="${esc(c.id)}" title="${esc(c.model || "")}">
      <td><b>${esc(c.size)}</b> @ ${c.budget} GiB</td><td><span class="v ${v.verdict}">${v.verdict}</span></td>
      <td class="fpc"><div class="fp" style="--c:${col}"><i class="z" style="left:${X(0)}"></i><i class="ln" style="left:${X(v.ci[0])};right:calc(100% - ${X(v.ci[1])})"></i><i class="pt" style="left:${X(v.diff)}"></i></div></td>
      <td class="num"><b>${sgn(v.diff)}</b> <span class="muted">[${sgn(v.ci[0], 1)}, ${sgn(v.ci[1], 1)}]</span></td>
      <td class="num">${h.ours.accuracy.toFixed(1)} <span class="muted">/ ${h.preset_test.accuracy.toFixed(1)} / ${h.original.accuracy.toFixed(1)}</span></td>
      <td class="num">${p ? `${p.ram.toFixed(2)}<span class="muted">/${c.budget}</span>` : '<span class="muted">—</span>'}</td></tr>`;
    if (ui.open.has(c.id)) s += `<tr class="det"><td colspan="6">${detail(c, p)}</td></tr>`;
  }
  $("#contests").innerHTML = s + "</tbody></table>";
  $("#contests").onclick = (e) => {
    const r = e.target.closest("tr.row"); if (!r) return;
    const id = r.dataset.id; ui.open.has(id) ? ui.open.delete(id) : ui.open.add(id);
    contests(ui.state);
  };
}
function detail(c, p) {
  const h = c.held_out, total = c.types.reduce((a, t) => a + t[1], 0);
  const rows = [["Original", h.original], ["Ours", h.ours], [`Preset ${c.verdict.preset}`, h.preset_test]];
  return `<div class="detail">
    <table><thead><tr><th></th><th>accuracy</th>${TASKS.map(([, n]) => `<th>${n}</th>`).join("")}<th>KL</th><th>flips %</th></tr></thead><tbody>
      ${rows.map(([n, r]) => `<tr${n === "Ours" ? ' class="ours"' : ""}><td>${esc(n)}</td><td class="num">${r.accuracy.toFixed(2)}</td>${TASKS.map(([k]) => `<td class="num">${r.per_task[k]?.toFixed(1) ?? "–"}</td>`).join("")}<td class="num">${r.kl != null ? r.kl.toFixed(3) : "–"}</td><td class="num">${r.flips != null && n !== "Original" ? r.flips.toFixed(1) : "–"}</td></tr>`).join("")}
    </tbody></table>
    <div class="mix" title="our mix of types">${c.types.map(([t, n]) => `<i title="${t} ×${n}" style="width:${(n / total) * 100}%;background:${typeColor(t)}"></i>`).join("")}</div>
    <div class="muted small">${esc(c.model || "")} · ${c.types.map(([t, n]) => `${t} ×${n}`).join(", ")}${p ? ` · on the Pi: ${p.ram.toFixed(2)} GiB peak, ${p.decode.toFixed(2)} tokens/s writing` : ""}</div>
  </div>`;
}

// --- the log ------------------------------------------------------------------------------------------------------------
function kind(x) {
  if (/ note: /.test(x)) return "note";
  if (/FAILED/.test(x)) return "fail";
  if (/held-out test vs|QAT .*held-out|: done$/.test(x)) return "res";
  return "quiet";
}
function feed(st) {
  const items = (st.events || []).map((x) => ({ x, k: kind(x) })).filter(({ k }) => ui.all || k !== "quiet");
  $("#feed").innerHTML = items.slice(0, ui.all ? 80 : 6).map(({ x, k }) => `<li class="${k}"><time>${esc(x.slice(0, 5))}</time><span>${esc(x.slice(6).replace(/^note: /, ""))}</span></li>`).join("") || '<li class="quiet"><span>nothing yet</span></li>';
  $("#more").textContent = ui.all ? "key only" : "show all";
}

// --- the loop -------------------------------------------------------------------------------------------------------------
function render(st) { headline(st); experiment(st.qat); contests(st); feed(st); }
async function poll() {
  try {
    const st = await (await fetch("api/state", { cache: "no-store" })).json();
    ui.state = st; ui.last = Date.now();
    $("#dot").className = st.error ? "dot err" : "dot";
    render(st);
  } catch (e) { $("#dot").className = "dot err"; }
}
setInterval(() => {
  const s = Math.round((Date.now() - ui.last) / 1000);
  $("#age").textContent = ui.last ? (s < 12 ? "live" : `${s}s ago`) : "…";
  if (ui.last && s > 35) $("#dot").className = "dot stale";
}, 1000);
let rs; addEventListener("resize", () => { clearTimeout(rs); rs = setTimeout(() => ui.state && render(ui.state), 150); });
poll(); setInterval(poll, 10000);
