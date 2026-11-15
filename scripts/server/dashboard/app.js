// Panther dashboard: polls /api/state every 10 s and draws everything with plain DOM and SVG (no libraries).
"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const GiB = 2 ** 30;
const sgn = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const BPW = { IQ1_S: 1.56, IQ1_M: 1.75, IQ2_XXS: 2.06, IQ2_XS: 2.31, IQ2_S: 2.5, Q2_K: 2.63, IQ3_XXS: 3.06, Q3_K: 3.44, IQ3_S: 3.44,
  IQ4_XS: 4.25, Q4_K: 4.5, Q5_K: 5.5, Q6_K: 6.56, Q8_0: 8.5, Q8: 8.5 };
const TASKS = [["mmlu", "MMLU"], ["hellaswag", "HellaSw."], ["arc_challenge", "ARC"], ["winogrande", "WinoGr."], ["gsm8k", "GSM8K"]];
const ui = { open: new Set(), block: null, filter: "key", last: 0, state: null };
try { ui.filter = localStorage.getItem("panther-filter") || "key"; } catch (e) {}

// --- theme ------------------------------------------------------------------------------------------------------
$("#theme").onclick = () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("panther-theme", next); } catch (e) {}
  if (ui.state) render(ui.state);
};
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const typeColor = (t) => {
  const b = Math.min(1, Math.max(0, ((BPW[t] ?? 4) - 1.5) / 7));
  return `hsl(${200 - 175 * b} ${42 + 18 * b}% ${document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches) ? 58 : 44}%)`;
};

// --- a line chart in SVG ------------------------------------------------------------------------------------------
function nice(lo, hi, n = 4) {
  const span = hi - lo || Math.abs(hi) || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}
function chart(el, { series, refs = [], marks = [], xfmt = (x) => x, yfmt = (y) => y, empty = "no data yet" }) {
  const w = el.clientWidth || 400, h = el.clientHeight || 190, L = 46, R = 10, T = 10, B = 24;
  const pts = series.flatMap((s) => s.pts);
  if (!pts.length) { el.innerHTML = `<div class="muted" style="padding:70px 0;text-align:center;font-size:12.5px">${esc(empty)}</div>`; return; }
  let ys = pts.map((p) => p[1]).concat(refs.map((r) => r.y)), xs = pts.map((p) => p[0]);
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0) * 0.12 || Math.abs(y1) * 0.1 || 1; y0 -= pad; y1 += pad;
  const x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (w - L - R), Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (h - T - B);
  let s = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><g class="ticks">`;
  for (const t of nice(y0, y1)) s += `<line x1="${L}" x2="${w - R}" y1="${Y(t)}" y2="${Y(t)}" stroke-dasharray="2 4"/><text x="${L - 7}" y="${Y(t) + 3.5}" text-anchor="end">${esc(yfmt(t))}</text>`;
  for (const t of nice(x0, x1, 5)) s += `<text x="${X(t)}" y="${h - 6}" text-anchor="middle">${esc(xfmt(t))}</text>`;
  s += `</g>`;
  for (const r of refs) s += `<line x1="${L}" x2="${w - R}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="${r.color}" stroke-dasharray="5 4" stroke-width="1.2"/><text x="${w - R}" y="${Y(r.y) - 5}" text-anchor="end" font-size="10.5" fill="${r.color}">${esc(r.label)}</text>`;
  for (const se of series) {
    if (!se.pts.length) continue;
    const d = se.pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round"/>`;
    if (se.dots !== false) for (const p of se.pts) s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.3" fill="${se.color}"/>`;
  }
  for (const m of marks) s += `<circle cx="${X(m.x)}" cy="${Y(m.y)}" r="5.5" fill="none" stroke="${m.color}" stroke-width="2"/>`;
  s += `<line class="xh" y1="${T}" y2="${h - B}" stroke="${css("--muted")}" stroke-width="1" opacity="0"/></svg><div class="tip"></div>`;
  el.innerHTML = s;
  const svg = el.querySelector("svg"), tip = el.querySelector(".tip"), xh = el.querySelector(".xh");
  svg.onmousemove = (e) => {
    const r = svg.getBoundingClientRect(), mx = ((e.clientX - r.left) / r.width) * w;
    let best = null;
    for (const se of series) for (const p of se.pts) { const dx = Math.abs(X(p[0]) - mx); if (!best || dx < best.dx) best = { dx, p, se }; }
    if (!best) return;
    const at = best.p[0], lines = series.map((se) => { const q = se.pts.find((p) => p[0] === at); return q ? `${se.name} ${yfmt(q[1])}` : null; }).filter(Boolean);
    tip.textContent = `${xfmt(at)} · ${lines.join(" · ")}`;
    tip.style.left = `${(X(at) / w) * 100}%`; tip.style.top = `${(Y(best.p[1]) / h) * 100}%`; tip.style.opacity = 1;
    xh.setAttribute("x1", X(at)); xh.setAttribute("x2", X(at)); xh.setAttribute("opacity", .5);
  };
  svg.onmouseleave = () => { tip.style.opacity = 0; xh.setAttribute("opacity", 0); };
}

const trend = (pts, k = 2) => pts.map((p, i) => { const w = pts.slice(Math.max(0, i - k), i + k + 1); return [p[0], w.reduce((a, q) => a + q[1], 0) / w.length]; });

// --- instruments --------------------------------------------------------------------------------------------------
function gauges(st) {
  const sys = st.system || {}, q = st.qat, booked = (sys.bookings || []).reduce((a, b) => a + b.bytes, 0);
  const g = [];
  if (sys.gpu) {
    const u = sys.gpu.used / sys.gpu.total;
    g.push(`<div class="card gauge"><div class="k">GPU memory</div><div class="v">${(sys.gpu.used / GiB).toFixed(0)}<small>of ${(sys.gpu.total / GiB).toFixed(0)} GiB</small></div><div class="bar"><i style="width:${u * 100}%"></i></div><div class="foot">booked ${(booked / GiB).toFixed(0)} of 92 GiB · ${(sys.bookings || []).length} jobs</div></div>`);
    g.push(`<div class="card gauge"><div class="k">GPU busy</div><div class="v">${sys.gpu.util.toFixed(0)}<small>%</small></div><div class="bar"><i style="width:${sys.gpu.util}%;background:var(--run)"></i></div><div class="foot">right now</div></div>`);
  } else {
    g.push(`<div class="card gauge"><div class="k">GPU bookings</div><div class="v">${(booked / GiB).toFixed(0)}<small>of 92 GiB</small></div><div class="bar">${(sys.bookings || []).map((b, i, a) => `<i class="b" style="left:${(a.slice(0, i).reduce((s, x) => s + x.bytes, 0) / (92 * GiB)) * 100}%;width:${(b.bytes / (92 * GiB)) * 100}%"></i>`).join("")}</div><div class="foot">${(sys.bookings || []).length} jobs in the ledger</div></div>`);
    g.push(`<div class="card gauge"><div class="k">GPU busy</div><div class="v muted">—</div><div class="bar"></div><div class="foot">no nvidia-smi on this machine</div></div>`);
  }
  if (sys.disk) {
    const used = 1 - sys.disk.free / sys.disk.total;
    g.push(`<div class="card gauge"><div class="k">Disk free</div><div class="v">${(sys.disk.free / 1e9).toFixed(0)}<small>of ${(sys.disk.total / 1e9).toFixed(0)} GB</small></div><div class="bar"><i style="width:${used * 100}%;background:${used > 0.9 ? "var(--lose)" : "var(--accent)"}"></i></div><div class="foot">${(used * 100).toFixed(0)} % used</div></div>`);
  }
  if (q) {
    const done = q.blocks.filter((b) => b.after != null).length;
    g.push(`<div class="card gauge"><div class="k">QAT stage 1</div><div class="v">${done}<small>of ${q.total} blocks</small></div><div class="bar"><i style="width:${(done / q.total) * 100}%;background:var(--win)"></i></div><div class="foot">${q.minutes_per_block ? `${q.minutes_per_block.toFixed(0)} min a block · ` : ""}${q.stage1_eta && done < q.total ? `done ~${q.stage1_eta} IST` : done >= q.total ? "done" : ""}</div></div>`);
  }
  $("#gauges").innerHTML = g.join("");
  if (!ui.shown) [...$("#gauges").children].forEach((c, i) => { c.classList.add("reveal"); c.style.animationDelay = `${i * 0.04}s`; });
  ui.shown = true;
}

// --- the experiment -----------------------------------------------------------------------------------------------
function experiment(q) {
  const el = $("#exp");
  if (!q) { el.hidden = true; return; }
  el.hidden = false;
  const done = q.blocks.filter((b) => b.after != null), cur = q.blocks.find((b) => b.after == null);
  const s2 = q.stage2 || [], res = q.results || {};
  const st1 = done.length >= q.total || s2.length ? "done" : q.blocks.length ? "running" : "waiting";
  const st2 = res["stage 1 + 2"] ? "done" : s2.length ? (s2.at(-1).update >= s2.at(-1).of ? "done" : "running") : "waiting";
  const st3 = res["stage 1 + 2"] ? "done" : st2 === "done" ? "running" : "waiting";
  const frac = (s) => (s === "done" ? 100 : s === "waiting" ? 0 : null);
  if (ui.block == null || !q.blocks.some((b) => b.block === ui.block)) ui.block = (cur && cur.checks.length ? cur : done.at(-1) || cur || {}).block ?? null;
  const cells = [];
  for (let i = 0; i < q.total; i++) {
    const b = q.blocks.find((x) => x.block === i);
    let style = "", cls = "cell", title = `block ${i}: waiting`;
    if (b && b.after != null) {
      const p = (b.after - b.before) / b.before;
      title = `block ${i}: ${b.before.toFixed(4)} → ${b.after.toFixed(4)} (${sgn(p * 100, 0)}%)${b.stopped ? `, stopped early at ${b.stopped}` : ""}, best at step ${b.best}`;
      if (b.best === 0) cls += " kept";
      else style = `background:color-mix(in srgb, ${p < 0 ? "var(--win)" : "var(--lose)"} ${Math.round(18 + 70 * Math.min(1, Math.abs(p) / 0.45))}%, var(--sunk))`;
    } else if (b) { cls += " now"; title = `block ${i}: training, step ${b.checks.at(-1)?.step ?? 0} of ${b.of ?? 4096}`; }
    if (i === ui.block) cls += " sel";
    cells.push(`<div class="${cls}" style="${style}" title="${esc(title)}" data-b="${i}">${i}</div>`);
  }
  const avg = done.length ? done.reduce((a, b) => a + (b.after - b.before) / b.before, 0) / done.length : null;
  const rows = Object.entries(res).map(([name, r]) => {
    const h = r.held_out, w = r.vs_winner, p = r.vs_preset;
    return `<tr${name === "stage 1 + 2" ? ' class="ours"' : ""}><td>${esc(name)}</td><td>${h.accuracy.toFixed(2)}</td>
      <td>${sgn(w.diff)} <span class="muted">[${sgn(w.ci[0])}, ${sgn(w.ci[1])}]</span> <span class="v ${w.verdict}">${w.verdict}</span></td>
      <td>${sgn(p.diff)} <span class="muted">[${sgn(p.ci[0])}, ${sgn(p.ci[1])}]</span> <span class="v ${p.verdict}">${p.verdict}</span></td>
      <td>${h.kl.toFixed(3)}</td><td>${h.ppl.toFixed(2)}</td><td>${h.flips.toFixed(1)}</td>${TASKS.map(([k]) => `<td>${h.per_task[k].toFixed(1)}</td>`).join("")}</tr>`;
  }).join("");
  const ref = Object.values(res)[0];
  el.innerHTML = `
    <header><h2>Experiment</h2><h3>EfficientQAT on 14B @ 5 GiB</h3>${q.strategy ? `<span class="chip">${esc(q.strategy.split(",")[0])}</span>` : ""}
      <span class="sub" style="flex-basis:100%">Training the codes of our winner's mix of types, block by block (Stage 1), then its scales end to end (Stage 2); the control runs Stage 2 alone.</span></header>
    ${q.now ? `<div class="now">${esc(q.now)}</div>` : ""}
    <div class="track">
      ${[["Stage 1 · Block-AP", st1, `${done.length} of ${q.total} blocks${avg != null ? ` · mean ${sgn(avg * 100, 0)}%` : ""}`],
         ["Stage 2 · scales, KL", st2, s2.length ? `update ${s2.at(-1).update} of ${s2.at(-1).of}` : "after Stage 1"],
         ["Held-out tests", st3, res["stage 1 + 2"] ? "done" : "vs the winner and the preset"]].map(([t, s, d], i) => `
        <div class="stage ${s}"><span class="node"></span><div class="line"><i style="width:${frac(s) ?? (i === 0 ? (done.length / q.total) * 100 : s2.length ? (s2.at(-1).update / s2.at(-1).of) * 100 : 50)}%"></i></div>
        <div class="t">${t}</div><div class="d">${esc(d)}</div></div>`).join("")}
    </div>
    <div class="body">
      <div class="ph" style="display:flex;justify-content:space-between;font-size:12.5px;font-weight:650;margin-top:6px">Every block's validation error, after vs before training <span class="muted" style="font-weight:500;font-size:11.5px">click a block for its curve</span></div>
      <div class="strip" id="strip">${cells.join("")}</div>
      <div class="legend"><span><i class="sw" style="background:color-mix(in srgb,var(--win) 80%,var(--sunk))"></i>better (darker: more)</span><span><i class="sw" style="background-image:repeating-linear-gradient(135deg,var(--tie-soft) 0 3px,transparent 3px 6px)"></i>kept as it was</span><span><i class="sw" style="border-color:var(--run)"></i>training now</span><span><i class="sw"></i>waiting</span></div>
      <div class="grid2">
        <div class="panel"><div class="ph"><span style="color:var(--ink);font-weight:650;font-size:12.5px">Block <span class="mono" id="bnum"></span></span><span id="binfo"></span></div><div class="chart" id="c-block"></div></div>
        <div class="panel"><div class="ph">Stage 2 · KL to the original (top 64) <span>per update of 32 samples</span></div><div class="chart" id="c-s2"></div></div>
      </div>
      ${rows ? `<div class="ph" style="margin:20px 0 6px;font-size:12.5px;font-weight:650">Held-out results</div>
      <div class="scroll"><table><thead><tr><th>file</th><th>accuracy</th><th>vs current winner</th><th>vs preset</th><th>KL</th><th>perplexity</th><th>flips %</th>${TASKS.map(([, n]) => `<th>${n}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>
      <div class="muted" style="font-size:11.5px;margin-top:6px">Current winner ${ref.vs_winner.winner_accuracy.toFixed(2)} · preset ${esc(ref.vs_preset.preset)} ${ref.vs_preset.preset_accuracy.toFixed(2)} · original ${ref.original_accuracy.toFixed(2)}</div>` : ""}
      ${q.tests.length ? `<div class="ph" style="margin:20px 0 4px;font-size:12.5px;font-weight:650">Block-0 tests (how Stage 1's settings were chosen)</div><ul class="tests">${q.tests.map((t) => `<li class="${/worse/.test(t) ? "bad" : /-\d+%/.test(t) ? "good" : ""}">${esc(t)}</li>`).join("")}</ul>` : ""}
    </div>`;
  el.querySelector("#strip").onclick = (e) => { const c = e.target.closest(".cell"); if (c) { ui.block = +c.dataset.b; experiment(q); } };
  const b = q.blocks.find((x) => x.block === ui.block);
  $("#bnum").textContent = ui.block ?? "–";
  $("#binfo").textContent = b ? (b.after != null ? `${b.before.toFixed(4)} → ${b.after.toFixed(4)}${b.stopped ? ` · stopped at ${b.stopped}` : ""}` : "training") : "";
  chart($("#c-block"), {
    series: b ? [{ name: "validation", color: css("--accent"), pts: b.checks.filter((c) => c.val != null).map((c) => [c.step, c.val]) },
                 { name: "train", color: css("--muted"), dash: "4 4", width: 1.5, pts: b.checks.map((c) => [c.step, c.train]) }] : [],
    refs: b && b.before != null ? [{ y: b.before, label: "before", color: css("--tie") }] : [],
    marks: b && b.best && b.checks.find((c) => c.step === b.best) ? [{ x: b.best, y: b.checks.find((c) => c.step === b.best).val, color: css("--win") }] : [],
    xfmt: (x) => `step ${x}`, yfmt: (y) => (+y).toPrecision(3), empty: "this block has no checks yet",
  });
  chart($("#c-s2"), {
    series: [{ name: "control", color: css("--s2"), width: 1, dots: false, pts: (q.control || []).map((u) => [u.update, u.kl]) },
             { name: "control, trend", color: css("--s2"), width: 2.6, dots: false, pts: trend((q.control || []).map((u) => [u.update, u.kl])) },
             { name: "after Stage 1", color: css("--s1"), width: 1, dots: false, pts: s2.map((u) => [u.update, u.kl]) },
             { name: "after Stage 1, trend", color: css("--s1"), width: 2.6, dots: false, pts: trend(s2.map((u) => [u.update, u.kl])) }],
    xfmt: (x) => `update ${x}`, yfmt: (y) => (+y).toFixed(2), empty: "Stage 2 has not started",
  });
}

// --- contests ------------------------------------------------------------------------------------------------------
function forest(st) {
  const cs = st.contests || [], pis = Object.fromEntries((st.pi || []).map((p) => [p.contest, p]));
  const ci = cs.filter((c) => c.verdict).flatMap((c) => c.verdict.ci);
  const lo = Math.min(-1, ...ci) - 0.5, hi = Math.max(1, ...ci) + 0.5;
  const X = (v) => (((v - lo) / (hi - lo)) * 100).toFixed(3) + "%";
  const axis = nice(lo, hi, 6);
  let s = `<div class="axis"><span></span><span></span><div class="fx">${axis.map((t) => `<span style="left:${X(t)}">${t > 0 ? "+" : ""}${t}</span>`).join("")}</div><span class="x" style="text-align:right">ours − preset</span><span class="x">peak RAM on the Pi</span></div>`;
  for (const c of cs) {
    const v = c.verdict, p = pis[c.id], open = ui.open.has(c.id);
    const name = `<div class="name">${esc(c.size)} @ ${c.budget} GiB<small>${esc(c.model || "")}</small></div>`;
    if (!v) {
      const tag = c.state === "dropped" ? '<span class="v drop">dropped</span>' : c.state === "failed" ? '<span class="v LOSE">failed</span>' : `<span class="v run">${esc(c.state)}</span>`;
      s += `<div class="row dim">${name}${tag}<span class="muted" style="font-size:12px">${c.state === "dropped" ? "no longer searched (see the README)" : c.eta && c.eta !== "done" ? `ETA ${esc(c.eta)} IST` : ""}</span><span></span><span></span></div>`;
      continue;
    }
    const col = v.verdict === "WIN" ? "var(--win)" : v.verdict === "LOSE" ? "var(--lose)" : "var(--tie)";
    const svg = `<div class="fp" style="--c:${col}">${axis.map((t) => `<i class="g" style="left:${X(t)}"></i>`).join("")}<i class="z" style="left:${X(0)}"></i>
      <i class="ln" style="left:${X(v.ci[0])};right:calc(100% - ${X(v.ci[1])})"></i><i class="cap" style="left:${X(v.ci[0])}"></i><i class="cap" style="left:${X(v.ci[1])}"></i>
      <i class="pt" style="left:${X(v.diff)}" title="${sgn(v.diff)} [${sgn(v.ci[0])}, ${sgn(v.ci[1])}]"></i></div>`;
    const pi = p ? `<div class="pi">${p.ram.toFixed(2)} / ${c.budget} GiB<div class="meter"><i style="width:${Math.min(100, (p.ram / c.budget) * 100)}%"></i></div></div>` : '<div class="pi">not measured</div>';
    s += `<div class="row${open ? " open" : ""}" data-id="${esc(c.id)}">${name}<span class="v ${v.verdict}">${v.verdict}</span>${svg}
      <div class="ci num"><b>${sgn(v.diff)}</b> [${sgn(v.ci[0])}, ${sgn(v.ci[1])}]</div>${pi}</div>`;
    if (open) s += detail(c, p);
  }
  $("#forest").innerHTML = s;
  $("#forest").onclick = (e) => {
    const r = e.target.closest(".row[data-id]");
    if (!r) return;
    const id = r.dataset.id; ui.open.has(id) ? ui.open.delete(id) : ui.open.add(id);
    forest(ui.state);
  };
}
function detail(c, p) {
  const h = c.held_out, o = h.original.accuracy, rows = [["Original (16-bit)", h.original], ["Ours", h.ours], [`Preset ${c.verdict.preset}`, h.preset_test]];
  const total = c.types.reduce((a, t) => a + t[1], 0);
  return `<div class="detail">
    <div class="scroll"><table><thead><tr><th>held-out</th><th>accuracy</th><th>% of original</th>${TASKS.map(([, n]) => `<th>${n}</th>`).join("")}<th>KL</th><th>perplexity</th><th>flips %</th></tr></thead><tbody>
      ${rows.map(([n, r]) => `<tr${n === "Ours" ? ' class="ours"' : ""}><td>${esc(n)}</td><td>${r.accuracy.toFixed(2)}</td><td>${((r.accuracy / o) * 100).toFixed(1)}</td>${TASKS.map(([k]) => `<td>${r.per_task[k]?.toFixed(1) ?? "–"}</td>`).join("")}<td>${r.kl != null ? r.kl.toFixed(3) : "–"}</td><td>${r.ppl.toFixed(2)}</td><td>${r.flips != null && n !== "Original (16-bit)" ? r.flips.toFixed(1) : "–"}</td></tr>`).join("")}
    </tbody></table></div>
    <div><div class="ph" style="font-size:12px;font-weight:650;margin-bottom:6px">Our mix of types (${total} matrices, ${c.verdict.ram_gib.toFixed(3)} GiB predicted)</div>
      <div class="mix">${c.types.map(([t, n]) => `<i title="${t} ×${n}" style="width:${(n / total) * 100}%;background:${typeColor(t)}"></i>`).join("")}</div>
      <div class="mixlab">${c.types.map(([t, n]) => `<span><i class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;background:${typeColor(t)}"></i>${t} ×${n}</span>`).join("")}</div></div>
    ${p ? `<div class="kv">On the Pi 5: peak RAM <b class="num">${p.ram.toFixed(3)} GiB</b> (predicted ${p.predicted.toFixed(3)}, budget ${c.budget}) · writing <b class="num">${p.decode.toFixed(2)}</b> tokens/s · reading ${p.prefill.toFixed(2)} tokens/s${p.throttled ? " · throttled during the run" : ""}</div>` : ""}
    <div class="steps">${c.steps.map((x) => `<span class="step ${x.state}" title="${esc(x.about)}">${x.name}</span>`).join("")}</div>
    ${c.events.length ? `<ul class="tests">${c.events.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : ""}
  </div>`;
}

// --- the rail ------------------------------------------------------------------------------------------------------
function pis(st) {
  const order = (st.contests || []).map((c) => c.id), list = [...(st.pi || [])].sort((a, b) => order.indexOf(a.contest) - order.indexOf(b.contest));
  $("#pis").innerHTML = list.length ? list.map((p) => `<div class="pirow"><span class="mono" style="font-weight:650">${esc(p.contest)}</span>
      <div class="meter" title="${p.ram.toFixed(3)} of ${p.budget} GiB"><i style="width:${Math.min(100, (p.ram / p.budget) * 100)}%"></i></div>
      <span class="sp num">${p.ram.toFixed(2)}/${p.budget} GiB · ${p.decode.toFixed(2)} t/s</span></div>`).join("")
    : '<div class="muted">no Pi results yet</div>';
}
function kind(x) {
  if (/ note: /.test(x)) return "note";
  if (/FAILED/.test(x)) return "fail";
  if (/step \d+\/\d+, train MSE|distill update|: queued|: started$/.test(x)) return "quiet";
  if (/held-out|guard:|validation MSE .* -> |WIN|LOSE|TIE|: done|stopping early|strategy:|resuming/.test(x)) return "res";
  return "";
}
function feed(st) {
  const f = ui.filter, items = (st.events || []).map((x) => ({ x, k: kind(x) })).filter(({ k }) => f === "all" || (f === "notes" ? k === "note" : k !== "quiet"));
  $("#feed").innerHTML = items.slice(0, 60).map(({ x, k }) => `<li class="${k}"><time>${esc(x.slice(0, 5))}</time><span>${esc(x.slice(6).replace(/^note: /, ""))}</span></li>`).join("") || '<li class="quiet"><span>nothing yet</span></li>';
  for (const b of $("#filters").children) b.classList.toggle("on", b.dataset.f === f);
}
$("#filters").onclick = (e) => {
  const b = e.target.closest("button"); if (!b) return;
  ui.filter = b.dataset.f; try { localStorage.setItem("panther-filter", ui.filter); } catch (er) {}
  if (ui.state) feed(ui.state);
};

// --- the loop -------------------------------------------------------------------------------------------------------
function render(st) {
  const q = st.qat, done = q ? q.blocks.filter((b) => b.after != null).length : 0;
  const run = $("#running");
  if (q && done < q.total && q.blocks.length) { run.hidden = false; run.textContent = `QAT · stage 1 · block ${Math.min(done, q.total - 1)} of ${q.total}`; }
  else run.hidden = true;
  const ban = $("#banner"), pz = (st.system || {}).paused || [];
  ban.hidden = !pz.length && !st.error;
  ban.textContent = st.error ? `The server could not read the data: ${st.error}` : pz.length ? `Paused: ${pz.join(", ")}` : "";
  gauges(st); experiment(q); forest(st); pis(st); feed(st);
}
async function poll() {
  try {
    const r = await fetch("api/state", { cache: "no-store" });
    const st = await r.json();
    ui.state = st; ui.last = Date.now();
    $("#dot").className = st.error ? "dot err" : "dot";
    render(st);
    $("#clock").textContent = `${st.now || ""} IST`;
  } catch (e) {
    $("#dot").className = "dot err";
  }
}
setInterval(() => {
  const s = Math.round((Date.now() - ui.last) / 1000);
  $("#age").textContent = ui.last ? (s < 12 ? "live" : `updated ${s}s ago`) : "connecting…";
  if (ui.last && s > 35) $("#dot").className = "dot stale";
}, 1000);
let rs; addEventListener("resize", () => { clearTimeout(rs); rs = setTimeout(() => ui.state && render(ui.state), 150); });
poll(); setInterval(poll, 10000);
