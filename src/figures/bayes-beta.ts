// The beta-binomial model: a Beta prior on a coin's bias, set by its mean and
// its strength in flips (alpha + beta), and a count of heads and tails. The
// plot shows the prior (dashed), the likelihood rescaled to unit area
// (dotted), and the posterior Beta(alpha + h, beta + t) with its central 95%
// credible interval shaded. With the estimates switched on, it marks the
// maximum likelihood estimate, the posterior mode (MAP), and the posterior
// mean. The readout writes the posterior mean as a weighted average of the
// prior mean and the observed fraction of heads.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { betaPdf } from "./lib/stats.ts";
import { clip, frame, frameAxes, legend } from "./lib/plot.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "bias θ (probability of heads)",
    y: "density",
    prior: "prior",
    lik: "likelihood (rescaled)",
    post: "posterior",
    ci: "95% credible interval",
    mle: "MLE",
    map: "MAP",
    pm: "mean",
    priorLine: "prior Beta({a}, {b}): mean {m}, worth n₀ = {n0} flips",
    dataLine: "data: {h} heads, {t} tails",
    dataNone: "data: none yet (the posterior is the prior)",
    postLine: "posterior Beta({a}, {b}), 95% interval [{lo}, {hi}]",
    mixLine: "mean {pm} = {w} × prior mean {m} + {v} × data fraction {f}",
    mixNone: "mean {pm} = the prior mean; P(next flip heads) = {pm}",
    nextLine: "P(next flip heads) = posterior mean = {pm}",
    estLine: "MLE {mle} · MAP {map} · posterior mean {pm}",
    describe: "A Beta({a}, {b}) prior and {h} heads with {t} tails give a Beta({pa}, {pb}) posterior with mean {pm} and 95% credible interval from {lo} to {hi}.",
    undef: "undefined",
    none: "none",
  },
  zh: {
    x: "偏向 θ（正面朝上的概率）",
    y: "密度",
    prior: "先验",
    lik: "似然（缩放后）",
    post: "后验",
    ci: "95% 可信区间",
    mle: "MLE",
    map: "MAP",
    pm: "均值",
    priorLine: "先验 Beta({a}, {b})：均值 {m}，相当于 n₀ = {n0} 次抛掷",
    dataLine: "数据：{h} 次正面，{t} 次反面",
    dataNone: "数据：尚无（后验即先验）",
    postLine: "后验 Beta({a}, {b})，95% 区间 [{lo}, {hi}]",
    mixLine: "均值 {pm} = {w} × 先验均值 {m} + {v} × 正面比例 {f}",
    mixNone: "均值 {pm} = 先验均值；P(下一次为正面) = {pm}",
    nextLine: "P(下一次为正面) = 后验均值 = {pm}",
    estLine: "MLE {mle} · MAP {map} · 后验均值 {pm}",
    describe: "Beta({a}, {b}) 先验加上 {h} 次正面、{t} 次反面，得到 Beta({pa}, {pb}) 后验，均值为 {pm}，95% 可信区间为 {lo} 至 {hi}。",
    undef: "无定义",
    none: "无",
  },
};

const params = {
  m0: { kind: "range", label: { en: "Prior mean", zh: "先验均值" }, min: 0.1, max: 0.9, default: 0.5, step: 0.05 },
  n0: { kind: "range", label: { en: "Prior strength n₀ (flips)", zh: "先验强度 n₀（抛掷次数）" }, min: 1, max: 100, default: 2, step: 1 },
  heads: { kind: "range", label: { en: "Heads h", zh: "正面次数 h" }, min: 0, max: 100, default: 7, step: 1 },
  tails: { kind: "range", label: { en: "Tails t", zh: "反面次数 t" }, min: 0, max: 100, default: 3, step: 1 },
  estimates: { kind: "toggle", label: { en: "Show point estimates", zh: "显示点估计" }, default: false },
} as const;

type P = { m0: number; n0: number; heads: number; tails: number; estimates: boolean };

const XS = grid(0.0025, 0.9975, 399);

function model(p: P) {
  const a = p.m0 * p.n0, b = (1 - p.m0) * p.n0;
  const h = Math.round(p.heads), t = Math.round(p.tails), n = h + t;
  const pa = a + h, pb = b + t;
  const pm = pa / (pa + pb);
  const mle = n ? h / n : NaN;
  const map = pa > 1 && pb > 1 ? (pa - 1) / (pa + pb - 2) : pa <= 1 && pb > 1 ? 0 : pb <= 1 && pa > 1 ? 1 : NaN;
  // central 95% interval from the cumulative sum of the density on a fine grid
  const fine = grid(0, 1, 4001);
  const dens = fine.map((x, i) => (i === 0 || i === fine.length - 1 ? 0 : betaPdf(x, pa, pb)));
  const cdf: number[] = [0];
  for (let i = 1; i < fine.length; i++) cdf.push(cdf[i - 1] + 0.5 * (dens[i] + dens[i - 1]) * (fine[i] - fine[i - 1]));
  const tot = cdf[cdf.length - 1] || 1;
  const q = (pr: number) => { const k = cdf.findIndex((c) => c / tot >= pr); return fine[Math.max(0, k)]; };
  return { a, b, h, t, n, pa, pb, pm, mle, map, lo: q(0.025), hi: q(0.975), w: p.n0 / (p.n0 + n) };
}

function describe(st: State<P>): string {
  const m = model(st.p);
  return tpl(labels[st.lang ?? "en"].describe, { a: sig(m.a, 3), b: sig(m.b, 3), h: m.h, t: m.t, pa: sig(m.pa, 3), pb: sig(m.pb, 3), pm: fixed(m.pm, 2), lo: fixed(m.lo, 2), hi: fixed(m.hi, 2) });
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], w = st.w;
  const narrow = w < 480;
  const m = model(p);
  const prior = XS.map((x) => betaPdf(x, m.a, m.b));
  const lik = m.n ? XS.map((x) => betaPdf(x, m.h + 1, m.t + 1)) : [];
  const post = XS.map((x) => betaPdf(x, m.pa, m.pb));
  const finite = (v: number[]) => v.filter((x, i) => Number.isFinite(x) && i > 3 && i < XS.length - 4);
  const ymax = Math.min(Math.max(...finite(post), ...(lik.length ? finite(lik) : [0]), 1.2) * 1.1, 40);
  const lg = legend(narrow ? 8 : 52, 14, w - 16, [
    { kind: "dash", color: C.ink2, label: L.prior },
    ...(m.n ? [{ kind: "dash" as const, color: C.c5, label: L.lik }] : []),
    { kind: "line", color: C.model, label: L.post },
    { kind: "band", color: C.band, label: L.ci },
  ]);
  const top = 14 + lg.height + 6;
  const f = frame({ w, top, height: narrow ? 170 : 210, yDomain: [0, ymax], yTitle: L.y });
  const cid = `${st.uid}-clip`;
  const path = (ys: number[]) => linePath(XS.map((x, i) => [f.x(x), f.y(Math.min(ys[i], ymax * 1.5))]));
  const inCi = XS.map((x, i) => [x, post[i]] as [number, number]).filter(([x]) => x >= m.lo && x <= m.hi);
  const ciPath = inCi.length ? linePath(inCi.map(([x, v]) => [f.x(x), f.y(Math.min(v, ymax * 1.5))])) + `L${f.x(inCi[inCi.length - 1][0])},${f.y(0)}L${f.x(inCi[0][0])},${f.y(0)}Z` : "";
  const parts: string[] = [
    el("defs", {}, clip(f, cid)),
    lg.svg,
    frameAxes(f, { yTitle: L.y, xTitle: L.x, yCount: 3 }),
    g({ "clip-path": `url(#${cid})` },
      ciPath ? el("path", { d: ciPath, fill: C.band, stroke: "none" }) : "",
      el("path", { d: path(prior), fill: "none", stroke: C.ink2, "stroke-width": 1.6, "stroke-dasharray": "6 4" }),
      m.n ? el("path", { d: path(lik), fill: "none", stroke: C.c5, "stroke-width": 1.6, "stroke-dasharray": "2 3" }) : "",
      el("path", { d: path(post), fill: "none", stroke: C.model, "stroke-width": 2.4 }),
    ),
  ];
  if (p.estimates) {
    const marks: Array<[number, string, string]> = [];
    if (Number.isFinite(m.mle)) marks.push([m.mle, L.mle, C.c5]);
    if (Number.isFinite(m.map)) marks.push([m.map, L.map, C.ink]);
    marks.push([m.pm, L.pm, C.model]);
    // stagger labels that would collide: each label takes the first row
    // where no earlier label sits within 34 px
    const placed: Array<{ px: number; row: number }> = [];
    for (const [x, lab, color] of marks) {
      const px = f.x(x);
      let row = 0;
      while (placed.some((q) => q.row === row && Math.abs(q.px - px) < 34)) row++;
      placed.push({ px, row });
      parts.push(
        el("line", { x1: px, x2: px, y1: f.top + 4 + row * 13, y2: f.bottom, stroke: color, "stroke-width": 1.5, "stroke-dasharray": "3 2" }),
        text(px, f.top + row * 13, lab, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong fig-t-halo" }),
      );
    }
  }
  let y = f.bottom + 52;
  const x0 = narrow ? 8 : f.left;
  const line = (s: string, cls = "fig-t-muted") => { parts.push(text(x0, y, s, { "font-size": narrow ? TYPE.small : TYPE.body, class: `${cls} fig-t-num` })); y += 18; };
  line(tpl(L.priorLine, { a: sig(m.a, 3), b: sig(m.b, 3), m: fixed(p.m0, 2), n0: sig(p.n0, 3) }));
  line(m.n ? tpl(L.dataLine, { h: m.h, t: m.t }) : L.dataNone);
  line(tpl(L.postLine, { a: sig(m.pa, 3), b: sig(m.pb, 3), lo: fixed(m.lo, 2), hi: fixed(m.hi, 2) }), "fig-t-strong");
  if (m.n) {
    line(tpl(L.mixLine, { pm: fixed(m.pm, 2), w: fixed(m.w, 2), m: fixed(p.m0, 2), v: fixed(1 - m.w, 2), f: fixed(m.mle, 2) }));
    line(tpl(L.nextLine, { pm: fixed(m.pm, 2) }));
  } else line(tpl(L.mixNone, { pm: fixed(m.pm, 2) }));
  if (p.estimates) line(tpl(L.estLine, { mle: Number.isFinite(m.mle) ? fixed(m.mle, 2) : L.undef, map: Number.isFinite(m.map) ? fixed(m.map, 2) : L.none, pm: fixed(m.pm, 2) }));
  return svg(w, y - 4, describe(st), ...parts);
}


const bump = (p: P, key: "heads" | "tails"): P => ({ ...p, [key]: Math.min(100, p[key] + 1) });

export default defineFigure({
  name: "bayes-beta",
  title: { en: "Beta-binomial updating: prior strength against data", zh: "Beta-二项更新：先验强度与数据的较量" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Heads +1", zh: "正面 +1" }, primary: true, run: (p) => bump(p as P, "heads") },
    { label: { en: "Tails +1", zh: "反面 +1" }, primary: true, run: (p) => bump(p as P, "tails") },
    { label: { en: "Clear flips", zh: "清空抛掷" }, run: (p) => ({ ...p, heads: 0, tails: 0 }), enabled: (p) => p.heads + p.tails > 0 },
  ],
});
