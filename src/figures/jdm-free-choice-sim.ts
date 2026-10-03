// Why the classic free-choice paradigm cannot, on its own, show that choosing
// changes preferences. Simulated participants have fixed true liking for six
// items; ratings measure it with noise; choices between similarly rated items
// follow true liking (with a little noise of their own). Choosing then shifts
// liking by a true amount δ, split between the chosen and the rejected item.
// The classic design (rate, choose, rate) measures a spread of alternatives
// even at δ = 0, because the choice reveals which item the noisy first rating
// underrated. The control design (rate, rate, choose) measures that artifact
// alone; the difference between the two recovers δ. Monte Carlo with a fixed
// seed, so every render of the same parameters is identical.

import { defineFigure, type State } from "./types.ts";
import { el, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { memo, normal, rng } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    classic: "Rate, choose, rate (classic)",
    control: "Rate, rate, choose (control)",
    diff: "Classic minus control",
    truth: "true change δ",
    axis: "spread of alternatives (in SDs of true liking)",
    reliability: "test-retest reliability of one rating: {r}",
    describe: "With rating noise {s} (test-retest reliability {r}) and a true change of {d}, the classic design measures a spread of {a}, the control design {b}, and their difference {c}.",
  },
  zh: {
    classic: "评分、选择、评分（经典设计）",
    control: "评分、评分、选择（对照设计）",
    diff: "经典减去对照",
    truth: "真实改变 δ",
    axis: "选项分化（以真实喜爱程度的标准差为单位）",
    reliability: "单次评分的重测信度：{r}",
    describe: "评分噪声为 {s}（重测信度 {r}）、真实改变为 {d} 时，经典设计测得的选项分化为 {a}，对照设计为 {b}，两者之差为 {c}。",
  },
};

const params = {
  delta: { kind: "range", label: { en: "Change from choosing δ", zh: "选择带来的改变 δ" }, min: 0, max: 1, default: 0, step: 0.05 },
  noise: { kind: "range", label: { en: "Rating noise σ", zh: "评分噪声 σ" }, min: 0.2, max: 1.5, default: 0.8, step: 0.05 },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 17, step: 1, control: false },
} as const;

type P = { delta: number; noise: number; seed: number };

const PEOPLE = 3000;
const ITEMS = 6;
const CHOICE_NOISE = 0.5;

const simulate = memo((delta: number, noise: number, seed: number) => {
  const z = normal(rng(seed));
  let classic = 0, control = 0, n = 0;
  for (let p = 0; p < PEOPLE; p++) {
    const v = Array.from({ length: ITEMS }, () => z());
    const r1 = v.map((x) => x + noise * z());
    const r2pre = v.map((x) => x + noise * z()); // second rating before any choice (control)
    const rank = v.map((_, i) => i).sort((i, j) => r1[j] - r1[i]);
    const post = v.slice();
    const picks: Array<[number, number]> = [];
    for (const [a, b] of [[rank[1], rank[2]], [rank[3], rank[4]]]) {
      const ca = v[a] + CHOICE_NOISE * z(), cb = v[b] + CHOICE_NOISE * z();
      const [c, r] = ca >= cb ? [a, b] : [b, a];
      picks.push([c, r]);
      post[c] += delta / 2;
      post[r] -= delta / 2;
    }
    const r2 = post.map((x) => x + noise * z());
    for (const [c, r] of picks) {
      classic += (r2[c] - r1[c]) - (r2[r] - r1[r]);
      control += (r2pre[c] - r1[c]) - (r2pre[r] - r1[r]);
      n++;
    }
  }
  return { classic: classic / n, control: control / n };
}, 16);

function compute(p: P) {
  const s = simulate(p.delta, p.noise, p.seed);
  return { ...s, diff: s.classic - s.control, rel: 1 / (1 + p.noise * p.noise) };
}

function describe(st: State<P>): string {
  const c = compute(st.p);
  return tpl(labels[st.lang ?? "en"].describe, { s: fixed(st.p.noise, 2), r: fixed(c.rel, 2), d: fixed(st.p.delta, 2), a: fixed(c.classic, 2), b: fixed(c.control, 2), c: fixed(c.diff, 2) });
}

function render(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  const p = st.p, w = st.w, narrow = w < 480;
  const c = compute(p);
  const parts: string[] = [];
  const pad = narrow ? 8 : 16;
  const labelW = narrow ? 0 : 200;
  const left = pad + labelW, right = w - pad - 40;
  const hi = Math.max(1, c.classic, p.delta) * 1.08;
  const x = linear([-0.1, hi], [left, right]);
  const rowH = narrow ? 50 : 36;
  const top = 26;
  const rows: Array<[string, number, string]> = [[L.classic, c.classic, C.c1], [L.control, c.control, C.c7], [L.diff, c.diff, C.c6]];
  const bottom = top + rows.length * rowH + 6;
  parts.push(axis({ scale: x, orient: "bottom", at: bottom, span: [top, bottom], title: L.axis, count: narrow ? 4 : 6 }));
  rows.forEach(([name, v, color], i) => {
    const y0 = top + i * rowH;
    const by = narrow ? y0 + 20 : y0 + 8;
    const bh = narrow ? 18 : 20;
    if (narrow) parts.push(text(pad, y0 + 13, name, { "font-size": TYPE.small, class: "fig-t-strong" }));
    else parts.push(text(left - 10, by + 14, name, { "font-size": TYPE.body, "text-anchor": "end", class: "fig-t-strong" }));
    const x0 = x(0), x1 = x(v);
    parts.push(el("rect", { x: Math.min(x0, x1), y: by, width: Math.max(1, Math.abs(x1 - x0)), height: bh, rx: 3, fill: color, opacity: 0.85 }));
    parts.push(text(Math.max(x0, x1) + 6, by + bh / 2 + 4, fixed(v, 2), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  });
  // the true change, the quantity a good design should recover
  const xd = x(p.delta);
  parts.push(
    el("line", { x1: x(0), x2: x(0), y1: top, y2: bottom, stroke: C.rule }),
    el("line", { x1: xd, x2: xd, y1: top - 18, y2: bottom, stroke: C.truth, "stroke-width": 2, "stroke-dasharray": "5 3" }),
    text(xd + 4, top - 8, L.truth, { "font-size": TYPE.small, class: "fig-t-halo", style: `fill:${C.truth}` }),
  );
  const H = bottom + 62;
  parts.push(text(pad, H - 6, tpl(L.reliability, { r: fixed(c.rel, 2) }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  return svg(w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "jdm-free-choice-sim",
  title: { en: "The free-choice artifact, simulated", zh: "模拟自由选择范式中的伪影" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
