// The cover art: the book's two halves in one picture, computed rather than
// drawn. Above, fifty-six samples from a Gaussian process posterior on the running
// objective, pinched at six observations and fanning out between them, with
// the next query (largest expected improvement) as an orange hairline. Below,
// the same observations as a comparison graph: an arc for each answered pair,
// the preferred option filled. The title, subtitle, and author are HTML laid
// over the art (en/index.md), so they use the book's fonts and stay text.

import { fit, kernel, predict, samplePosterior } from "./figures/lib/gp.ts";
import { argmax, ei } from "./figures/lib/acq.ts";
import { rng } from "./figures/lib/random.ts";
import { running } from "./figures/lib/objectives.ts";

const W = 400, H = 600;
const PAPER = "#f4ecdc", ORANGE = "#ee8a3a";

export function coverArt(): string {
  const obsX = [0.07, 0.21, 0.4, 0.57, 0.74, 0.91];
  const obsY = obsX.map(running.f);
  const ym = obsY.reduce((a, b) => a + b, 0) / obsY.length;
  const gp = fit(kernel("rbf", 0.085, 0.9), obsX, obsY, 1e-5, ym);
  const xs = Array.from({ length: 141 }, (_, i) => -0.03 + (1.06 * i) / 140);
  const post = predict(gp, xs, true);
  const samples = samplePosterior(post, rng(20261002), 56);
  const best = Math.max(...obsY);
  const next = xs[argmax(post.mean.map((m, i) => ei(m, Math.sqrt(Math.max(post.var[i], 0)), best, 0.01)))];

  // Threads occupy y 262 to 482; the comparison graph hangs below y 508.
  const [lo, hi] = [-1.9, 2.1];
  const px = (x: number) => (x * W).toFixed(1);
  const py = (y: number) => (482 - ((y - lo) / (hi - lo)) * 220).toFixed(1);
  const line = (ys: number[]) => "M" + xs.map((x, i) => `${px(x)},${py(ys[i])}`).join("L");

  const threads = samples.map((s) => `<path d="${line(s)}"/>`).join("");
  const base = 508;
  const pairs: Array<[number, number]> = [[0, 2], [1, 2], [2, 3], [1, 4], [3, 4], [4, 5], [2, 5], [0, 1]];
  const arcs = pairs.map(([i, j]) => {
    const a = Number(px(obsX[i])), b = Number(px(obsX[j]));
    const rx = Math.abs(b - a) / 2, ry = Math.min(76, rx * 0.42);
    return `<path d="M${a},${base}A${rx.toFixed(1)},${ry.toFixed(1)} 0 0 0 ${b},${base}"/>`;
  }).join("");
  // An observation's mark on the baseline is filled if it won its first comparison.
  const wins = new Set(pairs.map(([i, j]) => (obsY[i] > obsY[j] ? i : j)));
  const stems = obsX.map((x, i) => `<line x1="${px(x)}" x2="${px(x)}" y1="${py(obsY[i])}" y2="${base}"/>`).join("");
  const marks = obsX.map((x, i) => wins.has(i)
    ? `<circle cx="${px(x)}" cy="${base}" r="3.4" fill="${PAPER}"/>`
    : `<circle cx="${px(x)}" cy="${base}" r="3" fill="#0e1a2c" stroke="${PAPER}" stroke-width="1.2"/>`).join("");
  const dots = obsX.map((x, i) => `<circle cx="${px(x)}" cy="${py(obsY[i])}" r="3.6"/>`).join("");

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid slice">
<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#15263f"/><stop offset="1" stop-color="#0b1524"/></linearGradient></defs>
<rect width="${W}" height="${H}" fill="url(#bg)"/>
<g fill="none" stroke="${PAPER}" stroke-width="0.7" stroke-opacity="0.16" stroke-linejoin="round">${threads}</g>
<path d="${line(post.mean)}" fill="none" stroke="${PAPER}" stroke-width="1.3" stroke-opacity="0.6"/>
<g stroke="${PAPER}" stroke-opacity="0.22" stroke-width="0.8" stroke-dasharray="1.5 3">${stems}</g>
<line x1="${px(next)}" x2="${px(next)}" y1="268" y2="${base}" stroke="${ORANGE}" stroke-width="1.3"/>
<circle cx="${px(next)}" cy="268" r="4.2" fill="none" stroke="${ORANGE}" stroke-width="1.5"/>
<g fill="${PAPER}">${dots}</g>
<g fill="none" stroke="${PAPER}" stroke-opacity="0.45" stroke-width="0.9">${arcs}</g>
${marks}
</svg>
`;
}
