// The cover art: the book's two halves in one picture, computed rather than
// drawn. Above, samples from a Gaussian process posterior on the running
// objective, pinched at the observations and fanning out between them. Below,
// the same observations as a comparison graph: an arc for each answered pair,
// the preferred option filled. In orange, the next question: the point of
// largest expected improvement, and a dashed arc pairing it with the best
// design so far. The title, subtitle, and author are HTML laid over the art
// (en/index.md), so they use the book's fonts and stay text.
//
// The build writes the default state as a still image (assets/cover-art.svg,
// and the link-preview cards); on the landing page, src/client/cover-live.ts
// draws the same art from changing states, so the cover runs the loop.

import { fit, kernel, predict, samplePosterior } from "./figures/lib/gp.ts";
import { argmax, ei } from "./figures/lib/acq.ts";
import { rng } from "./figures/lib/random.ts";
import { running } from "./figures/lib/objectives.ts";

const W = 400, H = 600;
const PAPER = "#f4ecdc", ORANGE = "#ee8a3a";
const KERNEL = kernel("rbf", 0.085, 0.9);
const XS = Array.from({ length: 141 }, (_, i) => -0.03 + (1.06 * i) / 140);

export interface CoverState {
  obsX: number[]; // observed inputs, in the order they were asked
  pairs: Array<[number, number]>; // answered comparisons, as indices into obsX
  seed: number; // for the posterior samples
  fresh?: boolean; // mark the newest observation and comparison for animation
}

export const DEFAULT_COVER: CoverState = {
  obsX: [0.07, 0.21, 0.4, 0.57, 0.74, 0.91],
  pairs: [[0, 2], [1, 2], [2, 3], [1, 4], [3, 4], [4, 5], [2, 5], [0, 1]],
  seed: 20261002,
};

export const objective = (x: number) => running.f(x);

function posterior(obsX: number[]) {
  const obsY = obsX.map(objective);
  const ym = obsY.reduce((a, b) => a + b, 0) / obsY.length;
  const post = predict(fit(KERNEL, obsX, obsY, 1e-5, ym), XS, true);
  return { obsY, post };
}

// The input of largest expected improvement, searched inside the cover's
// visible range.
export function nextQuery(obsX: number[]): number {
  const { obsY, post } = posterior(obsX);
  const best = Math.max(...obsY);
  const a = post.mean.map((m, i) => (XS[i] < 0.02 || XS[i] > 0.98 ? -1 : ei(m, Math.sqrt(Math.max(post.var[i], 0)), best, 0.01)));
  return XS[argmax(a)];
}

export function coverArt(st: CoverState = DEFAULT_COVER): string {
  const { obsX, pairs } = st;
  const { obsY, post } = posterior(obsX);
  const samples = samplePosterior(post, rng(st.seed), 56);
  const best = Math.max(...obsY);
  const next = nextQuery(obsX);

  // Threads occupy y 262 to 482; the comparison graph hangs below y 508.
  const [lo, hi] = [-1.9, 2.1];
  const px = (x: number) => (x * W).toFixed(1);
  const py = (y: number) => (482 - ((y - lo) / (hi - lo)) * 220).toFixed(1);
  const line = (ys: number[]) => "M" + XS.map((x, i) => `${px(x)},${py(ys[i])}`).join("L");
  const newest = st.fresh ? obsX.length - 1 : -1;
  const newestPair = st.fresh ? pairs.length - 1 : -1;

  const threads = samples.map((s) => `<path d="${line(s)}"/>`).join("");
  const base = 508;
  const arc = (x1: number, x2: number) => {
    const a = Math.min(x1, x2), b = Math.max(x1, x2);
    const rx = (b - a) / 2, ry = Math.min(76, rx * 0.42);
    return `M${a},${base}A${rx.toFixed(1)},${ry.toFixed(1)} 0 0 0 ${b},${base}`;
  };
  const arcs = pairs.map(([i, j], k) => `<path d="${arc(Number(px(obsX[i])), Number(px(obsX[j])))}"${k === newestPair ? ` class="bc-new-arc" pathLength="1"` : ""}/>`).join("");
  // An observation's mark on the baseline is filled if it won a comparison.
  const wins = new Set(pairs.map(([i, j]) => (obsY[i] > obsY[j] ? i : j)));
  const stems = obsX.map((x, i) => `<line x1="${px(x)}" x2="${px(x)}" y1="${py(obsY[i])}" y2="${base}"/>`).join("");
  const marks = obsX.map((x, i) => wins.has(i)
    ? `<circle cx="${px(x)}" cy="${base}" r="3.4" fill="${PAPER}"/>`
    : `<circle cx="${px(x)}" cy="${base}" r="3" fill="#0e1a2c" stroke="${PAPER}" stroke-width="1.2"/>`).join("");
  const dots = obsX.map((x, i) => `<circle cx="${px(x)}" cy="${py(obsY[i])}" r="3.6"${i === newest ? ` class="bc-new-dot"` : ""}/>`).join("");
  // The next question, in orange: the point of largest expected improvement
  // on the posterior mean, its stem down to the comparison row, and a dashed
  // arc to the best design so far, the comparison the person is asked next.
  const iNext = XS.indexOf(next);
  const nx = Number(px(next)), bx = Number(px(obsX[obsY.indexOf(best)]));
  const ask = `<g class="bc-ask"><line x1="${nx}" x2="${nx}" y1="${py(post.mean[iNext])}" y2="${base}" stroke="${ORANGE}" stroke-opacity="0.7" stroke-width="0.9" stroke-dasharray="1.5 3"/>
<path d="${arc(nx, bx)}" fill="none" stroke="${ORANGE}" stroke-width="1.1" stroke-dasharray="3 2.5"/>
<circle cx="${nx}" cy="${py(post.mean[iNext])}" r="3.6" fill="${ORANGE}"/>
<circle cx="${nx}" cy="${base}" r="3.2" fill="#0e1a2c" stroke="${ORANGE}" stroke-width="1.4"/></g>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid slice">
<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#15263f"/><stop offset="1" stop-color="#0b1524"/></linearGradient></defs>
<rect width="${W}" height="${H}" fill="url(#bg)"/>
<g fill="none" stroke="${PAPER}" stroke-width="0.7" stroke-opacity="0.16" stroke-linejoin="round">${threads}</g>
<path d="${line(post.mean)}" fill="none" stroke="${PAPER}" stroke-width="1.3" stroke-opacity="0.6"/>
<g stroke="${PAPER}" stroke-opacity="0.22" stroke-width="0.8" stroke-dasharray="1.5 3">${stems}</g>
<g fill="${PAPER}">${dots}</g>
<g fill="none" stroke="${PAPER}" stroke-opacity="0.45" stroke-width="0.9">${arcs}</g>
${marks}
${ask}
</svg>
`;
}
