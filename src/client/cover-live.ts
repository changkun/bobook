// The cover runs the loop it depicts. Every few seconds the orange next
// question is asked: it becomes an observation, the posterior threads pinch
// at it, and its comparison with the best design so far is drawn below.
// After a while the cover returns to a fresh start. Clicking (or tapping)
// the cover asks the question there instead. Without script, or with reduced
// motion requested, the still image (assets/cover-art.svg) stays.

import { coverArt, DEFAULT_COVER, nextQuery, objective, type CoverState } from "../cover.ts";
import { rng } from "../figures/lib/random.ts";

const STEP_MS = 2600; // between questions
const HOLD_MS = 5000; // pause before starting over
const MAX_OBS = 10;
const MIN_GAP = 0.015; // a new observation this close to an old one adds nothing

export function startCover(cover: HTMLElement) {
  const still = cover.querySelector("img");
  const art = document.createElement("div");
  art.className = "bc-art";
  art.setAttribute("aria-hidden", "true");
  cover.insertBefore(art, cover.querySelector(".bc-text"));
  cover.classList.add("is-live");

  let state: CoverState = { ...DEFAULT_COVER };
  let round = 0;
  let timer: number | undefined;
  let visible = true;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function draw(st: CoverState) {
    const layer = document.createElement("div");
    layer.className = "bc-layer";
    layer.innerHTML = coverArt(st);
    art.append(layer);
    // Cross-fade: the new layer fades in over the old, which then goes.
    requestAnimationFrame(() => layer.classList.add("is-shown"));
    const old = [...art.children].slice(0, -1);
    setTimeout(() => old.forEach((o) => o.remove()), reduced ? 0 : 900);
    if (still) still.style.visibility = "hidden";
  }

  function bestIndex(obsX: number[]) {
    const ys = obsX.map(objective);
    return ys.indexOf(Math.max(...ys));
  }

  // Ask at x: add the observation and its comparison with the best so far.
  function ask(x: number): boolean {
    if (state.obsX.some((o) => Math.abs(o - x) < MIN_GAP)) return false;
    const best = bestIndex(state.obsX);
    const obsX = [...state.obsX, x];
    state = { obsX, pairs: [...state.pairs, [obsX.length - 1, best]], seed: state.seed, fresh: true };
    draw(state);
    return true;
  }

  function restart() {
    round++;
    const r = rng(round * 7919 + 11);
    const a = 0.05 + 0.4 * r(), b = 0.55 + 0.4 * r();
    state = { obsX: [a, b], pairs: [[0, 1]], seed: DEFAULT_COVER.seed + round };
    draw(state);
  }

  function tick() {
    timer = undefined;
    if (!visible || document.hidden) return;
    if (state.obsX.length >= MAX_OBS || !ask(nextQuery(state.obsX))) {
      timer = window.setTimeout(() => { restart(); schedule(); }, HOLD_MS);
      return;
    }
    schedule();
  }

  function schedule(delay = STEP_MS) {
    if (reduced || timer !== undefined) return;
    timer = window.setTimeout(tick, delay);
  }

  cover.addEventListener("click", (e) => {
    const r = cover.getBoundingClientRect();
    const x = Math.min(0.98, Math.max(0.02, (e.clientX - r.left) / r.width));
    if (state.obsX.length >= MAX_OBS) restart();
    ask(x);
    if (timer !== undefined) { clearTimeout(timer); timer = undefined; }
    schedule(STEP_MS * 2);
  });

  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) schedule();
  }).observe(cover);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) schedule(); });

  draw(state);
  schedule();
}
