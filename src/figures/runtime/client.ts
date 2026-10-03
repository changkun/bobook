// Hydration: turn each static figure host into a live figure. The host
// already holds the build-time SVG; this adds the controls, actions, and the
// transport, re-renders at the measured column width, and re-renders on every
// parameter or timeline change.
//
// Pointer input: a figure with a `pointer` handler receives clicks and drags
// in SVG coordinates. Any element with data-fig-set="key=value" sets that
// parameter on click; data-fig-hit="name" names the target a pointer event
// reports.
//
// host.__fig exposes { params, t, set, seek, play, pause } for tests,
// screenshots, and debugging.

import { REGISTRY } from "virtual:figure-registry";
import { tr, type AnyFigure, type Lang } from "../types.ts";
import { coerce, defaults, type ParamRecord } from "../lib/params.ts";
import { sig } from "../lib/format.ts";
import { buildControls, h } from "./controls.ts";
import { Transport } from "./transport.ts";

interface FigHandle {
  params: ParamRecord;
  t: number;
  set(key: string, value: number | string | boolean): void;
  seek(t: number): void;
  play(): void;
  pause(): void;
}

declare global {
  interface HTMLElement { __fig?: FigHandle }
}

const reducedMotion = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

const RESET = { en: "Reset", zh: "重置" };

// Figures may link to pages with root-relative hrefs written as @@ROOT@@...;
// the build rewrites the static render, and the client rewrites live renders
// with the same prefix the page's stylesheet link uses.
const ROOT = typeof document !== "undefined"
  ? (document.querySelector('link[rel="stylesheet"][href*="assets/book.css"]')?.getAttribute("href") ?? "").replace(/assets\/book\.css.*$/, "")
  : "";

function mount(host: HTMLElement, fig: AnyFigure) {
  const lang: Lang = host.dataset.lang === "zh" ? "zh" : "en";
  const initial: ParamRecord = { ...defaults(fig), ...JSON.parse(host.dataset.params || "{}") };
  let p: ParamRecord = { ...initial };
  const tl = fig.timeline;
  const uid = `${host.closest("figure")?.id || fig.name}-live`;

  const stage = h("div", "fig-stage");
  const live = h("div", "fig-sr");
  live.setAttribute("aria-live", "polite");

  let t = 0;
  let width = 0;
  let announceTimer = 0;
  const announce = (extra = "") => {
    clearTimeout(announceTimer);
    announceTimer = window.setTimeout(() => {
      live.textContent = [extra, fig.describe({ p, t, w: width, uid, lang }, lang)].filter(Boolean).join(" ");
    }, 350);
  };

  const draw = () => {
    width = Math.max(260, Math.floor(stage.clientWidth || host.clientWidth || 660));
    stage.innerHTML = fig.render({ p, t, w: width, uid, lang }, lang).replaceAll("@@ROOT@@", ROOT);
    for (const b of actionButtons) b.sync();
  };

  let transport: Transport | undefined;
  const refreshTimeline = () => {
    if (tl && transport) {
      transport.configure(tl.duration(p), tl.keyframes(p, lang));
      t = transport.shown();
    }
  };
  const setAll = (next: ParamRecord, changed?: string) => {
    p = next;
    if (fig.update && changed) p = fig.update(p, changed);
    controls.sync(p);
    refreshTimeline();
    draw();
    announce();
  };
  const set = (key: string, value: number | string | boolean) => setAll({ ...p, [key]: coerce(fig, key, value) }, key);

  const controls = buildControls(fig, lang, p, set);
  const actionButtons: Array<{ sync(): void }> = [];
  const bar = h("div", "fig-actions");
  for (const a of fig.actions ?? []) {
    const b = h("button", a.primary ? "fig-btn" : "fig-btn fig-btn-quiet", tr(a.label, lang));
    b.type = "button";
    b.addEventListener("click", () => setAll(a.run({ ...p }) as ParamRecord));
    // An action that does not apply in the current state is hidden, not greyed
    // out, so the bar only ever shows what the reader can do now.
    actionButtons.push({ sync: () => { b.hidden = a.enabled ? !a.enabled(p) : false; } });
    bar.append(b);
  }
  const hasData = Object.values(fig.params).some((s) => (s as { kind: string }).kind === "data");
  if (fig.actions?.length || hasData || controls.root.childElementCount) {
    const reset = h("button", "fig-btn fig-btn-quiet", tr(RESET, lang));
    reset.type = "button";
    reset.addEventListener("click", () => { setAll({ ...initial }); if (tl && transport) transport.seek(host.dataset.t ? Number(host.dataset.t) : tl.poster(p), "user"); });
    bar.append(reset);
  }

  const bare = host.dataset.bare === "true";
  host.classList.add("fig-ready");
  host.setAttribute("role", "group");
  host.setAttribute("aria-label", tr(fig.title, lang));
  const parts: HTMLElement[] = [];
  const top = h("div", "fig-top");
  if (controls.root.childElementCount) top.append(controls.root);
  if (bar.childElementCount) top.append(bar);
  if (top.childElementCount && !bare) parts.push(top);
  if (fig.hint && !bare) parts.push(h("p", "fig-hint", tr(fig.hint, lang)));

  if (tl) {
    const reduced = reducedMotion();
    const t0 = host.dataset.t != null && host.dataset.t !== "" ? Number(host.dataset.t) : tl.poster(p);
    const unit = tl.unit;
    const time = unit && ((nt: number) => {
      const v = unit.value ? unit.value(nt, p, lang) : sig(nt, 3);
      const sym = tr(unit.symbol, lang);
      return sym ? `${v} ${sym}` : v;
    });
    transport = new Transport({
      lang, reduced, rate: tl.rate, discrete: tl.discrete ?? true, time,
      duration: tl.duration(p), keyframes: tl.keyframes(p, lang), t: t0,
      onSeek: (nt, cause) => {
        t = nt;
        draw();
        if (cause === "user") announce(transport!.currentKey()?.t === nt ? transport!.currentKey()!.label : "");
      },
    });
    t = transport.shown();
    if (!bare) parts.push(transport.root);
    // Autoplay: play while visible, and at the end start a fresh run with
    // the next seed, so a cover figure keeps telling new versions of its story.
    const autoplay = host.dataset.autoplay === "true" && !reduced;
    let restart = 0;
    if (autoplay) transport.onEnd = () => {
      clearTimeout(restart);
      restart = window.setTimeout(() => {
        if ("seed" in fig.params) set("seed", ((Number(p.seed) || 1) % 9999) + 1);
        transport!.seek(0, "play");
        transport!.play();
      }, 2200);
    };
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver((entries) => {
        for (const e of entries) {
          if (!e.isIntersecting && transport!.isPlaying) transport!.pause();
          else if (e.isIntersecting && autoplay && !transport!.isPlaying) transport!.play();
        }
      }, { threshold: 0.3 }).observe(host);
    }
  }

  parts.push(stage, live);
  host.querySelector(".fig-static")?.remove();
  host.append(...parts);
  draw();

  // Pointer input in SVG coordinates.
  const toSvg = (ev: PointerEvent | MouseEvent) => {
    const svgEl = stage.querySelector("svg");
    if (!svgEl) return undefined;
    const pt = svgEl.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    const m = svgEl.getScreenCTM();
    if (!m) return undefined;
    const q = pt.matrixTransform(m.inverse());
    const hitEl = (ev.target as Element).closest?.("[data-fig-hit]");
    const target = hitEl?.getAttribute("data-fig-hit") ?? undefined;
    const mapEl = dragMap ?? (ev.target as Element).closest?.("[data-fig-map]");
    let data: { x: number; y: number } | undefined;
    const map = mapEl?.getAttribute("data-fig-map")?.split(",").map(Number);
    if (map && map.length === 8) {
      const [px0, px1, dx0, dx1, py0, py1, dy0, dy1] = map;
      data = { x: dx0 + ((q.x - px0) / (px1 - px0)) * (dx1 - dx0), y: dy0 + ((q.y - py0) / (py1 - py0)) * (dy1 - dy0) };
    }
    return { x: q.x, y: q.y, w: width, target: dragTarget ?? target, data };
  };
  let dragMap: Element | null | undefined;
  let dragTarget: string | undefined;
  let dragging = false;
  let moved = false;
  let downAt: { x: number; y: number } | undefined;
  if (fig.pointer) {
    stage.classList.add("fig-interactive");
    stage.addEventListener("pointerdown", (ev) => {
      dragMap = undefined; dragTarget = undefined;
      const q = toSvg(ev);
      if (!q) return;
      dragMap = (ev.target as Element).closest?.("[data-fig-map]");
      dragTarget = q.target;
      dragging = true; moved = false; downAt = { x: ev.clientX, y: ev.clientY };
      const next = fig.pointer!(p, { phase: "down", ...q });
      if (next) { stage.setPointerCapture(ev.pointerId); setAll(next as ParamRecord); }
    });
    stage.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      if (downAt && Math.hypot(ev.clientX - downAt.x, ev.clientY - downAt.y) > 4) moved = true;
      if (!moved) return;
      const q = toSvg(ev);
      if (!q) return;
      const next = fig.pointer!(p, { phase: "move", ...q });
      if (next) setAll(next as ParamRecord);
    });
    const end = (ev: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      const q = toSvg(ev);
      if (!q) return;
      const next = fig.pointer!(p, { phase: moved ? "up" : "click", ...q });
      dragMap = undefined; dragTarget = undefined;
      if (next) setAll(next as ParamRecord);
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", () => { dragging = false; });
  }

  stage.addEventListener("click", (ev) => {
    const target = (ev.target as Element).closest?.("[data-fig-set]");
    if (!target) return;
    const [key, value] = (target.getAttribute("data-fig-set") ?? "").split("=");
    if (key && value != null && key in fig.params) set(key, value);
  });

  if (typeof ResizeObserver === "function") {
    let raf = 0;
    new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        if (Math.floor(stage.clientWidth) !== width) draw();
      });
    }).observe(stage);
  }

  host.__fig = {
    get params() { return { ...p }; },
    get t() { return t; },
    set,
    seek: (nt: number) => transport?.seek(nt, "user"),
    play: () => transport?.play(),
    pause: () => transport?.pause(),
  } as FigHandle;
}

// Load each embedded figure's module once, then mount every host that uses
// it. Until the module arrives, and if it fails, the static SVG stays visible.
export function mountFigures() {
  const hosts = [...document.querySelectorAll<HTMLElement>(".fig[data-figure]:not(.fig-ready):not(.fig-loading)")];
  for (const host of hosts) {
    const name = host.dataset.figure ?? "";
    const load = REGISTRY[name];
    if (!load) { host.dataset.figError = "unknown figure"; continue; }
    host.classList.add("fig-loading");
    // Hydrate when the figure comes near the viewport, so a long chapter does
    // not compute every figure at once.
    const start = () => load().then(({ default: fig }) => {
      host.classList.remove("fig-loading");
      if (!host.isConnected) return;
      try { mount(host, fig); } catch (e) {
        host.dataset.figError = String((e as Error).message ?? e);
        console.error(`figure ${name}:`, e);
      }
    }).catch((e: unknown) => {
      host.classList.remove("fig-loading");
      host.dataset.figError = String((e as Error)?.message ?? e);
      console.error(`figure ${name}:`, e);
    });
    if (typeof IntersectionObserver === "function") {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) { io.disconnect(); start(); }
      }, { rootMargin: "600px 0px" });
      io.observe(host);
    } else start();
  }
}
