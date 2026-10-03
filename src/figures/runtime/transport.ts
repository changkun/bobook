// The motion transport for animated figures: play/pause, step, a scrubber
// with a tick per keyframe, and the label of the latest keyframe. The figure
// never animates itself; the transport owns time and asks for a redraw. Under
// prefers-reduced-motion there is no play button and the step buttons move
// between keyframes.

import type { Keyframe, Lang } from "../types.ts";
import { h } from "./controls.ts";

export interface TransportOpts {
  lang: Lang;
  reduced: boolean;
  rate: number;
  discrete: boolean;
  duration: number;
  keyframes: Keyframe[];
  t: number;
  time?: (t: number) => string;
  onSeek(t: number, cause: "user" | "play"): void;
}

const L = {
  en: { play: "Play", pause: "Pause", back: "Step back", fwd: "Step forward", restart: "Restart", step: "step {i} of {n}", pos: "Position" },
  zh: { play: "播放", pause: "暂停", back: "后退一步", fwd: "前进一步", restart: "重新开始", step: "第 {i} 步，共 {n} 步", pos: "位置" },
};

export class Transport {
  root: HTMLElement;
  isPlaying = false;
  onEnd?: () => void;
  private t: number;
  private o: TransportOpts;
  private range: HTMLInputElement;
  private playBtn?: HTMLButtonElement;
  private readout: HTMLElement;
  private keyLabel: HTMLElement;
  private ticks: HTMLElement;
  private raf = 0;
  private last = 0;

  constructor(o: TransportOpts) {
    this.o = o;
    this.t = Math.min(o.t, o.duration);
    const lab = L[o.lang];
    this.root = h("div", "fig-transport");
    const btns = h("div", "fig-tr-buttons");
    const mk = (cls: string, label: string, glyph: string, fn: () => void) => {
      const b = h("button", `fig-tr-btn ${cls}`);
      b.type = "button";
      b.setAttribute("aria-label", label);
      b.title = label;
      b.innerHTML = glyph;
      b.addEventListener("click", fn);
      btns.append(b);
      return b;
    };
    mk("fig-tr-restart", lab.restart, "&#x23EE;", () => { this.pause(); this.seek(0, "user"); });
    mk("fig-tr-back", lab.back, "&#x23F4;", () => { this.pause(); this.step(-1); });
    if (!o.reduced) this.playBtn = mk("fig-tr-play", lab.play, "&#x25B6;", () => (this.isPlaying ? this.pause() : this.play()));
    mk("fig-tr-fwd", lab.fwd, "&#x23F5;", () => { this.pause(); this.step(1); });
    const scrub = h("div", "fig-tr-scrub");
    this.range = h("input");
    this.range.type = "range";
    this.range.min = "0";
    this.range.setAttribute("aria-label", lab.pos);
    this.range.addEventListener("input", () => { this.pause(); this.seek(Number(this.range.value), "user"); });
    this.ticks = h("div", "fig-tr-ticks");
    scrub.append(this.range, this.ticks);
    this.readout = h("span", "fig-tr-readout");
    this.keyLabel = h("span", "fig-tr-key");
    const info = h("div", "fig-tr-info");
    info.append(this.readout, this.keyLabel);
    this.root.append(btns, scrub, info);
    this.configure(o.duration, o.keyframes);
  }

  configure(duration: number, keyframes: Keyframe[]) {
    this.o.duration = duration;
    this.o.keyframes = keyframes;
    this.range.max = String(duration);
    this.range.step = this.o.discrete ? "1" : String(duration / 500 || 0.01);
    this.ticks.innerHTML = "";
    for (const k of keyframes) {
      const tk = h("span", "fig-tr-tick");
      tk.style.left = `${duration ? (100 * k.t) / duration : 0}%`;
      tk.title = k.label;
      this.ticks.append(tk);
    }
    if (this.t > duration) this.t = duration;
    this.paint();
  }

  shown(): number {
    return this.o.discrete ? Math.round(this.t) : this.t;
  }

  currentKey(): Keyframe | undefined {
    let cur: Keyframe | undefined;
    for (const k of this.o.keyframes) if (k.t <= this.shown() + 1e-9) cur = k;
    return cur;
  }

  seek(t: number, cause: "user" | "play") {
    this.t = Math.max(0, Math.min(this.o.duration, t));
    this.paint();
    this.o.onSeek(this.shown(), cause);
  }

  step(dir: 1 | -1) {
    const cur = this.shown();
    if (this.o.reduced && this.o.keyframes.length) {
      const ks = this.o.keyframes.map((k) => k.t);
      const next = dir > 0 ? ks.find((k) => k > cur + 1e-9) : [...ks].reverse().find((k) => k < cur - 1e-9);
      this.seek(next ?? (dir > 0 ? this.o.duration : 0), "user");
      return;
    }
    const d = this.o.discrete ? 1 : this.o.duration / 50;
    this.seek(cur + dir * d, "user");
  }

  play() {
    if (this.o.reduced) return;
    if (this.t >= this.o.duration) this.t = 0;
    this.isPlaying = true;
    this.last = performance.now();
    const tick = (now: number) => {
      if (!this.isPlaying) return;
      const dt = (now - this.last) / 1000;
      this.last = now;
      const before = this.shown();
      this.t = Math.min(this.o.duration, this.t + dt * this.o.rate);
      if (this.shown() !== before || !this.o.discrete) { this.paint(); this.o.onSeek(this.shown(), "play"); }
      if (this.t >= this.o.duration) { this.pause(); this.onEnd?.(); return; }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
    this.paint();
  }

  pause() {
    this.isPlaying = false;
    cancelAnimationFrame(this.raf);
    this.paint();
  }

  private paint() {
    const lab = L[this.o.lang];
    this.range.value = String(this.shown());
    const pos = this.o.time ? this.o.time(this.shown()) : lab.step.replace("{i}", String(Math.round(this.shown()))).replace("{n}", String(Math.round(this.o.duration)));
    this.readout.textContent = pos;
    this.range.setAttribute("aria-valuetext", pos);
    this.keyLabel.textContent = this.currentKey()?.label ?? "";
    if (this.playBtn) {
      this.playBtn.innerHTML = this.isPlaying ? "&#x23F8;" : "&#x25B6;";
      this.playBtn.setAttribute("aria-label", this.isPlaying ? lab.pause : lab.play);
      this.playBtn.title = this.isPlaying ? lab.pause : lab.play;
    }
  }
}
