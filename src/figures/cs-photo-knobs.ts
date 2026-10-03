// The six adjustments of the photo-enhancement case study by hand: one
// slider per adjustment, the edited photograph beside the original, and, for
// each adjustment, the picture with that one adjustment at either end of its
// range and the others where the reader left them. This is the manual
// baseline that preferential optimization replaces: six sliders that
// interact, explored by a person directly.

import { defineFigure, type State } from "./types.ts";
import { svg, text } from "./lib/svg.ts";
import { TYPE } from "./lib/theme.ts";
import { KNOBS, ORIGINAL, PHOTO_ASPECT, knobText, photo, settingsText, type Pt } from "./lib/cs-photo.ts";

const labels = {
  en: {
    edited: "your edit",
    original: "original",
    ends: "each adjustment at the ends of its range, the others as you set them",
    endsShort: "each adjustment at its two ends",
    describe: "The photograph edited by hand: {settings}.",
  },
  zh: {
    edited: "你编辑的版本",
    original: "原图",
    ends: "每项调整取其范围的两端，其余按你的设置",
    endsShort: "每项调整的两端",
    describe: "手动编辑的照片：{settings}。",
  },
};

const knob = (i: number) => ({ kind: "range", label: { en: KNOBS[i].label, zh: knobText(i, "zh").label }, min: 0, max: 1, default: 0.5, step: 0.01 }) as const;

const params = {
  exposure: knob(0),
  contrast: knob(1),
  saturation: knob(2),
  temperature: knob(3),
  tint: knob(4),
  shadows: knob(5),
} as const;

type P = { exposure: number; contrast: number; saturation: number; temperature: number; tint: number; shadows: number };
const toPt = (p: P): Pt => [p.exposure, p.contrast, p.saturation, p.temperature, p.tint, p.shadows];

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w;
  const narrow = w < 480;
  const pad = 8;
  const u = toPt(st.p);
  const id = (s: string) => `${st.uid}-${s}`;
  const parts: string[] = [];
  let y: number;
  if (narrow) {
    const bw = w - 2 * pad, bh = Math.round(bw / PHOTO_ASPECT);
    parts.push(text(pad, 14, L.edited, { "font-size": TYPE.small, class: "fig-t-strong" }), photo(pad, 20, bw, bh, u, id("e"), { rx: 8 }));
    const sw = Math.round(bw * 0.42), sh = Math.round(sw / PHOTO_ASPECT);
    const oy = 20 + bh + 22;
    parts.push(text(pad, oy - 6, L.original, { "font-size": TYPE.small, class: "fig-t-strong" }), photo(pad, oy, sw, sh, ORIGINAL, id("o")));
    KNOBS.forEach((k, i) => parts.push(text(pad + sw + 12, oy + 10 + i * 16, `${knobText(i, lang).label} ${k.fmt(u[i])}`, { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" })));
    y = oy + Math.max(sh, 6 * 16) + 16;
  } else {
    const bw = Math.round((w - 2 * pad) * 0.62), bh = Math.round(bw / PHOTO_ASPECT);
    parts.push(text(pad, 14, L.edited, { "font-size": TYPE.small, class: "fig-t-strong" }), photo(pad, 20, bw, bh, u, id("e"), { rx: 8 }));
    const ox = pad + bw + 16, sw = w - pad - ox, sh = Math.round(sw / PHOTO_ASPECT);
    parts.push(text(ox, 14, L.original, { "font-size": TYPE.small, class: "fig-t-strong" }), photo(ox, 20, sw, sh, ORIGINAL, id("o")));
    KNOBS.forEach((k, i) => parts.push(
      text(ox, 20 + sh + 22 + i * 17, knobText(i, lang).label, { "font-size": TYPE.small, class: "fig-t-muted" }),
      text(w - pad, 20 + sh + 22 + i * 17, k.fmt(u[i]), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num" }),
    ));
    y = 20 + Math.max(bh, sh + 22 + 6 * 17) + 26;
  }
  // Each adjustment at its two ends, the others as set.
  parts.push(text(pad, y, narrow ? L.endsShort : L.ends, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const cols = narrow ? 2 : 3;
  const gx = narrow ? 10 : 16;
  const cw = (w - 2 * pad - (cols - 1) * gx) / cols; // one column holds a pair of thumbnails
  const tw = (cw - 4) / 2, th = Math.round(tw / PHOTO_ASPECT);
  const rowH = th + 36;
  KNOBS.forEach((k, i) => {
    const cx = pad + (i % cols) * (cw + gx);
    const cy = y + 12 + Math.floor(i / cols) * rowH;
    const lo = u.slice(), hi = u.slice();
    lo[i] = 0; hi[i] = 1;
    const kt = knobText(i, lang);
    parts.push(
      text(cx, cy + 10, kt.label, { "font-size": TYPE.small, class: "fig-t-strong" }),
      photo(cx, cy + 16, tw, th, lo, id(`l${i}`), { rx: 3 }),
      photo(cx + tw + 4, cy + 16, tw, th, hi, id(`h${i}`), { rx: 3 }),
      text(cx, cy + 16 + th + 13, kt.lo, { "font-size": TYPE.small, class: "fig-t-faint" }),
      text(cx + cw, cy + 16 + th + 13, kt.hi, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
    );
  });
  const H = y + 12 + Math.ceil(KNOBS.length / cols) * rowH - 4;
  return svg(w, H, describe(st), ...parts);
}

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  return labels[lang].describe.replace("{settings}", settingsText(toPt(st.p), lang));
}

export default defineFigure({
  name: "cs-photo-knobs",
  title: { en: "The six photo adjustments, by hand", zh: "手动操作六项照片调整" },
  labels,
  params,
  hint: { en: "Move the sliders to edit the photograph. The small pictures show what each adjustment does at either end of its range.", zh: "移动滑块编辑照片。小图显示每项调整在其范围两端的效果。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
