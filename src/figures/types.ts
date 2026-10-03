// The figure module contract. A figure is one TypeScript module that declares
// its parameters, its optional timeline, its labels, and a pure render function
// from state to SVG markup. The same module renders the static fallback at
// build time (Node) and the live figure in the browser, so render must not
// touch the DOM, the clock, or Math.random.
//
// The contract follows the figure modules of "AI as an Infrastructure", with
// two additions that Bayesian optimization figures need: a `data` parameter
// that carries a serialized list (the observations a reader has added, the
// comparisons answered so far), and a `pointer` handler that turns a click or
// a drag on the plot into new parameters.

export type Lang = "en" | "zh";

// A string in each book language. The English edition comes first; the
// Chinese twin fills `zh` later, and `tr` falls back to English until then.
export interface Text { en: string; zh?: string }

export function tr(t: Text | undefined, lang: Lang): string {
  if (!t) return "";
  return (lang === "zh" ? t.zh : undefined) ?? t.en;
}

export interface Mark { value: number; label: Text }

export interface RangeParam {
  kind: "range";
  label: Text;
  min: number;
  max: number;
  default: number;
  step?: number; // linear step; log ranges move in 1/100 of a decade
  scale?: "linear" | "log";
  unit?: Text;
  marks?: ReadonlyArray<Mark>;
  control?: boolean; // false: settable from the chapter block, no on-page control
}

export interface ChoiceParam<V extends string | number = string | number> {
  kind: "choice";
  label: Text;
  options: ReadonlyArray<{ value: V; label: Text }>;
  default: V;
  control?: boolean | "buttons" | "select";
}

export interface ToggleParam {
  kind: "toggle";
  label: Text;
  default: boolean;
  control?: boolean;
}

// A serialized value with no slider: a list of points "0.1:0.4,0.7:-0.2", a
// list of duels "3>1,2>5". The figure parses it in render; `validate` rejects
// malformed values at build time. It never has an on-page control, but a
// figure's pointer handler or buttons may change it, and the reset button
// restores the default.
export interface DataParam {
  kind: "data";
  label: Text;
  default: string;
  validate?(v: string): string | undefined; // an error message, or undefined
}

export type ParamSpec = RangeParam | ChoiceParam | ToggleParam | DataParam;
export type ParamSpecs = Record<string, ParamSpec>;

// A choice's value is typed as string or number, not as the union of its
// option literals: figures keep their state in plain records, and the
// runtime validates the value against the options.
export type ParamValue<S> = S extends RangeParam ? number
  : S extends ToggleParam ? boolean
  : S extends DataParam ? string
  : S extends ChoiceParam<infer V> ? (V extends number ? number : string)
  : never;
export type Params<D extends ParamSpecs> = { -readonly [K in keyof D]: ParamValue<D[K]> };

// What render sees. `t` is the timeline position (0 without a timeline); `w`
// is the layout width in CSS pixels, so a figure can change its composition
// for a phone column; `uid` namespaces SVG ids because a page can hold several
// renders of the same figure.
export interface State<P> {
  p: P;
  t: number;
  w: number;
  uid: string;
  // The edition's language. The runtime and the static render set it, so a
  // module's own helpers can choose `labels[st.lang ?? "en"]` without
  // threading the second argument of render and describe through.
  lang?: Lang;
}

export interface Keyframe { t: number; label: string }

export interface TimelineUnit<P> {
  symbol: Text;
  value?(t: number, p: P, lang: Lang): string;
}

export interface Timeline<P> {
  duration(p: P): number;
  rate: number; // positions per second at 1x
  discrete?: boolean;
  unit?: TimelineUnit<P>;
  keyframes(p: P, lang: Lang): Keyframe[];
  // The position the page opens on and the build renders: the most
  // informative moment, not an empty t = 0.
  poster(p: P): number;
}

// A pointer event in the SVG's own coordinates (one unit is one CSS pixel at
// the rendered width). `phase` is "click" for a tap, or "down" / "move" / "up"
// for a drag. `target` is the nearest `data-fig-hit` value under the pointer.
// During a drag, `target` and `data` refer to the element where it started.
export interface PointerEvt {
  phase: "click" | "down" | "move" | "up";
  x: number;
  y: number;
  w: number;
  target?: string;
  // Data coordinates, when the pointer is over an element that declares its
  // pixel-to-data mapping (lib/plot.ts hitArea): the figure need not redo
  // its layout to interpret a click.
  data?: { x: number; y: number };
}

// A button the figure offers beside its controls: "Next query", "Clear".
// `run` returns the new parameters (and optionally a timeline position).
export interface Action<P> {
  label: Text;
  primary?: boolean; // the figure's main action ("Next query"), drawn filled
  run(p: P): P;
  enabled?(p: P): boolean;
}

export interface Figure<D extends ParamSpecs = ParamSpecs, L extends Record<string, string> = Record<string, string>> {
  name: string; // kebab-case, equal to the module file name
  title: Text; // accessible name of the figure group
  params: D;
  labels: { en: L; zh?: L };
  timeline?: Timeline<Params<D>>;
  // Pure: SVG markup with a viewBox and no fixed pixel width.
  render(s: State<Params<D>>, lang: Lang): string;
  // Plain-text summary of the current state: the SVG's accessible name and
  // what the live region announces after a change.
  describe(s: State<Params<D>>, lang: Lang): string;
  // Adjust dependent parameters after the reader changes `key`.
  update?(p: Params<D>, key: keyof D & string): Params<D>;
  // Turn a pointer event into new parameters, or null to ignore it. Every
  // pointer interaction also needs a keyboard path (an action or a control).
  pointer?(p: Params<D>, e: PointerEvt): Params<D> | null;
  actions?: Action<Params<D>>[];
  // A short instruction under the controls: "Click the plot to add an observation."
  hint?: Text;
  // States other than the defaults whose numbers the prose quotes ("Simulate
  // all, and ..."), each reached from the defaults. The describe snapshot
  // test pins them too.
  snapshots?: Record<string, (p: Params<D>) => Params<D>>;
}

export function defineFigure<const D extends ParamSpecs, L extends Record<string, string>>(
  f: Omit<Figure<D, L>, "labels"> & { labels: { en: L; zh?: NoInfer<L> } },
): Figure<D, L> {
  return f;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyFigure = Figure<any, any>;
