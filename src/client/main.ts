// The page script. Everything here is an enhancement: without script the
// book reads in full, with static figures, working links, and native
// <details> for collapsed derivations and reference lists.

import { mountFigures } from "../figures/runtime/client.ts";

// The page's edition decides the words the client writes.
const LANG = document.documentElement.lang === "zh" ? "zh" : "en";
const T = LANG === "zh"
  ? { stepThrough: "逐步展开", previous: "上一步", nextStep: "下一步", showAll: "全部显示", stepOf: (i: number, n: number) => `第 ${i} 步，共 ${n} 步`, noMatches: "没有找到结果。", matches: (n: number) => `${n} 条匹配` }
  : { stepThrough: "Step through", previous: "Previous", nextStep: "Next step", showAll: "Show all", stepOf: (i: number, n: number) => `step ${i} of ${n}`, noMatches: "No matches.", matches: (n: number) => `${n} match${n === 1 ? "" : "es"}` };

const root = document.documentElement;

// Theme: follows the system unless the reader picks one.
document.querySelector(".theme-btn")?.addEventListener("click", () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("bobook-theme", root.dataset.theme); } catch { /* private mode */ }
});

// Sidebar drawer on narrow screens.
const menu = document.querySelector<HTMLButtonElement>(".menu-btn");
menu?.addEventListener("click", () => {
  const open = root.classList.toggle("nav-open");
  menu.setAttribute("aria-expanded", String(open));
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { root.classList.remove("nav-open"); closeSearch(); }
});
document.querySelector(".sidebar [aria-current]")?.scrollIntoView({ block: "center" });

// Citation cards: hovering or focusing an inline citation shows the full
// reference from the chapter's list, so a reader never has to jump away.
let card: HTMLElement | undefined;
let hideTimer = 0;
function showCard(a: HTMLAnchorElement) {
  const key = a.dataset.cite;
  const li = key ? document.getElementById(`ref-${key}`) : null;
  if (!li) return;
  clearTimeout(hideTimer);
  card ??= Object.assign(document.createElement("div"), { className: "cite-card", role: "tooltip" });
  card.innerHTML = li.innerHTML.replace(/<span class="ref-back">[\s\S]*?<\/span>\s*$/, "");
  document.body.append(card);
  const r = a.getBoundingClientRect();
  const w = Math.min(420, window.innerWidth - 24);
  card.style.width = `${w}px`;
  const left = Math.max(12, Math.min(window.scrollX + r.left + r.width / 2 - w / 2, window.scrollX + window.innerWidth - w - 12));
  card.style.left = `${left}px`;
  const below = r.bottom + 8 + card.offsetHeight < window.innerHeight;
  card.style.top = `${window.scrollY + (below ? r.bottom + 8 : r.top - card.offsetHeight - 8)}px`;
  card.classList.add("on");
}
function hideCard() {
  hideTimer = window.setTimeout(() => card?.classList.remove("on"), 120);
}
for (const a of document.querySelectorAll<HTMLAnchorElement>("a.cite-link")) {
  a.addEventListener("mouseenter", () => showCard(a));
  a.addEventListener("focus", () => showCard(a));
  a.addEventListener("mouseleave", hideCard);
  a.addEventListener("blur", hideCard);
}
document.addEventListener("mouseover", (e) => {
  if (card && (e.target as Element).closest?.(".cite-card")) clearTimeout(hideTimer);
});
document.addEventListener("mouseout", (e) => {
  if (card && (e.target as Element).closest?.(".cite-card")) hideCard();
});

// The outline highlights the section being read.
const tocLinks = [...document.querySelectorAll<HTMLAnchorElement>(".toc a")];
if (tocLinks.length && "IntersectionObserver" in window) {
  const byId = new Map(tocLinks.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
  const heads = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean) as HTMLElement[];
  const visible = new Set<string>();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id));
    let current = heads[0]?.id;
    for (const h of heads) if (h.getBoundingClientRect().top < window.innerHeight * 0.3) current = h.id;
    for (const [id, a] of byId) a.classList.toggle("active", id === current);
  }, { rootMargin: "0px 0px -60% 0px" });
  for (const h of heads) io.observe(h);
}

// Search: a small client-side index of every section.
interface Rec { h: string; t: string; s: string; x: string }
let index: Rec[] | undefined;
const panel = document.querySelector<HTMLElement>(".search-panel");
const input = panel?.querySelector("input");
const results = panel?.querySelector(".search-results");
const rootPath = (document.querySelector<HTMLLinkElement>('link[rel="stylesheet"][href*="assets/book.css"]')?.getAttribute("href") ?? "").replace(/assets\/book\.css.*$/, "");
function closeSearch() { if (panel) panel.hidden = true; }
async function openSearch() {
  if (!panel || !input) return;
  panel.hidden = false;
  input.focus();
  if (!index) index = await fetch(`${rootPath}assets/search-${LANG}.json`).then((r) => r.json()).catch(() => []);
}
document.querySelector(".search-btn")?.addEventListener("click", openSearch);
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && !(e.target as Element).closest("input, textarea")) { e.preventDefault(); openSearch(); }
});
panel?.addEventListener("click", (e) => { if (e.target === panel) closeSearch(); });
input?.addEventListener("input", () => {
  if (!index || !results) return;
  const q = input.value.trim().toLowerCase();
  const words = q.split(/\s+/).filter(Boolean);
  if (!words.length) { results.innerHTML = ""; return; }
  const scored = index.map((r) => {
    const t = r.t.toLowerCase(), s = r.s.toLowerCase(), x = r.x.toLowerCase();
    let score = 0;
    for (const w of words) {
      if (!t.includes(w) && !s.includes(w) && !x.includes(w)) return { r, score: 0 };
      score += (t.includes(w) ? 5 : 0) + (s.includes(w) ? 8 : 0) + (x.includes(w) ? 1 : 0);
    }
    return { r, score };
  }).filter((a) => a.score > 0).sort((a, b) => b.score - a.score).slice(0, 20);
  const escH = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  results.innerHTML = scored.map(({ r }) => {
    const i = r.x.toLowerCase().indexOf(words[0]);
    const snip = i >= 0 ? `…${r.x.slice(Math.max(0, i - 60), i + 120)}…` : r.x.slice(0, 160);
    return `<li><a href="${rootPath}${r.h}"><span class="sr-title">${escH(r.t)}${r.s ? ` · ${escH(r.s)}` : ""}</span><span class="sr-snip">${escH(snip)}</span></a></li>`;
  }).join("") || `<li class="sr-none">${T.noMatches}</li>`;
});

// Bibliography filter: every word typed must appear in the entry.
const bibFilter = document.querySelector<HTMLInputElement>(".bib-filter");
if (bibFilter) {
  const count = document.querySelector<HTMLElement>(".bib-count");
  const groups = [...document.querySelectorAll<HTMLElement>(".bib-group")].map((g) => ({ g, items: [...g.querySelectorAll<HTMLElement>("li")].map((li) => ({ li, text: (li.textContent ?? "").toLowerCase() })) }));
  bibFilter.addEventListener("input", () => {
    const words = bibFilter.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    for (const { g, items } of groups) {
      let any = false;
      for (const { li, text } of items) {
        const ok = words.every((w) => text.includes(w));
        li.hidden = !ok;
        if (ok) { any = true; shown++; }
      }
      g.hidden = !any;
    }
    if (count) count.textContent = words.length ? T.matches(shown) : "";
  });
}

mountFigures();

// Step-through derivations. A derivation written as a numbered list can be
// read one step at a time: later steps stay visible but faded, so the reader
// sees where the argument is going and can try the next step before revealing
// it. Without script, or before the reader opts in, every step is shown.
for (const box of document.querySelectorAll<HTMLElement>(".callout-derivation, .callout-proof")) {
  const steps = [...box.querySelectorAll<HTMLLIElement>(":scope > .callout-body > ol > li")];
  if (steps.length < 3) continue;
  const bar = document.createElement("div");
  bar.className = "deriv-bar";
  const mk = (label: string, cls = "") => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `deriv-btn ${cls}`;
    b.textContent = label;
    bar.append(b);
    return b;
  };
  const start = mk(T.stepThrough, "deriv-start");
  const prev = mk(T.previous);
  const next = mk(T.nextStep, "deriv-next");
  const all = mk(T.showAll);
  const count = document.createElement("span");
  count.className = "deriv-count";
  bar.append(count);
  let k = -1; // -1: not stepping
  const paint = () => {
    const stepping = k >= 0;
    box.classList.toggle("deriv-stepping", stepping);
    steps.forEach((li, i) => li.classList.toggle("deriv-future", stepping && i > k));
    start.hidden = stepping;
    prev.hidden = next.hidden = all.hidden = count.hidden = !stepping;
    prev.disabled = k <= 0;
    next.disabled = k >= steps.length - 1;
    count.textContent = stepping ? T.stepOf(k + 1, steps.length) : "";
  };
  start.addEventListener("click", () => { k = 0; paint(); next.focus(); });
  prev.addEventListener("click", () => { k = Math.max(0, k - 1); paint(); });
  next.addEventListener("click", () => { k = Math.min(steps.length - 1, k + 1); paint(); if (k === steps.length - 1) all.focus(); });
  all.addEventListener("click", () => { k = -1; paint(); });
  box.querySelector(".callout-head")?.append(bar);
  paint();
}
