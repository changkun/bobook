// Screenshot pages of the built site and check them.
//
//   node tools/shots.ts en/gp/02-gp-regression.html [more pages...]
//   node tools/shots.ts --all            every page at both widths, checks only, no images
//   node tools/shots.ts --fig gp-posterior en/gp/02-gp-regression.html
//
// For each page: load it at 1280 px (light) and 390 px (dark), wait for every
// figure on the page to hydrate, and report figures that failed (data-fig-
// error), console errors, horizontal page overflow, a squeezed text column on
// a phone, KaTeX errors, and missing cross-references. Screenshots go to
// .cache/shots/. With --fig NAME, only that figure's element is captured, plus
// a static (no script) render.
// Uses the Chromium that Playwright has cached; set CHROME=/path to override.

import { createServer } from "node:http";
import { readFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { extname, join } from "node:path";
import { homedir } from "node:os";
import { chromium } from "playwright-core";

const argv = process.argv.slice(2);
const siteIdx = argv.indexOf("--site");
const SITE = siteIdx >= 0 ? argv[siteIdx + 1] : "_site";
const shotsIdx = argv.indexOf("--shots");
const OUT = shotsIdx >= 0 ? argv[shotsIdx + 1] : ".cache/shots";
const skip = new Set([siteIdx, siteIdx + 1, shotsIdx, shotsIdx + 1].filter((i, k) => (k < 2 ? siteIdx >= 0 : shotsIdx >= 0)));
const args = argv.filter((_, i) => !skip.has(i));
const all = args.includes("--all");
const figIdx = args.indexOf("--fig");
const figName = figIdx >= 0 ? args[figIdx + 1] : undefined;
const pages = args.filter((a, i) => !a.startsWith("--") && (figIdx < 0 || i !== figIdx + 1));

function chromePath(): string | undefined {
  if (process.env.CHROME) return process.env.CHROME;
  const base = join(homedir(), "Library/Caches/ms-playwright");
  if (!existsSync(base)) return undefined;
  const dirs = readdirSync(base).filter((d) => d.startsWith("chromium-")).sort().reverse();
  for (const d of dirs) {
    const p = join(base, d, "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing");
    if (existsSync(p)) return p;
    const q = join(base, d, "chrome-mac/Chromium.app/Contents/MacOS/Chromium");
    if (existsSync(q)) return q;
  }
  return undefined;
}

const TYPES: Record<string, string> = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json", ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".png": "image/png" };

function serve(): Promise<{ url: string; close(): void }> {
  return new Promise((resolve) => {
    const srv = createServer((req, res) => {
      let p = decodeURIComponent((req.url ?? "/").split("?")[0]);
      if (p.endsWith("/")) p += "index.html";
      const f = join(SITE, p);
      if (!existsSync(f) || statSync(f).isDirectory()) { res.writeHead(404); res.end("not found"); return; }
      res.writeHead(200, { "content-type": TYPES[extname(f)] ?? "application/octet-stream" });
      res.end(readFileSync(f));
    });
    srv.listen(0, () => {
      const a = srv.address();
      resolve({ url: `http://localhost:${typeof a === "object" && a ? a.port : 0}/`, close: () => srv.close() });
    });
  });
}

function allPages(dir = SITE, prefix = ""): string[] {
  const out: string[] = [];
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) { if (f !== "assets") out.push(...allPages(p, `${prefix}${f}/`)); }
    else if (f.endsWith(".html") && prefix) out.push(`${prefix}${f}`);
  }
  return out;
}

async function main() {
  const list = all ? allPages() : pages;
  if (!list.length) { console.error("usage: node tools/shots.ts <page.html ...> | --all"); process.exit(2); }
  mkdirSync(OUT, { recursive: true });
  const server = await serve();
  const browser = await chromium.launch({ executablePath: chromePath() });
  let problems = 0;
  const configs = [{ w: 1280, scheme: "light" as const }, { w: 390, scheme: "dark" as const }];
  for (const page of list) {
    for (const cfg of configs) {
      const ctx = await browser.newContext({ viewport: { width: cfg.w, height: 900 }, colorScheme: cfg.scheme, deviceScaleFactor: 1 });
      const tab = await ctx.newPage();
      const errors: string[] = [];
      tab.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
      tab.on("pageerror", (e) => errors.push(String(e)));
      await tab.goto(server.url + page, { waitUntil: "networkidle" });
      // Scroll through so every figure hydrates, then wait for them.
      await tab.evaluate(async () => {
        for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 30)); }
        window.scrollTo(0, 0);
      });
      await tab.waitForFunction(() => [...document.querySelectorAll(".fig[data-figure]")].every((f) => f.classList.contains("fig-ready") || (f as HTMLElement).dataset.figError), null, { timeout: 15000 }).catch(() => errors.push("timeout waiting for figures to hydrate"));
      const report = await tab.evaluate(() => ({
        figErrors: [...document.querySelectorAll<HTMLElement>(".fig[data-fig-error]")].map((f) => `${f.dataset.figure}: ${f.dataset.figError}`),
        overflow: document.documentElement.scrollWidth - window.innerWidth,
        // The text column should use the width it has: a layout rule that keeps
        // a hidden column on a phone squeezes the text without overflowing.
        articleWidth: Math.round(document.querySelector(".article")?.getBoundingClientRect().width ?? window.innerWidth),
        culprits: (() => {
          const out: string[] = [];
          for (const e of document.querySelectorAll<HTMLElement>("main *")) {
            const r = e.getBoundingClientRect();
            if (r.right > window.innerWidth + 1 && !e.closest(".table-wrap, .math-body, pre")) {
              const fig = e.closest<HTMLElement>(".fig[data-figure]")?.dataset.figure;
              out.push(`${fig ? `figure ${fig}: ` : ""}${e.tagName.toLowerCase()}.${(e.className && typeof e.className === "string" ? e.className : "").split(" ")[0]} right=${Math.round(r.right)}`);
            }
          }
          return [...new Set(out)].slice(0, 5);
        })(),
        katex: document.querySelectorAll(".math-error").length,
        xref: document.querySelectorAll(".xref-missing").length,
        figs: document.querySelectorAll(".fig[data-figure]").length,
      }));
      const tag = `${page} @${cfg.w} ${cfg.scheme}`;
      const issues = [
        ...report.figErrors.map((e) => `figure error ${e}`),
        ...errors.map((e) => `console: ${e.slice(0, 200)}`),
        report.overflow > 1 ? `page overflows horizontally by ${report.overflow}px (${report.culprits.join("; ")})` : "",
        report.katex ? `${report.katex} KaTeX errors` : "",
        cfg.w < 600 && report.articleWidth < cfg.w - 48 ? `text column is ${report.articleWidth}px wide in a ${cfg.w}px window` : "",
        report.xref ? `${report.xref} unresolved cross-references` : "",
      ].filter(Boolean);
      problems += issues.length;
      console.log(`${issues.length ? "✗" : "✓"} ${tag} (${report.figs} figures)${issues.map((i) => `\n    ${i}`).join("")}`);
      if (!all) {
        const base = `${OUT}/${page.replace(/[\/.]/g, "_")}-${cfg.w}-${cfg.scheme}`;
        if (figName) {
          const els = await tab.$$(`.fig[data-figure="${figName}"]`);
          for (let i = 0; i < els.length; i++) {
            await els[i].scrollIntoViewIfNeeded();
            await els[i].evaluate((e) => e.closest("figure")?.scrollIntoView({ block: "center" }));
            await (await els[i].evaluateHandle((e) => e.closest("figure") ?? e)).asElement()!.screenshot({ path: `${base}-${figName}-${i}.png` });
          }
        } else {
          // Segments of one tall viewport each, readable without stitching.
          const H = await tab.evaluate(() => document.documentElement.scrollHeight);
          const seg = cfg.w > 600 ? 1300 : 1500;
          const maxSegs = Number(process.env.SEGS ?? 40);
          for (let i = 0, y = 0; y < H && i < maxSegs; i++, y += seg) {
            await tab.screenshot({ path: `${base}-${String(i).padStart(2, "0")}.png`, fullPage: true, clip: { x: 0, y, width: cfg.w, height: Math.min(seg, H - y) } });
          }
        }
      }
      await ctx.close();
    }
    if (figName && !all) {
      // Static render: no script.
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, javaScriptEnabled: false, deviceScaleFactor: 2 });
      const tab = await ctx.newPage();
      await tab.goto(server.url + page, { waitUntil: "networkidle" });
      const els = await tab.$$(`.fig[data-figure="${figName}"]`);
      for (let i = 0; i < els.length; i++) await els[i].screenshot({ path: `${OUT}/${page.replace(/[\/.]/g, "_")}-static-${figName}-${i}.png` });
      await ctx.close();
    }
  }
  await browser.close();
  server.close();
  console.log(problems ? `${problems} problem(s)` : "no problems");
  if (problems) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
