// Render the link-preview images (og:image) for each edition: the book cover
// on the left, title, subtitle, and author on the right, 1200 × 630 PNG.
// Chat apps and social sites do not render SVG previews, so these are raster
// images, committed to the repo and copied by the build. Rerun after changing
// the cover or a title:
//
//   node tools/og-image.ts            writes src/styles/og-en.png and og-zh.png

import { chromium } from "playwright-core";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parse as parseYaml } from "yaml";
import { coverArt } from "../src/cover.ts";

function chromePath(): string | undefined {
  if (process.env.CHROME) return process.env.CHROME;
  const base = join(homedir(), "Library/Caches/ms-playwright");
  if (!existsSync(base)) return undefined;
  for (const d of readdirSync(base).filter((d) => d.startsWith("chromium-")).sort().reverse()) {
    for (const p of [join(base, d, "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"), join(base, d, "chrome-mac/Chromium.app/Contents/MacOS/Chromium")]) if (existsSync(p)) return p;
  }
  return undefined;
}

const art = `data:image/svg+xml;base64,${Buffer.from(coverArt()).toString("base64")}`;

function card(lang: "en" | "zh"): string {
  const y = parseYaml(readFileSync(`${lang}/book.yml`, "utf8")) as { title: string; subtitle: string; author: string };
  const zh = lang === "zh";
  const serif = zh ? `"Noto Serif SC", "Source Serif 4", serif` : `"Source Serif 4", serif`;
  const sans = zh ? `Inter, "Noto Sans SC", sans-serif` : `Inter, sans-serif`;
  return `<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&family=Noto+Sans+SC:wght@400;600&family=Noto+Serif+SC:wght@600&display=block">
<style>
  html, body { margin: 0; }
  body { width: 1200px; height: 630px; background: linear-gradient(180deg, #15263f, #0b1524); color: #f4ecdc; display: flex; align-items: center; overflow: hidden; }
  .cover { position: relative; width: 340px; height: 510px; margin-left: 84px; flex: none; border-radius: 4px; overflow: hidden; box-shadow: 0 18px 50px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(244, 236, 220, 0.12); }
  .cover img { position: absolute; inset: 0; width: 100%; height: 100%; }
  .cover .t { position: absolute; top: 7.5%; left: 9%; right: 8%; }
  .cover .a { font: 600 12.5px/1 ${sans}; letter-spacing: 0.2em; text-transform: uppercase; opacity: 0.7; margin-bottom: 29px; }
  .cover .h { font: 600 ${zh ? 44 : 41}px/1.05 ${serif}; }
  .cover .s { font: 400 15.5px/1.35 ${sans}; opacity: 0.8; margin-top: 17px; max-width: 22ch; }
  .text { margin-left: 72px; margin-right: 72px; }
  .title { font: 600 ${zh ? 78 : 70}px/1.08 ${serif}; letter-spacing: ${zh ? "0.02em" : "-0.01em"}; }
  .sub { font: 400 ${zh ? 34 : 32}px/1.3 ${sans}; opacity: 0.85; margin-top: 22px; text-wrap: balance; }
  .by { font: 600 22px/1 ${sans}; letter-spacing: ${zh ? "0.3em" : "0.16em"}; text-transform: uppercase; opacity: 0.7; margin-top: 46px; }
  .url { font: 400 22px/1 ${sans}; opacity: 0.55; margin-top: 18px; }
</style></head><body>
<div class="cover"><img src="${art}" alt=""><div class="t"><div class="a">${y.author}</div><div class="h">${y.title}</div><div class="s">${y.subtitle}</div></div></div>
<div class="text"><div class="title">${y.title}</div><div class="sub">${y.subtitle}</div><div class="by">${y.author}</div><div class="url">changkun.de/bobook${zh ? "/zh" : ""}</div></div>
</body></html>`;
}

const browser = await chromium.launch({ executablePath: chromePath() });
for (const lang of ["en", "zh"] as const) {
  if (!existsSync(`${lang}/book.yml`)) continue;
  const page = await (await browser.newContext({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 })).newPage();
  await page.setContent(card(lang), { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: `src/styles/og-${lang}.png`, type: "png" });
  console.log(`wrote src/styles/og-${lang}.png`);
}
await browser.close();
