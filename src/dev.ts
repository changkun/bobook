// Development server: build once, serve _site/ on http://localhost:4400, and
// rebuild in a fresh process whenever a chapter, a bibliography, a figure, or
// the pipeline changes. A fresh process per build means edited figure modules
// are always re-imported. The page reloads itself after each rebuild.

import { createServer } from "node:http";
import { existsSync, readFileSync, statSync, watch } from "node:fs";
import { extname, join } from "node:path";
import { spawnSync } from "node:child_process";

const PORT = Number(process.env.PORT ?? 4400);
const TYPES: Record<string, string> = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json", ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".png": "image/png" };

let version = 0;
function build() {
  const t = Date.now();
  const r = spawnSync(process.execPath, ["src/build.ts", "--lenient", "--dev"], { stdio: "inherit" });
  version++;
  console.log(r.status === 0 ? `rebuilt in ${Date.now() - t} ms` : "build failed");
}

const RELOAD = `<script>(()=>{let v=null;setInterval(async()=>{try{const r=await fetch("/__version");const n=await r.text();if(v!==null&&n!==v)location.reload();v=n}catch{}},800)})()</script>`;

build();
createServer((req, res) => {
  const url = decodeURIComponent((req.url ?? "/").split("?")[0]);
  if (url === "/__version") { res.end(String(version)); return; }
  let f = join("_site", url.endsWith("/") ? url + "index.html" : url);
  if (existsSync(f) && statSync(f).isDirectory()) f = join(f, "index.html");
  if (!existsSync(f)) { res.writeHead(404); res.end("not found"); return; }
  const type = TYPES[extname(f)] ?? "application/octet-stream";
  res.writeHead(200, { "content-type": type, "cache-control": "no-store" });
  const body = readFileSync(f);
  res.end(type.startsWith("text/html") ? body.toString().replace("</body>", `${RELOAD}</body>`) : body);
}).listen(PORT, () => console.log(`serving http://localhost:${PORT}/en/index.html`));

let timer: NodeJS.Timeout | undefined;
for (const dir of ["en", "refs", "src"]) {
  watch(dir, { recursive: true }, (_e, file) => {
    if (!file || /(^|\/)\./.test(file)) return;
    clearTimeout(timer);
    timer = setTimeout(build, 150);
  });
}
