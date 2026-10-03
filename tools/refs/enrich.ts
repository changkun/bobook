// Fill bibliographic metadata in refs/literature.bib from primary sources.
//
//   node tools/refs/enrich.ts [options] [refs/literature.bib]
//
//   --only arxiv,crossref,meta,search   run only these sources
//   --keys k1,k2                 only these entries
//   --force                      re-apply to entries that already have `checked`
//                                (never to checked = manual)
//   --offline                    use cached responses only, no network
//   --accept k1,k2               apply these records even though the title
//                                looks like another work (see report.json)
//   --dry                        report, do not write the .bib
//
// Sources, chosen per entry from its url and identifiers:
//
//   arxiv     The arXiv API, for entries with an `eprint`: the exact title and
//             the full author list. Batches of at most 100 ids, 3 s apart.
//   crossref  The Crossref REST API, for entries with a `doi`, a DOI readable
//             from the URL (publisher pages that embed it), or a PubMed/PMC id
//             that Europe PMC resolves to a DOI: title, authors, container,
//             volume, number, pages, year. About 5 requests per second.
//   meta      Highwire-style citation_* meta tags on the landing page (PMLR,
//             NeurIPS, JMLR, IJCAI, ACL Anthology, CVF, ...). At most 2
//             requests per second per host; a host that answers 403 or 429 is
//             skipped for the rest of the run.
//   search    For papers still unverified (an author's PDF, an OpenReview
//             forum): a Crossref, then arXiv, title search. Only a hit with
//             the same title and first author is used; its DOI or arXiv id is
//             added and the record applied as crossref or arxiv.
//
// The url and evidence fields are never changed. What was verified is recorded
// in `checked` (arxiv | crossref | meta; hand-verified entries say manual and
// are left alone). Raw responses are cached under .cache/refs/, so a rerun is
// cheap, works offline, and resumes where an interrupted run stopped. The
// .bib is rewritten after each source, and every title change that is more
// than cosmetic is listed in .cache/refs/report.json for review.

import { readFileSync, writeFileSync, existsSync, mkdirSync, renameSync } from "node:fs";
import { createHash } from "node:crypto";
import { parseBib, formatBib, people, type BibEntry } from "../../src/pipeline/bib.ts";

const args = process.argv.slice(2);
const flag = (name: string) => args.includes(`--${name}`);
const opt = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};
const FILE = args.find((a, i) => !a.startsWith("--") && !["--only", "--keys", "--accept"].includes(args[i - 1])) ?? "refs/literature.bib";
const ONLY = new Set((opt("only") ?? "arxiv,crossref,meta,search").split(","));
const KEYS = opt("keys") ? new Set(opt("keys")!.split(",")) : undefined;
const FORCE = flag("force");
const OFFLINE = flag("offline");
const DRY = flag("dry");
const CACHE = ".cache/refs";
const UA = "bobook-refs/0.1";

// ---------------------------------------------------------------------------
// Fetching: per-host spacing, retries, and a file cache.

const lastHit = new Map<string, number>();
const blockedHosts = new Set<string>();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function spaced(host: string, gapMs: number): Promise<void> {
  // Reserve the next slot before waiting so concurrent callers queue up.
  const now = Date.now();
  const next = Math.max(now, (lastHit.get(host) ?? 0) + gapMs);
  lastHit.set(host, next);
  if (next > now) await sleep(next - now);
}

interface Got { status: number; body: string; url: string }

async function get(url: string, gapMs: number, accept = "*/*"): Promise<Got> {
  const host = new URL(url).host;
  for (let attempt = 0; attempt < 4; attempt++) {
    await spaced(host, gapMs);
    try {
      const res = await fetch(url, {
        headers: { "User-Agent": UA, Accept: accept },
        redirect: "follow",
        signal: AbortSignal.timeout(60_000),
      });
      const body = await res.text();
      if (res.status === 429 || res.status >= 500) {
        const wait = Number(res.headers.get("retry-after")) * 1000 || 5000 * 2 ** attempt;
        console.warn(`  ${res.status} from ${host}, waiting ${Math.round(wait / 1000)} s`);
        if (attempt === 3) return { status: res.status, body, url: res.url };
        await sleep(wait);
        continue;
      }
      return { status: res.status, body, url: res.url };
    } catch (err) {
      console.warn(`  ${host}: ${(err as Error).message}`);
      if (attempt === 3) return { status: 0, body: "", url };
      await sleep(3000 * 2 ** attempt);
    }
  }
  return { status: 0, body: "", url };
}

function cachePath(kind: string, id: string, ext: string): string {
  const safe = id.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 120);
  const hash = createHash("sha1").update(id).digest("hex").slice(0, 8);
  return `${CACHE}/${kind}/${safe}-${hash}.${ext}`;
}

function readCache(path: string): Got | undefined {
  if (!existsSync(path)) return undefined;
  return JSON.parse(readFileSync(path, "utf8")) as Got;
}

function writeCache(path: string, got: Got): void {
  mkdirSync(path.slice(0, path.lastIndexOf("/")), { recursive: true });
  writeFileSync(path, JSON.stringify(got));
}

// Fetch through the cache. Definite answers (2xx, 404, 410) are cached;
// transient failures are not, so a rerun tries them again.
async function cachedGet(kind: string, id: string, url: string, gapMs: number, accept?: string): Promise<Got | undefined> {
  const path = cachePath(kind, id, "json");
  const hit = readCache(path);
  if (hit) return hit;
  if (OFFLINE) return undefined;
  const got = await get(url, gapMs, accept);
  if ((got.status >= 200 && got.status < 300) || got.status === 404 || got.status === 410) writeCache(path, got);
  return got;
}

// ---------------------------------------------------------------------------
// Text helpers.

const ENT: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—", rsquo: "’", lsquo: "‘", rdquo: "”", ldquo: "“", hellip: "…" };
// The Latin-1 letters by entity name (&Agrave; = U+00C0 ... &yuml; = U+00FF).
"Agrave Aacute Acirc Atilde Auml Aring AElig Ccedil Egrave Eacute Ecirc Euml Igrave Iacute Icirc Iuml ETH Ntilde Ograve Oacute Ocirc Otilde Ouml times Oslash Ugrave Uacute Ucirc Uuml Yacute THORN szlig agrave aacute acirc atilde auml aring aelig ccedil egrave eacute ecirc euml igrave iacute icirc iuml eth ntilde ograve oacute ocirc otilde ouml divide oslash ugrave uacute ucirc uuml yacute thorn yuml"
  .split(" ").forEach((n, i) => { ENT[n] = String.fromCodePoint(0xc0 + i); });

function decodeEntities(s: string): string {
  return s
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(parseInt(d, 10)))
    .replace(/&([a-z]+);/gi, (m, n) => ENT[n] ?? ENT[n.toLowerCase()] ?? m);
}

// Markup tags (<i>, </sub>, <mml:mi ...>) are removed; a bare "<" in a
// formula ("$a<b$") is not a tag and stays.
const TAG = /<\/?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?\/?>/g;

// Direction marks, zero-width spaces and soft hyphens that some records carry.
const INVISIBLE = /[\u00AD\u200B-\u200F\u202A-\u202E\u2060\uFEFF]/g;

function clean(s: string | undefined): string {
  if (!s) return "";
  let t = decodeEntities(s.replace(TAG, "")).normalize("NFC").replace(INVISIBLE, "").replace(/\s+/g, " ").trim();
  // A value must keep balanced braces or the .bib stops parsing.
  let depth = 0, ok = true;
  for (const c of t) { if (c === "{") depth++; else if (c === "}" && --depth < 0) ok = false; }
  if (!ok || depth) t = t.replace(/[{}]/g, "");
  return t;
}

function fold(s: string): string {
  return s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

function words(s: string): string[] {
  return fold(s).replace(/[^a-z0-9]+/g, " ").trim().split(" ").filter(Boolean);
}

// Dice similarity of the word sets, 0..1. Below about 0.6 the two titles are
// not the same wording of one title.
export function similarity(a: string, b: string): number {
  const A = new Set(words(a)), B = new Set(words(b));
  if (!A.size || !B.size) return 0;
  let both = 0;
  for (const w of A) if (B.has(w)) both++;
  return (2 * both) / (A.size + B.size);
}

const CJK = /[㐀-鿿]/;

// ---------------------------------------------------------------------------
// Names.

const PARTICLE = /^(van|von|der|den|de|del|della|degli|di|da|la|le|du|dos|das|ter|ten|zu|zur|af|av|bin|ibn|al)$/i;
const SUFFIX = /^(jr\.?|sr\.?|ii|iii|iv)$/i;

// Surnames the entry already used for this entry, keyed by their folded
// letters; they decide where a multi-word surname starts and catch the
// first/last swaps and dropped particles that some page metadata has.
type Hints = Map<string, string>;
const key = (s: string) => fold(s).replace(/[^a-z]/g, "");

function surnameHints(e: BibEntry): Hints {
  const out: Hints = new Map();
  const field = e.fields.author ?? "";
  for (const p of people(field).list) if (!p.corporate && p.last && !CJK.test(p.last)) out.set(key(p.last), p.last);
  // A list of bare surnames ("Bengs and El Mesaoudi-Paul") names each
  // surname whole, particles and all.
  if (!field.includes(",")) {
    for (const part of field.split(/\s+and\s+/)) {
      const t = part.trim();
      if (t && t !== "others" && !/[{}]/.test(t) && !CJK.test(t)) out.set(key(t), t);
    }
  }
  return out;
}

// The entry's spelling of a surname when the source has the same letters
// without diacritics ("Calauzenes" for Calauzènes).
function accent(last: string, hints: Hints): string {
  const orig = hints.get(key(last));
  if (!orig || orig === last) return last;
  const marks = (x: string) => /[\u0300-\u036f]/.test(x.normalize("NFKD"));
  return fold(orig) === fold(last) && marks(orig) && !marks(last) ? orig : last;
}

// "Given Middle Family" -> "Family, Given Middle"; "Family, Given" is kept.
export function bibName(raw: string, hints: Hints = new Map()): string {
  const s = clean(raw).replace(/\s*,\s*$/, "");
  if (!s) return "";
  if (s.includes(",")) {
    const [a, ...rest] = s.split(",").map((t) => t.trim());
    const b = rest.join(", ");
    if (!b) return a;
    if (hints.size && !hints.has(key(a))) {
      // Swapped halves ("Eric, Brochu").
      if (hints.has(key(b))) return `${b}, ${a}`;
      // A dropped particle ("Freitas, Nando" where the entry has de Freitas).
      for (const [k, orig] of hints) {
        if (k.length > key(a).length && k.endsWith(key(a)) && key(a).length >= 3 && orig.split(/\s+/).slice(0, -1).every((p) => PARTICLE.test(p))) return `${orig}, ${b}`;
      }
    }
    return `${accent(a, hints)}, ${b}`;
  }
  const w = s.split(/\s+/);
  let suffix = "";
  if (w.length > 2 && SUFFIX.test(w[w.length - 1])) suffix = w.pop()!;
  if (w.length === 1) return w[0];
  let j = -1;
  // The longest tail the entry already knows as a surname wins.
  for (let k = 1; k < w.length; k++) if (hints.has(key(w.slice(k).join(" ")))) { j = k; break; }
  if (j < 0) {
    j = w.length - 1;
    while (j > 1 && PARTICLE.test(w[j - 1])) j--;
  }
  const last = accent(w.slice(j).join(" "), hints), first = w.slice(0, j).join(" ");
  return suffix ? `${last}, ${suffix}, ${first}` : `${last}, ${first}`;
}

// "BROWN, ETIENNE" as some publishers deposit it -> "Brown, Etienne".
function nameCase(s: string): string {
  if (/\p{Ll}/u.test(s) || !/\p{Lu}{2}/u.test(s)) return s;
  return s.toLowerCase().replace(/(^|[\s\-'’.])(\p{L})/gu, (_, p, c) => p + c.toUpperCase());
}

function authorField(names: string[]): string {
  return names.filter(Boolean).join(" and ");
}

// ---------------------------------------------------------------------------
// Applying a record to an entry.

interface Record_ {
  source: "arxiv" | "crossref" | "meta";
  title: string;
  authors: string[]; // already in "Last, First" form
  type?: string;
  venueField?: "journal" | "booktitle" | "howpublished" | "publisher" | "institution" | "school";
  venue?: string;
  publisher?: string;
  volume?: string;
  number?: string;
  pages?: string;
  year?: string;
  years?: string[]; // other years the record supports (online, print, issued)
  series?: string;
  doi?: string;
}

interface Change { key: string; source: string; sim: number; old: string; new: string }
const report: {
  titleChanges: Change[];
  mismatches: Change[];
  yearChanges: Array<{ key: string; source: string; old: string; new: string }>;
  failures: Array<{ key: string; source: string; why: string }>;
} = { titleChanges: [], mismatches: [], yearChanges: [], failures: [] };

// Entries whose record is applied even though its title looks like another
// work, after a person has checked the pair in report.json.
const ACCEPT = new Set((opt("accept") ?? "").split(",").filter(Boolean));

// The gate every source passes: a record whose title shares too few words
// with the entry's (English, non-placeholder) title is probably a different
// work behind a wrong identifier, so it is reported and not applied.
function sameWork(e: BibEntry, r: Record_, min: number): boolean {
  if (isPlaceholderTitle(e) || ACCEPT.has(e.key)) return true;
  const sim = similarity(r.title, e.fields.title ?? "");
  if (sim >= min) return true;
  report.mismatches.push({ key: e.key, source: r.source, sim: +sim.toFixed(2), old: e.fields.title ?? "", new: r.title });
  return false;
}

const DROP_NOTES = [/Descriptive title; the full title was not recorded\.?/i, /Only title and venue verified\.?/i];
const VENUE_FIELDS = ["journal", "booktitle", "howpublished", "publisher", "institution", "school"];

function isPlaceholderTitle(e: BibEntry): boolean {
  return CJK.test(e.fields.title ?? "") || /Descriptive title/.test(e.fields.note ?? "");
}

function apply(e: BibEntry, r: Record_): void {
  const f = e.fields;
  const oldTitle = f.title ?? "";
  let title = r.title;
  // Keep the entry's casing when the source shouts the same title in capitals.
  if (title && !/[a-z]/.test(title) && similarity(title, oldTitle) > 0.9) title = oldTitle;
  if (title) {
    const sim = similarity(title, oldTitle);
    if (sim < 0.6) report.titleChanges.push({ key: e.key, source: r.source, sim: +sim.toFixed(2), old: oldTitle, new: title });
    f.title = title;
  }
  if (r.authors.length) f.author = authorField(r.authors);
  if (r.type && r.type !== e.type) {
    e.type = r.type;
  }
  if (r.venueField && r.venue) {
    for (const v of VENUE_FIELDS) if (v !== r.venueField && v !== "publisher") delete f[v];
    f[r.venueField] = r.venue;
  }
  if (r.publisher && (e.type === "book" || e.type === "incollection")) f.publisher = r.publisher;
  else if (r.venueField !== "publisher" && e.type !== "book" && e.type !== "incollection" && f.publisher && r.source === "crossref") delete f.publisher;
  for (const k of ["volume", "number", "pages", "series"] as const) if (r[k]) f[k] = r[k]!;
  // The entry's year stands when the record supports it or is within a year
  // of it: registries date an article by its online-first or early-access
  // release, a year before the issue it is cited by. An entry's year that is
  // missing or further off is replaced.
  if (r.year && f.year !== r.year && !(r.years ?? []).includes(f.year ?? "")) {
    const off = /^\d{4}$/.test(f.year ?? "") ? Math.abs(Number(f.year) - Number(r.year)) : Infinity;
    report.yearChanges.push({ key: e.key, source: r.source, old: f.year ?? "", new: off > 1 ? r.year : `${r.year} (kept ${f.year})` });
    if (off > 1) f.year = r.year;
  }
  if (r.doi && !f.doi) f.doi = r.doi;
  if (f.note) {
    let n = f.note;
    for (const re of DROP_NOTES) n = n.replace(re, "");
    n = n.replace(/^[\s.;]+|[\s;]+$/g, "").replace(/\.\s*\./g, ".");
    if (n) f.note = n; else delete f.note;
  }
  f.checked = r.source;
}

// ---------------------------------------------------------------------------
// arXiv.

function arxivId(e: BibEntry): string | undefined {
  const id = e.fields.eprint;
  return id ? id.replace(/v\d+$/, "") : undefined;
}

function parseAtom(xml: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const m of xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)) {
    const id = /<id>\s*https?:\/\/arxiv\.org\/abs\/([^<\s]+?)(?:v\d+)?\s*<\/id>/.exec(m[1]);
    if (id) out.set(id[1], m[0]);
  }
  return out;
}

function fromAtom(entryXml: string, hints: Hints): Record_ | undefined {
  const title = /<title>([\s\S]*?)<\/title>/.exec(entryXml)?.[1];
  if (!title || /^\s*Error\s*$/.test(title)) return undefined;
  const authors = [...entryXml.matchAll(/<author>\s*<name>([\s\S]*?)<\/name>/g)].map((m) => bibName(decodeEntities(m[1]), hints));
  // arXiv titles are plain text with TeX; nothing in them is markup.
  const t = decodeEntities(title).normalize("NFC").replace(/\s+/g, " ").trim();
  return { source: "arxiv", title: clean(t.replace(/</g, "\u0000")).replace(/\u0000/g, "<"), authors };
}

async function runArxiv(entries: BibEntry[]): Promise<number> {
  const todo = entries.filter((e) => arxivId(e));
  const need: string[] = [];
  for (const e of todo) if (!existsSync(cachePath("arxiv", arxivId(e)!, "xml"))) need.push(arxivId(e)!);
  const ids = [...new Set(need)];
  for (let i = 0; i < ids.length && !OFFLINE; i += 100) {
    const batch = ids.slice(i, i + 100);
    const url = `https://export.arxiv.org/api/query?id_list=${batch.join(",")}&max_results=${batch.length}`;
    console.log(`arxiv: ${i + batch.length}/${ids.length}`);
    const got = await get(url, 3000);
    if (got.status !== 200) {
      console.warn(`  arXiv batch failed with ${got.status}; ${got.body.slice(0, 200)}`);
      continue;
    }
    const found = parseAtom(got.body);
    for (const id of batch) {
      const x = found.get(id);
      if (x) { mkdirSync(`${CACHE}/arxiv`, { recursive: true }); writeFileSync(cachePath("arxiv", id, "xml"), x); }
      else report.failures.push({ key: id, source: "arxiv", why: "id not in API response" });
    }
  }
  let n = 0;
  for (const e of todo) {
    const p = cachePath("arxiv", arxivId(e)!, "xml");
    if (!existsSync(p)) continue;
    const r = fromAtom(readFileSync(p, "utf8"), surnameHints(e));
    if (!r) { report.failures.push({ key: e.key, source: "arxiv", why: "arXiv returned an error entry" }); continue; }
    if (!sameWork(e, r, 0.4)) continue;
    apply(e, r);
    n++;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Crossref.

// A DOI written into a publisher URL, or implied by it.
export function doiFromUrl(url: string): string | undefined {
  let u: URL;
  try { u = new URL(url); } catch { return undefined; }
  const host = u.host.replace(/^www\./, "");
  const path = decodeURIComponent(u.pathname + u.search);
  if (host === "doi.org" || host === "dx.doi.org") return path.slice(1);
  if (host === "nature.com") {
    // Palgrave titles hosted on nature.com (Humanities and Social Sciences
    // Communications, s41599) keep their own prefix.
    const m = /^\/articles\/([a-z0-9.\-]+)$/i.exec(u.pathname);
    return m ? `${/^s41599-/.test(m[1]) ? "10.1057" : "10.1038"}/${m[1]}` : undefined;
  }
  if (host === "elifesciences.org") {
    const m = /^\/articles\/(\d+)$/.exec(u.pathname);
    return m ? `10.7554/eLife.${m[1]}` : undefined;
  }
  if (host === "rips-irsp.com") {
    const m = /\/articles\/(10\.\d{4,9}\/[^/?#]+)/.exec(path);
    return m?.[1];
  }
  const m = /(10\.\d{4,9}\/[^\s?#&]+)/.exec(path);
  if (!m) return undefined;
  let d = m[1];
  if (host === "biorxiv.org" || host === "medrxiv.org") d = d.replace(/v\d+(\.full)?(\.pdf)?$/, "");
  return d.replace(/\.(pdf|full)$/i, "").replace(/\/$/, "");
}

// ScienceDirect blocks robots, but Crossref knows its article ids (PII).
async function piiDoi(e: BibEntry): Promise<string | undefined> {
  const pii = /sciencedirect\.com\/science\/article\/(?:abs\/)?pii\/(S?[0-9X]+)/i.exec(e.fields.url ?? "")?.[1];
  if (!pii) return undefined;
  const url = `https://api.crossref.org/works?filter=alternative-id:${pii}&rows=2`;
  const got = await cachedGet("crossref-pii", pii, url, 200, "application/json");
  if (!got || got.status !== 200) return undefined;
  const items = JSON.parse(got.body)?.message?.items ?? [];
  return items.length === 1 ? items[0].DOI : undefined;
}

async function pmcDoi(e: BibEntry): Promise<string | undefined> {
  const u = e.fields.url;
  let q: string | undefined;
  let m = /pmc\.ncbi\.nlm\.nih\.gov\/articles\/(PMC\d+)/i.exec(u) ?? /europepmc\.org\/(?:article|abstract)\/PMC\/(PMC\d+)/i.exec(u);
  if (m) q = `PMCID:${m[1]}`;
  else if ((m = /europepmc\.org\/(?:article|abstract)\/MED\/(\d+)/i.exec(u) ?? /pubmed\.ncbi\.nlm\.nih\.gov\/(\d+)/i.exec(u))) q = `EXT_ID:${m[1]} AND SRC:MED`;
  if (!q) return undefined;
  const url = `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=${encodeURIComponent(q)}&format=json&resultType=lite`;
  const got = await cachedGet("europepmc", q, url, 250, "application/json");
  if (!got || got.status !== 200) return undefined;
  const res = JSON.parse(got.body)?.resultList?.result?.[0];
  return res?.doi;
}

// Book series that Crossref lists next to the volume title of a chapter.
const SERIES = /^(Lecture Notes in|Communications in Computer|Advances in Intelligent Systems|Studies in |Springer Proceedings|Smart Innovation|IFIP Advances|Series on |Research in the Sociology of Organizations$)/i;
// Series that publish conference proceedings: a chapter there is a paper.
const PROCEEDINGS_SERIES = /^(Lecture Notes in (Computer Science|Artificial Intelligence)|Communications in Computer and Information Science|IFIP Advances)/i;

// "Oxford University PressOxford" and "WORLD SCIENTIFIC" as Crossref has them.
export function publisherName(s: string): string {
  let t = s.replace(/(Press)(?=[A-Z])[A-Za-z ,]*$/, "$1");
  if (t && !/[a-z]/.test(t)) t = t.toLowerCase().replace(/\b[a-z]/g, (c) => c.toUpperCase());
  return t;
}

function datePart(d: any): string | undefined {
  const y = d?.["date-parts"]?.[0]?.[0];
  return y ? String(y) : undefined;
}

function fromCrossref(msg: any, e: BibEntry): Record_ {
  const hints = surnameHints(e);
  let title = clean(msg.title?.[0]).replace(/''/g, '"');
  const sub = clean(msg.subtitle?.[0]);
  if (sub && !fold(title).includes(fold(sub))) title = /[?!:.]$/.test(title) ? `${title} ${sub}` : `${title}: ${sub}`;
  title = title.replace(/(?<!\.)\.$/, "");
  const authors = (msg.author ?? []).map((a: any) => {
    if (a.family) {
      const fam = nameCase(clean(a.family)), giv = nameCase(clean(a.given)), suf = clean(a.suffix);
      return [fam, suf, giv].filter(Boolean).join(", ");
    }
    if (a.name) return `{${clean(a.name)}}`;
    return a.given ? bibName(clean(a.given), hints) : "";
  });
  const containers: string[] = (msg["container-title"] ?? []).map(clean).filter(Boolean);
  const container = containers.find((c) => !SERIES.test(c)) ?? containers[0];
  const r: Record_ = {
    source: "crossref", title, authors,
    year: datePart(msg["published-print"]) ?? datePart(msg.issued),
    years: [msg["published-online"], msg["published-print"], msg.issued].map(datePart).filter((y): y is string => !!y),
  };
  if (msg.volume) r.volume = clean(msg.volume);
  if (msg.issue) r.number = clean(msg.issue);
  const page = clean(msg.page) || clean(msg["article-number"]);
  if (page) r.pages = page.replace(/\s*[-–]\s*/, "--");
  const publisher = publisherName(clean(msg.publisher));
  switch (msg.type) {
    case "journal-article":
      // Some conference proceedings are registered as journals (AAAI).
      if (container && /^Proceedings\b/.test(container) && /\bConference\b/.test(container)) { r.type = "inproceedings"; r.venueField = "booktitle"; r.venue = container; break; }
      r.type = "article"; r.venueField = "journal"; r.venue = container; break;
    case "proceedings-article":
      r.type = "inproceedings"; r.venueField = "booktitle"; r.venue = container; r.publisher = publisher; break;
    case "book-chapter": case "book-section": case "book-part": case "reference-entry":
      r.type = containers.some((c) => PROCEEDINGS_SERIES.test(c)) ? "inproceedings" : "incollection";
      r.venueField = "booktitle"; r.venue = container; r.publisher = publisher; break;
    case "book": case "monograph": case "edited-book": case "reference-book": case "book-set":
      r.type = "book"; r.venueField = "publisher"; r.venue = publisher; break;
    case "dissertation":
      r.type = "phdthesis"; r.venueField = "school"; r.venue = clean(msg.institution?.[0]?.name) || e.fields.school; break;
    case "report": case "report-component":
      r.type = "techreport"; r.venueField = "institution"; r.venue = clean(msg.institution?.[0]?.name) || publisher; break;
    case "posted-content": {
      // bioRxiv names itself as the institution and a subject as the group;
      // PsyArXiv names itself as the group.
      const where = clean(msg.institution?.[0]?.name) || clean(msg["group-title"]) || publisher;
      if (e.type === "techreport") { r.venueField = "institution"; r.venue = CJK.test(e.fields.institution ?? "") || !e.fields.institution ? where : e.fields.institution; }
      else { r.type = "misc"; r.venueField = "howpublished"; r.venue = where; }
      break;
    }
    default:
      // Leave the type and venue as the entry gave them.
      break;
  }
  if (r.venue === undefined) delete r.venueField;
  return r;
}

async function runCrossref(entries: BibEntry[]): Promise<number> {
  let n = 0, i = 0;
  for (const e of entries) {
    i++;
    if (e.fields.eprint) continue;
    let doi: string | undefined = e.fields.doi ?? doiFromUrl(e.fields.url ?? "");
    if (!doi && /pmc\.ncbi|europepmc|pubmed\.ncbi/.test(e.fields.url ?? "")) doi = await pmcDoi(e);
    if (!doi && /sciencedirect\.com/.test(e.fields.url ?? "")) doi = await piiDoi(e);
    if (!doi) continue;
    doi = doi.trim();
    const url = `https://api.crossref.org/works/${doi.split("/").map(encodeURIComponent).join("/")}`;
    const got = await cachedGet("crossref", doi.toLowerCase(), url, 200, "application/json");
    if (i % 50 === 0) console.log(`crossref: ${i}/${entries.length}`);
    if (!got) continue;
    if (got.status !== 200) { report.failures.push({ key: e.key, source: "crossref", why: `HTTP ${got.status} for ${doi}` }); continue; }
    const msg = JSON.parse(got.body).message;
    const r = fromCrossref(msg, e);
    // A DOI read from the URL needs a closer match than one the entry gave.
    if (!sameWork(e, r, e.fields.doi ? 0.4 : 0.5)) continue;
    if (!e.fields.doi) r.doi = doi;
    apply(e, r);
    n++;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Landing-page meta tags.

// Hosts whose pages carry no paper metadata, or describe a different work
// than the page itself (pith.science reviews arXiv papers and tags the page
// with the reviewed paper's metadata).
const NO_META_HOSTS = /(^|\.)(github\.com|pypi\.org|wikipedia\.org|arxiv\.org|semanticscholar\.org|openreview\.net|pith\.science)$/;

function metaTags(html: string): Map<string, string[]> {
  const out = new Map<string, string[]>();
  for (const m of html.matchAll(/<meta\b([^>]*)>/gi)) {
    const attrs: Record<string, string> = {};
    for (const a of m[1].matchAll(/([A-Za-z_:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>/]+))/g)) attrs[a[1].toLowerCase()] = a[2] ?? a[3] ?? a[4] ?? "";
    const name = (attrs.name ?? attrs.property ?? "").toLowerCase();
    if (!(name.startsWith("citation_") || name === "dcterms.modified") || attrs.content === undefined) continue;
    const v = clean(attrs.content);
    if (!v) continue;
    out.set(name, [...(out.get(name) ?? []), v]);
  }
  return out;
}

function fromMeta(tags: Map<string, string[]>, e: BibEntry): Record_ | undefined {
  const one = (k: string) => tags.get(k)?.[0];
  const title = one("citation_title");
  if (!title) return undefined;
  const hints = surnameHints(e);
  // One tag per author, or (RePEc) one tag listing them all.
  const authors = [...(tags.get("citation_author") ?? []), ...(tags.get("citation_authors") ?? [])]
    .flatMap((a) => (a.includes(";") ? a.split(";") : [a])).map((a) => bibName(a, hints)).filter(Boolean);
  const r: Record_ = { source: "meta", title, authors };
  // A RePEc working-paper listing (/p/) names a repository series, not the
  // venue the entry records.
  const series = /ideas\.repec\.org\/p\//.test(e.fields.url ?? "");
  const conf = series ? undefined : one("citation_conference_title") ?? one("citation_inbook_title");
  const journal = series ? undefined : one("citation_journal_title")?.replace(/\s*\(formerly [^)]*\)$/i, "");
  if (conf) { r.type = "inproceedings"; r.venueField = "booktitle"; r.venue = conf; }
  else if (journal) {
    if (e.type === "inproceedings") { r.venueField = "booktitle"; r.venue = journal; }
    else { r.type = "article"; r.venueField = "journal"; r.venue = journal; }
  } else if (one("citation_dissertation_institution")) {
    r.type = "phdthesis"; r.venueField = "school"; r.venue = one("citation_dissertation_institution");
  } else if (one("citation_technical_report_institution")) {
    r.type = "techreport"; r.venueField = "institution"; r.venue = one("citation_technical_report_institution");
  }
  // Software, books and other non-paper entries keep their type and venue.
  if (e.type === "misc" || e.type === "book") { delete r.type; delete r.venueField; delete r.venue; }
  const date = one("citation_publication_date") ?? one("citation_date") ?? one("citation_year") ?? one("citation_online_date");
  const y = date && /(\d{4})/.exec(date)?.[1];
  if (y) r.year = y;
  // A revised page (an encyclopedia entry) is cited by its revision year.
  const revised = one("dcterms.modified") && /(\d{4})/.exec(one("dcterms.modified")!)?.[1];
  if (revised) r.years = [revised];
  if (one("citation_volume")) r.volume = one("citation_volume");
  if (one("citation_issue")) r.number = one("citation_issue");
  // PMLR pages name the conference but not the volume, which is in the URL.
  const pmlr = /proceedings\.mlr\.press\/v(\d+)\//.exec(e.fields.url ?? "");
  if (pmlr && r.venueField === "booktitle") { r.series = "Proceedings of Machine Learning Research"; r.volume ??= pmlr[1]; }
  const fp = one("citation_firstpage"), lp = one("citation_lastpage");
  if (fp) r.pages = lp && lp !== fp ? `${fp}--${lp}` : fp;
  return r;
}

async function runMeta(entries: BibEntry[]): Promise<number> {
  const todo = entries.filter((e) => {
    if (e.fields.eprint || e.fields.doi) return false;
    if (e.fields.checked && !(FORCE && e.fields.checked === "meta")) return false;
    const u = e.fields.url ?? "";
    let host = "";
    try { host = new URL(u).host.replace(/^www\./, ""); } catch { return false; }
    if (NO_META_HOSTS.test(host)) return false;
    if (/\.pdf($|[?#])/i.test(u)) return false;
    return true;
  });
  // One queue per host, hosts in parallel.
  const byHost = new Map<string, BibEntry[]>();
  for (const e of todo) {
    const h = new URL(e.fields.url).host;
    byHost.set(h, [...(byHost.get(h) ?? []), e]);
  }
  let n = 0;
  await Promise.all([...byHost].map(async ([host, list]) => {
    for (const e of list) {
      if (blockedHosts.has(host)) { report.failures.push({ key: e.key, source: "meta", why: `${host} blocked` }); continue; }
      const got = await cachedGet("html", e.fields.url, e.fields.url, 500, "text/html");
      if (!got) continue;
      if (got.status === 403 || got.status === 429) { blockedHosts.add(host); report.failures.push({ key: e.key, source: "meta", why: `HTTP ${got.status} from ${host}` }); continue; }
      if (got.status !== 200) { report.failures.push({ key: e.key, source: "meta", why: `HTTP ${got.status}` }); continue; }
      const r = fromMeta(metaTags(got.body), e);
      if (!r || !sameWork(e, r, 0.5)) continue;
      apply(e, r);
      n++;
    }
  }));
  return n;
}

// ---------------------------------------------------------------------------
// Search: papers whose URL carries no identifier (an author's PDF, an
// OpenReview forum, a conference's virtual page). Crossref and then arXiv are
// searched by title; a hit counts only when its title is the entry's title
// word for word and its first author is the entry's first author. The hit's
// DOI or arXiv id is recorded, so later runs go straight to the record.

const PAPER = new Set(["peer-reviewed", "workshop", "preprint", "working-paper", "book"]);

function firstSurname(e: BibEntry): string {
  const p = people(e.fields.author).list[0];
  return p && !p.corporate ? key(p.last) : "";
}

function sameTitle(a: string, b: string): boolean {
  return words(a).join(" ") === words(b).join(" ");
}

async function runSearch(entries: BibEntry[]): Promise<number> {
  const todo = entries.filter((e) => !e.fields.checked && !e.fields.doi && !e.fields.eprint && PAPER.has(e.fields.evidence ?? "") && !CJK.test(e.fields.title ?? "") && firstSurname(e));
  let n = 0;
  for (const e of todo) {
    const title = e.fields.title!;
    const who = firstSurname(e);
    // Crossref first: a published version has a DOI.
    const q = `${title} ${people(e.fields.author).list[0].last}`;
    const got = await cachedGet("crossref-search", q, `https://api.crossref.org/works?query.bibliographic=${encodeURIComponent(q)}&rows=5`, 200, "application/json");
    const items: any[] = got?.status === 200 ? JSON.parse(got.body)?.message?.items ?? [] : [];
    const year = Number(e.fields.year);
    const hit = items.find((m) => {
      const t = clean(m.title?.[0]) + (m.subtitle?.[0] ? `: ${clean(m.subtitle[0])}` : "");
      if (!(sameTitle(t, title) || sameTitle(clean(m.title?.[0]), title))) return false;
      if (!key(clean(m.author?.[0]?.family ?? "")).endsWith(who)) return false;
      // The same version: a preprint server copy is not the journal paper,
      // and a record years away from the entry's is another release.
      if (m.type === "posted-content" && e.fields.evidence !== "preprint" && e.fields.evidence !== "working-paper") return false;
      const y = Number(datePart(m["published-print"]) ?? datePart(m.issued));
      return !year || !y || Math.abs(y - year) <= 1;
    });
    if (hit) {
      const r = fromCrossref(hit, e);
      r.doi = hit.DOI;
      apply(e, r);
      n++;
      continue;
    }
    // Then arXiv, for papers that never got a DOI (ICLR, workshops).
    // A contiguous run of the title's words; arXiv matches it as a phrase.
    const phrase = words(title).slice(0, 12).join(" ");
    const aq = `ti:"${phrase}"`;
    const ag = await cachedGet("arxiv-search", aq, `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(aq)}&max_results=10`, 3000, "application/atom+xml");
    if (!ag || ag.status !== 200) continue;
    for (const [id, x] of parseAtom(ag.body)) {
      const r = fromAtom(x, surnameHints(e));
      if (!r || !sameTitle(r.title, title) || !key(r.authors[0]?.split(",")[0] ?? "").endsWith(who)) continue;
      e.fields.eprint = id;
      e.fields.archiveprefix = "arXiv";
      apply(e, r);
      n++;
      break;
    }
  }
  return n;
}

// ---------------------------------------------------------------------------

function load(): { header: string; entries: BibEntry[] } {
  const src = readFileSync(FILE, "utf8");
  const header = src.slice(0, src.indexOf("@")).trim();
  return { header, entries: parseBib(src, FILE) };
}

function save(header: string, entries: BibEntry[]): void {
  if (DRY) return;
  const out = formatBib([...entries].sort((a, b) => a.key.localeCompare(b.key)).map(({ file: _f, ...e }) => e as BibEntry), header);
  parseBib(out, FILE); // never write a file the reader rejects
  writeFileSync(`${FILE}.tmp`, out);
  renameSync(`${FILE}.tmp`, FILE);
}

async function main() {
  const { header, entries } = load();
  const pick = (e: BibEntry) => (!KEYS || KEYS.has(e.key)) && e.fields.checked !== "manual" && (FORCE || !e.fields.checked);
  // url and evidence are invariant; assert it before writing.
  const frozen = new Map(entries.map((e) => [e.key, `${e.fields.url}\u0000${e.fields.evidence}`]));
  const guard = () => {
    for (const e of entries) if (frozen.get(e.key) !== `${e.fields.url}\u0000${e.fields.evidence}`) throw new Error(`url/evidence changed in ${e.key}`);
  };
  const counts: Record<string, number> = {};
  if (ONLY.has("arxiv")) { counts.arxiv = await runArxiv(entries.filter(pick)); guard(); save(header, entries); }
  if (ONLY.has("crossref")) { counts.crossref = await runCrossref(entries.filter(pick)); guard(); save(header, entries); }
  if (ONLY.has("meta")) { counts.meta = await runMeta(entries.filter(pick)); guard(); save(header, entries); }
  if (ONLY.has("search")) { counts.search = await runSearch(entries.filter(pick)); guard(); save(header, entries); }
  mkdirSync(CACHE, { recursive: true });
  writeFileSync(`${CACHE}/report.json`, JSON.stringify(report, null, 2));
  const by: Record<string, number> = {};
  for (const e of entries) { const c = e.fields.checked ?? "unchecked"; by[c] = (by[c] ?? 0) + 1; }
  console.log("applied this run:", counts);
  console.log("checked in file:", by);
  console.log(`title changes to review: ${report.titleChanges.length}; mismatches skipped: ${report.mismatches.length}; failures: ${report.failures.length} (see ${CACHE}/report.json)`);
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
