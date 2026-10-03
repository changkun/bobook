// Spacing in the Chinese edition. Markdown needs a space before a
// cross-reference or citation (`在 @sec-x 中`), but the rendered reference
// starts or ends with Chinese ("第 10 章", "（Lin 等，2022）"), and Chinese puts
// no space between characters. This pass removes a single space whose nearest
// visible neighbors on both sides are Chinese characters, or that touches
// full-width punctuation on either side, looking through tags. Other spaces
// next to Latin letters and digits stay, as the style rules want ("第 3 章",
// "使用 EUBO"). Math, code, and figures are left alone.

const WIDE = /[⺀-鿿　-〿＀-￯“”‘’]/;
const PUNCT = /[\u3000-\u303f\uff00-\uffef“”‘’]/;
const SKIP = /^<(pre|code|svg|script|style|textarea)\b/i;

export function tightenCjkSpaces(html: string): string {
  const parts = html.split(/(<[^>]+>)/);
  // For each part: is it text we may edit, and what visible text it carries.
  const editable: boolean[] = [];
  let skipTag: string | null = null, skipDepth = 0, katexDepth = 0;
  for (const p of parts) {
    if (p.startsWith("<")) {
      editable.push(false);
      if (skipTag) {
        if (new RegExp(`^<${skipTag}\\b`, "i").test(p)) skipDepth++;
        else if (new RegExp(`^</${skipTag}\\s*>`, "i").test(p) && --skipDepth === 0) skipTag = null;
      } else if (katexDepth) {
        if (/^<span\b/i.test(p)) katexDepth++;
        else if (/^<\/span\s*>/i.test(p)) katexDepth--;
      } else if (SKIP.test(p) && !/\/>$/.test(p)) {
        skipTag = SKIP.exec(p)![1]; skipDepth = 1;
      } else if (/^<span class="katex/.test(p)) katexDepth = 1;
      continue;
    }
    editable.push(!skipTag && !katexDepth);
  }
  // Visible neighbors: the last character before and the first after a
  // position, skipping tags; text inside skipped regions counts as non-Chinese.
  const visible = parts.map((p, i) => (p.startsWith("<") ? "" : editable[i] ? p : p ? "x" : ""));
  const prevChar = (i: number, k: number) => {
    if (k > 0) return parts[i][k - 1];
    for (let j = i - 1; j >= 0; j--) if (visible[j]) return visible[j].slice(-1);
    return "";
  };
  const nextChar = (i: number, k: number) => {
    if (k < parts[i].length - 1) return parts[i][k + 1];
    for (let j = i + 1; j < parts.length; j++) if (visible[j]) return visible[j][0];
    return "";
  };
  for (let i = 0; i < parts.length; i++) {
    if (!editable[i] || !parts[i].includes(" ")) continue;
    let out = "";
    const s = parts[i];
    for (let k = 0; k < s.length; k++) {
      if (s[k] === " " && s[k - 1] !== " " && s[k + 1] !== " ") {
        const a = prevChar(i, k), z = nextChar(i, k);
        // Between two Chinese characters, or next to full-width punctuation on
        // either side ("。** 2020 年" renders "。2020 年"): no space.
        if ((WIDE.test(a) && WIDE.test(z)) || PUNCT.test(a) || PUNCT.test(z)) continue;
      }
      out += s[k];
    }
    parts[i] = out;
    visible[i] = out;
  }
  return parts.join("");
}
