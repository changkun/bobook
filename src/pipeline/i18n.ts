// The words the build puts on pages, in each edition's language. Chapter text
// is translated in the Markdown; this file covers what the pipeline writes
// itself: callout labels, cross-reference names, the page shell, reference
// lists. Chinese places the number inside the phrase ("第 3 章", "图 3.1"),
// so names that carry a number are templates, not words.

export type Lang = "en" | "zh";

const ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];
const ZH_NUM = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"];

// "Part II" / "第二部分". Parts are numbered with Roman numerals in the book.
export function partName(lang: Lang, roman: string): string {
  if (lang === "en") return `Part ${roman}`;
  const i = ROMAN.indexOf(roman);
  return `第${i > 0 ? ZH_NUM[i] : roman}部分`;
}

type Num = (n: string) => string;

export interface UI {
  callout: Record<string, string>;
  // Numbered things, as a cross-reference names them.
  kind: Record<string, Num>;
  chapter: Num;
  appendix: Num;
  section: Num;
  equation: Num;
  figure: Num;
  // Page shell.
  onThisPage: string;
  chapterNav: string;
  skip: string;
  toc: string;
  search: string;
  theme: string;
  bookContents: string;
  appendices: string;
  bibliography: string;
  status: Record<string, [string, string]>;
  untranslated: string;
  otherEdition: { label: string; title: string };
  // Reference lists.
  citedInSection: (where: string) => string;
  theIntroduction: string;
  thisSection: string;
  citedIn: string;
  references: string;
  chaptersInPart: string;
  referencesForPart: (part: string) => string;
  worksInPart: (n: number) => string;
  chapterShort: Num;
  notWritten: string;
  contents: string;
  filterBib: (n: number) => string;
  jumpToLetter: string;
  filterBibLabel: string;
  pagerLabel: (kind: string, num: string, title: string, partRoman?: string) => string;
  evidence: Record<string, string>;
  and: string;
  etAl: string; // in running text: "Lin 等人（2022）"
  etAlParen: string; // inside parentheses and lists: "（Lin 等，2022）"
  eqSuppressed: (n: string) => string; // a bare equation number, "(3.2)" / "（3.2）"
}

export const UI: Record<Lang, UI> = {
  en: {
    callout: {
      definition: "Definition", theorem: "Theorem", lemma: "Lemma", proposition: "Proposition", corollary: "Corollary",
      example: "Example", exercise: "Exercise", algorithm: "Algorithm", table: "Table", solution: "Solution",
      proof: "Proof", derivation: "Derivation", note: "Note", aside: "Going deeper", pitfall: "Pitfall",
      keyidea: "Key idea", frontier: "Research status", recap: "Recap", code: "In code",
    },
    kind: {
      fig: (n) => `Figure ${n}`, tbl: (n) => `Table ${n}`, def: (n) => `Definition ${n}`, thm: (n) => `Theorem ${n}`,
      lem: (n) => `Lemma ${n}`, prop: (n) => `Proposition ${n}`, cor: (n) => `Corollary ${n}`, ex: (n) => `Example ${n}`,
      exr: (n) => `Exercise ${n}`, alg: (n) => `Algorithm ${n}`,
    },
    chapter: (n) => `Chapter ${n}`,
    appendix: (n) => `Appendix ${n}`,
    section: (n) => `Section ${n}`,
    equation: (n) => `Equation (${n})`,
    figure: (n) => `Figure ${n}`,
    onThisPage: "On this page",
    chapterNav: "Chapter navigation",
    skip: "Skip to content",
    toc: "Table of contents",
    search: "Search the book",
    theme: "Switch light or dark theme",
    bookContents: "Book contents",
    appendices: "Appendices",
    bibliography: "Bibliography",
    status: {
      outline: ["Outline", "headings and sources only; the prose is not written yet"],
      draft: ["Draft", "complete but not yet reviewed; expect revisions"],
      review: ["In review", "being checked"],
    },
    untranslated: "",
    otherEdition: { label: "中文", title: "Read this page in Chinese" },
    citedInSection: (where) => `Sources cited in ${where}`,
    theIntroduction: "the introduction",
    thisSection: "this section",
    citedIn: "Cited in",
    references: "References",
    chaptersInPart: "Chapters in this part",
    referencesForPart: (part) => `References for ${part}`,
    worksInPart: (n) => `${n} works cited across this part's chapters.`,
    chapterShort: (n) => `Ch. ${n}`,
    notWritten: "not written yet",
    contents: "Contents",
    filterBib: (n) => `Filter ${n} works by author, title, venue, or year`,
    jumpToLetter: "Jump to letter",
    filterBibLabel: "Filter the bibliography",
    pagerLabel: (kind, num, title, roman) => `${kind === "part" ? `Part ${roman}: ` : num ? `${num}. ` : ""}${title}`,
    evidence: {
      "peer-reviewed": "", preprint: "preprint", "working-paper": "working paper", workshop: "workshop paper",
      book: "", thesis: "thesis", software: "software", other: "non-peer-reviewed",
    },
    and: "and",
    etAl: "et al.",
    etAlParen: "et al.",
    eqSuppressed: (n) => `(${n})`,
  },
  zh: {
    callout: {
      definition: "定义", theorem: "定理", lemma: "引理", proposition: "命题", corollary: "推论",
      example: "例", exercise: "习题", algorithm: "算法", table: "表", solution: "解答",
      proof: "证明", derivation: "推导", note: "注", aside: "深入一步", pitfall: "易错点",
      keyidea: "要点", frontier: "研究现状", recap: "回顾", code: "代码实现",
    },
    kind: {
      fig: (n) => `图 ${n}`, tbl: (n) => `表 ${n}`, def: (n) => `定义 ${n}`, thm: (n) => `定理 ${n}`,
      lem: (n) => `引理 ${n}`, prop: (n) => `命题 ${n}`, cor: (n) => `推论 ${n}`, ex: (n) => `例 ${n}`,
      exr: (n) => `习题 ${n}`, alg: (n) => `算法 ${n}`,
    },
    chapter: (n) => `第 ${n} 章`,
    appendix: (n) => `附录 ${n}`,
    section: (n) => `第 ${n} 节`,
    equation: (n) => `式（${n}）`,
    figure: (n) => `图 ${n}`,
    onThisPage: "本页内容",
    chapterNav: "章节导航",
    skip: "跳到正文",
    toc: "目录",
    search: "搜索本书",
    theme: "切换浅色或深色主题",
    bookContents: "全书目录",
    appendices: "附录",
    bibliography: "参考文献",
    status: {
      outline: ["提纲", "仅有标题与资料来源，正文尚未写成"],
      draft: ["草稿", "内容完整但尚未审校，可能还会修改"],
      review: ["审校中", "正在核对"],
    },
    untranslated: "本页尚未译成中文，以下为英文原文。",
    otherEdition: { label: "EN", title: "Read this page in English" },
    citedInSection: (where) => `${where}引用的文献`,
    theIntroduction: "引言",
    thisSection: "本节",
    citedIn: "引用于",
    references: "参考文献",
    chaptersInPart: "本部分各章",
    referencesForPart: (part) => `${part}参考文献`,
    worksInPart: (n) => `本部分各章共引用 ${n} 篇文献。`,
    chapterShort: (n) => `第 ${n} 章`,
    notWritten: "尚未写成",
    contents: "目录",
    filterBib: (n) => `按作者、标题、出处或年份筛选 ${n} 篇文献`,
    jumpToLetter: "按首字母跳转",
    filterBibLabel: "筛选参考文献",
    pagerLabel: (kind, num, title, roman) => `${kind === "part" ? `${partName("zh", roman ?? "")}：` : kind === "chapter" ? `第 ${num} 章 ` : kind === "appendix" ? `附录 ${num} ` : ""}${title}`,
    evidence: {
      "peer-reviewed": "", preprint: "预印本", "working-paper": "工作论文", workshop: "研讨会论文",
      book: "", thesis: "学位论文", software: "软件", other: "非同行评审",
    },
    and: "与",
    etAl: "等人",
    etAlParen: "等",
    eqSuppressed: (n) => `（${n}）`,
  },
};
