// The figure registry. Every module in src/figures/*.ts other than the
// infrastructure files is a figure whose name is its file name. There is no
// hand-maintained list: the build imports every module for static rendering,
// and the client bundle gets a generated map of lazy imports, so a figure is
// downloaded only on pages that embed it.

import { readdirSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import type { AnyFigure } from "./types.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const INFRA = new Set(["types.ts", "static.ts", "registry.ts"]);

export function figureFiles(): Array<{ name: string; path: string }> {
  return readdirSync(HERE)
    .filter((f) => f.endsWith(".ts") && !f.endsWith(".test.ts") && !INFRA.has(f))
    .sort()
    .map((f) => ({ name: f.replace(/\.ts$/, ""), path: join(HERE, f) }));
}

// A module that fails to load is reported and skipped, so one broken figure
// does not stop everyone else's build; pages that embed it then fail with
// "unknown figure module", and a strict build fails on the load error itself.
export async function loadFigures(errors: string[] = []): Promise<Map<string, AnyFigure>> {
  const out = new Map<string, AnyFigure>();
  for (const { name, path } of figureFiles()) {
    try {
      const mod = await import(pathToFileURL(path).href);
      const fig = mod.default as AnyFigure | undefined;
      if (!fig || typeof fig.render !== "function") throw new Error("no default figure export");
      if (fig.name !== name) throw new Error(`declares name "${fig.name}"; the name must equal the file name`);
      out.set(name, fig);
    } catch (e) {
      errors.push(`src/figures/${name}.ts: ${(e as Error).message.split("\n")[0]}`);
    }
  }
  return out;
}

// Source of the client's virtual:figure-registry module.
export function registrySource(only?: Set<string>): string {
  const lines = figureFiles().filter(({ name }) => !only || only.has(name)).map(({ name, path }) => `  ${JSON.stringify(name)}: () => import(${JSON.stringify(path)}),`);
  return `export const REGISTRY = {\n${lines.join("\n")}\n};\n`;
}
