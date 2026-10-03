declare module "virtual:figure-registry" {
  import type { AnyFigure } from "./figures/types.ts";
  export const REGISTRY: Record<string, () => Promise<{ default: AnyFigure }>>;
}
declare module "markdown-it-attrs";
