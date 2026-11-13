// Index time: bring the index up to date with the help center. Same behaviour as ../python/index.py.
//   node index.js
// Re-embeds only the articles that are new or changed since the last build, reuses the vectors of the
// others, and drops removed articles. A different embedding model or different prefixes mean a full
// rebuild. The new index replaces ../data/index.json in one step (index_store.write), so an assistant
// that is serving keeps using the old version until the new one is complete.
import { pathToFileURL } from "node:url";
import * as store from "./index_store.js";
import { embed, embeddingModel, unload } from "./llm.js";

export async function update(helpCenterPath = store.HELP_CENTER_PATH, indexPath = store.INDEX_PATH) {
  const previous = store.read(indexPath);
  const [index, stats] = await store.build(store.readArticles(helpCenterPath), previous, store.settings(embeddingModel()),
    (texts) => embed(texts, { timeout: 300 }));
  if (previous === null || index.version !== previous.version) store.write(index, indexPath);
  stats.previous_version = previous ? previous.version : null;
  return [index, stats];
}

export function describe(index, stats) {
  const change = stats.previous_version === index.version ? "unchanged" : `was ${stats.previous_version || "none"}`;
  return `embedded ${stats.embedded}, reused ${stats.reused}, removed ${stats.removed} -> version ${index.version} (${change})`;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const start = performance.now();
  const [index, stats] = await update();
  await unload(embeddingModel());
  console.log(`${index.entries.length} articles, ${embeddingModel()} (${index.dimensions} dimensions): ` +
    `${describe(index, stats)}, ${((performance.now() - start) / 1000).toFixed(1)} s`);
}
