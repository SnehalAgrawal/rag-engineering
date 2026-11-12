// Index time: embed every help-center article once, and save the vectors. Same behaviour as ../python/index.py.
//   node index.js
// Writes ../data/index.json. Run it again whenever the articles, the embedding model, or the
// prefixes change; rag.js refuses to use an index built from different articles or settings.
import { writeFileSync } from "node:fs";
import { embed, embeddingModel, unload } from "./llm.js";
import { ARTICLES, DOCUMENT_PREFIX, INDEX_PATH, QUERY_PREFIX, documentText, fingerprint } from "./rag.js";

const start = performance.now();
const texts = ARTICLES.map(documentText);
const vectors = await embed(texts.map((t) => DOCUMENT_PREFIX + t));
await unload(embeddingModel());

const index = {
  embedding_model: embeddingModel(),
  document_prefix: DOCUMENT_PREFIX,
  query_prefix: QUERY_PREFIX,
  dimensions: vectors[0].length,
  articles: ARTICLES.map((a, i) => ({ id: a.id, sha256: fingerprint(texts[i]), vector: vectors[i] })),
};
writeFileSync(INDEX_PATH, JSON.stringify(index) + "\n");
console.log(`Indexed ${ARTICLES.length} articles with ${embeddingModel()} (${index.dimensions} dimensions) ` +
  `in ${((performance.now() - start) / 1000).toFixed(1)} s -> index.json`);
