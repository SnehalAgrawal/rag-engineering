// The index as a versioned build artifact. Same behaviour as ../python/index_store.py.
// An index is a snapshot: for every article, the text that was embedded, its SHA-256, and its vector,
// plus the settings the vectors depend on (embedding model and prefixes). Its version is a hash of
// those settings and of every article's fingerprint, so the same content always gets the same version.
//   build()     re-embeds only the articles that are new or changed, reuses the other vectors, and
//               drops articles that were removed. Different settings mean a full rebuild.
//   write()     writes to a temporary file and renames it over the old index. A reader sees either the
//               old index or the new one, never half of one.
//   staleness() counts the articles that changed in the help center since the index was built.
import { createHash } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const HELP_CENTER_PATH = fileURLToPath(new URL("../data/help_center.json", import.meta.url));
export const INDEX_PATH = fileURLToPath(new URL("../data/index.json", import.meta.url));

// nomic-embed-text was trained with task prefixes: one for questions, one for documents.
export const QUERY_PREFIX = process.env.QUERY_PREFIX ?? "search_query: ";
export const DOCUMENT_PREFIX = process.env.DOCUMENT_PREFIX ?? "search_document: ";

// What gets embedded for an article (without the prefix).
export const documentText = (article) => `${article.title}\n${article.text}`;
export const fingerprint = (text) => createHash("sha256").update(text, "utf8").digest("hex");

export const settings = (embeddingModel) => ({
  embedding_model: embeddingModel, document_prefix: DOCUMENT_PREFIX, query_prefix: QUERY_PREFIX,
});

export function versionOf(indexSettings, entries) {
  const lines = [indexSettings.embedding_model, indexSettings.document_prefix, indexSettings.query_prefix,
    ...entries.map((e) => `${e.id} ${e.sha256}`)];
  return fingerprint(lines.join("\n")).slice(0, 12);
}

export const readArticles = (path = HELP_CENTER_PATH) => JSON.parse(readFileSync(path, "utf8")).articles;
export const read = (path = INDEX_PATH) => (existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null);

// A new index for these articles, and what it took: { embedded, reused, removed }.
// embedDocuments(texts) returns one vector per text; it is called once, only for new or changed articles.
export async function build(articles, previous, indexSettings, embedDocuments) {
  let reusable = new Map();
  if (previous && Object.entries(indexSettings).every(([key, value]) => previous[key] === value)) {
    reusable = new Map(previous.entries.map((e) => [e.sha256, e.vector]));
  }
  const entries = [], toEmbed = [];
  for (const article of articles) {
    const text = documentText(article);
    const entry = { id: article.id, title: article.title, text: article.text, sha256: fingerprint(text) };
    if (reusable.has(entry.sha256)) entry.vector = reusable.get(entry.sha256);
    else toEmbed.push([entry, text]);
    entries.push(entry);
  }
  if (toEmbed.length) {
    const vectors = await embedDocuments(toEmbed.map(([, text]) => DOCUMENT_PREFIX + text));
    toEmbed.forEach(([entry], i) => { entry.vector = vectors[i]; });
  }
  const currentIds = new Set(articles.map((a) => a.id));
  const removed = previous ? previous.entries.filter((e) => !currentIds.has(e.id)).length : 0;
  const index = {
    ...indexSettings, version: versionOf(indexSettings, entries), dimensions: entries[0].vector.length, entries,
  };
  return [index, { embedded: toEmbed.length, reused: entries.length - toEmbed.length, removed }];
}

export function write(index, path = INDEX_PATH) {
  const temporary = `${path}.tmp`;
  writeFileSync(temporary, JSON.stringify(index) + "\n");
  renameSync(temporary, path); // atomic on the same file system
}

// How many articles were added, changed, or removed in the help center since this index was built.
export function staleness(index, articles) {
  const indexed = new Map(index.entries.map((e) => [e.id, e.sha256]));
  const current = new Map(articles.map((a) => [a.id, fingerprint(documentText(a))]));
  const ids = new Set([...indexed.keys(), ...current.keys()]);
  return [...ids].filter((id) => indexed.get(id) !== current.get(id)).length;
}
