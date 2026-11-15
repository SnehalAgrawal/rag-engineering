// The reranker. Same behaviour as ../python/rerank.py: a chat model reads the question and one candidate
// chunk together and answers Yes or No; the score is the log-odds of its first token, log P(yes) - log P(no).
// It is a model-derived relevance score, not a calibrated probability.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as llm from "./llm.js";

const DATA = join(dirname(fileURLToPath(import.meta.url)), "..", "data");

export const loadPrompt = () => JSON.parse(readFileSync(join(DATA, "prompt.json"), "utf8"));

export const messages = (prompt, question, passage) => [
  { role: "system", content: prompt.system },
  { role: "user", content: prompt.user.replace("{question}", question).replace("{passage}", passage) },
];

function logsumexp(values) {
  const top = Math.max(...values);
  return top + Math.log(values.reduce((s, v) => s + Math.exp(v - top), 0));
}

// log P(yes tokens) - log P(no tokens); a label missing from the alternatives gets the lowest listed value.
export function score(top, settings) {
  const floor = Math.min(...Object.values(top));
  const pick = (labels) => {
    const found = labels.filter((t) => t in top).map((t) => top[t]);
    return found.length ? found : [floor];
  };
  return logsumexp(pick(settings.yes)) - logsumexp(pick(settings.no));
}

export async function judge(prompt, question, passage, settings) {
  const result = await llm.firstTokenLogprobs(messages(prompt, question, passage), {
    options: settings.options, topLogprobs: settings.top_logprobs,
  });
  return { ...result, score: score(result.top, settings) };
}
