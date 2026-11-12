// Ask the assistant one question, and see every step: what was retrieved, what the model
// returned, and what the customer would see. Same behaviour as ../python/ask.py.
//   node ask.js "Can I pay with PayPal?"
//   node ask.js "Can I pay with PayPal?" --k 5
//   node ask.js "Can I pay with PayPal?" --mode all
import { parseArgs } from "node:util";
import { answer } from "./rag.js";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { mode: { type: "string", default: "rag" }, k: { type: "string", default: "3" }, "num-ctx": { type: "string" } },
});
if (positionals.length !== 1 || !["rag", "all"].includes(values.mode)) {
  console.error('Usage: node ask.js "question" [--mode rag|all] [--k N] [--num-ctx N]');
  process.exit(2);
}

const numCtx = values["num-ctx"] ? Number(values["num-ctx"]) : null;
const result = await answer(positionals[0], { mode: values.mode, k: Number(values.k), numCtx });
if (result.retrieved.length) {
  console.log("Retrieved (cosine similarity):");
  for (const r of result.retrieved) console.log(`  ${r.score.toFixed(4)}  ${r.id}`);
}
console.log(`\nModel output: ${result.model_output}`);
console.log(`\nShown to the customer (${result.status}, ${result.log.outcome}): ${result.message}`);
if (result.sources.length) console.log(`Sources: ${result.sources.join(", ")}`);
console.log(`\nLog: ${JSON.stringify(result.log)}`);
