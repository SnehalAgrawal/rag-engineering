// Ask the assistant one question, and see what the pipeline did: what was retrieved, what the gate
// decided, what the model returned, what the customer sees, and the trace. Same behaviour as ../python/ask.py.
//   node ask.js "Can I pay with PayPal?"
//   node ask.js "Do you have a phone number?" --profile basic
import { parseArgs } from "node:util";
import { Assistant, PROFILES } from "./pipeline.js";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { profile: { type: "string", default: "production" } },
});
if (positionals.length !== 1 || !(values.profile in PROFILES)) {
  console.error(`Usage: node ask.js "question" [--profile ${Object.keys(PROFILES).join("|")}]`);
  process.exit(2);
}

const result = await new Assistant(values.profile).answer(positionals[0]);
const trace = result.trace;
if (trace.retrieved.length) {
  console.log("Retrieved (cosine similarity):");
  for (const r of trace.retrieved) console.log(`  ${r.score.toFixed(4)}  ${r.id}`);
}
if (trace.gate) {
  const gate = trace.gate;
  const decision = gate.threshold === null ? "no gate" : gate.passed ? "passed" : "blocked";
  console.log(`Gate: top score ${gate.top_score.toFixed(4)}, threshold ${gate.threshold ?? "None"}: ${decision}`);
}
console.log(`\nModel output: ${result.model_output ?? "None"}`);
console.log(`\nShown to the customer (${result.status}, ${trace.outcome}): ${result.message}`);
if (result.sources.length) console.log(`Sources: ${result.sources.join(", ")}`);
console.log(`\nTrace: ${JSON.stringify(trace)}`);
