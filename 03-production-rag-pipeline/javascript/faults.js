// Fault injection: what does a customer see, and how long do they wait, when a model call fails?
// Same behaviour and output as ../python/faults.py.
//   node faults.js
// A small stand-in server sits between the assistant and Ollama. For each scenario it either forwards a
// request to the real Ollama server or injects a fault on one endpoint (/api/embed or /api/chat):
//   hang       accepts the request and never answers
//   slow:N     waits N seconds, then forwards the request
//   error      answers HTTP 500
//   malformed  answers HTTP 200 with an HTML page instead of JSON
//   reset      closes the connection without answering
// "Ollama not running" points the assistant at a port where nothing listens. Each scenario asks the same
// question through both profiles: basic (01-basic-rag's fixed timeouts: 60 s for the embedding call,
// 120 s for the chat call) and production (one 10-second deadline for the whole request).
// Uses only Node's http module for the stand-in server. The hang scenarios take about 3.5 minutes in all.
import { writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { baseUrl, chatModel, embeddingModel, unload } from "./llm.js";
import { Assistant } from "./pipeline.js";
import { pyRepr } from "./text.js";

const QUESTION = "How long is the password reset link valid?";
const SCENARIOS = [
  ["healthy", {}],
  ["Ollama not running", null],
  ["embedding call hangs", { "/api/embed": "hang" }],
  ["embedding call returns HTTP 500", { "/api/embed": "error" }],
  ["chat call hangs", { "/api/chat": "hang" }],
  ["chat call takes 12 s", { "/api/chat": "slow:12" }],
  ["chat call returns HTML, not JSON", { "/api/chat": "malformed" }],
  ["chat connection closed, no reply", { "/api/chat": "reset" }],
];

let faults = {};
const upstream = baseUrl();
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const server = createServer(async (req, res) => {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const fault = faults[req.url] || "";
  const respond = (status, body, contentType) => {
    if (res.destroyed) return; // the client gave up and closed the connection
    res.writeHead(status, { "Content-Type": contentType, "Content-Length": Buffer.byteLength(body) }).end(body);
  };
  if (fault === "hang") return; // never answer; the client's timeout ends the request
  if (fault === "reset") return req.socket.destroy(); // close the connection with no response
  if (fault === "error") return respond(500, '{"error":"injected fault"}', "application/json");
  if (fault === "malformed") return respond(200, "<html><body>502 Bad Gateway</body></html>", "text/html");
  if (fault.startsWith("slow:")) {
    await sleep(Number(fault.split(":")[1]) * 1000);
    if (res.destroyed || req.socket.destroyed) return; // the client gave up: don't send Ollama work nobody waits for
  }
  try {
    const response = await fetch(upstream + req.url, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: Buffer.concat(chunks),
      signal: AbortSignal.timeout(300_000),
    });
    respond(response.status, Buffer.from(await response.arrayBuffer()), "application/json");
  } catch {
    if (!res.destroyed) res.destroy();
  }
});

// A port where nothing listens: open one, note it, close it.
async function freePort() {
  const probe = createServer();
  await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const { port } = probe.address();
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const standIn = `http://127.0.0.1:${server.address().port}`;
const closed = `http://127.0.0.1:${await freePort()}`;

// Warm up both models through the real server, so the timings show the faults, not model loading.
await new Assistant("production").answer(QUESTION);
console.log(`Question: ${pyRepr(QUESTION)}\n`);
console.log("scenario".padEnd(34) + "profile".padEnd(12) + "waited".padStart(8) + "status".padStart(8) + "  " + "outcome".padEnd(13) + "failed call");
const records = [];
for (const [name, scenarioFaults] of SCENARIOS) {
  faults = scenarioFaults || {};
  for (const profile of ["basic", "production"]) {
    const assistant = new Assistant(profile, { url: scenarioFaults === null ? closed : standIn });
    const began = performance.now();
    const result = await assistant.answer(QUESTION);
    const waited = (performance.now() - began) / 1000;
    const { trace } = result;
    const failed = trace.error ? `${trace.error.stage}: ${trace.error.kind}` : "";
    console.log((name.padEnd(34) + profile.padEnd(12) + waited.toFixed(1).padStart(6) + " s" + String(result.status).padStart(8) +
      "  " + trace.outcome.padEnd(13) + failed).trimEnd());
    records.push({ scenario: name, profile, message: result.message, trace });
  }
}
server.closeAllConnections();
server.close();
await unload(chatModel());
await unload(embeddingModel());

for (const outcome of ["answered", "unavailable"]) {
  const messages = [...new Set(records.filter((r) => r.trace.outcome === outcome).map((r) => r.message))].sort();
  console.log(`\n${outcome}: the customer saw ${messages.map(pyRepr).join(" / ")}`);
}
writeFileSync("results-faults.jsonl", records.map((r) => JSON.stringify(r)).join("\n") + "\n");
console.log("\nEvery trace was saved to results-faults.jsonl.");
