"""
Fault injection: what does a customer see, and how long do they wait, when a model call fails?

  python faults.py

A small stand-in server sits between the assistant and Ollama. For each scenario it either forwards a
request to the real Ollama server or injects a fault on one endpoint (/api/embed or /api/chat):

  hang       accepts the request and never answers
  slow:N     waits N seconds, then forwards the request
  error      answers HTTP 500
  malformed  answers HTTP 200 with an HTML page instead of JSON
  reset      closes the connection without answering

"Ollama not running" points the assistant at a port where nothing listens. Each scenario asks the same
question through both profiles: basic (01-basic-rag's fixed timeouts: 60 s for the embedding call,
120 s for the chat call) and production (one 10-second deadline for the whole request).
Uses only the standard library for the stand-in server. The hang scenarios take about 3.5 minutes in all.
"""
import json
import select
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm import base_url, chat_model, embedding_model, unload
from pipeline import Assistant

QUESTION = "How long is the password reset link valid?"
HANG_SECONDS = 150  # longer than any timeout used here
SCENARIOS = [
    ("healthy", {}),
    ("Ollama not running", None),
    ("embedding call hangs", {"/api/embed": "hang"}),
    ("embedding call returns HTTP 500", {"/api/embed": "error"}),
    ("chat call hangs", {"/api/chat": "hang"}),
    ("chat call takes 12 s", {"/api/chat": "slow:12"}),
    ("chat call returns HTML, not JSON", {"/api/chat": "malformed"}),
    ("chat connection closed, no reply", {"/api/chat": "reset"}),
]


def client_gone(connection: socket.socket) -> bool:
    """True if the client has closed the connection (a closed socket reads as empty)."""
    readable, _, _ = select.select([connection], [], [], 0)
    return bool(readable) and connection.recv(1, socket.MSG_PEEK) == b""


class StandIn(BaseHTTPRequestHandler):
    upstream = base_url()
    faults: dict = {}

    def log_message(self, *args):  # keep the output to the results table
        pass

    def respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        fault = self.faults.get(self.path, "")
        if fault == "hang":
            time.sleep(HANG_SECONDS)
            self.close_connection = True
            return
        if fault == "reset":
            self.close_connection = True  # returning without writing closes the connection with no response
            return
        if fault == "error":
            return self.respond(500, b'{"error":"injected fault"}', "application/json")
        if fault == "malformed":
            return self.respond(200, b"<html><body>502 Bad Gateway</body></html>", "text/html")
        if fault.startswith("slow:"):
            time.sleep(float(fault.split(":")[1]))
            if client_gone(self.connection):
                return  # the client gave up: don't send Ollama work nobody waits for
        request = urllib.request.Request(self.upstream + self.path, data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as upstream:
                self.respond(upstream.status, upstream.read(), "application/json")
        except urllib.error.HTTPError as error:
            self.respond(error.code, error.read(), "application/json")
        except OSError:
            pass  # the client gave up and closed the connection


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # nothing listens here once the socket is closed


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StandIn)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    stand_in = f"http://127.0.0.1:{server.server_address[1]}"
    closed = f"http://127.0.0.1:{free_port()}"

    # Warm up both models through the real server, so the timings show the faults, not model loading.
    Assistant("production").answer(QUESTION)
    print(f"Question: {QUESTION!r}\n")
    print(f"{'scenario':<34}{'profile':<12}{'waited':>8}{'status':>8}  {'outcome':<13}failed call")
    records = []
    for name, faults in SCENARIOS:
        StandIn.faults = faults or {}
        for profile in ("basic", "production"):
            assistant = Assistant(profile, url=closed if faults is None else stand_in)
            began = time.perf_counter()
            result = assistant.answer(QUESTION)
            waited = time.perf_counter() - began
            trace = result["trace"]
            error = trace["error"]
            failed = f"{error['stage']}: {error['kind']}" if error else ""
            print(f"{name:<34}{profile:<12}{waited:>6.1f} s{result['status']:>8}  {trace['outcome']:<13}{failed}".rstrip())
            records.append({"scenario": name, "profile": profile, "message": result["message"], "trace": trace})
    server.shutdown()
    unload(chat_model())
    unload(embedding_model())

    for outcome in ("answered", "unavailable"):
        messages = sorted({r["message"] for r in records if r["trace"]["outcome"] == outcome})
        print(f"\n{outcome}: the customer saw {' / '.join(repr(m) for m in messages)}")
    with open("results-faults.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nEvery trace was saved to results-faults.jsonl.")


if __name__ == "__main__":
    main()
