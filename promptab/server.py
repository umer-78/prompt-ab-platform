"""HTTP API over a set of experiments, standard library only.

GET  /experiments/<name>/assign    {"variant", "version", "template"} for one request
POST /experiments/<name>/score     {"variant", "score"}: how that request went, 0 to 1
GET  /experiments/<name>           variants, scores, drops, winner
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from .experiment import Experiment, Prompt


def load(config):
    """{name: (Experiment, {variant: Prompt})} from a YAML file (see examples/experiments.yaml)."""
    out = {}
    for name, cfg in yaml.safe_load(Path(config).read_text())["experiments"].items():
        prompts = {v: Prompt(v, t) for v, t in cfg.pop("variants").items()}
        out[name] = (Experiment(name, list(prompts), **cfg), prompts)
    return out


def make_server(experiments, host="127.0.0.1", port=8000):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def route(self, method):
            parts = self.path.split("?")[0].strip("/").split("/")
            if len(parts) not in (2, 3) or parts[0] != "experiments" or parts[1] not in experiments:
                return self.send(404, {"error": "no such experiment"})
            (ex, prompts), action = experiments[parts[1]], parts[2] if len(parts) == 3 else ""
            try:
                with lock:
                    if (method, action) == ("GET", ""):
                        return self.send(200, ex.status())
                    if (method, action) == ("GET", "assign"):
                        p = prompts[ex.assign()]
                        return self.send(200, {"variant": p.name, "version": p.version, "template": p.template})
                    if (method, action) == ("POST", "score"):
                        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                        ex.record(body["variant"], float(body["score"]))
                        return self.send(200, ex.status())
            except (KeyError, ValueError, TypeError) as e:
                return self.send(400, {"error": f"{type(e).__name__}: {e}"})
            self.send(404, {"error": "not found"})

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)
