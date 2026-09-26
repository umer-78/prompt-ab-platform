"""python -m promptab replay EXPERIMENT [--allocation thompson] [--seed N]   one experiment, with its log
python -m promptab bench [--runs N]                                     all experiments, written to results/
python -m promptab serve CONFIG [--port P]                              the HTTP API"""
import argparse
import sys

from . import data
from .replay import bench, matrix, replay
from .server import load, make_server


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promptab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("experiment", help="model/task, e.g. gpt-j-6b/civil_comments")
    r.add_argument("--allocation", choices=("uniform", "thompson"), default="uniform")
    r.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bench")
    b.add_argument("--runs", type=int, default=20)
    s = sub.add_parser("serve")
    s.add_argument("config")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd == "bench":
        print(bench(args.runs))
    elif args.cmd == "serve":
        server = make_server(load(args.config), args.host, args.port)
        print(f"serving on http://{args.host}:{server.server_address[1]}")
        server.serve_forever()
    else:
        e = data.load()[args.experiment]
        names, scores = matrix(e)
        for v, mu in zip(names, scores.mean(1)):
            p = e["prompts"][v]
            print(f"  {v:13s} {100 * mu:5.1f}   {p['instructions']!r} {p['input_prefix']!r} ... {p['output_prefix']!r}")
        out = replay(names, scores, args.allocation, args.seed)
        for entry in out["log"]:
            print(f"  request {entry['at']:>6,}: {entry['event']}")
        print(f"Chose {out['winner']} ({100 * out['gap']:.1f} points below the best); shortfall over the horizon "
              f"{out['shortfall']:,.0f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
