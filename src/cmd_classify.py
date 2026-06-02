"""Conservative classifier for "this shell command will not terminate on its own".

Used by the bash tool to REFUSE running a long-running/non-terminating command in
the foreground (where it would wedge the whole agent turn) and tell the model to
re-run it with the `#!bg` marker instead.

Design constraints (deliberately narrow — a false positive REFUSES a command the
user may have wanted, so precision matters more than recall):
  * Only HIGH-confidence, name-based signatures (known servers, watch/follow
    flags, REPLs). No guessing behind indirection (`make`, `./script.sh`,
    `npm run <custom>`); those fall through to the 600s timeout backstop.
  * Pure function, stdlib only, fails OPEN (returns "not non-terminating") on any
    parse error so it can never itself block a command.

The companion runtime backstops are the foreground wall-clock timeout and the
optional idle-kill in src/tool_execution.py.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass

ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Command prefixes that wrap another command without changing whether it ends.
WRAPPERS = {"nohup", "setsid", "time", "exec", "command", "stdbuf", "env"}

# Programs that run until killed regardless of arguments.
SERVER_PROGS = {
    "uvicorn", "gunicorn", "daphne", "hypercorn", "vite", "nuxt", "ngrok",
    "cloudflared", "localtunnel", "lt", "http-server", "live-server", "serve",
    "nodemon", "webpack-dev-server",
}
PKG_RUNNERS = {"npm", "pnpm", "yarn", "bun"}
# Build/test tools whose --watch / -w mode never exits.
WATCHABLE = {
    "webpack", "tsc", "rollup", "esbuild", "tailwindcss", "jest", "vitest",
    "parcel", "snowpack", "wmr",
}


@dataclass
class Classification:
    non_terminating: bool
    confidence: str   # "high" | "none"
    reason: str       # human-readable, surfaced to the model
    matched: str      # the decisive command segment


def _basename(prog: str) -> str:
    return os.path.basename(prog) if prog else prog


def _has_short_flag(args, ch: str) -> bool:
    """True if any short-option token bundles `ch` (e.g. -f, -nf, -fn)."""
    for a in args:
        if a.startswith("--") or not a.startswith("-") or len(a) < 2:
            continue
        if ch in a[1:]:
            return True
    return False


def _has_long_flag(args, name: str) -> bool:
    return name in args


def _tokenize_pipeline_groups(command: str):
    """Split a command line into pipelines at top-level `&&` `||` `;` newline,
    each pipeline into `|`-separated stages, tracking a trailing `&` (background).
    Quotes and parens are respected; subshell bodies are kept opaque (one stage)."""
    pipelines = []
    stages = []
    buf = []
    quote = None
    depth = 0
    i, n = 0, len(command)

    def flush_stage():
        s = "".join(buf).strip()
        buf.clear()
        if s:
            stages.append(s)

    def flush_pipeline(bg: bool):
        flush_stage()
        if stages:
            pipelines.append({"stages": list(stages), "backgrounded": bg})
        stages.clear()

    while i < n:
        c = command[i]
        nxt = command[i + 1] if i + 1 < n else ""
        if quote:
            buf.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "\\" and nxt:
            buf.append(c)
            buf.append(nxt)
            i += 2
            continue
        if c == "(":
            depth += 1
            buf.append(c)
            i += 1
            continue
        if c == ")":
            depth = max(0, depth - 1)
            buf.append(c)
            i += 1
            continue
        if depth == 0:
            if c == "&" and nxt == "&":
                flush_pipeline(False)
                i += 2
                continue
            if c == "|" and nxt == "|":
                flush_pipeline(False)
                i += 2
                continue
            if c in (";", "\n"):
                flush_pipeline(False)
                i += 1
                continue
            if c == "&":
                flush_pipeline(True)
                i += 1
                continue
            if c == "|":
                flush_stage()
                i += 1
                continue
        buf.append(c)
        i += 1
    flush_pipeline(False)
    return pipelines


def _leaf_argv(stage: str):
    """argv of a stage with leading env-assignments and wrapper commands stripped."""
    try:
        toks = shlex.split(stage)
    except ValueError:
        toks = stage.split()
    k = 0
    while k < len(toks) and ENV_ASSIGN.match(toks[k]):
        k += 1
    toks = toks[k:]
    while toks and _basename(toks[0]) in WRAPPERS:
        w = _basename(toks[0])
        toks = toks[1:]
        if w == "env":
            while toks and ENV_ASSIGN.match(toks[0]):
                toks = toks[1:]
        elif w in ("stdbuf", "time"):
            while toks and toks[0].startswith("-"):
                toks = toks[1:]
    return toks


def _runner_serves(args) -> bool:
    if not args:
        return False
    a0 = args[0]
    if a0 == "start":
        return True
    if a0 == "run" and len(args) >= 2 and args[1] in ("dev", "serve", "start", "watch"):
        return True
    if a0 in ("dev", "serve", "watch"):  # yarn dev / pnpm serve shorthand
        return True
    return False


def _special_server(prog: str, args):
    joined = " ".join(args)
    if prog == "rails" and args[:1] and args[0] in ("server", "s"):
        return "rails server"
    if prog == "php" and "-S" in args:
        return "php -S dev server"
    if prog in ("python", "python3"):
        if "http.server" in args:
            return "python -m http.server"
        if "flask" in args and "run" in args:
            return "flask run"
        if "uvicorn" in args or "gunicorn" in args:
            return "python -m <asgi/wsgi server>"
        if "manage.py" in joined and "runserver" in args:
            return "django runserver"
    if prog == "flask" and "run" in args:
        return "flask run"
    if prog == "next" and args[:1] == ["dev"]:
        return "next dev"
    if prog == "ng" and args[:1] == ["serve"]:
        return "ng serve"
    if prog == "hugo" and "server" in args:
        return "hugo server"
    if prog == "jekyll" and "serve" in args:
        return "jekyll serve"
    if prog == "docker" and args[:1] == ["compose"] and "up" in args and not _has_detach(args):
        return "docker compose up (no -d)"
    if prog == "docker-compose" and "up" in args and not _has_detach(args):
        return "docker-compose up (no -d)"
    if prog in WATCHABLE and args[:1] == ["serve"]:  # e.g. `webpack serve`
        return f"{prog} serve"
    return None


def _has_detach(args) -> bool:
    return "-d" in args or "--detach" in args


def _watch_follow(prog: str, args):
    if prog in WATCHABLE and (_has_long_flag(args, "--watch") or _has_short_flag(args, "w")):
        return ("watch", f"`{prog} --watch` rebuilds and never exits")
    if prog == "cargo" and args[:1] == ["watch"]:
        return ("watch", "`cargo watch` runs forever")
    if prog == "tail" and (_has_short_flag(args, "f") or _has_short_flag(args, "F") or _has_long_flag(args, "--follow")):
        return ("follow", "`tail -f` follows a file forever")
    if prog == "journalctl" and (_has_short_flag(args, "f") or _has_long_flag(args, "--follow")):
        return ("follow", "`journalctl -f` streams forever")
    if prog in ("kubectl", "docker", "oc") and "logs" in args and (_has_short_flag(args, "f") or _has_long_flag(args, "--follow")):
        return ("follow", f"`{prog} logs -f` streams forever")
    if prog == "watch":
        return ("follow", "`watch` repeats forever")
    if prog == "ping" and not _has_short_flag(args, "c") and not _has_long_flag(args, "--count"):
        return ("follow", "`ping` without -c runs forever")
    if prog == "ssh" and _has_short_flag(args, "N"):
        return ("follow", "`ssh -N` holds a tunnel/forward open")
    return (None, None)


def _repl(prog: str, args):
    nonflag = [a for a in args if not a.startswith("-")]
    if prog in ("python", "python3"):
        if "-m" in args or any(a == "-c" for a in args) or nonflag:
            return None
        return "an interactive Python REPL (stdin is not a TTY here, so it hangs)"
    if prog == "node":
        if "-e" in args or "--eval" in args or "-p" in args or "--print" in args or nonflag:
            return None
        return "an interactive Node REPL"
    if prog in ("bash", "sh", "zsh", "fish"):
        if "-c" in args or nonflag:
            return None
        return f"an interactive {prog} shell"
    if prog == "irb":
        return "an interactive Ruby REPL"
    if prog == "redis-cli":
        return None if nonflag else "an interactive redis-cli session"
    if prog in ("psql", "mysql"):
        if "-c" in args or "-e" in args or "-f" in args or "--command" in args or "--execute" in args:
            return None
        return f"an interactive {prog} session"
    if prog == "sqlite3":
        return None if len(nonflag) >= 2 else "an interactive sqlite3 session"
    if prog in ("mongo", "mongosh"):
        return None if "--eval" in args else f"an interactive {prog} session"
    return None


def _leaf_kind(toks):
    """(kind, reason) for a single command's argv, or (None, None)."""
    if not toks:
        return (None, None)
    prog = _basename(toks[0])
    args = toks[1:]
    if prog in SERVER_PROGS:
        return ("server", f"`{prog}` is a long-running server/process")
    if prog in PKG_RUNNERS and _runner_serves(args):
        return ("server", f"`{prog} {args[0]}` starts a dev server/watcher")
    s = _special_server(prog, args)
    if s:
        return ("server", f"`{s}` runs until killed")
    kind, reason = _watch_follow(prog, args)
    if kind:
        return (kind, reason)
    r = _repl(prog, args)
    if r:
        return ("repl", r)
    return (None, None)


def _is_bounded_consumer(toks) -> bool:
    """A pipe consumer that closes the pipe early, terminating an infinite producer."""
    if not toks:
        return False
    prog = _basename(toks[0])
    args = toks[1:]
    if prog == "head":
        return True
    if prog == "grep" and (
        _has_short_flag(args, "m") or _has_long_flag(args, "--max-count")
        or _has_short_flag(args, "q") or _has_long_flag(args, "--quiet")
        or _has_long_flag(args, "--silent")
    ):
        return True
    return False


def _pipeline_blocks(stages):
    leaves = [_leaf_argv(s) for s in stages]
    kinds = [_leaf_kind(lv) for lv in leaves]
    # A REPL/interactive program reads stdin. As a non-first pipe stage its stdin
    # is the upstream pipe, so it consumes that input and exits at EOF — not an
    # interactive hang (e.g. `curl ... | sh`). Only flag REPLs in the lead stage.
    for idx in range(1, len(kinds)):
        if kinds[idx][0] == "repl":
            kinds[idx] = (None, None)
    last_kind, last_reason = kinds[-1]
    if last_kind:
        return last_reason
    # A non-terminating PRODUCER earlier in the pipe still wedges unless a
    # downstream stage closes the pipe (head, grep -m/-q).
    for idx in range(len(stages) - 1):
        kind, reason = kinds[idx]
        if kind:
            if not any(_is_bounded_consumer(leaves[j]) for j in range(idx + 1, len(stages))):
                return reason
    return None


def classify(command: str) -> Classification:
    if not command or not command.strip():
        return Classification(False, "none", "", "")
    try:
        pipelines = _tokenize_pipeline_groups(command)
        for pl in pipelines:
            if pl["backgrounded"] or not pl["stages"]:
                continue
            reason = _pipeline_blocks(pl["stages"])
            if reason:
                return Classification(True, "high", reason, pl["stages"][-1])
    except Exception:
        # Fail open: never let a classifier bug block a command.
        return Classification(False, "none", "", "")
    return Classification(False, "none", "", "")
