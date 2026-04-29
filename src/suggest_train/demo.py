"""Interactive REPL for the suggest models.

Type characters and watch the top-k completions update live. Three backends
are supported; pick one with ``--model``:

  * ``lm``     — beam-searched neural LM (slowest, most interesting).
  * ``mpc``    — sorted-prefix lookup over training targets (sub-ms).
  * ``hybrid`` — RRF fusion of LM and MPC top-K (the production-shaped path).

Suggestions recompute on every keystroke. Latency for the most recent query
is shown next to the input line.

Examples
--------

    suggest-demo --model mpc \\
        --mpc-path /.../data/suggest/mpc/pooled.npz

    suggest-demo --model lm \\
        --lm-ckpt /.../checkpoints/<run>/best-*.ckpt

    suggest-demo --model hybrid \\
        --lm-ckpt /.../best-*.ckpt \\
        --mpc-path /.../mpc/pooled.npz
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import (
    BufferControl,
    FormattedTextControl,
)
from prompt_toolkit.layout.dimension import D


# --- backends ---------------------------------------------------------------

def _fmt_score(v: float) -> str:
    """Format a score for display. Units vary by backend; we branch on sign
    and magnitude: negatives are LM log-probs, ``[0, 1)`` is an RRF score,
    ``>= 1`` is an MPC event count."""
    if v < 0:
        return f"{v:+.3f}"
    if v < 1:
        return f"{v:.4f}"
    return f"{int(v):,}"


# A backend returns ``(text, score)`` tuples. The score's units differ per
# backend (LM: log-prob, MPC: event count, hybrid: RRF score) — the demo
# formats them generically.
TopKFn = Callable[[str, int], list[tuple[str, float]]]


def _build_lm_backend(
    ckpt: Path,
    tokenizer_dir: Path | None,
    beam_width: int,
    device: str,
    lev_budget: int = 0,
) -> TopKFn:
    from .infer import load_lm_searcher

    searcher = load_lm_searcher(
        ckpt_path=ckpt,
        tokenizer_dir=tokenizer_dir,
        beam_width=beam_width,
        device=device,
        lev_budget=lev_budget,
    )
    return searcher.topk_with_scores


def _build_mpc_backend(mpc_path: Path) -> TopKFn:
    from .mpc import MPC

    mpc = MPC.load(mpc_path)

    def topk(prefix: str, k: int) -> list[tuple[str, float]]:
        return [(s, float(c)) for s, c in mpc.topk_with_counts(prefix, k)]

    return topk


def _build_hybrid_backend(
    lm_fn: TopKFn,
    mpc_path: Path,
    k_each: int,
    rrf_c: float,
) -> TopKFn:
    """RRF-fuse LM and MPC top-K. Mirrors hybrid.py without the eval harness."""
    from .mpc import MPC

    mpc = MPC.load(mpc_path)

    def topk(prefix: str, k: int) -> list[tuple[str, float]]:
        lm_list = lm_fn(prefix, k_each)
        mpc_list = mpc.topk_with_counts(prefix, k_each)
        scores: dict[str, float] = {}
        for r, (cand, _) in enumerate(lm_list):
            scores[cand] = scores.get(cand, 0.0) + 1.0 / (rrf_c + r + 1)
        for r, (cand, _) in enumerate(mpc_list):
            scores[cand] = scores.get(cand, 0.0) + 1.0 / (rrf_c + r + 1)
        ordered = sorted(scores.keys(), key=lambda x: -scores[x])[:k]
        return [(c, scores[c]) for c in ordered]

    return topk


# --- UI ---------------------------------------------------------------------


class DemoApp:
    def __init__(self, topk_fn: TopKFn, k: int, label: str) -> None:
        self.topk_fn = topk_fn
        self.k = k
        self.label = label
        self.results: list[tuple[str, float]] = []
        self.last_ms: float | None = None
        self.last_prefix: str = ""
        self.in_flight: asyncio.Task | None = None
        self.dirty_prefix: str | None = None

        self.input_buffer = Buffer(
            multiline=False,
            on_text_changed=self._on_text_changed,
        )
        self.results_control = FormattedTextControl(self._render_results)
        self.status_control = FormattedTextControl(self._render_status)

        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-d")
        def _exit(event):
            event.app.exit()

        layout = Layout(
            HSplit(
                [
                    Window(self.status_control, height=1),
                    Window(
                        BufferControl(buffer=self.input_buffer),
                        height=1,
                    ),
                    Window(height=1, char="─"),
                    Window(self.results_control, height=D(min=1)),
                ]
            ),
            focused_element=self.input_buffer,
        )
        self.app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
        )

    # ---- rendering --------------------------------------------------------

    def _render_status(self):
        latency = (
            f"{self.last_ms:6.1f} ms" if self.last_ms is not None else "  --   "
        )
        return [
            ("class:label", f"[{self.label}] "),
            ("", "type to search  "),
            ("class:dim", "(Ctrl-C to quit)  "),
            ("class:latency", f"latency: {latency}"),
        ]

    def _render_results(self):
        if not self.last_prefix:
            return [("class:dim", "(start typing…)")]
        if not self.results:
            return [
                ("class:dim", f"no completions for {self.last_prefix!r}"),
            ]
        out: list[tuple[str, str]] = []
        for i, (text, score) in enumerate(self.results, 1):
            out.append(("class:rank", f"  {i:2d}. "))
            out.append(("class:score", f"{_fmt_score(score):>10}  "))
            # Highlight the prefix portion of each completion.
            if text.startswith(self.last_prefix):
                out.append(("class:prefix", self.last_prefix))
                out.append(("", text[len(self.last_prefix) :]))
            else:
                out.append(("", text))
            out.append(("", "\n"))
        return out

    # ---- input → query ----------------------------------------------------

    def _on_text_changed(self, _buffer: Buffer) -> None:
        text = self.input_buffer.text
        self.dirty_prefix = text
        if self.in_flight is None or self.in_flight.done():
            self.in_flight = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        """Drain the dirty-prefix queue, always running the most recent text.

        Each keystroke can fire while a previous query is still running.
        We don't want to spawn one task per keystroke — we want exactly one
        worker that, whenever it finishes, immediately picks up whatever the
        user has typed *since* it started. That's this loop.
        """
        while self.dirty_prefix is not None:
            prefix = self.dirty_prefix
            self.dirty_prefix = None
            if prefix == "":
                self.results = []
                self.last_prefix = ""
                self.last_ms = None
                self.app.invalidate()
                continue
            t0 = time.perf_counter()
            try:
                results = await asyncio.to_thread(
                    self.topk_fn, prefix, self.k
                )
            except Exception as e:  # noqa: BLE001 — surface in the UI
                results = [f"[error: {e!r}]"]
            self.last_ms = (time.perf_counter() - t0) * 1000
            self.last_prefix = prefix
            self.results = results
            self.app.invalidate()

    def run(self) -> None:
        self.app.run()


# --- entrypoint -------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=("lm", "mpc", "hybrid"),
        default="lm",
        help="Which backend to demo.",
    )
    p.add_argument("--lm-ckpt", type=Path, default=None)
    p.add_argument("--lm-tokenizer", type=Path, default=None)
    p.add_argument("--lm-beam-width", type=int, default=20)
    p.add_argument("--lm-device", default="cuda")
    p.add_argument(
        "--lev-budget", type=int, default=0,
        help="LM beam-search edit budget (0=exact, 1=tolerate 1 typo).",
    )
    p.add_argument("--mpc-path", type=Path, default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--hybrid-k-each", type=int, default=20,
        help="(hybrid only) candidates pulled from each backend before fusion.",
    )
    p.add_argument("--hybrid-rrf-c", type=float, default=60.0)
    args = p.parse_args()

    if args.model in ("lm", "hybrid") and args.lm_ckpt is None:
        p.error("--lm-ckpt is required for --model lm/hybrid")
    if args.model in ("mpc", "hybrid") and args.mpc_path is None:
        p.error("--mpc-path is required for --model mpc/hybrid")

    print(f"loading {args.model} backend…", flush=True)
    if args.model == "lm":
        fn = _build_lm_backend(
            args.lm_ckpt, args.lm_tokenizer,
            args.lm_beam_width, args.lm_device,
            lev_budget=args.lev_budget,
        )
        label = f"lm lev={args.lev_budget}: {args.lm_ckpt.name}"
    elif args.model == "mpc":
        fn = _build_mpc_backend(args.mpc_path)
        label = f"mpc: {args.mpc_path.name}"
    else:
        lm_fn = _build_lm_backend(
            args.lm_ckpt, args.lm_tokenizer,
            args.lm_beam_width, args.lm_device,
            lev_budget=args.lev_budget,
        )
        fn = _build_hybrid_backend(
            lm_fn, args.mpc_path,
            k_each=args.hybrid_k_each,
            rrf_c=args.hybrid_rrf_c,
        )
        label = f"hybrid lev={args.lev_budget}: {args.lm_ckpt.name} + {args.mpc_path.name}"

    DemoApp(fn, k=args.k, label=label).run()


if __name__ == "__main__":
    main()
