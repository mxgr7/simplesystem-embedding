"""Inference for the suggest LM.

Beam search with byte-level prefix constraint
---------------------------------------------
We start from ``<s>`` and at each step:

  1. Run the model on every active beam in parallel.
  2. Take the top-``beam_width`` tokens per beam by log-probability.
  3. For each candidate ``(beam, token)``:
       - decode the surface form of the partial sequence,
       - keep it only if the surface still satisfies the prefix constraint
         (``prefix.startswith(decoded)`` while we're below the prefix, or
         ``decoded.startswith(prefix)`` once we've already passed it).
  4. Move ``</s>``-emitting candidates to the finished pool.
  5. Keep the top ``beam_width`` survivors as the new active beam set.

We expand until all beams finish or until ``max_steps`` is reached. Final
ranking is by raw cumulative log-probability (no length normalisation —
short, popular completions are exactly what we want to favour).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .data import TOKENIZER_DIR
from .tokenizer import Tokenizer


class _Beam:
    __slots__ = ("token_ids", "log_prob", "surface", "dp_row", "prefix_matched")

    def __init__(
        self,
        token_ids: list[int],
        log_prob: float,
        surface: str = "",
        dp_row: list[int] | None = None,
        prefix_matched: bool = False,
    ) -> None:
        self.token_ids = token_ids
        self.log_prob = log_prob
        # Cached decoded surface (the completion-region string).
        self.surface = surface
        # Wagner-Fischer one-row vector. dp_row[j] = Levenshtein(surface,
        # prefix[:j]). Only populated when lev_budget > 0; None otherwise.
        self.dp_row = dp_row
        # Sticky flag: True once the full prefix has been matched within
        # the edit budget. Past this point the beam is free-running and
        # the constraint is dropped.
        self.prefix_matched = prefix_matched


def _advance_dp(
    prefix: str, dp_row: list[int], delta: str, budget: int
) -> tuple[list[int], bool]:
    """Advance a Wagner-Fischer DP row through ``delta`` characters.

    Returns ``(new_row, matched)``. ``matched`` is True iff at any point
    during the advance ``new_row[len(prefix)] <= budget`` — i.e. the full
    prefix has been matched within the edit budget. We stop early in that
    case because the beam exits the constraint regime.
    """
    n = len(prefix)
    cur = list(dp_row)
    if cur[n] <= budget:
        return cur, True
    for c in delta:
        new = [cur[0] + 1] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if c == prefix[j - 1] else 1
            new[j] = min(
                cur[j] + 1,        # delete c from surface
                new[j - 1] + 1,    # insert prefix[j-1]
                cur[j - 1] + cost, # match / substitute
            )
        cur = new
        if cur[n] <= budget:
            return cur, True
    return cur, False


def _load_lit_module(ckpt_path: Path, device: str):
    """Reconstruct the LightningModule from a checkpoint.

    We bypass Lightning's auto-loader (which struggles with our DictConfig
    constructor) and rebuild from the saved ``hyper_parameters`` dict.
    """
    from .train import SuggestLMModule  # local import to dodge cycles

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["hyper_parameters"])
    module = SuggestLMModule(cfg)
    module.load_state_dict(ckpt["state_dict"])
    module.eval()
    module._cfg_variant = str(cfg.data.get("variant", "a")).lower()
    return module.to(device)


class SuggestBeamSearcher:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        beam_width: int = 20,
        max_steps: int = 60,
        candidates_per_beam: int = 200,
        prefix_batch_size: int = 64,
        device: str = "cuda",
        variant: str = "a",
        lev_budget: int = 0,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.beam_width = int(beam_width)
        self.max_steps = int(max_steps)
        self.candidates_per_beam = int(candidates_per_beam)
        self.prefix_batch_size = int(prefix_batch_size)
        self.device = device
        self.variant = str(variant).lower()
        if self.variant not in ("a", "b"):
            raise ValueError(f"Unknown variant {self.variant!r}")
        self.lev_budget = int(lev_budget)
        if self.lev_budget < 0:
            raise ValueError("lev_budget must be >= 0")

    def _initial_dp_row(self, prefix: str) -> list[int] | None:
        if self.lev_budget == 0:
            return None
        return list(range(len(prefix) + 1))

    def _check_extend(
        self, parent: "_Beam", decoded: str, prefix: str
    ) -> tuple[bool, list[int] | None, bool]:
        """Decide whether a candidate extension survives the prefix constraint.

        Returns (keep, new_dp_row, new_prefix_matched). The DP row is None
        when ``lev_budget == 0`` — we use the cheap exact predicate then.
        """
        if parent.prefix_matched:
            return True, parent.dp_row, True
        if self.lev_budget == 0:
            if prefix.startswith(decoded):
                return True, None, False
            if decoded.startswith(prefix):
                return True, None, True
            return False, None, False
        delta = decoded[len(parent.surface):]
        new_dp, matched = _advance_dp(
            prefix, parent.dp_row, delta, self.lev_budget
        )
        if matched:
            return True, new_dp, True
        if min(new_dp) <= self.lev_budget:
            return True, new_dp, False
        return False, None, False

    def _accept_completion(
        self, parent: "_Beam", surface: str, prefix: str
    ) -> bool:
        """Decide whether an EOS (or context-limit) emission yields a valid
        completion for this beam."""
        if self.lev_budget == 0:
            return surface.startswith(prefix) and len(surface) > len(prefix)
        return parent.prefix_matched and len(surface) > len(prefix) - self.lev_budget

    def _initial_state(self, prefix: str) -> tuple[list[int], int]:
        """Returns (initial token_ids, completion_offset).

        ``completion_offset`` is the index into ``token_ids`` from which
        the surface (decoded text) is computed. Newly generated tokens
        are appended after this offset.
        """
        bos = self.tokenizer.bos_id
        if self.variant == "a":
            return [bos], 1  # surface decoded from token_ids[1:]
        # variant 'b': BOS prefix... SEP, decode from token_ids after SEP.
        sep = self.tokenizer.sep_id
        p_ids = self.tokenizer.sp.encode(prefix, out_type=int)
        return [bos, *p_ids, sep], 2 + len(p_ids)

    @torch.inference_mode()
    def topk_with_scores(
        self, prefix: str, k: int
    ) -> list[tuple[str, float]]:
        if not prefix:
            return []
        eos = self.tokenizer.eos_id
        pad = self.tokenizer.pad_id
        beam_width = self.beam_width
        per_beam = min(self.candidates_per_beam, self.tokenizer.vocab_size)

        max_len = int(getattr(self.model.cfg, "max_seq_len", 64))
        init_ids, completion_offset = self._initial_state(prefix)
        if len(init_ids) >= max_len:
            return []
        # Active beams (surface still being built / extended).
        active: list[_Beam] = [
            _Beam(
                list(init_ids), 0.0,
                surface="",
                dp_row=self._initial_dp_row(prefix),
                prefix_matched=False,
            )
        ]
        finished: list[tuple[float, str]] = []

        for _step in range(self.max_steps):
            if not active:
                break

            # `pad_len` is the right-pad width for the current batch; distinct
            # from `max_len`, which is the model's context window.
            pad_len = max(len(b.token_ids) for b in active)
            n = len(active)
            input_ids = torch.full(
                (n, pad_len), pad, dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros(
                (n, pad_len), dtype=torch.long, device=self.device
            )
            for i, beam in enumerate(active):
                t = len(beam.token_ids)
                input_ids[i, :t] = torch.tensor(
                    beam.token_ids, dtype=torch.long, device=self.device
                )
                attention_mask[i, :t] = 1

            logits = self.model(input_ids, attention_mask=attention_mask)
            last_idx = attention_mask.sum(dim=1) - 1
            last_logits = logits[torch.arange(n, device=self.device), last_idx]
            log_probs = F.log_softmax(last_logits.float(), dim=-1)

            # Pull a wide top-N per beam so we have plenty of candidates to
            # filter against the prefix constraint. `per_beam` defaults to
            # ~200, which is enough to escape the EOS-trap in early-training
            # checkpoints without the cost of scanning the full vocab.
            top_lp, top_idx = log_probs.topk(per_beam, dim=-1)
            top_lp_cpu = top_lp.cpu().tolist()
            top_idx_cpu = top_idx.cpu().tolist()

            new_active: list[_Beam] = []
            # Process beams in interleaved best-first order so we don't
            # pour all of one beam's children into the survivor set before
            # reaching the next beam.
            ranked_candidates: list[tuple[float, int, int]] = []
            for i in range(n):
                for j in range(per_beam):
                    ranked_candidates.append(
                        (
                            active[i].log_prob + float(top_lp_cpu[i][j]),
                            i,
                            int(top_idx_cpu[i][j]),
                        )
                    )
            ranked_candidates.sort(key=lambda x: -x[0])

            for new_lp, beam_idx, tok in ranked_candidates:
                if len(new_active) >= beam_width:
                    break
                parent = active[beam_idx]
                if tok == eos:
                    surface = parent.surface
                    # An EOS is only a finishing event when the beam's
                    # current surface (without the EOS) already covers the
                    # prefix within the edit budget and is long enough.
                    if self._accept_completion(parent, surface, prefix):
                        finished.append((new_lp, surface))
                    continue
                if len(parent.token_ids) + 1 > max_len:
                    surface = parent.surface
                    if self._accept_completion(parent, surface, prefix):
                        finished.append((new_lp, surface))
                    continue
                seq = parent.token_ids + [tok]
                decoded = self.tokenizer.decode(seq[completion_offset:])
                keep, new_dp, new_matched = self._check_extend(
                    parent, decoded, prefix
                )
                if keep:
                    new_active.append(
                        _Beam(
                            seq, new_lp,
                            surface=decoded,
                            dp_row=new_dp,
                            prefix_matched=new_matched,
                        )
                    )

            active = new_active

            # Pruning: drop active beams whose log-prob is already worse
            # than the worst finished candidate by a wide margin (cheap
            # safeguard against runaway exploration).
            if len(finished) >= k * 2 and active:
                worst_finished = sorted(finished, key=lambda x: -x[0])[:k][-1][0]
                active = [b for b in active if b.log_prob > worst_finished - 1.0]

        finished.sort(key=lambda x: -x[0])
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for lp, surface in finished:
            if surface in seen:
                continue
            seen.add(surface)
            out.append((surface, float(lp)))
            if len(out) >= k:
                break
        return out

    @torch.inference_mode()
    def topk(self, prefix: str, k: int) -> list[str]:
        return [s for s, _ in self.topk_with_scores(prefix, k)]

    @torch.inference_mode()
    def topk_batch(
        self, prefixes: list[str], k: int
    ) -> list[list[str]]:
        """Batched beam search across many prefixes simultaneously.

        Active beams from every prefix in the chunk are packed into a
        single forward pass each step, so the model sees a meaningful
        batch (vs. a 1-prefix-at-a-time loop where the GPU is idle).
        """
        return [
            [s for s, _ in res]
            for res in self.topk_batch_with_scores(prefixes, k)
        ]

    @torch.inference_mode()
    def topk_batch_with_scores(
        self, prefixes: list[str], k: int
    ) -> list[list[tuple[str, float]]]:
        if not prefixes:
            return []
        results: list[list[tuple[str, float]]] = [[] for _ in prefixes]
        bs = self.prefix_batch_size
        for chunk_start in range(0, len(prefixes), bs):
            chunk = prefixes[chunk_start : chunk_start + bs]
            chunk_results = self._topk_chunk(chunk, k)
            for offset, result in enumerate(chunk_results):
                results[chunk_start + offset] = result
        return results

    @torch.inference_mode()
    def _topk_chunk(
        self, prefixes: list[str], k: int
    ) -> list[list[str]]:
        eos = self.tokenizer.eos_id
        pad = self.tokenizer.pad_id
        beam_width = self.beam_width
        per_beam = min(self.candidates_per_beam, self.tokenizer.vocab_size)
        # Model's context window. Don't shadow this inside the step loop —
        # we shadowed it as the per-step pad width once and the resulting
        # "context overflow" test fired on every candidate, killing all
        # beams at step 1.
        max_len = int(getattr(self.model.cfg, "max_seq_len", 64))

        # Per-prefix state and completion offsets.
        offsets: list[int] = [0] * len(prefixes)
        active: list[list[_Beam]] = []
        for pi, prefix in enumerate(prefixes):
            if not prefix:
                active.append([])
                continue
            init_ids, off = self._initial_state(prefix)
            # Skip prefixes whose conditioning already fills the context.
            if len(init_ids) >= max_len:
                active.append([])
                continue
            offsets[pi] = off
            active.append([
                _Beam(
                    list(init_ids), 0.0,
                    surface="",
                    dp_row=self._initial_dp_row(prefix),
                    prefix_matched=False,
                )
            ])
        finished: list[list[tuple[float, str]]] = [[] for _ in prefixes]

        for _step in range(self.max_steps):
            # Flatten all active beams into one batch.
            flat: list[tuple[int, _Beam]] = []
            for pi, beams in enumerate(active):
                for beam in beams:
                    flat.append((pi, beam))

            if not flat:
                break

            n = len(flat)
            pad_len = max(len(beam.token_ids) for _, beam in flat)
            input_ids = torch.full(
                (n, pad_len), pad, dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros(
                (n, pad_len), dtype=torch.long, device=self.device
            )
            for i, (_, beam) in enumerate(flat):
                t = len(beam.token_ids)
                input_ids[i, :t] = torch.tensor(
                    beam.token_ids, dtype=torch.long, device=self.device
                )
                attention_mask[i, :t] = 1

            logits = self.model(input_ids, attention_mask=attention_mask)
            last_idx = attention_mask.sum(dim=1) - 1
            last_logits = logits[torch.arange(n, device=self.device), last_idx]
            log_probs = F.log_softmax(last_logits.float(), dim=-1)

            top_lp, top_idx = log_probs.topk(per_beam, dim=-1)
            top_lp_cpu = top_lp.cpu().tolist()
            top_idx_cpu = top_idx.cpu().tolist()

            # Group flat indices by prefix so we apply the per-prefix
            # constraint and survivor cap independently.
            indices_by_prefix: list[list[int]] = [[] for _ in prefixes]
            for flat_idx, (pi, _beam) in enumerate(flat):
                indices_by_prefix[pi].append(flat_idx)

            for pi, prefix in enumerate(prefixes):
                if not indices_by_prefix[pi]:
                    continue
                ranked: list[tuple[float, _Beam, int]] = []
                for fi in indices_by_prefix[pi]:
                    parent = flat[fi][1]
                    parent_lp = parent.log_prob
                    for j in range(per_beam):
                        ranked.append(
                            (
                                parent_lp + float(top_lp_cpu[fi][j]),
                                parent,
                                int(top_idx_cpu[fi][j]),
                            )
                        )
                ranked.sort(key=lambda x: -x[0])

                new_active: list[_Beam] = []
                off = offsets[pi]
                for new_lp, parent, tok in ranked:
                    if len(new_active) >= beam_width:
                        break
                    if tok == eos:
                        surface = parent.surface
                        if self._accept_completion(parent, surface, prefix):
                            finished[pi].append((new_lp, surface))
                        continue
                    if len(parent.token_ids) + 1 > max_len:
                        # Force-finish at the context limit if the surface
                        # already covers the prefix; otherwise drop.
                        surface = parent.surface
                        if self._accept_completion(parent, surface, prefix):
                            finished[pi].append((new_lp, surface))
                        continue
                    seq = parent.token_ids + [tok]
                    decoded = self.tokenizer.decode(seq[off:])
                    keep, new_dp, new_matched = self._check_extend(
                        parent, decoded, prefix
                    )
                    if keep:
                        new_active.append(
                            _Beam(
                                seq, new_lp,
                                surface=decoded,
                                dp_row=new_dp,
                                prefix_matched=new_matched,
                            )
                        )

                if len(finished[pi]) >= k * 2 and new_active:
                    worst = sorted(finished[pi], key=lambda x: -x[0])[:k][-1][0]
                    new_active = [
                        b for b in new_active if b.log_prob > worst - 1.0
                    ]

                active[pi] = new_active

        out: list[list[tuple[str, float]]] = []
        for pi, prefix in enumerate(prefixes):
            if not prefix:
                out.append([])
                continue
            ranked_finished = sorted(finished[pi], key=lambda x: -x[0])
            seen: set[str] = set()
            picks: list[tuple[str, float]] = []
            for lp, surface in ranked_finished:
                if surface in seen:
                    continue
                seen.add(surface)
                picks.append((surface, float(lp)))
                if len(picks) >= k:
                    break
            out.append(picks)
        return out


def load_lm_searcher(
    ckpt_path: Path,
    tokenizer_dir: Path | None = None,
    beam_width: int = 20,
    prefix_batch_size: int = 64,
    device: str = "cuda",
    lev_budget: int = 0,
) -> SuggestBeamSearcher:
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    module = _load_lit_module(Path(ckpt_path), device=device)
    tokenizer = module.tokenizer
    if tokenizer_dir is not None:
        tokenizer = Tokenizer.load(Path(tokenizer_dir) / "spm.model")
    variant = getattr(module, "_cfg_variant", "a")
    return SuggestBeamSearcher(
        module.model,
        tokenizer,
        beam_width=beam_width,
        max_steps=int(module.lm_cfg.max_seq_len),
        prefix_batch_size=prefix_batch_size,
        device=device,
        variant=variant,
        lev_budget=lev_budget,
    )


def load_lm_topk(
    ckpt_path: Path,
    tokenizer_dir: Path | None = None,
    beam_width: int = 20,
    batch_size: int = 128,
    device: str = "cuda",
    lev_budget: int = 0,
):
    """Return a ``model_fn`` callable suitable for ``eval.evaluate``."""
    del batch_size  # only used by the batched precomputation path
    searcher = load_lm_searcher(
        ckpt_path=ckpt_path,
        tokenizer_dir=tokenizer_dir,
        beam_width=beam_width,
        device=device,
        lev_budget=lev_budget,
    )

    def topk(prefix: str, k: int) -> list[str]:
        return searcher.topk(prefix, k)

    return topk


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, default=None)
    p.add_argument("--beam-width", type=int, default=20)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--lev-budget", type=int, default=0,
        help="Max Levenshtein edits between surface and prefix during "
             "beam search (0 = exact, 1 = one typo, 2 = pricey).",
    )
    p.add_argument("prefixes", nargs="+",
                   help="Prefixes to autocomplete.")
    args = p.parse_args()

    fn = load_lm_topk(
        ckpt_path=args.ckpt,
        tokenizer_dir=args.tokenizer_dir,
        beam_width=args.beam_width,
        device=args.device,
        lev_budget=args.lev_budget,
    )
    for prefix in args.prefixes:
        t0 = time.time()
        results = fn(prefix, args.k)
        print(f"{prefix!r}  ({(time.time()-t0)*1000:.1f}ms)")
        for i, r in enumerate(results, 1):
            print(f"  {i:2d}. {r}")
        print()


if __name__ == "__main__":
    main()
